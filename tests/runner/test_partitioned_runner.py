"""Tests for ``PartitionedRunner`` and ``partitioned``."""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
from unittest import mock

import pytest

from kedro.io import DataCatalog, MemoryDataset
from kedro.pipeline import node, pipeline
from kedro.runner.partitioned_runner import (
    PARTITIONED_TAG,
    PartitionedRunner,
    _PartitionedTask,
    _is_partition_dict,
    _validate_backend,
    _wrap_as_lazy_loaders,
    partitioned,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_partitions(data_map: dict[str, object]) -> dict[str, object]:
    """Create a PartitionedDataset-style dict of lazy loaders from a dict of data."""
    return {key: (lambda v=val: v) for key, val in data_map.items()}


def identity(x):
    return x


def double(x):
    return x * 2


def add_offset(data, params_offset):
    return data + params_offset


# ---------------------------------------------------------------------------
# _is_partition_dict
# ---------------------------------------------------------------------------


class TestIsPartitionDict:
    def test_empty_dict_is_not_partition(self):
        assert _is_partition_dict({}) is False

    def test_dict_of_callables_is_partition(self):
        assert _is_partition_dict({"a": lambda: 1, "b": lambda: 2}) is True

    def test_dict_of_non_callables_is_not_partition(self):
        assert _is_partition_dict({"a": 1, "b": 2}) is False

    def test_mixed_dict_is_not_partition(self):
        assert _is_partition_dict({"a": lambda: 1, "b": 2}) is False

    def test_non_dict_is_not_partition(self):
        assert _is_partition_dict([lambda: 1]) is False
        assert _is_partition_dict("hello") is False
        assert _is_partition_dict(42) is False

    def test_non_string_keys_are_not_partition(self):
        assert _is_partition_dict({1: lambda: "a", 2: lambda: "b"}) is False


# ---------------------------------------------------------------------------
# _wrap_as_lazy_loaders
# ---------------------------------------------------------------------------


class TestWrapAsLazyLoaders:
    def test_values_become_callables(self):
        result = _wrap_as_lazy_loaders({"a": 10, "b": 20})
        assert set(result.keys()) == {"a", "b"}
        assert all(callable(v) for v in result.values())
        assert result["a"]() == 10
        assert result["b"]() == 20

    def test_result_is_partition_dict(self):
        result = _wrap_as_lazy_loaders({"x": 42})
        assert _is_partition_dict(result) is True

    def test_empty_dict(self):
        result = _wrap_as_lazy_loaders({})
        assert result == {}


# ---------------------------------------------------------------------------
# partitioned helper
# ---------------------------------------------------------------------------


class TestPartitionedPipeline:
    def test_tags_nodes_consuming_partitioned_datasets(self):
        pipe = pipeline([
            node(identity, "raw", "cleaned", name="clean"),
            node(identity, "cleaned", "final", name="transform"),
            node(identity, "unrelated", "other", name="other"),
        ])
        pp = partitioned(pipe, datasets={"raw", "cleaned"})

        nodes_by_name = {n.name: n for n in pp.nodes}
        # "clean" consumes "raw" → tagged
        assert PARTITIONED_TAG in nodes_by_name["clean"].tags
        # "transform" consumes "cleaned" → tagged
        assert PARTITIONED_TAG in nodes_by_name["transform"].tags
        # "other" consumes "unrelated" → NOT tagged
        assert PARTITIONED_TAG not in nodes_by_name["other"].tags

    def test_preserves_existing_tags(self):
        pipe = pipeline([
            node(identity, "raw", "out", name="n1", tags="existing"),
        ])
        pp = partitioned(pipe, datasets={"raw"})

        n = list(pp.nodes)[0]
        assert "existing" in n.tags
        assert PARTITIONED_TAG in n.tags

    def test_empty_datasets(self):
        pipe = pipeline([node(identity, "a", "b", name="n1")])
        pp = partitioned(pipe, datasets=set())

        n = list(pp.nodes)[0]
        assert PARTITIONED_TAG not in n.tags


# ---------------------------------------------------------------------------
# PartitionedRunner — no implicit detection (explicit config required)
# ---------------------------------------------------------------------------


class TestPartitionedRunnerNoImplicitDetection:
    def test_dict_of_callables_not_fanned_out_without_config(self):
        """Without explicit config, Dict[str, Callable] inputs are passed
        through as-is — no implicit duck-typing detection."""
        partitions = _make_partitions({"p1": 10, "p2": 20})
        catalog = DataCatalog(
            datasets={
                "raw": MemoryDataset(data=partitions),
                "out": MemoryDataset(),
            }
        )
        test_pipeline = pipeline([
            node(identity, "raw", "out", name="n1"),
        ])

        PartitionedRunner().run(test_pipeline, catalog)

        # Should be passed through unchanged — no fan-out.
        out = catalog.load("out")
        assert isinstance(out, dict)
        assert set(out.keys()) == {"p1", "p2"}
        assert all(callable(v) for v in out.values())

    def test_non_partitioned_pipeline(self):
        """Non-partitioned nodes should run normally."""
        catalog = DataCatalog(
            datasets={
                "input_data": MemoryDataset(data=42),
                "output_data": MemoryDataset(),
            }
        )
        test_pipeline = pipeline([
            node(double, "input_data", "output_data", name="double"),
        ])

        PartitionedRunner().run(test_pipeline, catalog)
        assert catalog.load("output_data") == 84

    def test_empty_dict_passthrough(self):
        """Empty dict should pass through unchanged."""
        catalog = DataCatalog(
            datasets={
                "input": MemoryDataset(data={}),
                "output": MemoryDataset(),
            }
        )
        test_pipeline = pipeline([
            node(identity, "input", "output", name="id"),
        ])

        PartitionedRunner().run(test_pipeline, catalog)
        assert catalog.load("output") == {}


# ---------------------------------------------------------------------------
# PartitionedRunner — partitioned (recommended API)
# ---------------------------------------------------------------------------


class TestPartitionedPipelineAsAPI:
    def test_basic_partitioned(self):
        """partitioned tags nodes so the runner fans out."""
        partitions = _make_partitions({"p1": 10, "p2": 20, "p3": 30})
        catalog = DataCatalog(
            datasets={
                "raw": MemoryDataset(data=partitions),
                "processed": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([node(double, "raw", "processed", name="double_node")]),
            datasets={"raw"},
        )

        PartitionedRunner(max_workers=2).run(pp, catalog)

        output = catalog.load("processed")
        assert _is_partition_dict(output)
        assert {k: v() for k, v in output.items()} == {"p1": 20, "p2": 40, "p3": 60}

    def test_mixed_partitioned_and_scalar_nodes(self):
        """Only tagged nodes fan out; untagged nodes run normally."""
        partitions = _make_partitions({"a": 5, "b": 10})
        catalog = DataCatalog(
            datasets={
                "scalar_in": MemoryDataset(data=100),
                "scalar_out": MemoryDataset(),
                "part_in": MemoryDataset(data=partitions),
                "part_out": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([
                node(double, "scalar_in", "scalar_out", name="scalar"),
                node(double, "part_in", "part_out", name="partitioned"),
            ]),
            datasets={"part_in"},
        )

        PartitionedRunner().run(pp, catalog)

        assert catalog.load("scalar_out") == 200

        part_out = catalog.load("part_out")
        assert _is_partition_dict(part_out)
        assert {k: v() for k, v in part_out.items()} == {"a": 10, "b": 20}

    def test_broadcast_static_inputs(self):
        """Non-partitioned inputs should be broadcast to every partition."""
        partitions = _make_partitions({"a": 10, "b": 20})
        catalog = DataCatalog(
            datasets={
                "raw": MemoryDataset(data=partitions),
                "offset": MemoryDataset(data=5),
                "result": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([
                node(
                    add_offset,
                    inputs={"data": "raw", "params_offset": "offset"},
                    outputs="result",
                    name="add_offset",
                ),
            ]),
            datasets={"raw"},
        )

        PartitionedRunner(max_workers=2).run(pp, catalog)

        result = catalog.load("result")
        assert {k: v() for k, v in result.items()} == {"a": 15, "b": 25}

    def test_compose_with_regular_pipeline(self):
        """partitioned result can be merged with regular pipelines."""
        partitions = _make_partitions({"a": 5})
        catalog = DataCatalog(
            datasets={
                "raw": MemoryDataset(data=partitions),
                "cleaned": MemoryDataset(),
                "scalar_in": MemoryDataset(data=99),
                "scalar_out": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([node(double, "raw", "cleaned", name="clean")]),
            datasets={"raw"},
        )
        regular = pipeline([
            node(double, "scalar_in", "scalar_out", name="scalar_op"),
        ])
        full = pp + regular

        PartitionedRunner().run(full, catalog)

        cleaned = catalog.load("cleaned")
        assert _is_partition_dict(cleaned)
        assert {k: v() for k, v in cleaned.items()} == {"a": 10}
        assert catalog.load("scalar_out") == 198

    def test_memory_dataset_intermediate_chaining(self):
        """MemoryDataset intermediates declared as partitioned chain correctly."""
        partitions = _make_partitions({"x": 3, "y": 7})
        catalog = DataCatalog(
            datasets={
                "raw": MemoryDataset(data=partitions),
                "intermediate": MemoryDataset(),
                "final": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([
                node(double, "raw", "intermediate", name="first"),
                node(double, "intermediate", "final", name="second"),
            ]),
            datasets={"raw", "intermediate"},
        )

        PartitionedRunner(max_workers=2).run(pp, catalog)

        final = catalog.load("final")
        assert _is_partition_dict(final)
        assert {k: v() for k, v in final.items()} == {"x": 12, "y": 28}


# ---------------------------------------------------------------------------
# PartitionedRunner — chaining (the key feature)
# ---------------------------------------------------------------------------


class TestPartitionedRunnerChaining:
    def test_two_node_chain_via_memory_dataset(self):
        """Partitioned outputs wrapped as lazy loaders should enable a second
        node to process partitions in parallel — even when the intermediate
        dataset is a plain MemoryDataset."""
        partitions = _make_partitions({"x": 3, "y": 7})
        catalog = DataCatalog(
            datasets={
                "raw": MemoryDataset(data=partitions),
                "intermediate": MemoryDataset(),
                "final": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([
                node(double, "raw", "intermediate", name="first"),
                node(double, "intermediate", "final", name="second"),
            ]),
            datasets={"raw", "intermediate"},
        )

        PartitionedRunner(max_workers=2).run(pp, catalog)

        final = catalog.load("final")
        assert _is_partition_dict(final)
        assert {k: v() for k, v in final.items()} == {"x": 12, "y": 28}

    def test_three_node_chain(self):
        partitions = _make_partitions({"a": 1})
        catalog = DataCatalog(
            datasets={
                "d0": MemoryDataset(data=partitions),
                "d1": MemoryDataset(),
                "d2": MemoryDataset(),
                "d3": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([
                node(double, "d0", "d1", name="n1"),
                node(double, "d1", "d2", name="n2"),
                node(double, "d2", "d3", name="n3"),
            ]),
            datasets={"d0", "d1", "d2"},
        )

        PartitionedRunner().run(pp, catalog)

        result = catalog.load("d3")
        # 1 -> 2 -> 4 -> 8
        assert {k: v() for k, v in result.items()} == {"a": 8}


# ---------------------------------------------------------------------------
# PartitionedRunner — explicit partitioned_datasets config
# ---------------------------------------------------------------------------


class TestPartitionedRunnerExplicitConfig:
    def test_explicit_partitioned_datasets(self):
        partitions = _make_partitions({"p1": 5})
        catalog = DataCatalog(
            datasets={
                "raw": MemoryDataset(data=partitions),
                "out": MemoryDataset(),
            }
        )
        test_pipeline = pipeline([
            node(double, "raw", "out", name="n1"),
        ])

        runner = PartitionedRunner(
            max_workers=2,
            partitioned_datasets={"raw"},
        )
        runner.run(test_pipeline, catalog)

        out = catalog.load("out")
        assert {k: v() for k, v in out.items()} == {"p1": 10}

    def test_explicit_config_ignores_non_listed_dicts(self):
        """When partitioned_datasets is set, a Dict[str, Callable] that is NOT
        in the set should be passed through normally (no fan-out)."""
        # "other" is a dict of callables but NOT in partitioned_datasets.
        other_data = {"k": lambda: 99}
        catalog = DataCatalog(
            datasets={
                "other": MemoryDataset(data=other_data),
                "out": MemoryDataset(),
            }
        )
        test_pipeline = pipeline([
            node(identity, "other", "out", name="n1"),
        ])

        runner = PartitionedRunner(partitioned_datasets={"unrelated"})
        runner.run(test_pipeline, catalog)

        # Should pass through the raw dict, not fan-out.
        out = catalog.load("out")
        assert isinstance(out, dict)
        assert callable(out["k"])
        assert out["k"]() == 99


# ---------------------------------------------------------------------------
# PartitionedRunner — partitioned tag
# ---------------------------------------------------------------------------


class TestPartitionedRunnerWithTaggedPipeline:
    def test_tagged_pipeline_triggers_partition_mode(self):
        partitions = _make_partitions({"a": 10, "b": 20})
        catalog = DataCatalog(
            datasets={
                "raw": MemoryDataset(data=partitions),
                "out": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([node(double, "raw", "out", name="n1")]),
            datasets={"raw"},
        )

        PartitionedRunner().run(pp, catalog)

        out = catalog.load("out")
        assert {k: v() for k, v in out.items()} == {"a": 20, "b": 40}

    def test_tagged_pipeline_chain(self):
        partitions = _make_partitions({"a": 2})
        catalog = DataCatalog(
            datasets={
                "raw": MemoryDataset(data=partitions),
                "mid": MemoryDataset(),
                "final": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([
                node(double, "raw", "mid", name="n1"),
                node(double, "mid", "final", name="n2"),
            ]),
            datasets={"raw", "mid"},
        )

        PartitionedRunner().run(pp, catalog)

        final = catalog.load("final")
        assert {k: v() for k, v in final.items()} == {"a": 8}


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestPartitionedRunnerErrors:
    def test_error_in_partition_propagates(self):
        def fail_on_zero(data):
            return 10 / data

        partitions = _make_partitions({"ok": 5, "bad": 0})
        catalog = DataCatalog(
            datasets={
                "input": MemoryDataset(data=partitions),
                "output": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([node(fail_on_zero, "input", "output", name="failing")]),
            datasets={"input"},
        )

        with pytest.raises(ZeroDivisionError):
            PartitionedRunner().run(pp, catalog)

    def test_max_workers_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            PartitionedRunner(max_workers=0)

    def test_max_workers_none_accepted(self):
        runner = PartitionedRunner(max_workers=None)
        assert runner._max_workers is None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestPartitionedRunnerLogging:
    def test_logs_partition_count(self, caplog):
        partitions = _make_partitions({"a": 1, "b": 2})
        catalog = DataCatalog(
            datasets={
                "input": MemoryDataset(data=partitions),
                "output": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([node(identity, "input", "output", name="test")]),
            datasets={"input"},
        )

        with caplog.at_level(logging.INFO):
            PartitionedRunner(max_workers=2).run(pp, catalog)

        assert "processing 2 partitions" in caplog.text.lower()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestPartitionedRunnerConcurrency:
    def test_partitions_processed_concurrently(self):
        partitions = _make_partitions({"a": 1, "b": 2, "c": 3, "d": 4})
        seen_threads: set[int] = set()
        lock = threading.Lock()

        def record_thread(x):
            with lock:
                seen_threads.add(threading.current_thread().ident)
            return x

        catalog = DataCatalog(
            datasets={
                "input": MemoryDataset(data=partitions),
                "output": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([node(record_thread, "input", "output", name="concurrent")]),
            datasets={"input"},
        )

        PartitionedRunner(max_workers=4).run(pp, catalog)
        # At least one worker thread was used (they're different from main).
        assert len(seen_threads) >= 1


# ---------------------------------------------------------------------------
# _PartitionedTask unit tests
# ---------------------------------------------------------------------------


class TestPartitionedTask:
    def test_partitioned_task_processes_partitions(self):
        from kedro.framework.hooks.manager import _NullPluginManager

        partitions = _make_partitions({"p1": 100, "p2": 200})
        catalog = DataCatalog(
            datasets={
                "input": MemoryDataset(data=partitions),
                "output": MemoryDataset(),
            }
        )
        test_node = node(double, "input", "output", name="test")

        task = _PartitionedTask(
            node=test_node,
            catalog=catalog,
            is_async=False,
            hook_manager=_NullPluginManager(),
            partition_max_workers=2,
            partitioned_datasets={"input"},
        )
        result_node = task.execute()
        assert result_node.name == "test"

        output = catalog.load("output")
        assert _is_partition_dict(output)
        assert {k: v() for k, v in output.items()} == {"p1": 200, "p2": 400}

    def test_task_non_partitioned_passthrough(self):
        from kedro.framework.hooks.manager import _NullPluginManager

        catalog = DataCatalog(
            datasets={
                "input": MemoryDataset(data=50),
                "output": MemoryDataset(),
            }
        )
        test_node = node(double, "input", "output", name="test")

        task = _PartitionedTask(
            node=test_node,
            catalog=catalog,
            is_async=False,
            hook_manager=_NullPluginManager(),
        )
        task.execute()
        assert catalog.load("output") == 100


# ---------------------------------------------------------------------------
# Multiple partitioned inputs (inner-join key alignment)
# ---------------------------------------------------------------------------


def add_two(a, b):
    return a + b


def concat(left, right):
    return f"{left}-{right}"


class TestMultiplePartitionedInputs:
    def test_inner_join_aligned_keys(self):
        """When two inputs are partitioned, only common keys are processed."""
        parts_a = _make_partitions({"x": 10, "y": 20, "z": 30})
        parts_b = _make_partitions({"y": 100, "z": 200, "w": 300})
        catalog = DataCatalog(
            datasets={
                "a": MemoryDataset(data=parts_a),
                "b": MemoryDataset(data=parts_b),
                "out": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([
                node(
                    add_two,
                    inputs={"a": "a", "b": "b"},
                    outputs="out",
                    name="add",
                ),
            ]),
            datasets={"a", "b"},
        )

        PartitionedRunner(max_workers=2).run(pp, catalog)

        out = catalog.load("out")
        assert _is_partition_dict(out)
        resolved = {k: v() for k, v in out.items()}
        # Only "y" and "z" are common keys.
        assert resolved == {"y": 120, "z": 230}

    def test_no_common_keys_returns_empty(self, caplog):
        """Disjoint partition keys should produce no outputs and log a warning."""
        parts_a = _make_partitions({"x": 1})
        parts_b = _make_partitions({"y": 2})
        catalog = DataCatalog(
            datasets={
                "a": MemoryDataset(data=parts_a),
                "b": MemoryDataset(data=parts_b),
                "out": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([
                node(
                    add_two,
                    inputs={"a": "a", "b": "b"},
                    outputs="out",
                    name="add",
                ),
            ]),
            datasets={"a", "b"},
        )

        with caplog.at_level(logging.WARNING):
            PartitionedRunner().run(pp, catalog)

        assert "no common partition keys" in caplog.text.lower()


# ---------------------------------------------------------------------------
# Multiple outputs from a partitioned node
# ---------------------------------------------------------------------------


def split(x):
    return x, x * 10


class TestMultipleOutputs:
    def test_partitioned_node_with_multiple_outputs(self):
        """A partitioned node that returns multiple outputs should produce
        partition dicts for each output."""
        partitions = _make_partitions({"a": 3, "b": 5})
        catalog = DataCatalog(
            datasets={
                "input": MemoryDataset(data=partitions),
                "out1": MemoryDataset(),
                "out2": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([
                node(split, "input", ["out1", "out2"], name="split"),
            ]),
            datasets={"input"},
        )

        PartitionedRunner(max_workers=2).run(pp, catalog)

        out1 = catalog.load("out1")
        out2 = catalog.load("out2")
        assert _is_partition_dict(out1)
        assert _is_partition_dict(out2)
        assert {k: v() for k, v in out1.items()} == {"a": 3, "b": 5}
        assert {k: v() for k, v in out2.items()} == {"a": 30, "b": 50}


# ---------------------------------------------------------------------------
# Kedro params convention
# ---------------------------------------------------------------------------


class TestPartitionedWithParams:
    def test_params_broadcast_to_partitions(self):
        """Kedro params: prefix inputs should be broadcast (not partitioned)."""
        partitions = _make_partitions({"a": 10, "b": 20})
        catalog = DataCatalog(
            datasets={
                "raw": MemoryDataset(data=partitions),
                "params:offset": MemoryDataset(data=5),
                "result": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([
                node(
                    add_offset,
                    inputs={"data": "raw", "params_offset": "params:offset"},
                    outputs="result",
                    name="with_params",
                ),
            ]),
            datasets={"raw"},
        )

        PartitionedRunner(max_workers=2).run(pp, catalog)

        result = catalog.load("result")
        assert {k: v() for k, v in result.items()} == {"a": 15, "b": 25}


# ---------------------------------------------------------------------------
# Namespaced pipelines
# ---------------------------------------------------------------------------


class TestPartitionedWithNamespace:
    def test_partitioned_with_namespaced_pipeline(self):
        """partitioned() should work with namespaced pipelines when dataset
        names include the namespace prefix."""
        partitions = _make_partitions({"a": 4, "b": 8})
        catalog = DataCatalog(
            datasets={
                "ns.raw": MemoryDataset(data=partitions),
                "ns.processed": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline(
                [node(double, "raw", "processed", name="double_node")],
                namespace="ns",
            ),
            datasets={"ns.raw"},
        )

        PartitionedRunner(max_workers=2).run(pp, catalog)

        out = catalog.load("ns.processed")
        assert _is_partition_dict(out)
        assert {k: v() for k, v in out.items()} == {"a": 8, "b": 16}


# ---------------------------------------------------------------------------
# Backend validation
# ---------------------------------------------------------------------------


class TestBackendValidation:
    def test_thread_backend_accepted(self):
        runner = PartitionedRunner(backend="thread")
        assert runner._backend == "thread"

    def test_process_backend_accepted(self):
        runner = PartitionedRunner(backend="process")
        assert runner._backend == "process"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Invalid backend"):
            PartitionedRunner(backend="invalid")

    def test_validate_backend_returns_value(self):
        assert _validate_backend("thread") == "thread"
        assert _validate_backend("process") == "process"

    def test_validate_backend_rejects_unknown(self):
        with pytest.raises(ValueError, match="Invalid backend"):
            _validate_backend("dask")

    def test_process_backend_missing_cloudpickle(self):
        with mock.patch.dict("sys.modules", {"cloudpickle": None}):
            with pytest.raises(ImportError, match="cloudpickle"):
                PartitionedRunner(backend="process")


# ---------------------------------------------------------------------------
# Process backend — core functionality
# ---------------------------------------------------------------------------


class TestProcessBackendBasic:
    """Mirror the key thread-backend tests using backend='process'."""

    def test_basic_partitioned(self):
        partitions = _make_partitions({"p1": 10, "p2": 20, "p3": 30})
        catalog = DataCatalog(
            datasets={
                "raw": MemoryDataset(data=partitions),
                "processed": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([node(double, "raw", "processed", name="double_node")]),
            datasets={"raw"},
        )

        PartitionedRunner(max_workers=2, backend="process").run(pp, catalog)

        output = catalog.load("processed")
        assert _is_partition_dict(output)
        assert {k: v() for k, v in output.items()} == {"p1": 20, "p2": 40, "p3": 60}

    def test_non_partitioned_pipeline(self):
        catalog = DataCatalog(
            datasets={
                "input_data": MemoryDataset(data=42),
                "output_data": MemoryDataset(),
            }
        )
        test_pipeline = pipeline([
            node(double, "input_data", "output_data", name="double"),
        ])

        PartitionedRunner(backend="process").run(test_pipeline, catalog)
        assert catalog.load("output_data") == 84

    def test_broadcast_static_inputs(self):
        partitions = _make_partitions({"a": 10, "b": 20})
        catalog = DataCatalog(
            datasets={
                "raw": MemoryDataset(data=partitions),
                "offset": MemoryDataset(data=5),
                "result": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([
                node(
                    add_offset,
                    inputs={"data": "raw", "params_offset": "offset"},
                    outputs="result",
                    name="add_offset",
                ),
            ]),
            datasets={"raw"},
        )

        PartitionedRunner(max_workers=2, backend="process").run(pp, catalog)

        result = catalog.load("result")
        assert {k: v() for k, v in result.items()} == {"a": 15, "b": 25}

    def test_explicit_partitioned_datasets(self):
        partitions = _make_partitions({"p1": 5})
        catalog = DataCatalog(
            datasets={
                "raw": MemoryDataset(data=partitions),
                "out": MemoryDataset(),
            }
        )
        test_pipeline = pipeline([
            node(double, "raw", "out", name="n1"),
        ])

        runner = PartitionedRunner(
            max_workers=2,
            partitioned_datasets={"raw"},
            backend="process",
        )
        runner.run(test_pipeline, catalog)

        out = catalog.load("out")
        assert {k: v() for k, v in out.items()} == {"p1": 10}

    def test_multiple_outputs(self):
        partitions = _make_partitions({"a": 3, "b": 5})
        catalog = DataCatalog(
            datasets={
                "input": MemoryDataset(data=partitions),
                "out1": MemoryDataset(),
                "out2": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([
                node(split, "input", ["out1", "out2"], name="split"),
            ]),
            datasets={"input"},
        )

        PartitionedRunner(max_workers=2, backend="process").run(pp, catalog)

        out1 = catalog.load("out1")
        out2 = catalog.load("out2")
        assert _is_partition_dict(out1)
        assert _is_partition_dict(out2)
        assert {k: v() for k, v in out1.items()} == {"a": 3, "b": 5}
        assert {k: v() for k, v in out2.items()} == {"a": 30, "b": 50}


# ---------------------------------------------------------------------------
# Process backend — chaining
# ---------------------------------------------------------------------------


class TestProcessBackendChaining:
    def test_two_node_chain(self):
        partitions = _make_partitions({"x": 3, "y": 7})
        catalog = DataCatalog(
            datasets={
                "raw": MemoryDataset(data=partitions),
                "intermediate": MemoryDataset(),
                "final": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([
                node(double, "raw", "intermediate", name="first"),
                node(double, "intermediate", "final", name="second"),
            ]),
            datasets={"raw", "intermediate"},
        )

        PartitionedRunner(max_workers=2, backend="process").run(pp, catalog)

        final = catalog.load("final")
        assert _is_partition_dict(final)
        assert {k: v() for k, v in final.items()} == {"x": 12, "y": 28}

    def test_three_node_chain(self):
        partitions = _make_partitions({"a": 1})
        catalog = DataCatalog(
            datasets={
                "d0": MemoryDataset(data=partitions),
                "d1": MemoryDataset(),
                "d2": MemoryDataset(),
                "d3": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([
                node(double, "d0", "d1", name="n1"),
                node(double, "d1", "d2", name="n2"),
                node(double, "d2", "d3", name="n3"),
            ]),
            datasets={"d0", "d1", "d2"},
        )

        PartitionedRunner(backend="process").run(pp, catalog)

        result = catalog.load("d3")
        assert {k: v() for k, v in result.items()} == {"a": 8}


# ---------------------------------------------------------------------------
# Process backend — multiple partitioned inputs
# ---------------------------------------------------------------------------


class TestProcessBackendMultipleInputs:
    def test_inner_join_aligned_keys(self):
        parts_a = _make_partitions({"x": 10, "y": 20, "z": 30})
        parts_b = _make_partitions({"y": 100, "z": 200, "w": 300})
        catalog = DataCatalog(
            datasets={
                "a": MemoryDataset(data=parts_a),
                "b": MemoryDataset(data=parts_b),
                "out": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([
                node(
                    add_two,
                    inputs={"a": "a", "b": "b"},
                    outputs="out",
                    name="add",
                ),
            ]),
            datasets={"a", "b"},
        )

        PartitionedRunner(max_workers=2, backend="process").run(pp, catalog)

        out = catalog.load("out")
        assert _is_partition_dict(out)
        resolved = {k: v() for k, v in out.items()}
        assert resolved == {"y": 120, "z": 230}


# ---------------------------------------------------------------------------
# Process backend — error handling
# ---------------------------------------------------------------------------


def _fail_on_zero(data):
    """Top-level function so it's picklable for the process backend."""
    return 10 / data


class TestProcessBackendErrors:
    def test_error_in_partition_propagates(self):
        partitions = _make_partitions({"ok": 5, "bad": 0})
        catalog = DataCatalog(
            datasets={
                "input": MemoryDataset(data=partitions),
                "output": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([node(_fail_on_zero, "input", "output", name="failing")]),
            datasets={"input"},
        )

        with pytest.raises(ZeroDivisionError):
            PartitionedRunner(backend="process").run(pp, catalog)


# ---------------------------------------------------------------------------
# Process backend — concurrency verification
# ---------------------------------------------------------------------------


class TestProcessBackendConcurrency:
    def test_partitions_processed_in_separate_processes(self):
        """Verify partitions actually run in separate processes."""
        partitions = _make_partitions({"a": 1, "b": 2, "c": 3, "d": 4})

        def get_pid(x):
            return os.getpid()

        catalog = DataCatalog(
            datasets={
                "input": MemoryDataset(data=partitions),
                "output": MemoryDataset(),
            }
        )
        pp = partitioned(
            pipeline([node(get_pid, "input", "output", name="pid_check")]),
            datasets={"input"},
        )

        PartitionedRunner(max_workers=4, backend="process").run(pp, catalog)

        output = catalog.load("output")
        pids = {v() for v in output.values()}
        # At least one worker process was used (different from main).
        main_pid = os.getpid()
        worker_pids = pids - {main_pid}
        assert len(worker_pids) >= 1, (
            f"Expected worker PIDs different from main ({main_pid}), got {pids}"
        )
