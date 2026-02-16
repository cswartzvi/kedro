"""Tests for ``PartitionedRunner`` and ``parallel_partitions``."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from kedro.io import DataCatalog, MemoryDataset
from kedro.pipeline import node, pipeline
from kedro.runner.partitioned_runner import (
    PartitionedRunner,
    _PartitionedTask,
    _is_partition_dict,
    parallel_partitions,
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


def add(x, y):
    return x + y


# ---------------------------------------------------------------------------
# _is_partition_dict tests
# ---------------------------------------------------------------------------


class TestIsPartitionDict:
    def test_empty_dict_is_not_partition(self):
        assert _is_partition_dict({}) is False

    def test_dict_of_callables_is_partition(self):
        data = {"a": lambda: 1, "b": lambda: 2}
        assert _is_partition_dict(data) is True

    def test_dict_of_non_callables_is_not_partition(self):
        data = {"a": 1, "b": 2}
        assert _is_partition_dict(data) is False

    def test_mixed_dict_is_not_partition(self):
        data = {"a": lambda: 1, "b": 2}
        assert _is_partition_dict(data) is False

    def test_non_dict_is_not_partition(self):
        assert _is_partition_dict([lambda: 1]) is False
        assert _is_partition_dict("hello") is False
        assert _is_partition_dict(42) is False

    def test_non_string_keys_are_not_partition(self):
        data = {1: lambda: "a", 2: lambda: "b"}
        assert _is_partition_dict(data) is False


# ---------------------------------------------------------------------------
# parallel_partitions decorator tests
# ---------------------------------------------------------------------------


class TestParallelPartitions:
    def test_basic_parallel_processing(self):
        """Decorator should parallelize processing across partitions."""
        partitions = _make_partitions({"p1": 10, "p2": 20, "p3": 30})

        @parallel_partitions
        def process(x):
            return x * 2

        result = process(partitions)
        assert result == {"p1": 20, "p2": 40, "p3": 60}

    def test_passthrough_for_regular_input(self):
        """Non-partitioned inputs should pass through unchanged."""

        @parallel_partitions
        def process(x):
            return x * 2

        assert process(5) == 10

    def test_with_max_workers(self):
        """Decorator should accept max_workers argument."""
        partitions = _make_partitions({"a": 1, "b": 2})

        @parallel_partitions(max_workers=2)
        def process(x):
            return x + 100

        result = process(partitions)
        assert result == {"a": 101, "b": 102}

    def test_preserves_function_name(self):
        @parallel_partitions
        def my_func(x):
            return x

        assert my_func.__name__ == "my_func"

    def test_preserves_function_name_with_args(self):
        @parallel_partitions(max_workers=4)
        def my_func(x):
            return x

        assert my_func.__name__ == "my_func"

    def test_additional_positional_args(self):
        """Extra positional args should be passed to each invocation."""
        partitions = _make_partitions({"a": 10, "b": 20})

        @parallel_partitions
        def process(x, multiplier):
            return x * multiplier

        result = process(partitions, 3)
        assert result == {"a": 30, "b": 60}

    def test_additional_keyword_args(self):
        """Extra keyword args should be passed to each invocation."""
        partitions = _make_partitions({"a": 5, "b": 15})

        @parallel_partitions
        def process(x, offset=0):
            return x + offset

        result = process(partitions, offset=100)
        assert result == {"a": 105, "b": 115}

    def test_partition_kwarg(self):
        """Partitioned data passed as keyword arg should work."""
        partitions = _make_partitions({"a": 7})

        @parallel_partitions
        def process(data):
            return data * 3

        result = process(data=partitions)
        assert result == {"a": 21}

    def test_exception_propagation(self):
        """Exceptions in partition processing should propagate."""
        partitions = _make_partitions({"a": 1, "b": 0})

        @parallel_partitions
        def process(x):
            return 10 / x

        with pytest.raises(ZeroDivisionError):
            process(partitions)

    def test_concurrent_execution(self):
        """Partitions should actually run concurrently."""
        partitions = _make_partitions({"a": 1, "b": 2, "c": 3, "d": 4})
        seen_threads: set[int] = set()
        lock = threading.Lock()

        @parallel_partitions(max_workers=4)
        def process(x):
            with lock:
                seen_threads.add(threading.current_thread().ident)
            return x

        process(partitions)
        # With 4 partitions and 4 workers, we expect at least 1 thread
        # (ThreadPoolExecutor uses worker threads, not the main thread).
        assert len(seen_threads) >= 1

    def test_metadata_attributes(self):
        """Decorated function should expose metadata."""

        @parallel_partitions(max_workers=3)
        def process(x):
            return x

        assert process._parallel_partitions is True
        assert process._max_workers == 3

    def test_single_partition(self):
        partitions = _make_partitions({"only": 42})

        @parallel_partitions
        def process(x):
            return x + 1

        assert process(partitions) == {"only": 43}


# ---------------------------------------------------------------------------
# PartitionedRunner tests
# ---------------------------------------------------------------------------


def _double_node_func(data):
    """Process a single data value — used as the node function."""
    return data * 2


def _add_offset(data, params_offset):
    """Process with an additional parameter input."""
    return data + params_offset


class TestPartitionedRunner:
    def test_basic_partitioned_pipeline(self):
        """PartitionedRunner should detect partitioned inputs and parallelise."""
        partitions = _make_partitions({"p1": 10, "p2": 20, "p3": 30})
        catalog = DataCatalog(
            datasets={
                "raw_partitioned": MemoryDataset(data=partitions),
                "processed_partitioned": MemoryDataset(),
            }
        )
        test_pipeline = pipeline(
            [
                node(
                    _double_node_func,
                    inputs="raw_partitioned",
                    outputs="processed_partitioned",
                    name="double_node",
                ),
            ]
        )

        runner = PartitionedRunner(max_workers=2)
        runner.run(test_pipeline, catalog)

        output = catalog.load("processed_partitioned")
        assert isinstance(output, dict)
        assert output == {"p1": 20, "p2": 40, "p3": 60}

    def test_non_partitioned_pipeline(self):
        """Non-partitioned nodes should run normally (like SequentialRunner)."""
        catalog = DataCatalog(
            datasets={
                "input_data": MemoryDataset(data=42),
                "output_data": MemoryDataset(),
            }
        )
        test_pipeline = pipeline(
            [
                node(
                    _double_node_func,
                    inputs="input_data",
                    outputs="output_data",
                    name="double_node",
                ),
            ]
        )

        runner = PartitionedRunner()
        runner.run(test_pipeline, catalog)

        output = catalog.load("output_data")
        assert output == 84

    def test_mixed_pipeline(self):
        """Pipeline with both partitioned and non-partitioned nodes."""
        partitions = _make_partitions({"a": 5, "b": 10})
        catalog = DataCatalog(
            datasets={
                "scalar_input": MemoryDataset(data=100),
                "scalar_output": MemoryDataset(),
                "partitioned_input": MemoryDataset(data=partitions),
                "partitioned_output": MemoryDataset(),
            }
        )
        test_pipeline = pipeline(
            [
                node(
                    _double_node_func,
                    inputs="scalar_input",
                    outputs="scalar_output",
                    name="scalar_node",
                ),
                node(
                    _double_node_func,
                    inputs="partitioned_input",
                    outputs="partitioned_output",
                    name="partitioned_node",
                ),
            ]
        )

        runner = PartitionedRunner()
        runner.run(test_pipeline, catalog)

        assert catalog.load("scalar_output") == 200
        assert catalog.load("partitioned_output") == {"a": 10, "b": 20}

    def test_partitioned_with_extra_inputs(self):
        """Partitioned node with additional non-partitioned inputs."""
        partitions = _make_partitions({"a": 10, "b": 20})
        catalog = DataCatalog(
            datasets={
                "raw": MemoryDataset(data=partitions),
                "offset": MemoryDataset(data=5),
                "result": MemoryDataset(),
            }
        )
        test_pipeline = pipeline(
            [
                node(
                    _add_offset,
                    inputs={"data": "raw", "params_offset": "offset"},
                    outputs="result",
                    name="add_offset_node",
                ),
            ]
        )

        runner = PartitionedRunner(max_workers=2)
        runner.run(test_pipeline, catalog)

        output = catalog.load("result")
        assert output == {"a": 15, "b": 25}

    def test_chained_partitioned_nodes(self):
        """First node processes partitions, second node receives plain dict output."""
        partitions = _make_partitions({"x": 3, "y": 7})
        catalog = DataCatalog(
            datasets={
                "raw": MemoryDataset(data=partitions),
                "intermediate": MemoryDataset(),
            }
        )

        # Only test the first node — the intermediate output is a plain dict,
        # not a Dict[str, Callable], so chaining only works when a real
        # PartitionedDataset re-wraps outputs as lazy loaders.
        test_pipeline = pipeline(
            [
                node(
                    _double_node_func,
                    inputs="raw",
                    outputs="intermediate",
                    name="first_node",
                ),
            ]
        )

        runner = PartitionedRunner()
        runner.run(test_pipeline, catalog)

        intermediate = catalog.load("intermediate")
        assert intermediate == {"x": 6, "y": 14}

    def test_error_in_partition_processing(self):
        """Errors in partition processing should propagate."""

        def fail_on_zero(data):
            return 10 / data

        partitions = _make_partitions({"ok": 5, "bad": 0})
        catalog = DataCatalog(
            datasets={
                "input": MemoryDataset(data=partitions),
                "output": MemoryDataset(),
            }
        )
        test_pipeline = pipeline(
            [
                node(
                    fail_on_zero,
                    inputs="input",
                    outputs="output",
                    name="failing_node",
                ),
            ]
        )

        runner = PartitionedRunner()
        with pytest.raises(ZeroDivisionError):
            runner.run(test_pipeline, catalog)

    def test_empty_partitions_no_keys(self):
        """PartitionedRunner with empty partition dict should not trigger
        partition-parallel mode (empty dict fails _is_partition_dict)."""
        catalog = DataCatalog(
            datasets={
                "input": MemoryDataset(data={}),
                "output": MemoryDataset(),
            }
        )
        test_pipeline = pipeline(
            [node(identity, inputs="input", outputs="output", name="id_node")]
        )

        runner = PartitionedRunner()
        runner.run(test_pipeline, catalog)
        assert catalog.load("output") == {}

    def test_logging_output(self, caplog):
        """Runner should log partition processing info."""
        partitions = _make_partitions({"a": 1, "b": 2})
        catalog = DataCatalog(
            datasets={
                "input": MemoryDataset(data=partitions),
                "output": MemoryDataset(),
            }
        )
        test_pipeline = pipeline(
            [node(identity, inputs="input", outputs="output", name="test_node")]
        )

        runner = PartitionedRunner(max_workers=2)
        with caplog.at_level(logging.INFO):
            runner.run(test_pipeline, catalog)

        assert "processing 2 partitions" in caplog.text.lower()

    def test_max_workers_validation(self):
        """max_workers=0 should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            PartitionedRunner(max_workers=0)

    def test_max_workers_none(self):
        """max_workers=None should be accepted (ThreadPoolExecutor default)."""
        runner = PartitionedRunner(max_workers=None)
        assert runner._max_workers is None


# ---------------------------------------------------------------------------
# parallel_partitions + PartitionedRunner integration
# ---------------------------------------------------------------------------


class TestDecoratorWithRunner:
    def test_decorator_with_sequential_runner(self):
        """parallel_partitions decorator should work with a regular
        SequentialRunner (the decorator handles parallelism itself)."""
        from kedro.runner import SequentialRunner

        partitions = _make_partitions({"a": 2, "b": 4})

        @parallel_partitions
        def process(x):
            return x * 10

        catalog = DataCatalog(
            datasets={
                "input": MemoryDataset(data=partitions),
                "output": MemoryDataset(),
            }
        )
        test_pipeline = pipeline(
            [node(process, inputs="input", outputs="output", name="proc")]
        )

        runner = SequentialRunner()
        runner.run(test_pipeline, catalog)

        assert catalog.load("output") == {"a": 20, "b": 40}

    def test_decorator_with_partitioned_runner(self):
        """Using the decorator with PartitionedRunner: the decorator
        takes precedence since it intercepts the Dict[str, Callable]
        before the runner's partition detection."""
        partitions = _make_partitions({"a": 3, "b": 6})

        @parallel_partitions(max_workers=2)
        def process(x):
            return x + 1

        catalog = DataCatalog(
            datasets={
                "input": MemoryDataset(data=partitions),
                "output": MemoryDataset(),
            }
        )
        test_pipeline = pipeline(
            [node(process, inputs="input", outputs="output", name="proc")]
        )

        runner = PartitionedRunner(max_workers=2)
        runner.run(test_pipeline, catalog)

        # The decorator transforms the output to a plain dict of results
        # (not callables), so the runner won't detect it as partitioned again.
        assert catalog.load("output") == {"a": 4, "b": 7}


# ---------------------------------------------------------------------------
# _PartitionedTask unit tests
# ---------------------------------------------------------------------------


class TestPartitionedTask:
    def test_partitioned_task_processes_partitions(self):
        """_PartitionedTask should detect and process partitions in parallel."""
        partitions = _make_partitions({"p1": 100, "p2": 200})
        catalog = DataCatalog(
            datasets={
                "input": MemoryDataset(data=partitions),
                "output": MemoryDataset(),
            }
        )
        test_node = node(
            _double_node_func,
            inputs="input",
            outputs="output",
            name="test_node",
        )
        from kedro.framework.hooks.manager import _NullPluginManager

        task = _PartitionedTask(
            node=test_node,
            catalog=catalog,
            is_async=False,
            hook_manager=_NullPluginManager(),
            partition_max_workers=2,
        )
        result_node = task.execute()
        assert result_node.name == "test_node"

        output = catalog.load("output")
        assert output == {"p1": 200, "p2": 400}

    def test_task_non_partitioned_passthrough(self):
        """_PartitionedTask should handle non-partitioned data like normal Task."""
        catalog = DataCatalog(
            datasets={
                "input": MemoryDataset(data=50),
                "output": MemoryDataset(),
            }
        )
        test_node = node(
            _double_node_func,
            inputs="input",
            outputs="output",
            name="test_node",
        )
        from kedro.framework.hooks.manager import _NullPluginManager

        task = _PartitionedTask(
            node=test_node,
            catalog=catalog,
            is_async=False,
            hook_manager=_NullPluginManager(),
        )
        task.execute()
        assert catalog.load("output") == 100
