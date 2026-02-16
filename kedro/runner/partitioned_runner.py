"""``PartitionedRunner`` and ``parallel_partitions`` provide parallel
processing of PartitionedDataset partitions within Kedro pipelines.

This module addresses a key limitation in Kedro: pipelines are static DAGs,
so there is no way to dynamically create N parallel nodes for N partitions
at runtime. Instead, a single node receives all partitions as a
``Dict[str, Callable]`` and must iterate them sequentially.

This module provides two complementary approaches:

1. ``parallel_partitions`` decorator: Transforms a per-partition function
   into one that loads and processes all partitions concurrently. Works
   with any runner.

2. ``PartitionedRunner``: A runner that automatically detects partitioned
   inputs (``Dict[str, Callable]``) and parallelizes partition loading
   and processing within each node using a thread pool.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

from kedro.runner.runner import AbstractRunner
from kedro.runner.task import Task

if TYPE_CHECKING:
    from pluggy import PluginManager

    from kedro.io import CatalogProtocol
    from kedro.pipeline import Pipeline
    from kedro.pipeline.node import Node

logger = logging.getLogger(__name__)


def _is_partition_dict(data: Any) -> bool:
    """Detect whether data matches the PartitionedDataset.load() signature.

    PartitionedDataset.load() returns ``Dict[str, Callable]`` where each
    value is a lazy-loading function for that partition.  We use a duck-typing
    check: a non-empty dict whose keys are all strings and values are all
    callable.

    We require at least one entry to avoid false positives on empty dicts.
    """
    return (
        isinstance(data, dict)
        and len(data) > 0
        and all(isinstance(k, str) for k in data)
        and all(callable(v) for v in data.values())
    )


# ---------------------------------------------------------------------------
# parallel_partitions decorator
# ---------------------------------------------------------------------------


def parallel_partitions(
    func: Callable | None = None,
    *,
    max_workers: int | None = None,
) -> Callable:
    """Decorator that parallelizes processing of PartitionedDataset partitions.

    Transforms a function that processes **a single partition's data** into
    one that processes all partitions concurrently using a thread pool.

    The decorated function should accept a single positional argument (the
    loaded partition data) plus any additional arguments, and return the
    processed result for that partition.

    When the decorated function is called with a ``Dict[str, Callable]``
    (the return value of ``PartitionedDataset.load()``), it will:

    1. Load each partition concurrently via the thread pool.
    2. Apply the original function to each loaded partition concurrently.
    3. Return ``Dict[str, result]`` suitable for ``PartitionedDataset.save()``.

    When called with regular (non-partitioned) data, the original function
    is invoked normally with no parallelism.

    Can be used with or without arguments::

        @parallel_partitions
        def process(data):
            return data.dropna()

        @parallel_partitions(max_workers=8)
        def process(data):
            return data.dropna()

    Args:
        func: The function to decorate (supplied automatically when the
            decorator is used without parentheses).
        max_workers: Maximum number of threads for concurrent partition
            processing.  Defaults to ``None`` (the ``ThreadPoolExecutor``
            default, typically ``min(32, os.cpu_count() + 4)``).

    Returns:
        A wrapper that transparently handles partitioned or regular inputs.
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Locate the partitioned argument (Dict[str, Callable]).
            partition_arg_idx: int | None = None
            partition_kwarg_key: str | None = None

            for i, arg in enumerate(args):
                if _is_partition_dict(arg):
                    partition_arg_idx = i
                    break

            if partition_arg_idx is None:
                for key, val in kwargs.items():
                    if _is_partition_dict(val):
                        partition_kwarg_key = key
                        break

            # No partitioned input found -- pass through.
            if partition_arg_idx is None and partition_kwarg_key is None:
                return fn(*args, **kwargs)

            # Extract the partitions dict and remaining arguments.
            if partition_arg_idx is not None:
                partitions: dict[str, Callable] = args[partition_arg_idx]
                other_args = (*args[:partition_arg_idx], *args[partition_arg_idx + 1 :])
            else:
                assert partition_kwarg_key is not None
                partitions = kwargs.pop(partition_kwarg_key)
                other_args = args

            logger.info(
                "parallel_partitions: processing %d partitions with max_workers=%s",
                len(partitions),
                max_workers,
            )

            results: dict[str, Any] = {}

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_key: dict[Future, str] = {}
                for partition_key, load_fn in partitions.items():
                    future = executor.submit(
                        _load_and_process,
                        fn,
                        load_fn,
                        other_args,
                        kwargs,
                        partition_arg_idx,
                        partition_kwarg_key,
                    )
                    future_to_key[future] = partition_key

                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    results[key] = future.result()  # propagates exceptions

            return results

        # Expose metadata so callers can introspect.
        wrapper._parallel_partitions = True  # type: ignore[attr-defined]
        wrapper._max_workers = max_workers  # type: ignore[attr-defined]
        return wrapper

    # Support both @parallel_partitions and @parallel_partitions(...)
    if func is not None:
        return decorator(func)
    return decorator


def _load_and_process(
    fn: Callable,
    load_fn: Callable,
    other_args: tuple,
    kwargs: dict[str, Any],
    partition_arg_idx: int | None,
    partition_kwarg_key: str | None,
) -> Any:
    """Load a single partition and apply the processing function."""
    data = load_fn()
    if partition_arg_idx is not None:
        full_args = (*other_args[:partition_arg_idx], data, *other_args[partition_arg_idx:])
        return fn(*full_args, **kwargs)
    else:
        assert partition_kwarg_key is not None
        return fn(*other_args, **{**kwargs, partition_kwarg_key: data})


# ---------------------------------------------------------------------------
# PartitionedRunner
# ---------------------------------------------------------------------------


class _PartitionedTask(Task):
    """A :class:`Task` subclass that detects partitioned inputs and processes
    them in parallel using a thread pool.

    When a node input is a ``Dict[str, Callable]`` (the signature of
    ``PartitionedDataset.load()``), the task will:

    1. Load all partitions concurrently.
    2. Call the node function once per partition with the loaded data.
    3. Collect the per-partition results into a ``Dict[str, result]``.

    For non-partitioned inputs the behaviour is identical to the base
    :class:`Task`.
    """

    def __init__(
        self,
        node: Node,
        catalog: CatalogProtocol,
        is_async: bool,
        hook_manager: PluginManager | None = None,
        run_id: str | None = None,
        parallel: bool = False,
        partition_max_workers: int | None = None,
    ):
        super().__init__(
            node=node,
            catalog=catalog,
            is_async=is_async,
            hook_manager=hook_manager,
            run_id=run_id,
            parallel=parallel,
        )
        self._partition_max_workers = partition_max_workers

    def _run_node_sequential(
        self,
        node: Node,
        catalog: CatalogProtocol,
        hook_manager: PluginManager,
        run_id: str | None = None,
    ) -> Node:
        """Override to detect partitioned inputs and parallelise processing."""
        inputs: dict[str, Any] = {}

        for name in node.inputs:
            hook_manager.hook.before_dataset_loaded(dataset_name=name, node=node)
            inputs[name] = catalog.load(name)
            hook_manager.hook.after_dataset_loaded(
                dataset_name=name, data=inputs[name], node=node
            )

        is_async = False
        additional_inputs = self._collect_inputs_from_hook(
            node, catalog, inputs, is_async, hook_manager, run_id=run_id
        )
        inputs.update(additional_inputs)

        # Detect partitioned inputs.
        partitioned_names = [
            name for name, data in inputs.items() if _is_partition_dict(data)
        ]

        if partitioned_names:
            outputs = self._run_node_over_partitions(
                node, catalog, inputs, partitioned_names, is_async, hook_manager, run_id
            )
        else:
            outputs = self._call_node_run(
                node, catalog, inputs, is_async, hook_manager, run_id=run_id
            )

        for name, data in outputs.items():
            hook_manager.hook.before_dataset_saved(
                dataset_name=name, data=data, node=node
            )
            catalog.save(name, data)
            hook_manager.hook.after_dataset_saved(
                dataset_name=name, data=data, node=node
            )
        return node

    def _run_node_over_partitions(  # noqa: PLR0913
        self,
        node: Node,
        catalog: CatalogProtocol,
        inputs: dict[str, Any],
        partitioned_names: list[str],
        is_async: bool,
        hook_manager: PluginManager,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the node function once per partition, in parallel.

        If more than one input is partitioned, the partitions are aligned by
        key (inner-join semantics — only keys present in ALL partitioned inputs
        are processed).

        Non-partitioned inputs are broadcast to every invocation unchanged.
        """
        # Collect partition keys (intersection of all partitioned inputs).
        partition_key_sets = [
            set(inputs[name].keys()) for name in partitioned_names
        ]
        partition_keys = sorted(set.intersection(*partition_key_sets))

        if not partition_keys:
            logger.warning(
                "PartitionedRunner: no common partition keys found across "
                "inputs %s — returning empty outputs.",
                partitioned_names,
            )
            return {}

        logger.info(
            "PartitionedRunner: processing %d partitions for node '%s' "
            "with max_workers=%s",
            len(partition_keys),
            node.name,
            self._partition_max_workers,
        )

        # Separate static (broadcast) inputs from partitioned ones.
        static_inputs = {
            name: data for name, data in inputs.items() if name not in partitioned_names
        }
        partitioned_inputs: dict[str, dict[str, Callable]] = {
            name: inputs[name] for name in partitioned_names
        }

        # Process partitions concurrently.
        per_partition_results: dict[str, dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=self._partition_max_workers) as executor:
            future_to_key: dict[Future, str] = {}
            for pk in partition_keys:
                # Build per-partition inputs: load partitioned data, merge static.
                future = executor.submit(
                    self._process_single_partition,
                    node,
                    catalog,
                    pk,
                    partitioned_inputs,
                    static_inputs,
                    is_async,
                    hook_manager,
                    run_id,
                )
                future_to_key[future] = pk

            for future in as_completed(future_to_key):
                pk = future_to_key[future]
                per_partition_results[pk] = future.result()

        # Pivot: {partition_key: {output_name: value}} -> {output_name: {partition_key: value}}
        merged_outputs: dict[str, dict[str, Any]] = {}
        for pk, outputs in per_partition_results.items():
            for output_name, value in outputs.items():
                merged_outputs.setdefault(output_name, {})[pk] = value

        return merged_outputs

    def _process_single_partition(  # noqa: PLR0913
        self,
        node: Node,
        catalog: CatalogProtocol,
        partition_key: str,
        partitioned_inputs: dict[str, dict[str, Callable]],
        static_inputs: dict[str, Any],
        is_async: bool,
        hook_manager: PluginManager,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Load one partition from each partitioned input, merge with static
        inputs, and run the node function."""
        partition_inputs = dict(static_inputs)
        for name, partitions_dict in partitioned_inputs.items():
            load_fn = partitions_dict[partition_key]
            partition_inputs[name] = load_fn()

        try:
            outputs = node.run(partition_inputs)
        except Exception as exc:
            hook_manager.hook.on_node_error(
                error=exc,
                node=node,
                catalog=catalog,
                inputs=partition_inputs,
                is_async=is_async,
                run_id=run_id,
            )
            raise
        return outputs


class PartitionedRunner(AbstractRunner):
    """``PartitionedRunner`` executes pipeline nodes sequentially but
    processes partitions within each node **in parallel** using threads.

    When a node's input is detected as a ``Dict[str, Callable]`` (the
    standard output of ``PartitionedDataset.load()``), the runner
    automatically:

    1. Loads each partition concurrently.
    2. Calls the node function once per partition.
    3. Pivots the per-partition results into ``Dict[str, result]``
       dictionaries keyed by partition ID, suitable for saving with
       ``PartitionedDataset``.

    For nodes that do not consume partitioned data, the behaviour is
    identical to :class:`SequentialRunner`.

    This runner works around Kedro's static-pipeline limitation by
    providing **data-level parallelism** within a single pipeline node,
    without requiring dynamic node creation.

    Example::

        from kedro.runner import PartitionedRunner

        runner = PartitionedRunner(max_workers=8)
        runner.run(pipeline, catalog)

    Or in ``settings.py``::

        from kedro.runner import PartitionedRunner

        SESSION_STORE_ARGS = {}
        RUNNER = PartitionedRunner(max_workers=4)
    """

    def __init__(
        self,
        max_workers: int | None = None,
        is_async: bool = False,
    ):
        """Instantiate the runner.

        Args:
            max_workers: Maximum number of threads for concurrent partition
                processing within each node.  Defaults to ``None`` (the
                ``ThreadPoolExecutor`` default).
            is_async: If True, the node inputs and outputs are loaded and
                saved asynchronously with threads.  Defaults to False.
        """
        super().__init__(is_async=is_async)
        self._max_workers = (
            self._validate_max_workers(max_workers)
            if max_workers is not None
            else max_workers
        )

    def _get_executor(self, max_workers: int) -> None:
        return None

    def _run(
        self,
        pipeline: Pipeline,
        catalog: CatalogProtocol,
        hook_manager: PluginManager | None = None,
        run_id: str | None = None,
    ) -> None:
        """Run the pipeline sequentially, using :class:`_PartitionedTask`
        for partition-level parallelism within each node."""
        nodes = pipeline.nodes

        self._validate_catalog(catalog)
        self._validate_nodes(nodes)
        self._set_manager_datasets(catalog)

        load_counts = __import__("collections").Counter(
            __import__("itertools").chain.from_iterable(n.inputs for n in nodes)
        )
        done_nodes: set[Node] = set()

        if not self._is_async:
            self._logger.info(
                "Using synchronous mode for loading and saving data. "
                "Use the --async flag for potential performance gains. "
                "https://docs.kedro.org/en/stable/build/run_a_pipeline/"
                "#load-and-save-asynchronously"
            )

        for node in nodes:
            try:
                _PartitionedTask(
                    node=node,
                    catalog=catalog,
                    hook_manager=hook_manager,
                    is_async=self._is_async,
                    run_id=run_id,
                    partition_max_workers=self._max_workers,
                ).execute()
                done_nodes.add(node)
            except Exception:
                self._suggest_resume_scenario(pipeline, done_nodes, catalog)
                raise
            self._logger.info("Completed node: %s", node.name)
            self._logger.info(
                "Completed %d out of %d tasks", len(done_nodes), len(nodes)
            )
            self._release_datasets(node, catalog, load_counts, pipeline)
