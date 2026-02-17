"""``PartitionedRunner`` provides parallel processing of PartitionedDataset
partitions within Kedro pipelines.

This module addresses a key limitation in Kedro: pipelines are static DAGs,
so there is no way to dynamically create N parallel nodes for N partitions
at runtime. Instead, a single node receives all partitions as a
``Dict[str, Callable]`` and must iterate them sequentially.

``PartitionedRunner`` is a runner that parallelizes partition loading and
processing within each node using a thread pool.  Users write simple
per-partition functions and the runner handles the fan-out / fan-in.

The ``partitioned_pipeline`` helper explicitly tags nodes that consume
partitioned datasets — this is the recommended way to declare which
datasets carry partitioned data.  The resulting pipeline composes
naturally with other pipelines via ``+``.

Partitioned datasets can be backed by ``PartitionedDataset`` on disk
**or** by ``MemoryDataset`` intermediates — any dataset name listed in
``partitioned_datasets`` will be treated as carrying partition data
(``Dict[str, Callable]``), regardless of the underlying storage.
"""

from __future__ import annotations

import collections
import itertools
import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable

from kedro.runner.runner import AbstractRunner
from kedro.runner.task import Task

if TYPE_CHECKING:
    from pluggy import PluginManager

    from kedro.io import CatalogProtocol
    from kedro.pipeline import Pipeline
    from kedro.pipeline.node import Node

logger = logging.getLogger(__name__)

# Tag applied to nodes whose inputs include partitioned datasets.
PARTITIONED_TAG = "kedro.partitioned"


def _is_partition_dict(data: Any) -> bool:
    """Detect whether *data* matches the ``PartitionedDataset.load()`` shape.

    ``PartitionedDataset.load()`` returns ``Dict[str, Callable]`` where each
    value is a lazy-loading function for that partition.  We use a duck-typing
    check: a non-empty dict whose keys are all strings and values are all
    callable.
    """
    return (
        isinstance(data, dict)
        and len(data) > 0
        and all(isinstance(k, str) for k in data)
        and all(callable(v) for v in data.values())
    )


def _wrap_as_lazy_loaders(partition_dict: dict[str, Any]) -> dict[str, Callable]:
    """Wrap a ``{key: value}`` dict into ``{key: lambda: value}``.

    This makes the output of a partitioned node look like the load result of
    a ``PartitionedDataset``, enabling downstream nodes (even through a
    ``MemoryDataset`` intermediate) to chain partition-parallel processing.

    ``PartitionedDataset.save()`` already handles both raw values **and**
    callables, so this is safe for persistent outputs too.
    """
    return {key: (lambda v=val: v) for key, val in partition_dict.items()}


# ---------------------------------------------------------------------------
# partitioned_pipeline helper
# ---------------------------------------------------------------------------


def partitioned_pipeline(
    pipe: Pipeline,
    partitioned_datasets: set[str],
) -> Pipeline:
    """Return a copy of *pipe* where every node that consumes at least one
    dataset in *partitioned_datasets* is tagged with :data:`PARTITIONED_TAG`.

    This makes the pipeline definition explicit about which nodes should
    receive partition-parallel treatment from :class:`PartitionedRunner`,
    without requiring any changes to the node functions themselves.

    Example::

        from kedro.pipeline import node, pipeline
        from kedro.runner.partitioned_runner import (
            PartitionedRunner,
            partitioned_pipeline,
        )

        def clean(data):
            return data.dropna()

        def transform(data):
            return data * 2

        my_pipeline = partitioned_pipeline(
            pipeline([
                node(clean, "raw", "cleaned", name="clean"),
                node(transform, "cleaned", "final", name="transform"),
            ]),
            partitioned_datasets={"raw", "cleaned", "final"},
        )

        PartitionedRunner(max_workers=4).run(my_pipeline, catalog)

    Args:
        pipe: The source pipeline.
        partitioned_datasets: Names of datasets that carry partitioned data
            (i.e. their loaded form is ``Dict[str, Callable]``).

    Returns:
        A new :class:`Pipeline` where the relevant nodes carry the
        ``kedro.partitioned`` tag.
    """
    from kedro.pipeline import pipeline as make_pipeline

    tagged_nodes = []
    for n in pipe.nodes:
        if set(n.inputs) & partitioned_datasets:
            tagged_nodes.append(n.tag(PARTITIONED_TAG))
        else:
            tagged_nodes.append(n)
    return make_pipeline(tagged_nodes)


# ---------------------------------------------------------------------------
# _PartitionedTask
# ---------------------------------------------------------------------------


class _PartitionedTask(Task):
    """A :class:`Task` subclass that detects partitioned inputs and processes
    them in parallel using a thread pool.

    When a node input is a ``Dict[str, Callable]`` (the signature of
    ``PartitionedDataset.load()``), the task will:

    1. Load all partitions concurrently.
    2. Call the node function once per partition with the loaded data.
    3. Collect the per-partition results into a ``Dict[str, Callable]``
       (lazy loaders) so that downstream nodes can chain.

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
        partitioned_datasets: set[str] | None = None,
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
        self._partitioned_datasets = partitioned_datasets or set()

    # -- helpers -----------------------------------------------------------

    def _is_partitioned_input(self, name: str, data: Any) -> bool:
        """Decide whether a loaded input should be treated as partitioned.

        Uses explicit configuration only: either the dataset name was passed
        to ``PartitionedRunner(partitioned_datasets=...)`` or the node was
        tagged via :func:`partitioned_pipeline`.  No duck-typing fallback.
        """
        # 1. Explicit runner-level configuration.
        if name in self._partitioned_datasets:
            return True
        # 2. Pipeline-level tag applied by partitioned_pipeline().
        if PARTITIONED_TAG in self.node.tags and _is_partition_dict(data):
            return True
        return False

    # -- overrides ---------------------------------------------------------

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
            name
            for name, data in inputs.items()
            if self._is_partitioned_input(name, data)
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

    # -- partition fan-out / fan-in ----------------------------------------

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
        key (inner-join semantics — only keys present in **all** partitioned
        inputs are processed).

        Non-partitioned inputs are broadcast to every invocation unchanged.

        Outputs are wrapped as lazy loaders (``Dict[str, Callable]``) so that
        downstream nodes can chain partition-parallel processing even through
        ``MemoryDataset`` intermediates.
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
            name: data
            for name, data in inputs.items()
            if name not in partitioned_names
        }
        partitioned_inputs: dict[str, dict[str, Callable]] = {
            name: inputs[name] for name in partitioned_names
        }

        # Process partitions concurrently.
        per_partition_results: dict[str, dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=self._partition_max_workers) as executor:
            future_to_key: dict[Future, str] = {}
            for pk in partition_keys:
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

        # Pivot: {partition_key: {output_name: val}} -> {output_name: {pk: val}}
        merged_outputs: dict[str, dict[str, Any]] = {}
        for pk, outputs in per_partition_results.items():
            for output_name, value in outputs.items():
                merged_outputs.setdefault(output_name, {})[pk] = value

        # Wrap as lazy loaders so downstream nodes can chain.
        return {
            name: _wrap_as_lazy_loaders(partition_dict)
            for name, partition_dict in merged_outputs.items()
        }

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


# ---------------------------------------------------------------------------
# PartitionedRunner
# ---------------------------------------------------------------------------


class PartitionedRunner(AbstractRunner):
    """``PartitionedRunner`` executes pipeline nodes sequentially but
    processes partitions within each node **in parallel** using threads.

    When a node's input is detected as partitioned, the runner:

    1. Loads each partition concurrently.
    2. Calls the node function once per partition.
    3. Wraps per-partition results as lazy loaders so that downstream
       nodes can chain partition-parallel processing — even through
       ``MemoryDataset`` intermediates.

    For nodes that do not consume partitioned data, the behaviour is
    identical to :class:`SequentialRunner`.

    **Detection requires explicit configuration** (checked in order):

    1. *Runner-level* — dataset names passed via ``partitioned_datasets``.
    2. *Pipeline-level* — nodes tagged ``kedro.partitioned`` by
       :func:`partitioned_pipeline`.

    The recommended approach is :func:`partitioned_pipeline`, which makes
    the partitioned contract visible in the pipeline definition and
    composes naturally with other pipelines via ``+``.

    Example — pipeline-level tagging (recommended)::

        from kedro.runner.partitioned_runner import (
            PartitionedRunner,
            partitioned_pipeline,
        )

        my_pipeline = partitioned_pipeline(
            pipeline([
                node(clean, "raw", "cleaned"),
                node(transform, "cleaned", "final"),
            ]),
            partitioned_datasets={"raw", "cleaned", "final"},
        )

        # Compose with other pipelines normally.
        full_pipeline = my_pipeline + reporting_pipeline

        PartitionedRunner(max_workers=4).run(full_pipeline, catalog)

    Example — runner-level dataset names::

        runner = PartitionedRunner(
            max_workers=4,
            partitioned_datasets={"raw", "cleaned", "final"},
        )
        runner.run(pipeline, catalog)
    """

    def __init__(
        self,
        max_workers: int | None = None,
        is_async: bool = False,
        partitioned_datasets: set[str] | None = None,
    ):
        """Instantiate the runner.

        Args:
            max_workers: Maximum number of threads for concurrent partition
                processing within each node.  Defaults to ``None`` (the
                ``ThreadPoolExecutor`` default).
            is_async: If True, the node inputs and outputs are loaded and
                saved asynchronously with threads.  Defaults to False.
            partitioned_datasets: Optional set of dataset names whose loaded
                form is ``Dict[str, Callable]`` (i.e. from a
                ``PartitionedDataset`` or a ``MemoryDataset`` carrying
                partition data).  When provided, only these inputs trigger
                partition-parallel processing.  Alternatively, use
                :func:`partitioned_pipeline` to declare partitioned datasets
                at the pipeline level.
        """
        super().__init__(is_async=is_async)
        self._max_workers = (
            self._validate_max_workers(max_workers)
            if max_workers is not None
            else max_workers
        )
        self._partitioned_datasets = partitioned_datasets or set()

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

        load_counts = collections.Counter(
            itertools.chain.from_iterable(n.inputs for n in nodes)
        )
        done_nodes: set[Node] = set()

        if not self._is_async:
            self._logger.info(
                "Using synchronous mode for loading and saving data. "
                "Use the --async flag for potential performance gains. "
                "https://docs.kedro.org/en/stable/build/run_a_pipeline/"
                "#load-and-save-asynchronously"
            )

        for exec_node in nodes:
            try:
                _PartitionedTask(
                    node=exec_node,
                    catalog=catalog,
                    hook_manager=hook_manager,
                    is_async=self._is_async,
                    run_id=run_id,
                    partition_max_workers=self._max_workers,
                    partitioned_datasets=self._partitioned_datasets,
                ).execute()
                done_nodes.add(exec_node)
            except Exception:
                self._suggest_resume_scenario(pipeline, done_nodes, catalog)
                raise
            self._logger.info("Completed node: %s", exec_node.name)
            self._logger.info(
                "Completed %d out of %d tasks", len(done_nodes), len(nodes)
            )
            self._release_datasets(exec_node, catalog, load_counts, pipeline)
