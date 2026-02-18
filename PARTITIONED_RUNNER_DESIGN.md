# PartitionedRunner v2 — Tag-Based Design

## Overview

A Kedro runner plugin that uses **three tags** to express partition-level
execution semantics.  No monkey-patching, no wrapper functions — just tags
on nodes and a runner that knows what to do with them.

```python
pipeline([
    node(clean, "raw", "cleaned", tags=["partitioned.parallel"]),
    node(transform, "cleaned", "features", tags=["partitioned.parallel"]),
    node(
        concat_frames,
        inputs={"acc": "empty_df", "chunk": "features"},
        outputs="combined",
        tags=["partitioned.reduce"],
    ),
    node(make_report, "combined", "report"),  # normal node, no tag
])
```

---

## Tags

### `partitioned.parallel`

**Intent:** "Call me once per partition, independently."

**Inputs:** At least one must be partitioned (`Dict[str, Callable]`).
Non-partitioned inputs are broadcast to every invocation.

**Outputs:** `Dict[str, Callable]` (lazy loaders) — one entry per
partition key.  Downstream parallel/reduce/collect nodes can chain.

**Execution:**

```
for each partition key (in parallel):
    load partition data from each partitioned input
    call node.run({partitioned_arg: data, **static_args})
    collect output
wrap outputs as lazy loaders
```

**Multi-input alignment:** When multiple inputs are partitioned, only
common keys are processed (inner-join semantics).

---

### `partitioned.collect`

**Intent:** "Materialize all partitions and give me the dict."

**Inputs:** Exactly one partitioned input.  The runner calls every lazy
loader and passes `Dict[str, loaded_value]` to the node function.
Non-partitioned inputs are passed through normally.

**Outputs:** Whatever the function returns — not wrapped as partitions.
This is an "exit ramp" from the partition chain.

**Execution:**

```
materialized = {key: loader() for key, loader in partitioned_input.items()}
node.run({partitioned_arg: materialized, **static_args})
```

**Loading can be parallelized** (thread/process pool to call loaders
concurrently), then the materialized dict is passed to the function in
one call.

**Use case:** When you need all partitions simultaneously — e.g. joining
them, computing cross-partition statistics, building a report.

**Memory:** O(N) — all partitions in memory at once.

---

### `partitioned.reduce`

**Intent:** "Fold across partitions, one at a time, threading an
accumulator."

**Inputs:** Exactly two inputs:

| Input      | Role         | Description                                |
|------------|--------------|--------------------------------------------|
| Partitioned| Iterable     | `Dict[str, Callable]` — the partitions     |
| Scalar     | Initializer  | The starting accumulator value             |

The runner identifies which is which by checking the catalog
(`PartitionedDataset`) or the runtime data shape (`Dict[str, Callable]`).

**Function signature:** `fn(accumulator, current_partition) -> new_accumulator`
— mirrors `functools.reduce`.  Parameter names are the user's choice;
the runner maps them based on which input is partitioned.

**Outputs:** A single value (the final accumulator).  Not wrapped as
partitions.  This is an "exit ramp."

**Execution:**

```
acc = catalog.load(initializer_input)
for key in sorted(partition_keys):
    partition_data = partitioned_input[key]()
    acc = node.run({acc_param: acc, partition_param: partition_data})
catalog.save(output_name, acc)
```

**Memory:** O(1) in partition data — only the accumulator + one partition
loaded at a time (assuming the accumulator stays bounded).

**Loading:** Partitions are loaded one at a time during the fold.
Pre-loading could be added as an optimization (load next partition while
current fold step runs), but not required for MVP.

**Use cases:**
- `pd.concat([acc, chunk])` over hundreds of Parquet files
- Running totals / aggregations
- Incremental model training
- Any accumulation that shouldn't load all data at once

---

## Detection & Validation

### At runner startup (fail fast)

| Check | Tag | Error |
|---|---|---|
| Node has more than one of the three tags | any | `"Node 'X' has conflicting partition tags: ..."` |
| Reduce node doesn't have exactly 2 inputs | reduce | `"Reduce node 'X' must have exactly 2 inputs (initializer + partitioned), got N"` |
| Reduce node doesn't have exactly 1 output | reduce | `"Reduce node 'X' must have exactly 1 output, got N"` |

### At runtime (during execution)

| Check | Tag | Error |
|---|---|---|
| No partitioned inputs found | parallel, collect, reduce | `"Node 'X' tagged partitioned.Y but no inputs are partitioned"` |
| Collect node has >1 partitioned input | collect | `"Collect node 'X' must have exactly 1 partitioned input, got N"` |
| Reduce: can't distinguish initializer from iterable | reduce | `"Reduce node 'X': could not identify which input is the initializer vs partitioned"` |

### How the runner identifies partitioned inputs

Checked in order:

1. **Catalog type** — if the dataset is a `PartitionedDataset` instance,
   it's partitioned.
2. **Runtime shape** — if the loaded data is `Dict[str, Callable]`
   (non-empty, all string keys, all callable values), it's partitioned.
   This catches `MemoryDataset` intermediates from upstream parallel nodes.

For reduce nodes, exactly one input should be partitioned and one should
not.  If both are or neither is, raise a clear error.

---

## Execution Model

```
PartitionedRunner._run(pipeline, catalog)
│
├── for each node in topological order:
│   │
│   ├── no partition tag → execute normally (SequentialRunner behavior)
│   │
│   ├── partitioned.parallel →
│   │   ├── load inputs, identify partitioned ones
│   │   ├── compute partition keys (intersection if multiple)
│   │   ├── fan-out: submit one task per partition to executor
│   │   │   └── executor is ThreadPool or ProcessPool (backend param)
│   │   ├── collect results
│   │   └── wrap outputs as Dict[str, Callable] lazy loaders
│   │
│   ├── partitioned.collect →
│   │   ├── load inputs, identify the partitioned one
│   │   ├── materialize: call all lazy loaders (optionally parallel)
│   │   ├── replace partitioned input with Dict[str, loaded_value]
│   │   └── call node.run() once with the materialized dict
│   │
│   └── partitioned.reduce →
│       ├── load inputs, identify partitioned vs initializer
│       ├── acc = initializer value
│       ├── for key in sorted(partition_keys):
│       │   ├── data = load partition
│       │   └── acc = node.run({acc_param: acc, data_param: data})[output]
│       └── save acc as final output
│
└── release datasets, track progress
```

---

## API

### Tags (constants)

```python
PARTITIONED_PARALLEL = "partitioned.parallel"
PARTITIONED_COLLECT  = "partitioned.collect"
PARTITIONED_REDUCE   = "partitioned.reduce"
```

### Runner

```python
from kedro_partitioned import PartitionedRunner

runner = PartitionedRunner(
    max_workers=4,        # thread/process pool size
    backend="thread",     # "thread" or "process" (requires cloudpickle)
)
runner.run(pipeline, catalog)
```

No `partitioned_datasets` parameter.  No `partitioned()` helper.  The
tags are the entire API surface.

### User code examples

#### Parallel processing

```python
def clean(raw_data):
    """Called once per partition.  Receives a single DataFrame."""
    return raw_data.dropna()

pipeline([
    node(clean, "raw_data", "cleaned", tags=[PARTITIONED_PARALLEL]),
])
```

#### Chaining parallel nodes

```python
pipeline([
    node(clean, "raw", "cleaned", tags=[PARTITIONED_PARALLEL]),
    node(featurize, "cleaned", "features", tags=[PARTITIONED_PARALLEL]),
])
# "cleaned" is a MemoryDataset carrying Dict[str, Callable].
# The runner detects this at runtime via shape check.
```

#### Parallel with broadcast

```python
def scale(data, params_factor):
    return data * params_factor

pipeline([
    node(
        scale,
        inputs={"data": "raw", "params_factor": "params:factor"},
        outputs="scaled",
        tags=[PARTITIONED_PARALLEL],
    ),
])
# "params:factor" is scalar — broadcast to every partition.
```

#### Collect (materialize all)

```python
def summarize(all_partitions: dict[str, pd.DataFrame]):
    combined = pd.concat(all_partitions.values())
    return combined.describe()

pipeline([
    node(clean, "raw", "cleaned", tags=[PARTITIONED_PARALLEL]),
    node(
        summarize,
        "cleaned",
        "summary_stats",
        tags=[PARTITIONED_COLLECT],
    ),
])
```

#### Reduce (fold with initializer)

```python
def concat_frames(accumulated, chunk):
    return pd.concat([accumulated, chunk])

pipeline([
    node(clean, "raw", "cleaned", tags=[PARTITIONED_PARALLEL]),
    node(
        concat_frames,
        inputs={"accumulated": "empty_df", "chunk": "cleaned"},
        outputs="combined",
        tags=[PARTITIONED_REDUCE],
    ),
])

# catalog.yml:
#   empty_df:
#     type: pandas.CSVDataset  # or MemoryDataset with an empty DataFrame
#   raw:
#     type: PartitionedDataset
#     path: data/01_raw
#     dataset: pandas.CSVDataset
```

#### Full MapReduce pipeline

```python
pipeline([
    # parallel: clean each partition
    node(clean, "raw", "cleaned", name="clean",
         tags=[PARTITIONED_PARALLEL]),

    # parallel: featurize each partition
    node(featurize, "cleaned", "features", name="featurize",
         tags=[PARTITIONED_PARALLEL]),

    # reduce: fold partitions into one DataFrame
    node(concat_frames,
         inputs={"accumulated": "empty_df", "chunk": "features"},
         outputs="combined", name="combine",
         tags=[PARTITIONED_REDUCE]),

    # normal node: train on the combined data
    node(train_model, "combined", "model", name="train"),
])
```

---

## Plugin Structure

```
kedro-partitioned/
├── pyproject.toml
├── src/
│   └── kedro_partitioned/
│       ├── __init__.py          # public API: runner, tag constants
│       ├── tags.py              # PARTITIONED_PARALLEL, COLLECT, REDUCE
│       ├── runner.py            # PartitionedRunner
│       ├── task.py              # _PartitionedTask (parallel, collect, reduce)
│       ├── _cloudpickle.py      # cloudpickle helpers for process backend
│       └── _validation.py       # tag/input validation logic
└── tests/
    ├── test_parallel.py
    ├── test_collect.py
    ├── test_reduce.py
    ├── test_chaining.py
    ├── test_validation.py
    └── test_process_backend.py
```

---

## Open Questions

1. **Reduce without initializer.** Should we support single-input reduce
   (first partition as seed, like `functools.reduce` without initializer)?
   Simpler API for the common case where accumulator type == partition
   type, but diverges from the "exactly 2 inputs" rule.

2. **Reduce output naming.** `node.run()` returns a dict keyed by output
   name.  For reduce, each iteration produces `{output_name: new_acc}`.
   The runner extracts the single value.  Should we validate exactly 1
   output at startup?

3. **Collect parallelism.** Should collect's materialization step use the
   same thread/process pool as parallel?  Probably yes — loading 500
   partitions sequentially before passing them in defeats the purpose.

4. **Reduce + process backend.** Reduce is inherently sequential, so the
   process backend doesn't help the compute.  But if partitions are on
   remote storage, could we pre-fetch the next partition while the
   current fold step runs?  Nice optimization, not MVP.

5. **Mixed tags in a pipeline.** A pipeline can have parallel, collect,
   and reduce nodes mixed together.  The runner handles each based on its
   tag.  No restriction on ordering — the DAG determines execution order.

6. **`partitioned()` helper.** Keep it as a convenience for applying
   `PARTITIONED_PARALLEL` to multiple nodes at once?  Or drop it in
   favor of explicit tags?  Tags are more explicit but more verbose for
   large pipelines.
