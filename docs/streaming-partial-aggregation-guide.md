# Streaming Partial Aggregation in DataFusion — Implementation Guide

This document describes how to implement streaming partial aggregation as a DataFusion extension. It requires **zero changes to DataFusion core** — everything uses public extension APIs.

## Problem

DataFusion's standard aggregation pipeline is:

```
AggregateExec(Partial) → Repartition → AggregateExec(Final)
```

The `Final` stage blocks until all partial results are collected, then emits complete results. For use cases like real-time dashboards or continuous monitoring, you want progressive partial results instead — the client receives intermediate aggregation state as it's computed and merges it incrementally.

The goal: let a user opt in to partial-only aggregation **per query** through SQL syntax, producing only:

```
AggregateExec(Partial) → Client
```

## Approach: Marker UDF

A `partial_agg()` function wraps aggregate expressions in SQL to signal that the query should skip the Final stage:

```sql
-- Standard aggregation (blocking):
SELECT bucket, SUM(value) as total FROM metrics GROUP BY bucket;

-- Partial-only aggregation (streaming):
SELECT bucket, partial_agg(SUM(value)) as total FROM metrics GROUP BY bucket;
```

`partial_agg()` is not a real function — it's a syntactic marker that is stripped during analysis. By registering it as a `ScalarUDF`, the SQL parser accepts it without any parser modifications.

## Architecture

Four DataFusion extension points are used:

```
SQL: SELECT bucket, partial_agg(SUM(value)) FROM t GROUP BY bucket
        │
        ▼
[SQL Parser]  →  Projection [ partial_agg(col_ref) AS alias ]
                   └─ Aggregate [ SUM(value) ]    ← partial_agg is NOT inside Aggregate
        │
        ▼
[PartialAggregateRule — AnalyzerRule]
                 →  Projection [ col_ref AS alias ]        ← partial_agg stripped
                      └─ PartialOnlyAggregate [ SUM(value) ]  ← custom Extension node
        │
        ▼
[PartialOnlyAggregatePlanner — ExtensionPlanner]
                 →  ProjectionExec (column renaming)
                      └─ AggregateExec(mode=Partial)       ← no Final stage
                           └─ DataSourceExec
        │
        ▼
[Client merges partial results by group key]
```

### Critical Subtlety: Plan Splitting

When DataFusion's SQL planner processes `partial_agg(SUM(value))`, it does NOT keep `partial_agg` inside the `Aggregate` node. Instead, it **splits** the expression:

- `SUM(value)` is extracted into the `Aggregate` node's `aggr_expr`
- `partial_agg(column_ref)` remains in a `Projection` node above the `Aggregate`

So the analyzer rule must match the **Projection → Aggregate** pattern, not look inside the Aggregate's expressions. This is the single most important implementation detail.

### Schema Mismatch: Column Renaming

`AggregateExec` in `Partial` mode outputs columns with intermediate state names:
- Group columns: `metrics.bucket` (fully qualified) instead of `bucket`
- Aggregate state: `sum(metrics.value)[sum]` instead of `sum(metrics.value)`

The logical plan's schema expects the non-intermediate names. Without correction, DataFusion's schema validation rejects the plan. The fix is a `ProjectionExec` wrapper that renames physical columns to match the logical schema.

## Implementation Steps

### Step 1: Marker UDF

Create a scalar UDF that accepts any single argument and passes through its type. The `invoke_with_args` method should return an error — if it's ever called, the analyzer failed to strip it.

```rust
struct PartialAggUdf {
    signature: Signature,
}

impl PartialAggUdf {
    fn new() -> Self {
        Self {
            // Accept any single argument type
            signature: Signature::any(1, Volatility::Volatile),
        }
    }
}

impl ScalarUDFImpl for PartialAggUdf {
    fn name(&self) -> &str { "partial_agg" }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        // Pass through unchanged
        Ok(arg_types[0].clone())
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Err(DataFusionError::Internal(
            "partial_agg() marker should have been removed by the analyzer".into(),
        ))
    }

    // ... as_any, signature
}
```

Register it:
```rust
ctx.register_udf(ScalarUDF::new_from_impl(PartialAggUdf::new()));
```

### Step 2: Custom Logical Node

Define a `PartialOnlyAggregate` that carries the same information as a standard `Aggregate` node — group expressions, aggregate expressions, schema — but signals to the physical planner that only a Partial stage should be created.

```rust
#[derive(Debug, PartialEq, Eq, Hash)]
struct PartialOnlyAggregate {
    input: LogicalPlan,
    group_expr: Vec<Expr>,
    aggr_expr: Vec<Expr>,
    schema: DFSchemaRef,
}
```

Key implementation notes for `UserDefinedLogicalNodeCore`:

- **`expressions()`**: Return group expressions followed by aggregate expressions (concatenated). The split point is tracked by `self.group_expr.len()`.
- **`with_exprs_and_inputs()`**: Split the incoming `exprs` vec at `self.group_expr.len()` to reconstruct group vs aggregate expressions. Rebuild the schema via `Aggregate::try_new` to ensure correctness.
- **`schema()`**: Must match the output schema of a standard `Aggregate` with the same expressions. Compute it by constructing a temporary `Aggregate::try_new` and using its `.schema`.
- **`PartialOrd`**: `DFSchemaRef` doesn't implement `PartialOrd`, so you must implement `PartialOrd` manually instead of deriving it. Compare by expressions only.

### Step 3: Analyzer Rule

The analyzer rule walks the plan bottom-up via `plan.transform_up()`. For each node:

1. Check if it's a `Projection`
2. Check if any projection expression contains `partial_agg(...)` — look for `Expr::ScalarFunction` (or `Expr::Alias(Expr::ScalarFunction(...))`) where `func.name() == "partial_agg"`
3. Check that the Projection's input is an `Aggregate`
4. Strip `partial_agg()` wrappers from the projection expressions (preserving aliases)
5. Replace the `Aggregate` child with `PartialOnlyAggregate`
6. Rebuild the Projection pointing to the new extension node

```rust
impl AnalyzerRule for PartialAggregateRule {
    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        plan.transform_up(|node| {
            let LogicalPlan::Projection(ref proj) = node else {
                return Ok(Transformed::no(node));
            };

            // Check projection exprs for partial_agg() wrappers...
            // If found and child is Aggregate:
            //   1. Strip partial_agg() from projection
            //   2. Replace Aggregate with PartialOnlyAggregate Extension
            //   3. Rebuild Projection
        })
        .map(|t| t.data)
    }
}
```

The `unwrap_partial_agg` helper must handle both:
- `partial_agg(expr)` — direct call
- `partial_agg(expr) AS alias` — aliased call (`Expr::Alias` wrapping `Expr::ScalarFunction`)

### Step 4: Extension Planner

The `ExtensionPlanner` maps `PartialOnlyAggregate` to physical execution nodes:

1. **Build `PhysicalGroupBy`** from group expressions using `PhysicalGroupBy::new_single()`
2. **Build aggregate expressions** using `create_aggregate_expr_and_maybe_filter()` — this is the same function DataFusion's standard planner uses
3. **Create `AggregateExec`** with `AggregateMode::Partial` — no Final stage
4. **Wrap with `ProjectionExec`** to rename columns from intermediate state format to logical schema names

```rust
impl ExtensionPlanner for PartialOnlyAggregatePlanner {
    async fn plan_extension(
        &self,
        planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let partial_node = node.as_any().downcast_ref::<PartialOnlyAggregate>()?;

        // Build groups, aggregates, filters (same as standard planner)...

        let partial_agg = Arc::new(AggregateExec::try_new(
            AggregateMode::Partial,
            groups, aggregates, filters,
            input_exec, physical_input_schema,
        )?);

        // Rename columns: physical partial schema → logical schema
        let projection = ProjectionExec::try_new(rename_exprs, partial_agg)?;
        Ok(Some(Arc::new(projection)))
    }
}
```

### Step 5: Query Planner Wiring

Create a custom `QueryPlanner` that registers the `ExtensionPlanner` with the `DefaultPhysicalPlanner`:

```rust
struct PartialAggQueryPlanner;

impl QueryPlanner for PartialAggQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let physical_planner = DefaultPhysicalPlanner::with_extension_planners(vec![
            Arc::new(PartialOnlyAggregatePlanner),
        ]);
        physical_planner.create_physical_plan(logical_plan, session_state).await
    }
}
```

### Step 6: Session Setup

Wire everything together in a `SessionContext`:

```rust
fn make_partial_agg_context() -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_analyzer_rule(Arc::new(PartialAggregateRule))
        .with_query_planner(Arc::new(PartialAggQueryPlanner))
        .build();

    let ctx = SessionContext::new_with_state(state);
    ctx.register_udf(ScalarUDF::new_from_impl(PartialAggUdf::new()));
    ctx
}
```

## Client-Side Merging

The client receives partial batches where the same group key may appear multiple times across batches. Merging rules for common aggregates:

| Aggregate | Intermediate State | Merge Rule |
|-----------|-------------------|------------|
| `SUM(x)` | Running sum (same type as x) | Sum the partial sums |
| `COUNT(*)` | Running count (Int64) | Sum the partial counts |
| `MIN(x)` | Running min | Take the minimum |
| `MAX(x)` | Running max | Take the maximum |
| `AVG(x)` | (sum, count) — two columns | Sum the sums and counts, divide at end |

Example merge loop:

```rust
let mut merged: HashMap<String, (f64, i64)> = HashMap::new();

while let Some(batch) = stream.next().await {
    let batch = batch?;
    for row in 0..batch.num_rows() {
        let key = get_group_key(&batch, row);
        let entry = merged.entry(key).or_default();
        entry.0 += get_sum(&batch, row);     // SUM: add partial sums
        entry.1 += get_count(&batch, row);   // COUNT: add partial counts
    }
    // Client state is updated incrementally during execution
}
```

## Resulting Physical Plan

```
ProjectionExec: expr=[metrics.bucket@0 as bucket,
                      sum(metrics.value)[sum]@1 as total,
                      count(*)[count]@2 as cnt]
  AggregateExec: mode=Partial, gby=[bucket@0 as metrics.bucket],
                 aggr=[sum(metrics.value), count(*)]
    MemoryExec: partitions=2, partition_sizes=[20, 20]
```

No `Final`/`FinalPartitioned` stage. The `ProjectionExec` handles the column name mapping between the Partial aggregate's intermediate output and the logical schema.

## Emission Behavior: StreamingPartialAggExec

The reference implementation uses a custom `StreamingPartialAggExec` operator that replaces DataFusion's built-in `AggregateExec(Partial)`. It flushes accumulated state every N input batches, giving predictable streaming behavior.

### Why Not Use AggregateExec(Partial)?

DataFusion's built-in `AggregateExec(Partial)` buffers all groups before emitting a single batch at the end. It only emits early as a side effect of memory pressure (`emit_early_if_necessary` in `row_hash.rs`), which is imprecise and data-dependent. There are no knobs for "emit every N batches." A custom operator is needed for controlled emission.

### How StreamingPartialAggExec Works

```rust
struct StreamingPartialAggExec {
    input: Arc<dyn ExecutionPlan>,
    group_exprs: Vec<Arc<dyn PhysicalExpr>>,     // GROUP BY columns
    aggr_exprs: Vec<Arc<AggregateFunctionExpr>>,  // SUM, COUNT, etc.
    emit_every: usize,                            // flush every N input batches
    schema: SchemaRef,
}
```

Execution loop per partition:

1. **Evaluate** group-by expressions and aggregate input expressions on each input batch
2. **Group rows by key** using `ScalarValue` vectors as hash keys
3. **Update accumulators** per group using `Accumulator::update_batch()` with `arrow::compute::take()` to extract per-group sub-arrays
4. **Every N batches**, call `Accumulator::evaluate()` on all groups, emit a `RecordBatch`, clear state
5. **On input exhaustion**, flush remaining accumulated state

### Reusing DataFusion's Accumulator API

The key insight: you don't need to reimplement SUM, COUNT, AVG, etc. DataFusion's `Accumulator` trait is public API:

```rust
// Create accumulators from the same AggregateFunctionExpr the planner produces
let accumulator: Box<dyn Accumulator> = agg_expr.create_accumulator()?;

// Feed it data (arrays, not row-by-row)
accumulator.update_batch(&[values_array])?;

// Extract the result
let result: ScalarValue = accumulator.evaluate()?;
```

What you **do** need to implement yourself:
- **Group key hashing**: DataFusion's `GroupValues` is `pub(crate)` (internal). The reference implementation uses `HashMap<Vec<ScalarValue>, Vec<Box<dyn Accumulator>>>` — simple and correct, though not as fast as DataFusion's specialized hash tables.
- **Row-to-group routing**: For each input row, extract group key values via `ScalarValue::try_from_array()`, look up the group in the hash map, then use `arrow::compute::take()` to build per-group sub-arrays for `update_batch()`.

### Choosing an Emission Trigger

Batch count (`emit_every: usize`) is the right default:

- **Deterministic and testable** — same input always produces same number of flushes
- **Adapts to input rate** — faster data means more frequent flushes
- **No async complexity** — no timers, no `tokio::time::interval`

For time-based triggers, you'd add a `last_flush: Instant` field and check `last_flush.elapsed() > threshold` after each batch. This is straightforward to add but harder to test deterministically.

### Resulting Physical Plan

```
ProjectionExec: expr=[bucket@0 as bucket,
                      sum(metrics.value)@1 as total,
                      count(*)@2 as cnt]
  StreamingPartialAggExec: emit_every=4, group_by=[metrics.bucket],
                           aggr=[sum(metrics.value), count(*)]
    MemoryExec: partitions=2, partition_sizes=[20, 20]
```

The `StreamingPartialAggExec` outputs the final aggregate schema directly (not intermediate state), so the `ProjectionExec` above it only handles aliasing (`total`, `cnt`), not column format translation.

## Required Dependencies

```toml
[dependencies]
datafusion = "45"
arrow = "54"
async-trait = "0.1"
futures = "0.3"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

## Extension Points Summary

| Component | DataFusion Trait | Registration |
|-----------|-----------------|-------------|
| `partial_agg()` UDF | `ScalarUDFImpl` | `ctx.register_udf()` |
| Analyzer rule | `AnalyzerRule` | `SessionStateBuilder::with_analyzer_rule()` |
| Custom logical node | `UserDefinedLogicalNodeCore` | Inserted by analyzer rule as `LogicalPlan::Extension` |
| Extension planner | `ExtensionPlanner` | `DefaultPhysicalPlanner::with_extension_planners()` via custom `QueryPlanner` |

## Reference Implementation

See `datafusion-examples/examples/partial_aggregation_streaming.rs` for a complete, working, self-contained implementation with correctness verification against standard aggregation.
