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

## Emission Behavior and Controlling Flush Frequency

### What the Marker UDF Gives You

The marker UDF approach removes the Final stage so partial results flow directly to the client. However, `AggregateExec(Partial)` is still DataFusion's built-in hash aggregate. Its emission behavior is:

- **Default**: It accumulates all input batches into a hash table, then emits one output batch per input partition at the end. This is technically "partial" (the state format is intermediate), but it's not truly streaming — the client still waits for all input to be consumed.
- **Under memory pressure**: DataFusion's `emit_early_if_necessary` (in `row_hash.rs`) spills partial state when the memory pool is exhausted. This produces multiple partial batches for the same group keys, but it's a side effect of memory management, not a deliberate emission strategy.

In other words: removing the Final stage is necessary but not sufficient for true streaming. The built-in `AggregateExec(Partial)` doesn't have knobs for "emit every N batches" or "emit every T seconds."

### Approaches for Controlled Emission

For production use, you'll likely want one of these:

**Option A: Custom ExecutionPlan operator.** Replace `AggregateExec(Partial)` in the `ExtensionPlanner` with a custom `ExecutionPlan` that wraps the input scan and implements its own hash aggregation with explicit flush triggers:

```rust
struct StreamingPartialAggregateExec {
    input: Arc<dyn ExecutionPlan>,
    group_exprs: Vec<Arc<dyn PhysicalExpr>>,
    aggr_exprs: Vec<Arc<AggregateFunctionExpr>>,
    /// Flush partial state every N input batches
    emit_every_n_batches: usize,
    /// Or flush when this duration has elapsed since last flush
    emit_interval: Option<Duration>,
}
```

The `execute()` method would consume input batches, accumulate into a hash table, and periodically flush (clear and emit) the accumulated state based on whichever trigger fires first. The reference example in `streaming_partial_aggregate.rs` demonstrates this pattern with a hardcoded `emit_every` batch count.

This gives you full control over emission but requires reimplementing the hash aggregation logic (or wrapping DataFusion's accumulators).

**Option B: Memory pool tuning (pragmatic hack).** Keep the built-in `AggregateExec(Partial)` and configure a small `GreedyMemoryPool` to force frequent early emission:

```rust
let runtime = RuntimeEnvBuilder::new()
    .with_memory_pool(Arc::new(GreedyMemoryPool::new(300_000)))
    .build_arc()?;
```

This is simple but imprecise — emission frequency depends on data characteristics (number of distinct groups, row sizes) rather than explicit triggers. It's useful for prototyping but not recommended for production.

**Option C: Extend AggregateExec (upstream contribution).** Add batch-count or time-based emission triggers to DataFusion's `AggregateExec(Partial)` itself. This would be the cleanest long-term solution but requires changes to DataFusion core.

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
