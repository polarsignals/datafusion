# Partial / Online Aggregations in DataFusion

## Problem Statement

We want to execute a GROUP BY aggregation such as:

```sql
SELECT date_bin('5 minutes', ts) AS bucket, SUM(value)
FROM metrics
GROUP BY bucket
```

and have **intermediate, in-progress results emitted periodically during execution** — not just at the end. The use case is a metrics dashboard that redraws progressively as the query executes, showing partial aggregation values that the client stitches together.

---

## 1. DataFusion Aggregation Internals

### 1.1 AggregateMode — Multi-Phase Pipeline

DataFusion supports multi-phase aggregation via the `AggregateMode` enum
(`datafusion/physical-plan/src/aggregates/mod.rs:65-97`):

```rust
pub enum AggregateMode {
    Partial,           // Phase 1: parallel partial aggregation per partition
    Final,             // Phase 2: combine partial results into final output
    FinalPartitioned,  // Final agg on pre-partitioned data (hash-repartitioned)
    Single,            // Entire aggregation in one operator
    SinglePartitioned, // Single-phase on pre-partitioned data
}
```

A typical two-phase pipeline looks like:

```text
                          ▲
                          │  evaluate()
              ┌───────────────────────┐
              │  GroupBy (Final)      │   merge_batch() combines states
              └───────────────────────┘
                          ▲
              ┌───────────┴───────────┐
              │   Repartition HASH    │   shuffle by hash(group keys)
              └───────────────────────┘
                          ▲
        ┌─────────────────┴─────────────────┐
        │                                   │
┌───────────────────┐           ┌───────────────────┐
│ GroupBy (Partial)  │           │ GroupBy (Partial)  │  update_batch() + state()
└───────────────────┘           └───────────────────┘
        ▲                                   ▲
   Input Partition 0                   Input Partition 1
```

- **First-stage** modes (`Partial`, `Single`, `SinglePartitioned`) call `update_batch()` to
  process raw input rows.
- **Second-stage** modes (`Final`, `FinalPartitioned`) call `merge_batch()` to combine
  intermediate states from the first stage.

### 1.2 Accumulator Trait

The `Accumulator` trait (`datafusion/expr-common/src/accumulator.rs:52`) handles state
for a **single group**:

| Method | Purpose |
|--------|---------|
| `update_batch(&mut self, values: &[ArrayRef])` | Process input rows, update internal state |
| `evaluate(&mut self) -> ScalarValue` | Produce final aggregate value |
| `state(&mut self) -> Vec<ScalarValue>` | Serialize intermediate state for multi-phase aggregation |
| `merge_batch(&mut self, states: &[ArrayRef])` | Merge intermediate states from other accumulators |
| `size(&self) -> usize` | Report memory usage for resource management |

The key insight: `state()` serializes the accumulator's internal state into values that can
be sent between execution phases. For example:
- **SUM**: state is `[running_sum]` — the state IS the partial result
- **AVG**: state is `[sum, count]` — the client divides sum/count after merging
- **COUNT**: state is `[count]` — merge adds counts together
- **MIN/MAX**: state is `[current_min]` / `[current_max]`

### 1.3 GroupsAccumulator Trait

The `GroupsAccumulator` trait (`datafusion/expr-common/src/groups_accumulator.rs:108`) is a
**vectorized version** that manages state for *all groups simultaneously*:

```rust
pub trait GroupsAccumulator: Send {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()>;

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef>;

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>>;

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()>;
}
```

Note the `EmitTo` parameter on `evaluate()` and `state()` — this controls **which groups**
to emit:

```rust
pub enum EmitTo {
    All,          // Emit all groups, reset state
    First(usize), // Emit first n groups, shift remaining indices down by n
}
```

`EmitTo::First(n)` is the mechanism that enables incremental emission: it emits only the
first N groups while preserving state for remaining groups.

### 1.4 GroupedHashAggregateStream — The Execution Loop

The core execution logic lives in `GroupedHashAggregateStream`
(`datafusion/physical-plan/src/aggregates/row_hash.rs`). Its `poll_next` method
(line ~650) follows this flow for each input batch:

```
1. Read batch from input stream
2. group_aggregate_batch(batch)     ← calls update_batch() on accumulators
3. Check hit_soft_group_limit()     ← if yes, emit all groups and stop
4. Check group_ordering.emit_to()  ← if sorted input, emit completed groups
5. emit_early_if_necessary()        ← if memory pressure (Partial only), emit first N groups
6. When input exhausted → emit all remaining groups
```

The `emit()` method (line 941) does the actual output:
- In `Partial` mode: calls `acc.state(emit_to)` — emits intermediate state columns
- In `Final`/`Single` mode: calls `acc.evaluate(emit_to)` — emits final values

### 1.5 GroupOrdering — Sorted Input Optimization

When input is sorted by group keys, DataFusion tracks this via `GroupOrdering`
(`datafusion/physical-plan/src/aggregates/order/mod.rs`):

```rust
pub enum GroupOrdering {
    None,                           // No ordering, cannot emit early (except memory pressure)
    Partial(GroupOrderingPartial),   // Some group key columns are sorted
    Full(GroupOrderingFull),         // All group key columns are sorted
}
```

With `Full` ordering, as soon as a new group key is seen, all previous groups are complete
and can be emitted via `EmitTo::First(n)`. This is truly incremental — but **each group
is emitted only once, when it's complete**. It does not re-emit a group with updated values.

---

## 2. Existing Early Emission Mechanisms

DataFusion has several mechanisms that cause groups to be emitted before all input is
consumed. None of them achieve the "periodic snapshot" behavior we want.

### 2.1 Sorted Input Emission

**How it works**: When input is sorted by group keys (e.g., data sorted by timestamp for a
`date_bin` GROUP BY), `GroupOrdering::Full` detects when a group is complete and emits it
immediately.

**Limitation for our use case**: Each group is emitted **exactly once** when complete.
If the input for a single `date_bin` bucket spans many batches, you won't see intermediate
progress for that bucket — only the final value once all its rows have been processed.

**When useful**: If your data is sorted by timestamp AND each `date_bin` bucket is small
relative to the input, groups complete quickly and you get near-streaming behavior. But for
a large bucket still being filled, you see nothing until it's done.

### 2.2 Memory Pressure Emission (`emit_early_if_necessary`)

**How it works** (`row_hash.rs:1040-1052`):

```rust
fn emit_early_if_necessary(&mut self) -> Result<()> {
    if self.group_values.len() >= self.batch_size
        && matches!(self.group_ordering, GroupOrdering::None)
        && self.update_memory_reservation().is_err()
    {
        let n = self.group_values.len() / self.batch_size * self.batch_size;
        if let Some(batch) = self.emit(EmitTo::First(n), false)? {
            self.exec_state = ExecutionState::ProducingOutput(batch);
        };
    }
    Ok(())
}
```

**Conditions**: Only fires when (a) in `Partial` mode, (b) `GroupOrdering::None` (unsorted),
(c) memory reservation fails, (d) enough groups accumulated.

**Limitation**: Non-deterministic timing. The same group key may appear in multiple emitted
batches (requiring downstream merge). And it only fires under memory pressure — if your data
fits in memory, it never fires.

### 2.3 Soft Group Limit

**How it works** (`row_hash.rs:1094-1098`):

```rust
fn hit_soft_group_limit(&self) -> bool {
    let Some(group_values_soft_limit) = self.group_values_soft_limit else {
        return false;
    };
    group_values_soft_limit <= self.group_values.len()
}
```

When hit, the stream emits all groups and transitions to `Done` — it **stops reading input**.

**Limitation**: This is only pushed down for pure DISTINCT queries (no aggregate functions
like SUM) by the `LimitedDistinctAggregation` optimizer rule
(`datafusion/physical-optimizer/src/limited_distinct_aggregation.rs`). It requires no
aggregate expressions, no filters, and no ordering — so it **cannot be used for
`SUM(value) GROUP BY bucket`**.

### 2.4 Spilling

When memory is exhausted in `Final`/`FinalPartitioned` modes, groups are spilled to disk
as sorted Arrow IPC files, then merged via a streaming merge sort. This is a memory
management mechanism, not an early-emission mechanism.

### Summary

| Mechanism | Emits in-progress groups? | Periodic? | Works with SUM/AVG? |
|-----------|:------------------------:|:---------:|:-------------------:|
| Sorted input | No (only complete groups) | N/A | Yes |
| Memory pressure | Partially (fragmented) | No (only under pressure) | Only in Partial mode |
| Soft group limit | No (stops reading) | No | No (DISTINCT only) |
| Spilling | No (memory management) | No | Yes |

**None of these achieve periodic snapshots of all in-progress groups.**

---

## 3. SQL-Level Tricks Analysis

Can we write SQL in a way that tricks the execution engine into emitting groups
early/periodically?

### 3.1 LIMIT on Aggregation with Aggregate Functions

```sql
SELECT date_bin('5 min', ts) AS bucket, SUM(value)
FROM metrics GROUP BY bucket LIMIT 5
```

**Does not work.** The `LimitedDistinctAggregation` optimizer rule
(`datafusion/physical-optimizer/src/limited_distinct_aggregation.rs`) only pushes the
LIMIT into the aggregate when there are **no aggregate expressions**. It explicitly
checks for empty aggregate lists. With `SUM(value)`, the aggregation must see all input
to produce correct values.

### 3.2 UNION ALL of Chunked Subqueries

```sql
SELECT bucket, SUM(value) FROM (SELECT * FROM t LIMIT 1000) GROUP BY bucket
UNION ALL
SELECT bucket, SUM(value) FROM (SELECT * FROM t OFFSET 1000 LIMIT 1000) GROUP BY bucket
```

**Partially works** — each arm executes independently. However:
- Each aggregation must finish before the UNION emits its results
- You need to know the data size in advance to write the SQL
- The client must merge matching bucket keys across UNION arms
- Not truly streaming; requires manual SQL construction per chunk

### 3.3 Window Functions

```sql
SELECT *, SUM(value) OVER (PARTITION BY bucket ORDER BY ts) AS running_sum
FROM metrics
```

**Does not work for our purpose.** Window functions execute in a separate
`WindowAggExec` plan node after the aggregation. They don't produce GROUP BY results;
they produce per-row values. The semantics are different.

### 3.4 Low Memory Limit + Small Batch Size

```sql
SET datafusion.execution.batch_size = 100;
-- + configure a very small memory pool
```

**Hacky but partially functional.** This can force `emit_early_if_necessary()` to fire
frequently in `Partial` mode. However:
- Timing is non-deterministic (depends on data distribution and memory usage)
- Only works in `Partial` mode (requires a downstream `Final` operator to merge)
- The same group key appears in multiple output batches
- Performance degrades significantly with very small batch sizes

### 3.5 skip_partial_aggregation Configs

```sql
SET datafusion.execution.skip_partial_aggregation_probe_ratio_threshold = 0.0;
```

**Wrong semantics.** This causes the aggregation to skip aggregation entirely and output
raw rows as intermediate state. It doesn't emit partial aggregation results — it bypasses
aggregation altogether.

### 3.6 TABLESAMPLE

**Not implemented.** The syntax is documented but no implementation exists in the codebase.

### Conclusion

**No SQL-level trick achieves periodic emission of in-progress aggregate state for queries
with aggregate functions (SUM, COUNT, AVG, etc.).** The fundamental issue is that
DataFusion's aggregation operators are designed to emit groups either when complete
(sorted input) or under memory pressure — never on a periodic schedule.

---

## 4. Recommended Approach: Custom ExecutionPlan

**Yes, this is achievable via a custom `ExecutionPlan`.** DataFusion's `ExecutionPlan` trait
is the primary extensibility point and gives you full control over execution behavior.

### 4.1 Architecture

```text
                        Client (dashboard)
                              ▲
                     receives stream of
                     partial state batches,
                     merges by group key
                              ▲
              ┌───────────────────────────────┐
              │  PartialAggregateExec         │
              │  (custom ExecutionPlan)        │
              │                               │
              │  For each chunk of N batches:  │
              │   1. Feed to accumulators      │
              │      via update_batch()        │
              │   2. Call state(EmitTo::All)   │
              │   3. Emit partial state batch  │
              │   4. Reset accumulators        │
              │   5. Continue with next chunk  │
              └───────────────────────────────┘
                              ▲
                        Input Stream
                     (table scan, etc.)
```

### 4.2 How It Works

The custom `PartialAggregateExec` wraps the child input plan and produces a stream that:

1. **Reads N input batches** (configurable `chunk_size`)
2. For each batch, calls `group_aggregate_batch()` — evaluating group-by expressions,
   computing group indices, and calling `update_batch()` on each accumulator
3. After N batches, calls `state(EmitTo::All)` on all accumulators to snapshot their
   intermediate state
4. **Emits a RecordBatch** with columns `[group_keys..., accumulator_state_columns...]`
5. Resets all accumulators and group values
6. Repeats from step 1

When the input stream is exhausted, it emits one final partial state batch for any
remaining data.

### 4.3 Output Schema

For `SUM(value) GROUP BY date_bin('5 min', ts)`:

Each emitted batch has columns:
```
┌──────────────────────┬──────────────────┐
│ bucket (Timestamp)   │ sum_state (F64)  │
├──────────────────────┼──────────────────┤
│ 2024-01-01 00:00:00  │ 150.0            │
│ 2024-01-01 00:05:00  │ 230.0            │
│ 2024-01-01 00:10:00  │ 45.0             │
└──────────────────────┴──────────────────┘
```

For `AVG(value)`, the state has two columns:
```
┌──────────────────────┬──────────────────┬──────────────────┐
│ bucket (Timestamp)   │ avg_sum (F64)    │ avg_count (U64)  │
├──────────────────────┼──────────────────┼──────────────────┤
│ 2024-01-01 00:00:00  │ 150.0            │ 3                │
│ 2024-01-01 00:05:00  │ 230.0            │ 5                │
└──────────────────────┴──────────────────┴──────────────────┘
```

The client merges by summing `avg_sum` and `avg_count` per bucket, then computing
`final_avg = total_sum / total_count`.

### 4.4 Key DataFusion APIs to Reuse

| API | Location | Purpose |
|-----|----------|---------|
| `GroupValues` trait | `physical-plan/src/aggregates/group_values/mod.rs` | Track and intern group keys |
| `create_group_accumulator()` | Via `AggregateFunctionExpr` | Create vectorized accumulators |
| `evaluate_group_by()` | `physical-plan/src/aggregates/mod.rs` | Evaluate GROUP BY expressions on a batch |
| `evaluate_many()` | `physical-plan/src/aggregates/mod.rs` | Evaluate aggregate input expressions |
| `state(EmitTo::All)` | `GroupsAccumulator` trait | Snapshot intermediate state for all groups |
| `GroupValues::intern()` | `GroupValues` trait | Map rows to group indices |
| `GroupValues::emit(EmitTo::All)` | `GroupValues` trait | Emit group key columns |

### 4.5 Pseudocode

```rust
struct PartialAggregateStream {
    input: SendableRecordBatchStream,
    group_by: PhysicalGroupBy,
    accumulators: Vec<Box<dyn GroupsAccumulator>>,
    group_values: Box<dyn GroupValues>,
    current_group_indices: Vec<usize>,
    chunk_size: usize,        // emit after this many input batches
    batches_since_emit: usize,
    schema: SchemaRef,        // output schema: group keys + state fields
    aggregate_arguments: Vec<Vec<Arc<dyn PhysicalExpr>>>,
    filter_expressions: Vec<Option<Arc<dyn PhysicalExpr>>>,
    finished: bool,
}

impl Stream for PartialAggregateStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }

        loop {
            match self.input.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(batch))) => {
                    // 1. Evaluate group-by expressions
                    let group_by_values = evaluate_group_by(&self.group_by, &batch)?;
                    let input_values = evaluate_many(&self.aggregate_arguments, &batch)?;
                    let filter_values = evaluate_optional(&self.filter_expressions, &batch)?;

                    for group_values in &group_by_values {
                        // 2. Intern group keys, get group indices
                        self.group_values.intern(group_values, &mut self.current_group_indices)?;
                        let total_num_groups = self.group_values.len();

                        // 3. Update accumulators
                        for ((acc, values), opt_filter) in
                            self.accumulators.iter_mut()
                                .zip(input_values.iter())
                                .zip(filter_values.iter())
                        {
                            acc.update_batch(
                                values,
                                &self.current_group_indices,
                                opt_filter.as_ref().map(|f| f.as_boolean()),
                                total_num_groups,
                            )?;
                        }
                    }

                    self.batches_since_emit += 1;

                    // 4. Emit partial state if chunk_size reached
                    if self.batches_since_emit >= self.chunk_size {
                        let batch = self.emit_partial_state()?;
                        self.batches_since_emit = 0;
                        if let Some(batch) = batch {
                            return Poll::Ready(Some(Ok(batch)));
                        }
                    }
                }

                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),

                Poll::Ready(None) => {
                    // Input exhausted — emit remaining state
                    self.finished = true;
                    let batch = self.emit_partial_state()?;
                    return Poll::Ready(batch.map(Ok));
                }

                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl PartialAggregateStream {
    fn emit_partial_state(&mut self) -> Result<Option<RecordBatch>> {
        if self.group_values.is_empty() {
            return Ok(None);
        }

        // Emit group key columns
        let mut output = self.group_values.emit(EmitTo::All)?;

        // Emit accumulator state columns
        for acc in self.accumulators.iter_mut() {
            output.extend(acc.state(EmitTo::All)?);
        }

        let batch = RecordBatch::try_new(Arc::clone(&self.schema), output)?;
        Ok(Some(batch))
    }
}
```

### 4.6 Integration Pattern

To use this in practice:

```rust
// 1. Build the logical plan normally
let df = ctx.sql("SELECT date_bin('5 min', ts) AS bucket, SUM(value) FROM metrics GROUP BY bucket").await?;

// 2. Get the physical plan
let plan = df.create_physical_plan().await?;

// 3. Replace the AggregateExec with PartialAggregateExec
//    (or build the physical plan manually)
let partial_plan = PartialAggregateExec::new(
    input_plan,          // the table scan
    group_by_exprs,      // date_bin('5 min', ts)
    aggregate_exprs,     // SUM(value)
    chunk_size: 10,      // emit after every 10 input batches
);

// 4. Execute and consume the stream
let stream = partial_plan.execute(0, task_ctx)?;
while let Some(batch) = stream.next().await {
    let batch = batch?;
    // Send to dashboard — each batch contains partial state
    // Client merges by bucket key:
    //   for SUM: add the state values per bucket
    //   for AVG: add sums and counts, then divide
    send_to_dashboard(batch);
}
```

### 4.7 Aggregate State Semantics by Function

Understanding how to merge partial states on the client side:

| Function | State Fields | Merge Operation | Final Computation |
|----------|-------------|-----------------|-------------------|
| SUM | `[sum]` | `total_sum += chunk_sum` | `total_sum` |
| COUNT | `[count]` | `total_count += chunk_count` | `total_count` |
| AVG | `[sum, count]` | `total_sum += sum; total_count += count` | `total_sum / total_count` |
| MIN | `[min]` | `total_min = min(total_min, chunk_min)` | `total_min` |
| MAX | `[max]` | `total_max = max(total_max, chunk_max)` | `total_max` |

---

## 5. Alternative Approaches Comparison

| Approach | Complexity | Performance | Control | Maintenance |
|----------|:----------:|:-----------:|:-------:|:-----------:|
| **Custom ExecutionPlan** (recommended) | Medium | Good | Full control over emission timing | Medium — depends on internal APIs but they're stable |
| Fork `GroupedHashAggregateStream` | High | Best | Full | High — must track upstream changes |
| Low memory limit hack | Low | Poor | None (non-deterministic) | Low |
| UNION ALL of chunked queries | Low | Poor (re-reads data) | Manual | Low |
| Custom UDAF only | Low | N/A | None — UDAFs don't control emission timing | Low |
| Multiple separate queries | Low | Worst (full re-scan per query) | Manual | Low |

The custom `ExecutionPlan` approach is recommended because it:
- Uses DataFusion's public extensibility point (`ExecutionPlan` trait)
- Gives full control over emission timing
- Reuses existing accumulator infrastructure (no need to reimplement SUM, AVG, etc.)
- Can be integrated without forking DataFusion
- Output is standard `RecordBatch` stream, compatible with Arrow Flight or any other transport
