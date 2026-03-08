// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! # Streaming Partial Aggregate — Custom Operator Proof-of-Concept
//!
//! Implements a custom `ExecutionPlan` node (`StreamingPartialAggregateExec`)
//! that periodically flushes partial aggregation state to the client without
//! waiting for all input to be consumed. This proves that:
//!
//! 1. Time-to-first-row is dramatically lower than a full aggregation.
//! 2. The client receives multiple partial batches for the same group keys.
//! 3. Client-side state can be updated incrementally during execution.
//!
//! ## Architecture
//!
//! ```text
//! [MemTable scan]
//!       |
//!       v
//! [StreamingPartialAggregateExec]   ← flushes every N input batches
//!       |
//!       v
//! [Client merges partial SUM/COUNT by group key]
//! ```

use std::any::Any;
use std::collections::HashMap;
use std::fmt::{self, Formatter};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{
    Array, Int64Array, RecordBatch, TimestampNanosecondArray, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit};
use datafusion::error::Result;
use datafusion::execution::context::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion::prelude::*;
use futures::{stream, StreamExt};

// ────────────────────────────────────────────────────────────────────────────
// Custom operator: StreamingPartialAggregateExec
// ────────────────────────────────────────────────────────────────────────────

/// A custom ExecutionPlan that performs GROUP BY date_bin(..., ts), SUM(value)
/// and flushes partial state every `emit_every` input batches.
///
/// This is intentionally specialized for the proof-of-concept query shape:
///   SELECT date_bin(interval, ts) AS bucket, SUM(value) AS total
///   FROM t GROUP BY 1
#[derive(Debug)]
struct StreamingPartialAggregateExec {
    input: Arc<dyn ExecutionPlan>,
    /// How often (in input batches) to flush partial state.
    emit_every: usize,
    /// The date_bin interval in nanoseconds.
    bin_interval_ns: i64,
    /// Output schema: (bucket: TimestampNanosecond, total: Int64, count: UInt64)
    schema: SchemaRef,
    cache: PlanProperties,
}

impl StreamingPartialAggregateExec {
    fn new(
        input: Arc<dyn ExecutionPlan>,
        emit_every: usize,
        bin_interval_ns: i64,
    ) -> Self {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "bucket",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                false,
            ),
            Field::new("total", DataType::Int64, false),
            Field::new("count", DataType::UInt64, false),
        ]));
        let cache = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            // Preserve the input's partition count so we run one stream per partition.
            input.output_partitioning().clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            input,
            emit_every,
            bin_interval_ns,
            schema,
            cache,
        }
    }

    /// Convert the in-memory accumulator to a RecordBatch and clear it.
    fn flush(
        accum: &mut HashMap<i64, (i64, u64)>,
        schema: &SchemaRef,
    ) -> Result<RecordBatch> {
        let len = accum.len();
        let mut buckets = Vec::with_capacity(len);
        let mut totals = Vec::with_capacity(len);
        let mut counts = Vec::with_capacity(len);
        for (&bucket_ns, &(sum, cnt)) in accum.iter() {
            buckets.push(bucket_ns);
            totals.push(sum);
            counts.push(cnt);
        }
        accum.clear();
        Ok(RecordBatch::try_new(
            Arc::clone(schema),
            vec![
                Arc::new(TimestampNanosecondArray::from(buckets)),
                Arc::new(Int64Array::from(totals)),
                Arc::new(UInt64Array::from(counts)),
            ],
        )?)
    }
}

impl DisplayAs for StreamingPartialAggregateExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "StreamingPartialAggregateExec(emit_every={}, bin_ns={})",
            self.emit_every, self.bin_interval_ns
        )
    }
}

impl ExecutionPlan for StreamingPartialAggregateExec {
    fn name(&self) -> &'static str {
        "StreamingPartialAggregateExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            Arc::clone(&children[0]),
            self.emit_every,
            self.bin_interval_ns,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, context)?;
        let emit_every = self.emit_every;
        let bin_ns = self.bin_interval_ns;
        let schema = Arc::clone(&self.schema);
        let schema2 = Arc::clone(&self.schema);

        let output_stream = stream::unfold(
            (input_stream, HashMap::<i64, (i64, u64)>::new(), 0usize, false),
            move |(mut stream, mut accum, mut batch_count, done)| {
                let schema = Arc::clone(&schema);
                async move {
                    if done {
                        return None;
                    }
                    loop {
                        match stream.next().await {
                            Some(Ok(batch)) => {
                                // --- Compute date_bin on the timestamp column ---
                                let ts_col = batch
                                    .column(0)
                                    .as_any()
                                    .downcast_ref::<TimestampNanosecondArray>()
                                    .expect("column 0 must be TimestampNanosecond");
                                let val_col = batch
                                    .column(1)
                                    .as_any()
                                    .downcast_ref::<Int64Array>()
                                    .expect("column 1 must be Int64");

                                for i in 0..batch.num_rows() {
                                    let ts = ts_col.value(i);
                                    // date_bin: floor to interval boundary
                                    let bucket = ts - ts.rem_euclid(bin_ns);
                                    let val = val_col.value(i);
                                    let entry = accum.entry(bucket).or_insert((0i64, 0u64));
                                    entry.0 += val;
                                    entry.1 += 1;
                                }

                                batch_count += 1;
                                if batch_count % emit_every == 0 && !accum.is_empty() {
                                    let rb = Self::flush(&mut accum, &schema).ok()?;
                                    return Some((Ok(rb), (stream, accum, batch_count, false)));
                                }
                            }
                            Some(Err(e)) => {
                                return Some((Err(e), (stream, accum, batch_count, true)));
                            }
                            None => {
                                // Input exhausted — flush remaining state.
                                if !accum.is_empty() {
                                    let rb = Self::flush(&mut accum, &schema).ok()?;
                                    return Some((
                                        Ok(rb),
                                        (stream, accum, batch_count, true),
                                    ));
                                }
                                return None;
                            }
                        }
                    }
                }
            },
        );

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema2,
            output_stream,
        )))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Client-side state
// ────────────────────────────────────────────────────────────────────────────

/// Simulates the client-side in-memory state that gets updated as partial
/// results stream in.
struct ClientState {
    /// bucket_ns → (running_sum, running_count)
    groups: HashMap<i64, (i64, u64)>,
    /// Tracks how many times each group key has been seen across batches.
    /// A count > 1 proves the same group key appeared in multiple partial batches.
    group_seen_count: HashMap<i64, usize>,
    batches_received: usize,
    first_batch_at: Option<Instant>,
    started_at: Instant,
}

impl ClientState {
    fn new() -> Self {
        Self {
            groups: HashMap::new(),
            group_seen_count: HashMap::new(),
            batches_received: 0,
            first_batch_at: None,
            started_at: Instant::now(),
        }
    }

    /// Merge a partial batch into the running state.
    /// This is called **during** stream execution, proving the client can
    /// update its state incrementally before all results are available.
    fn merge_batch(&mut self, batch: &RecordBatch) {
        if self.first_batch_at.is_none() {
            self.first_batch_at = Some(Instant::now());
        }
        self.batches_received += 1;

        let buckets = batch
            .column(0)
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .unwrap();
        let totals = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let counts = batch
            .column(2)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();

        for i in 0..batch.num_rows() {
            let bucket = buckets.value(i);
            let entry = self.groups.entry(bucket).or_insert((0, 0));
            entry.0 += totals.value(i);
            entry.1 += counts.value(i);
            *self.group_seen_count.entry(bucket).or_insert(0) += 1;
        }
    }

    /// Number of group keys that appeared in more than one partial batch.
    fn duplicate_group_count(&self) -> usize {
        self.group_seen_count.values().filter(|&&c| c > 1).count()
    }

    fn time_to_first_row(&self) -> std::time::Duration {
        self.first_batch_at
            .map(|t| t.duration_since(self.started_at))
            .unwrap_or_default()
    }

    fn total_time(&self) -> std::time::Duration {
        self.started_at.elapsed()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Test data generation
// ────────────────────────────────────────────────────────────────────────────

/// Generate 1M rows of (ts: TimestampNanosecond, value: Int64).
/// Timestamps are randomly distributed over a ~116-day range so that
/// date_bin('1000 seconds') produces ~10,000 distinct groups.
fn make_test_data(num_rows: usize) -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
        Field::new("value", DataType::Int64, false),
    ]));

    // 10,000 groups × 1,000 seconds/group = 10,000,000 seconds ≈ 116 days
    let range_ns: i64 = 10_000_000 * 1_000_000_000; // 10M seconds in nanos

    // Simple deterministic pseudo-random via LCG
    let mut rng_state: u64 = 42;
    let mut next_rand = move || -> u64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        rng_state
    };

    let batch_size = 8192;
    let mut batches = Vec::new();
    let mut remaining = num_rows;

    while remaining > 0 {
        let n = remaining.min(batch_size);
        let ts_data: Vec<i64> = (0..n)
            .map(|_| (next_rand() % range_ns as u64) as i64)
            .collect();
        let val_data: Vec<i64> = (0..n).map(|_| (next_rand() % 1000) as i64).collect();

        batches.push(
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(TimestampNanosecondArray::from(ts_data)),
                    Arc::new(Int64Array::from(val_data)),
                ],
            )
            .unwrap(),
        );
        remaining -= n;
    }
    batches
}

// ────────────────────────────────────────────────────────────────────────────
// Baseline: normal DataFusion aggregation (Single mode, full blocking)
// ────────────────────────────────────────────────────────────────────────────

async fn run_baseline(batches: Vec<RecordBatch>) -> Result<(std::time::Duration, HashMap<i64, (i64, u64)>)> {
    let schema = batches[0].schema();
    let ctx = SessionContext::new();
    let table =
        datafusion::datasource::MemTable::try_new(schema, vec![batches])?;
    ctx.register_table("t", Arc::new(table))?;

    let df = ctx
        .sql(
            "SELECT date_bin(INTERVAL '1000 seconds', ts) AS bucket, \
             SUM(value) AS total, COUNT(*) AS cnt \
             FROM t GROUP BY 1",
        )
        .await?;

    let start = Instant::now();
    let mut first_row_time = None;
    let mut groups: HashMap<i64, (i64, u64)> = HashMap::new();

    let plan = df.create_physical_plan().await?;
    let task_ctx = ctx.task_ctx();
    let num_partitions = plan.output_partitioning().partition_count();

    for partition in 0..num_partitions {
        let mut stream = plan.execute(partition, Arc::clone(&task_ctx))?;
        while let Some(result) = stream.next().await {
            let batch = result?;
            if first_row_time.is_none() && batch.num_rows() > 0 {
                first_row_time = Some(start.elapsed());
            }
            let buckets = batch
                .column(0)
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .unwrap();
            let totals = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let counts = batch.column(2);
            let counts = if let Some(a) = counts.as_any().downcast_ref::<Int64Array>() {
                a.values().iter().map(|&v| v as u64).collect::<Vec<_>>()
            } else if let Some(a) = counts.as_any().downcast_ref::<UInt64Array>() {
                a.values().to_vec()
            } else {
                panic!("unexpected count type: {:?}", counts.data_type());
            };
            for i in 0..batch.num_rows() {
                let entry = groups.entry(buckets.value(i)).or_insert((0, 0));
                entry.0 += totals.value(i);
                entry.1 += counts[i];
            }
        }
    }

    Ok((first_row_time.unwrap_or(start.elapsed()), groups))
}

// ────────────────────────────────────────────────────────────────────────────
// Streaming: custom operator (no Final stage, periodic flush)
// ────────────────────────────────────────────────────────────────────────────

async fn run_streaming(
    batches: Vec<RecordBatch>,
    emit_every: usize,
) -> Result<ClientState> {
    let schema = batches[0].schema();
    let ctx = SessionContext::new();
    let table =
        datafusion::datasource::MemTable::try_new(Arc::clone(&schema), vec![batches])?;
    ctx.register_table("t", Arc::new(table))?;

    // Build a physical plan for just a table scan (no aggregation).
    let df = ctx.sql("SELECT ts, value FROM t").await?;
    let scan_plan = df.create_physical_plan().await?;

    // 1000-second bins in nanoseconds
    let bin_ns: i64 = 1_000 * 1_000_000_000;

    // Wrap the scan with our custom streaming partial aggregate.
    let streaming_plan: Arc<dyn ExecutionPlan> = Arc::new(
        StreamingPartialAggregateExec::new(scan_plan, emit_every, bin_ns),
    );

    let mut client = ClientState::new();
    let task_ctx = ctx.task_ctx();
    let num_partitions = streaming_plan.output_partitioning().partition_count();

    for partition in 0..num_partitions {
        let mut stream = streaming_plan.execute(partition, Arc::clone(&task_ctx))?;
        while let Some(result) = stream.next().await {
            let batch = result?;
            // *** Client-side state update happens here, mid-stream ***
            client.merge_batch(&batch);
        }
    }

    Ok(client)
}

// ────────────────────────────────────────────────────────────────────────────
// Main: run both approaches and compare
// ────────────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let num_rows = 1_000_000;
    let emit_every = 4; // flush partial state every 4 input batches

    println!("Generating {num_rows} rows of test data...");
    let data = make_test_data(num_rows);
    let data_clone = data.clone();

    // --- Streaming (custom operator) ---
    println!("\n=== Streaming partial aggregate (custom operator) ===");
    let client = run_streaming(data, emit_every).await?;

    println!(
        "  Time to first row:  {:?}",
        client.time_to_first_row()
    );
    println!(
        "  Total time:         {:?}",
        client.total_time()
    );
    println!(
        "  Partial batches:    {}",
        client.batches_received
    );
    println!(
        "  Distinct groups:    {}",
        client.groups.len()
    );
    println!(
        "  Groups seen >1 time (duplicates): {}",
        client.duplicate_group_count()
    );

    // --- Baseline (normal DataFusion aggregation) ---
    println!("\n=== Baseline (normal full aggregation) ===");
    let baseline_start = Instant::now();
    let (baseline_ttfr, baseline_groups) = run_baseline(data_clone).await?;
    let baseline_total = baseline_start.elapsed();

    println!("  Time to first row:  {:?}", baseline_ttfr);
    println!("  Total time:         {:?}", baseline_total);
    println!(
        "  Distinct groups:    {}",
        baseline_groups.len()
    );

    // --- Verify correctness: streaming merged state must match baseline ---
    println!("\n=== Verification ===");
    let mut mismatches = 0;
    for (bucket, (b_sum, b_cnt)) in &baseline_groups {
        match client.groups.get(bucket) {
            Some((s_sum, s_cnt)) if s_sum == b_sum && s_cnt == b_cnt => {}
            Some((s_sum, s_cnt)) => {
                if mismatches < 5 {
                    println!(
                        "  MISMATCH bucket={}: baseline=({}, {}), streaming=({}, {})",
                        bucket, b_sum, b_cnt, s_sum, s_cnt
                    );
                }
                mismatches += 1;
            }
            None => {
                if mismatches < 5 {
                    println!("  MISSING bucket={} in streaming results", bucket);
                }
                mismatches += 1;
            }
        }
    }

    if mismatches == 0 && client.groups.len() == baseline_groups.len() {
        println!(
            "  ✓ All {} groups match between streaming and baseline!",
            client.groups.len()
        );
    } else {
        println!("  ✗ {} mismatches found", mismatches);
    }

    // --- Assertions for the proof-of-concept ---
    // 1. Streaming should have emitted multiple partial batches.
    assert!(
        client.batches_received > 1,
        "Expected multiple partial batches, got {}",
        client.batches_received
    );

    // 2. Time-to-first-row for streaming should be significantly lower.
    //    (We don't hard-assert a ratio since CI machines vary, but print it.)
    let speedup = baseline_ttfr.as_secs_f64() / client.time_to_first_row().as_secs_f64();
    println!(
        "\n  Time-to-first-row speedup: {:.1}x  (streaming {:?} vs baseline {:?})",
        speedup,
        client.time_to_first_row(),
        baseline_ttfr
    );

    // 3. Many group keys must appear in more than one partial batch.
    //    This proves the client receives partial (incomplete) results
    //    for the same group key across multiple emissions.
    let dupes = client.duplicate_group_count();
    println!(
        "  Duplicate group keys across batches: {} / {}",
        dupes,
        client.groups.len()
    );
    assert!(
        dupes > 0,
        "Expected some group keys to appear in multiple partial batches"
    );

    // 4. Correctness: merged streaming results must match baseline.
    assert_eq!(
        client.groups.len(),
        baseline_groups.len(),
        "Group count mismatch"
    );
    assert_eq!(mismatches, 0, "Some groups did not match");

    println!("\nAll assertions passed.");
    Ok(())
}
