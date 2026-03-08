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

//! # Streaming Partial Aggregation Example
//!
//! Demonstrates how to get progressive/streaming partial aggregation results
//! from DataFusion by using a low memory limit to force early emission of
//! partial aggregate state.
//!
//! ## How it works
//!
//! DataFusion's `AggregateMode::Partial` operator emits intermediate results
//! early when memory pressure is detected (see `emit_early_if_necessary` in
//! `row_hash.rs`). By configuring a small memory pool, we force frequent
//! emission of partial results that a client can merge incrementally.
//!
//! The key insight: for simple aggregates like SUM and COUNT, the intermediate
//! state format is identical to the final result (just the running sum/count),
//! so client-side merging is trivial — sum the partial SUMs, sum the partial
//! COUNTs.
//!
//! ## Architecture
//!
//! ```text
//! [MemTable / DataSource]
//!         |
//!         v
//! [AggregateExec (Partial mode)]  ← emits early under memory pressure
//!         |
//!         v
//!    [Client merges partial results by group key]
//! ```
//!
//! We bypass the Final aggregation node by extracting and executing only
//! the Partial aggregate from the physical plan. This way partial batches
//! flow directly to the client instead of being buffered by a Final node.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::execution::memory_pool::GreedyMemoryPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use datafusion::prelude::*;
use futures::StreamExt;

/// Generate test data: many batches of (group_key, value) pairs.
/// Creates enough data that memory pressure will force early emission.
fn make_test_batches(num_batches: usize, rows_per_batch: usize) -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("bucket", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
    ]));

    // Use many distinct groups so the hash table grows large enough
    // to trigger memory pressure and early emission.
    let num_distinct_groups = 10_000;

    (0..num_batches)
        .map(|batch_idx| {
            let bucket_data: Vec<String> = (0..rows_per_batch)
                .map(|row| {
                    format!(
                        "group_{:05}",
                        (batch_idx * rows_per_batch + row) % num_distinct_groups
                    )
                })
                .collect();
            let value_data: Vec<f64> = (0..rows_per_batch)
                .map(|row| ((batch_idx * rows_per_batch + row) % 100) as f64)
                .collect();

            let bucket_refs: Vec<&str> = bucket_data.iter().map(|s| s.as_str()).collect();
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(StringArray::from(bucket_refs)),
                    Arc::new(Float64Array::from(value_data)),
                ],
            )
            .unwrap()
        })
        .collect()
}

/// Walk the physical plan tree and find the first AggregateExec node.
/// Returns the node and whether it is in Partial mode.
fn find_partial_aggregate(
    plan: &Arc<dyn ExecutionPlan>,
) -> Option<Arc<dyn ExecutionPlan>> {
    if let Some(agg) = plan.as_any().downcast_ref::<AggregateExec>() {
        if agg.mode() == &AggregateMode::Partial {
            return Some(Arc::clone(plan));
        }
    }
    for child in plan.children() {
        if let Some(found) = find_partial_aggregate(child) {
            return Some(found);
        }
    }
    None
}

#[tokio::main]
async fn main() -> datafusion::common::Result<()> {
    // 1. Create a session with a very small memory pool to force early emission.
    //    Tune this value based on your data size and desired emission frequency.
    //    Smaller = more frequent partial emissions.
    // With 10,000 distinct groups, each group needs ~30 bytes for key + 8 bytes for
    // SUM + 8 bytes for COUNT ≈ ~46 bytes/group. 10K groups ≈ 460 KB of accumulator
    // state. Set the pool to ~300 KB so memory pressure triggers before all groups
    // are accumulated, forcing multiple early emissions.
    let memory_limit_bytes = 300_000; // 300 KB
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(GreedyMemoryPool::new(memory_limit_bytes)))
        .build_arc()?;

    let config = SessionConfig::new()
        // Multiple partitions forces Partial -> Final plan structure.
        // (With 1 partition the optimizer picks Single mode, which
        // does NOT support early emission.)
        .with_target_partitions(2)
        // Large-ish batches so each batch uses meaningful memory
        .with_batch_size(4096);

    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_runtime_env(runtime)
        .with_default_features()
        .build();
    let ctx = SessionContext::from(state);

    // 2. Register test data (split across 2 partitions to match target_partitions)
    let batches1 = make_test_batches(50, 4096); // partition 1: 50 batches
    let batches2 = make_test_batches(50, 4096); // partition 2: 50 batches
    let schema = batches1[0].schema();
    let table = datafusion::datasource::MemTable::try_new(
        schema,
        vec![batches1, batches2],
    )?;
    ctx.register_table("metrics", Arc::new(table))?;

    // 3. Create the physical plan via SQL
    let df = ctx.sql("SELECT bucket, SUM(value) as total, COUNT(*) as cnt FROM metrics GROUP BY bucket").await?;
    let plan = df.create_physical_plan().await?;

    println!("Full physical plan:\n{}\n", datafusion::physical_plan::displayable(plan.as_ref()).indent(true));

    // 4. Find and execute ONLY the Partial aggregate node.
    //    This bypasses the Final node so partial results stream to us directly.
    let partial_agg = find_partial_aggregate(&plan)
        .expect("Expected to find a Partial AggregateExec in the plan");

    println!("Executing partial aggregate directly (all partitions)...\n");

    // 5. Consume partial results from ALL partitions and merge client-side.
    //    For SUM: sum the partial sums per group.
    //    For COUNT: sum the partial counts per group.
    let mut merged_sum: HashMap<String, f64> = HashMap::new();
    let mut merged_count: HashMap<String, i64> = HashMap::new();
    let mut batch_num = 0;

    let num_partitions = partial_agg.output_partitioning().partition_count();
    for partition in 0..num_partitions {
        let task_ctx = ctx.task_ctx();
        let mut stream = partial_agg.execute(partition, task_ctx)?;

        while let Some(result) = stream.next().await {
            let batch = result?;
            batch_num += 1;

            // Partial mode output schema for "SUM(value), COUNT(*)" is:
            //   col 0: bucket (group key)
            //   col 1: SUM intermediate state (same as the running sum for SUM)
            //   col 2: COUNT intermediate state (same as the running count for COUNT)
            let buckets = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let sums = batch
                .column(1)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let counts = batch.column(2);

            println!(
                "Partition {partition}, Batch {batch_num}: {} rows, {} groups emitted",
                batch.num_rows(),
                buckets.len()
            );

            for row in 0..batch.num_rows() {
                let key = buckets.value(row).to_string();
                let sum_val = sums.value(row);
                let count_val = extract_i64(counts.as_ref(), row);

                *merged_sum.entry(key.clone()).or_default() += sum_val;
                *merged_count.entry(key).or_default() += count_val;
            }
        }
    }

    // Print first 10 groups of final merged results
    println!("\n--- Final merged results after {batch_num} partial batches ---");
    let mut keys: Vec<_> = merged_sum.keys().cloned().collect();
    keys.sort();
    for key in keys.iter().take(10) {
        println!(
            "  bucket={}: sum={:.1}, count={}",
            key,
            merged_sum[key],
            merged_count[key]
        );
    }
    println!("  ... ({} total groups)", keys.len());

    // 6. Verify against a normal (non-streaming) aggregation with unlimited memory.
    //    Use the same data layout (2 partitions of 50 batches each).
    println!("\n--- Verification (unlimited memory) ---");
    let ctx2 = SessionContext::new();
    let v_batches1 = make_test_batches(50, 4096);
    let v_batches2 = make_test_batches(50, 4096);
    let v_schema = v_batches1[0].schema();
    let table2 = datafusion::datasource::MemTable::try_new(
        v_schema,
        vec![v_batches1, v_batches2],
    )?;
    ctx2.register_table("metrics", Arc::new(table2))?;
    let df2 = ctx2
        .sql("SELECT bucket, SUM(value) as total, COUNT(*) as cnt FROM metrics GROUP BY bucket ORDER BY bucket")
        .await?;
    let results = df2.collect().await?;
    let mut shown = 0;
    for batch in &results {
        let buckets = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let sums = batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let counts = batch.column(2);
        for row in 0..batch.num_rows() {
            if shown < 10 {
                let key = buckets.value(row);
                let sum_val = sums.value(row);
                let count_val = extract_i64(counts.as_ref(), row);
                println!("  bucket={}: sum={:.1}, count={}", key, sum_val, count_val);
                shown += 1;
            }
        }
    }
    println!("  ... ({} total groups)", results.iter().map(|b| b.num_rows()).sum::<usize>());

    Ok(())
}

/// Extract an i64 value from an array that may be Int32, Int64, UInt32, etc.
fn extract_i64(array: &dyn Array, row: usize) -> i64 {
    if let Some(a) = array.as_any().downcast_ref::<arrow::array::Int64Array>() {
        a.value(row)
    } else if let Some(a) = array.as_any().downcast_ref::<Int32Array>() {
        a.value(row) as i64
    } else if let Some(a) = array.as_any().downcast_ref::<arrow::array::UInt64Array>() {
        a.value(row) as i64
    } else {
        panic!(
            "Unexpected count column type: {:?}",
            array.data_type()
        );
    }
}
