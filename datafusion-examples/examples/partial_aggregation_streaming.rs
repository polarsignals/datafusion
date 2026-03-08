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

//! # Streaming Partial Aggregation via Marker UDF
//!
//! This example demonstrates how to implement a `partial_agg()` marker UDF
//! that instructs DataFusion to produce only the **Partial** aggregation
//! stage — skipping the Final/FinalPartitioned stage entirely. This lets
//! a client receive progressive partial results and merge them
//! incrementally.
//!
//! ## Approach
//!
//! 1. **Marker UDF (`partial_agg`)**: A no-op scalar function that wraps
//!    aggregate expressions in SQL:
//!    ```sql
//!    SELECT bucket, partial_agg(SUM(value)) AS total
//!    FROM metrics GROUP BY bucket
//!    ```
//!    It exists only as a syntactic signal; it is stripped before execution.
//!
//! 2. **Analyzer Rule (`PartialAggregateRule`)**: Walks the logical plan
//!    tree. When it finds an `Aggregate` node whose aggregate expressions
//!    contain `partial_agg(...)` wrappers, it:
//!    - Strips the `partial_agg(...)` wrapper from each expression.
//!    - Replaces the `Aggregate` node with a custom
//!      `PartialOnlyAggregate` extension node.
//!
//! 3. **Custom Logical Node (`PartialOnlyAggregate`)**: Implements
//!    `UserDefinedLogicalNodeCore`. Carries the same group/aggregate
//!    expressions as a normal `Aggregate`, but tells the physical planner
//!    to emit only `AggregateExec(Partial)`.
//!
//! 4. **Extension Planner (`PartialOnlyAggregatePlanner`)**: Implements
//!    `ExtensionPlanner`. Maps `PartialOnlyAggregate` to a single
//!    `AggregateExec` in `Partial` mode — no Final stage.
//!
//! ## Architecture
//!
//! ```text
//! SQL: SELECT bucket, partial_agg(SUM(value)) FROM t GROUP BY bucket
//!         |
//!         v
//! [Parser]  → Aggregate { aggr_expr: [partial_agg(SUM(value))] }
//!         |
//!         v
//! [PartialAggregateRule]  → PartialOnlyAggregate { aggr_expr: [SUM(value)] }
//!         |
//!         v
//! [PartialOnlyAggregatePlanner]  → AggregateExec(Partial)
//!         |
//!         v
//! [Client merges partial results by group key]
//! ```
//!
//! ## Why This Approach
//!
//! This is designed to be implemented as an **extension** in a downstream
//! crate that depends on DataFusion. It requires:
//! - No changes to DataFusion core
//! - No custom SQL syntax / parser changes
//! - Only public APIs: `AnalyzerRule`, `UserDefinedLogicalNodeCore`,
//!   `ExtensionPlanner`, `ScalarUDF`

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};

use async_trait::async_trait;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{DFSchemaRef, Result};
use datafusion::execution::context::{QueryPlanner, SessionState};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::logical_expr::{
    Aggregate, Expr, Extension, LogicalPlan, ScalarUDF, ScalarUDFImpl, Signature,
    UserDefinedLogicalNode, UserDefinedLogicalNodeCore, Volatility,
};
use datafusion::optimizer::AnalyzerRule;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::{ColumnarValue, ExecutionPlan, ExecutionPlanProperties};
use datafusion::physical_planner::{
    create_aggregate_expr_and_maybe_filter, DefaultPhysicalPlanner, ExtensionPlanner,
    PhysicalPlanner,
};
use datafusion::prelude::*;

use datafusion::common::config::ConfigOptions;
use datafusion::logical_expr::ScalarFunctionArgs;

use futures::StreamExt;

// ============================================================================
// 1. Marker UDF: partial_agg()
// ============================================================================

/// Create the `partial_agg` scalar UDF.
///
/// This function is a no-op pass-through: `partial_agg(x)` returns `x`.
/// It exists purely as a syntactic marker in SQL. The `PartialAggregateRule`
/// analyzer strips it before the plan reaches the optimizer/planner.
///
/// By registering it as a real UDF, the SQL parser and type checker accept
/// it without any parser modifications.
fn make_partial_agg_udf() -> ScalarUDF {
    ScalarUDF::new_from_impl(PartialAggUdf::new())
}

#[derive(Debug)]
struct PartialAggUdf {
    signature: Signature,
}

impl PartialAggUdf {
    fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Volatile),
        }
    }
}

impl ScalarUDFImpl for PartialAggUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "partial_agg"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        // Pass through the inner type unchanged.
        Ok(arg_types[0].clone())
    }

    fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        // This should never be called — the analyzer strips partial_agg()
        // before execution.
        Err(datafusion::error::DataFusionError::Internal(
            "partial_agg() marker UDF should have been removed by the analyzer".to_string(),
        ))
    }
}

// ============================================================================
// 2. Custom Logical Node: PartialOnlyAggregate
// ============================================================================

/// A logical plan node that represents an aggregation that should only
/// produce partial (intermediate) results — no Final stage.
///
/// This is semantically equivalent to `Aggregate`, but the physical planner
/// maps it to `AggregateExec(Partial)` only.
#[derive(Debug, PartialEq, Eq, Hash)]
struct PartialOnlyAggregate {
    input: LogicalPlan,
    group_expr: Vec<Expr>,
    aggr_expr: Vec<Expr>,
    schema: DFSchemaRef,
}

impl PartialOrd for PartialOnlyAggregate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Compare by group and aggregate expressions only (schema is derived).
        match self.group_expr.partial_cmp(&other.group_expr) {
            Some(std::cmp::Ordering::Equal) => {}
            ord => return ord,
        }
        self.aggr_expr.partial_cmp(&other.aggr_expr)
    }
}

impl UserDefinedLogicalNodeCore for PartialOnlyAggregate {
    fn name(&self) -> &str {
        "PartialOnlyAggregate"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        self.group_expr
            .iter()
            .chain(self.aggr_expr.iter())
            .cloned()
            .collect()
    }

    fn prevent_predicate_push_down_columns(&self) -> HashSet<String> {
        HashSet::new()
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "PartialOnlyAggregate: group_expr=[{}], aggr_expr=[{}]",
            self.group_expr.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(", "),
            self.aggr_expr.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(", "),
        )
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        let (group_expr, aggr_expr) = exprs.split_at(self.group_expr.len());
        // Rebuild the schema via Aggregate::try_new so it matches.
        let input = inputs.swap_remove(0);
        let agg = Aggregate::try_new(
            Arc::new(input.clone()),
            group_expr.to_vec(),
            aggr_expr.to_vec(),
        )?;
        Ok(Self {
            input,
            group_expr: group_expr.to_vec(),
            aggr_expr: aggr_expr.to_vec(),
            schema: agg.schema,
        })
    }
}

// ============================================================================
// 3. Analyzer Rule: PartialAggregateRule
// ============================================================================

/// Analyzer rule that detects `partial_agg(...)` markers inside
/// `Aggregate` nodes and replaces the `Aggregate` with a
/// `PartialOnlyAggregate` extension node.
#[derive(Debug)]
struct PartialAggregateRule;

impl PartialAggregateRule {
    /// Check if an expression is `partial_agg(inner)` and return the inner
    /// expression if so.
    fn unwrap_partial_agg(expr: &Expr) -> Option<Expr> {
        // Handle: partial_agg(agg_fn(...)) and partial_agg(agg_fn(...)) AS alias
        let (alias_name, inner) = match expr {
            Expr::Alias(alias) => (Some(alias.name.clone()), alias.expr.as_ref()),
            other => (None, other),
        };
        if let Expr::ScalarFunction(sf) = inner {
            if sf.func.name() == "partial_agg" && sf.args.len() == 1 {
                let unwrapped = sf.args[0].clone();
                return Some(match alias_name {
                    Some(name) => unwrapped.alias(name),
                    None => unwrapped,
                });
            }
        }
        None
    }
}

impl AnalyzerRule for PartialAggregateRule {
    fn name(&self) -> &str {
        "partial_aggregate"
    }

    fn analyze(&self, plan: LogicalPlan, _config: &ConfigOptions) -> Result<LogicalPlan> {
        // The SQL planner splits `partial_agg(SUM(value))` into:
        //   Projection [partial_agg(column_ref) AS alias]
        //     Aggregate [SUM(value)]
        //
        // So we look for Projection → Aggregate patterns where the
        // Projection contains partial_agg() wrappers, then:
        //  1. Strip partial_agg() from the Projection expressions
        //  2. Replace the Aggregate with PartialOnlyAggregate
        plan.transform_up(|node| {
            let LogicalPlan::Projection(ref proj) = node else {
                return Ok(Transformed::no(node));
            };

            // Check if any projection expression contains partial_agg().
            let mut has_partial = false;
            let mut new_proj_expr = Vec::with_capacity(proj.expr.len());

            for expr in &proj.expr {
                if let Some(unwrapped) = Self::unwrap_partial_agg(expr) {
                    has_partial = true;
                    new_proj_expr.push(unwrapped);
                } else {
                    new_proj_expr.push(expr.clone());
                }
            }

            if !has_partial {
                return Ok(Transformed::no(node));
            }

            // The child must be an Aggregate for this to make sense.
            let LogicalPlan::Aggregate(ref agg) = *proj.input else {
                return Ok(Transformed::no(node));
            };

            // Replace the Aggregate with PartialOnlyAggregate.
            let new_agg = Aggregate::try_new(
                Arc::clone(&agg.input),
                agg.group_expr.clone(),
                agg.aggr_expr.clone(),
            )?;

            let partial_node = PartialOnlyAggregate {
                input: agg.input.as_ref().clone(),
                group_expr: agg.group_expr.clone(),
                aggr_expr: agg.aggr_expr.clone(),
                schema: new_agg.schema,
            };

            let extension = LogicalPlan::Extension(Extension {
                node: Arc::new(partial_node),
            });

            // Rebuild the Projection with partial_agg() stripped,
            // pointing to the new extension node.
            let new_proj = LogicalPlan::Projection(
                datafusion::logical_expr::Projection::try_new(
                    new_proj_expr,
                    Arc::new(extension),
                )?
            );

            Ok(Transformed::yes(new_proj))
        })
        .map(|t| t.data)
    }
}

// ============================================================================
// 4. Extension Planner: PartialOnlyAggregatePlanner
// ============================================================================

/// Physical planner for `PartialOnlyAggregate` nodes.
///
/// Creates a single `AggregateExec` in `Partial` mode — no Final stage.
/// The output schema matches the intermediate state format of each
/// aggregate function.
#[derive(Debug)]
struct PartialOnlyAggregatePlanner;

#[async_trait]
impl ExtensionPlanner for PartialOnlyAggregatePlanner {
    async fn plan_extension(
        &self,
        planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(partial_node) = node.as_any().downcast_ref::<PartialOnlyAggregate>()
        else {
            return Ok(None);
        };

        let input_exec = physical_inputs[0].clone();
        let logical_input_schema = partial_node.input.schema();
        let physical_input_schema = input_exec.schema();

        // Build PhysicalGroupBy from group expressions.
        let groups = PhysicalGroupBy::new_single(
            partial_node
                .group_expr
                .iter()
                .map(|e| {
                    let phys = planner.create_physical_expr(
                        e,
                        logical_input_schema,
                        session_state,
                    )?;
                    let name = e.schema_name().to_string();
                    Ok((phys, name))
                })
                .collect::<Result<Vec<_>>>()?,
        );

        // Build aggregate expressions.
        let agg_filter: Vec<_> = partial_node
            .aggr_expr
            .iter()
            .map(|e| {
                create_aggregate_expr_and_maybe_filter(
                    e,
                    logical_input_schema,
                    &physical_input_schema,
                    session_state.execution_props(),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let mut aggregates = Vec::with_capacity(agg_filter.len());
        let mut filters = Vec::with_capacity(agg_filter.len());
        for (agg, filter, _order_by) in agg_filter {
            aggregates.push(agg);
            filters.push(filter);
        }

        // Create ONLY the Partial aggregate — no Final stage.
        let partial_agg = Arc::new(AggregateExec::try_new(
            AggregateMode::Partial,
            groups,
            aggregates,
            filters,
            input_exec,
            physical_input_schema,
        )?);

        // AggregateExec(Partial) outputs intermediate state columns with
        // different names than the logical schema (e.g., "metrics.bucket"
        // and "sum(value)[sum]" instead of "bucket" and "sum(value)").
        // Add a ProjectionExec to rename columns to match the logical schema.
        let partial_schema = partial_agg.schema();
        let logical_schema: &DFSchemaRef =
            UserDefinedLogicalNodeCore::schema(partial_node);
        let projection_exprs: Vec<(Arc<dyn datafusion::physical_plan::PhysicalExpr>, String)> =
            (0..partial_schema.fields().len())
                .map(|i| {
                    let col: Arc<dyn datafusion::physical_plan::PhysicalExpr> = Arc::new(
                        datafusion::physical_plan::expressions::Column::new(
                            partial_schema.field(i).name(),
                            i,
                        ),
                    );
                    let logical_name = logical_schema.field(i).name().clone();
                    (col, logical_name)
                })
                .collect();

        let projection = ProjectionExec::try_new(projection_exprs, partial_agg)?;
        Ok(Some(Arc::new(projection)))
    }
}

// ============================================================================
// 5. Query Planner that wires in the extension planner
// ============================================================================

#[derive(Debug)]
struct PartialAggQueryPlanner;

#[async_trait]
impl QueryPlanner for PartialAggQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let physical_planner = DefaultPhysicalPlanner::with_extension_planners(vec![
            Arc::new(PartialOnlyAggregatePlanner),
        ]);
        physical_planner
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}

// ============================================================================
// 6. Helper: create a SessionContext with the partial_agg infrastructure
// ============================================================================

/// Build a `SessionContext` with the `partial_agg()` marker UDF,
/// the analyzer rule, and the extension planner all registered.
///
/// This is the single entry point that downstream users need to call.
fn make_partial_agg_context() -> SessionContext {
    let config = SessionConfig::new()
        .with_target_partitions(2)
        .with_batch_size(4096);

    let state = SessionStateBuilder::new()
        .with_config(config)
        .with_default_features()
        .with_analyzer_rule(Arc::new(PartialAggregateRule))
        .with_query_planner(Arc::new(PartialAggQueryPlanner))
        .build();

    let ctx = SessionContext::new_with_state(state);
    ctx.register_udf(make_partial_agg_udf());
    ctx
}

// ============================================================================
// 7. Test data generation
// ============================================================================

fn make_test_batches(num_batches: usize, rows_per_batch: usize) -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("bucket", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
    ]));

    let num_distinct_groups = 1_000;

    (0..num_batches)
        .map(|batch_idx| {
            let bucket_data: Vec<String> = (0..rows_per_batch)
                .map(|row| {
                    format!(
                        "group_{:04}",
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

// ============================================================================
// 8. Main: demonstrate the marker UDF approach end-to-end
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // --- Setup ---
    let ctx = make_partial_agg_context();

    let batches1 = make_test_batches(20, 4096);
    let batches2 = make_test_batches(20, 4096);
    let schema = batches1[0].schema();
    let table =
        datafusion::datasource::MemTable::try_new(schema, vec![batches1, batches2])?;
    ctx.register_table("metrics", Arc::new(table))?;

    // --- 1. Show EXPLAIN with partial_agg() ---
    println!("=== EXPLAIN with partial_agg() marker ===\n");
    let df = ctx
        .sql(
            "EXPLAIN SELECT bucket, partial_agg(SUM(value)) as total, \
             partial_agg(COUNT(*)) as cnt \
             FROM metrics GROUP BY bucket",
        )
        .await?;
    let results = df.collect().await?;
    for batch in &results {
        let plan_types = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let plans = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..batch.num_rows() {
            println!("{}: {}", plan_types.value(i), plans.value(i));
        }
    }

    // --- 2. Execute the partial-only query and merge client-side ---
    println!("\n=== Executing partial_agg() query ===\n");
    let df = ctx
        .sql(
            "SELECT bucket, partial_agg(SUM(value)) as total, \
             partial_agg(COUNT(*)) as cnt \
             FROM metrics GROUP BY bucket",
        )
        .await?;
    let plan = df.create_physical_plan().await?;

    println!(
        "Physical plan:\n{}\n",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
    );

    // Verify the plan has only Partial mode (no Final/FinalPartitioned aggregate).
    assert_no_final_aggregate(&plan);

    // The inner AggregateExec(Partial) has Incremental emission.
    // The outer ProjectionExec (for column renaming) may report Final,
    // but the streaming behavior is preserved.
    println!("Top-level emission type: {:?}", plan.properties().emission_type);

    // Execute and merge partial results client-side.
    let mut merged_sum: HashMap<String, f64> = HashMap::new();
    let mut merged_count: HashMap<String, i64> = HashMap::new();
    let mut batch_num = 0;

    let num_partitions = plan.output_partitioning().partition_count();
    for partition in 0..num_partitions {
        let task_ctx = ctx.task_ctx();
        let mut stream = plan.execute(partition, task_ctx)?;

        while let Some(result) = stream.next().await {
            let batch = result?;
            batch_num += 1;

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
                "Partition {partition}, Batch {batch_num}: {} rows",
                batch.num_rows(),
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

    // --- 3. Show merged results ---
    println!("\n--- Merged results ({batch_num} partial batches) ---");
    let mut keys: Vec<_> = merged_sum.keys().cloned().collect();
    keys.sort();
    for key in keys.iter().take(10) {
        println!(
            "  {}: sum={:.1}, count={}",
            key, merged_sum[key], merged_count[key]
        );
    }
    println!("  ... ({} total groups)\n", keys.len());

    // --- 4. Verify against a normal (non-partial) aggregation ---
    println!("=== Verification against normal aggregation ===\n");
    let ctx2 = SessionContext::new();
    let v1 = make_test_batches(20, 4096);
    let v2 = make_test_batches(20, 4096);
    let vschema = v1[0].schema();
    let vtable = datafusion::datasource::MemTable::try_new(vschema, vec![v1, v2])?;
    ctx2.register_table("metrics", Arc::new(vtable))?;
    let df2 = ctx2
        .sql(
            "SELECT bucket, SUM(value) as total, COUNT(*) as cnt \
             FROM metrics GROUP BY bucket ORDER BY bucket",
        )
        .await?;
    let baseline = df2.collect().await?;

    let mut baseline_sum: HashMap<String, f64> = HashMap::new();
    let mut baseline_count: HashMap<String, i64> = HashMap::new();
    for batch in &baseline {
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
        for row in 0..batch.num_rows() {
            let key = buckets.value(row).to_string();
            baseline_sum.insert(key.clone(), sums.value(row));
            baseline_count.insert(key, extract_i64(counts.as_ref(), row));
        }
    }

    // Compare
    assert_eq!(
        merged_sum.len(),
        baseline_sum.len(),
        "Group count mismatch: partial={} vs baseline={}",
        merged_sum.len(),
        baseline_sum.len()
    );
    let mut mismatches = 0;
    for (key, &expected_sum) in &baseline_sum {
        let actual_sum = merged_sum.get(key).copied().unwrap_or(f64::NAN);
        let actual_count = merged_count.get(key).copied().unwrap_or(-1);
        let expected_count = baseline_count[key];
        if (actual_sum - expected_sum).abs() > 1e-6 || actual_count != expected_count {
            if mismatches < 5 {
                println!(
                    "MISMATCH {}: expected sum={:.1} count={}, got sum={:.1} count={}",
                    key, expected_sum, expected_count, actual_sum, actual_count
                );
            }
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "{mismatches} groups did not match");
    println!(
        "All {} groups match between partial_agg() and normal aggregation.",
        baseline_sum.len()
    );

    // --- 5. Show that a query WITHOUT partial_agg() still works normally ---
    println!("\n=== Normal query (no partial_agg) still works ===\n");
    let df3 = ctx
        .sql(
            "SELECT bucket, SUM(value) as total \
             FROM metrics GROUP BY bucket ORDER BY bucket LIMIT 5",
        )
        .await?;
    let normal_results = df3.collect().await?;
    for batch in &normal_results {
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
        for row in 0..batch.num_rows() {
            println!("  {}: sum={:.1}", buckets.value(row), sums.value(row));
        }
    }
    println!("\nAll assertions passed.");

    Ok(())
}

// ============================================================================
// Helpers
// ============================================================================

/// Walk the physical plan tree and assert there is no Final/FinalPartitioned
/// AggregateExec. The plan should contain only Partial mode aggregates.
fn assert_no_final_aggregate(plan: &Arc<dyn ExecutionPlan>) {
    if let Some(agg) = plan.as_any().downcast_ref::<AggregateExec>() {
        assert_eq!(
            agg.mode(),
            &AggregateMode::Partial,
            "Expected only Partial aggregate, found {:?}",
            agg.mode()
        );
    }
    for child in plan.children() {
        assert_no_final_aggregate(child);
    }
}

/// Extract an i64 from an array that may be Int32, Int64, or UInt64.
fn extract_i64(array: &dyn Array, row: usize) -> i64 {
    if let Some(a) = array.as_any().downcast_ref::<arrow::array::Int64Array>() {
        a.value(row)
    } else if let Some(a) = array.as_any().downcast_ref::<Int32Array>() {
        a.value(row) as i64
    } else if let Some(a) = array.as_any().downcast_ref::<arrow::array::UInt64Array>() {
        a.value(row) as i64
    } else {
        panic!("Unexpected count column type: {:?}", array.data_type());
    }
}
