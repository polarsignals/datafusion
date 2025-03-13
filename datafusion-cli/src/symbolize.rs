use std::any::Any;
use std::fmt::{self, Debug};
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::ArrayRef;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::common::{internal_err, DFSchemaRef, Result};
use datafusion::execution::context::{QueryPlanner, TaskContext};
use datafusion::execution::SessionState;
use datafusion::logical_expr::{
    Expr, Extension, Join, LogicalPlan, UserDefinedLogicalNode,
    UserDefinedLogicalNodeCore,
};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, Partitioning,
    PlanProperties, RecordBatchStream, SendableRecordBatchStream, Statistics,
};
use datafusion::physical_planner::{
    DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner,
};
use datafusion_common::tree_node::Transformed;
use futures::{Stream, StreamExt};
// -----------------------------------------------------------------------------
// 1. Logical Extension Node: SymbolizePlanNode
// -----------------------------------------------------------------------------

/// JoinInputPlanNode is a logical plan node that converts a list of location
/// structs to locations and only keeps the locations with null lines (to be
/// joined).
#[derive(PartialEq, PartialOrd, Eq, Hash)]
struct JoinInputPlanNode {
    /// The input logical plan.
    input: LogicalPlan,
}

impl Debug for JoinInputPlanNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Use a simple explain format.
        write!(f, "SymbolizeJoinInput")
    }
}

impl UserDefinedLogicalNodeCore for JoinInputPlanNode {
    fn name(&self) -> &str {
        "SymbolizeJoinInput"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        // TODO(asubiotto): This is technically incorrect, we should only return
        // locations.
        self.input.schema()
    }

    fn check_invariants(
        &self,
        _check: datafusion::logical_expr::InvariantLevel,
        _plan: &LogicalPlan,
    ) -> Result<()> {
        Ok(())
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SymbolizeJoinInput")
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        assert_eq!(inputs.len(), 1, "input size inconsistent");
        Ok(Self {
            input: inputs.pop().unwrap(),
        })
    }

    fn supports_limit_pushdown(&self) -> bool {
        false
    }
}

// -----------------------------------------------------------------------------
// 2. Optimizer Rule: SymbolizeOptimizerRule
// -----------------------------------------------------------------------------

#[derive(Default, Debug)]
pub struct SymbolizeOptimizerRule {}

impl OptimizerRule for SymbolizeOptimizerRule {
    fn name(&self) -> &str {
        "SymbolizeOptimizerRule"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        if let LogicalPlan::Join(ref join) = &plan {
            if is_debuginfo_table(&join.right) {
                if let LogicalPlan::Extension(ref e) = join.left.as_ref() {
                    if e.node
                        .as_any()
                        .downcast_ref::<JoinInputPlanNode>()
                        .is_some()
                    {
                        // Already rewritten; skip further rewriting.
                        return Ok(Transformed::no(plan));
                    }
                }
                //println!("join node: {:#?}", join);
                //println!("left join input: {:?}", join.left.schema());
                //println!("right join input: {:?}", join.right.schema());
                //println!("join output schema: {:?}", join.schema);
                return Ok(Transformed::yes(LogicalPlan::Join(Join {
                    left: Arc::new(LogicalPlan::Extension(Extension {
                        node: Arc::new(JoinInputPlanNode {
                            input: join.left.as_ref().clone(),
                        }),
                    })),
                    right: join.right.clone(),
                    on: join.on.clone(),
                    filter: join.filter.clone(),
                    join_type: join.join_type,
                    join_constraint: join.join_constraint,
                    schema: join.schema.clone(),
                    null_equals_null: join.null_equals_null,
                })));
            }
        }
        Ok(Transformed::no(plan))
    }
}

/// Helper function to detect if a plan node corresponds to the debuginfo table.
fn is_debuginfo_table(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::TableScan(scan) => {
            println!(
                "TableScan: {:?} table name {:?}",
                scan,
                scan.table_name.table()
            );
            scan.table_name.table() == "debuginfo"
        }
        LogicalPlan::SubqueryAlias(alias) => is_debuginfo_table(&alias.input),
        _ => false,
    }
}

// -----------------------------------------------------------------------------
// 3. Extension Planner: SymbolizeQueryPlanner and SymbolizePlanner
// -----------------------------------------------------------------------------

#[derive(Debug)]
pub struct SymbolizeQueryPlanner {}

#[async_trait]
impl QueryPlanner for SymbolizeQueryPlanner {
    /// Given a `LogicalPlan` created from above, create an
    /// `ExecutionPlan` suitable for execution
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Teach the default physical planner how to plan TopK nodes.
        let physical_planner =
            DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(
                SymbolizePlanner {},
            )]);
        // Delegate most work of physical planning to the default physical planner
        physical_planner
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}

#[derive(Debug)]
struct SymbolizePlanner {}

#[async_trait]
impl ExtensionPlanner for SymbolizePlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        if let Some(_symbolize_node) = node.as_any().downcast_ref::<JoinInputPlanNode>() {
            // We expect one input
            if physical_inputs.len() != 1 {
                return internal_err!(
                    "SymbolizeJoinInput node expects exactly one physical input"
                );
            }
            Ok(Some(Arc::new(JoinInputExec::new(
                physical_inputs[0].clone(),
            ))))
        } else {
            Ok(None)
        }
    }
}

// -----------------------------------------------------------------------------
// 4. Physical Operator: SymbolizeExec and its Stream (SymbolizeReader)
// -----------------------------------------------------------------------------

struct JoinInputExec {
    input: Arc<dyn ExecutionPlan>,
    cache: PlanProperties,
}

impl JoinInputExec {
    fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        let cache = Self::compute_properties(input.schema());
        Self { input, cache }
    }

    fn compute_properties(schema: SchemaRef) -> PlanProperties {
        // TODO(asubiotto): Properties are copied from the TopK operator
        // example. Do we need to do anything custom here?
        PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        )
    }
}

impl Debug for JoinInputExec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "JoinInputExec")
    }
}

impl DisplayAs for JoinInputExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "JoinInputExec")
            }
            DisplayFormatType::TreeRender => write!(f, "JoinInputExec"),
        }
    }
}

#[async_trait]
impl ExecutionPlan for JoinInputExec {
    fn name(&self) -> &'static str {
        "JoinInputExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        self.input.required_input_distribution()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!("JoinInputExec expects exactly one child");
        }
        Ok(Arc::new(JoinInputExec::new(children[0].clone())))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        Ok(Box::pin(JoinInputReader::new(
            self.input.execute(partition, context)?,
        )))
    }

    fn statistics(&self) -> Result<Statistics> {
        self.input.statistics()
    }
}

/// The stream wrapper that applies our transformation.
struct JoinInputReader {
    input: SendableRecordBatchStream,
}

impl JoinInputReader {
    fn new(input: SendableRecordBatchStream) -> Self {
        Self { input }
    }
}

impl RecordBatchStream for JoinInputReader {
    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }
}

impl Stream for JoinInputReader {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        match self.input.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let schema = batch.schema();
                let mut new_columns = Vec::with_capacity(batch.num_columns());

                for (i, field) in schema.fields().iter().enumerate() {
                    if field.name() == "locations" {
                        let array = batch.column(i);
                        // Apply our placeholder transformation.
                        let new_array = nullify_lines_in_locations(array)
                            .unwrap_or_else(|_| array.clone());
                        new_columns.push(new_array);
                    } else {
                        new_columns.push(batch.column(i).clone());
                    }
                }

                let new_batch = RecordBatch::try_new(schema, new_columns);
                Poll::Ready(Some(new_batch.map_err(Into::into)))
            }
            other => other,
        }
    }
}

/// A helper function that “nullifies” the `lines` field in the locations array.
/// In a full implementation you would use Arrow’s array builders and kernels
/// to iterate over the array elements and set the `lines` field to null.
fn nullify_lines_in_locations(array: &ArrayRef) -> Result<ArrayRef> {
    // For demonstration purposes, we simply return the input array.
    // Insert your transformation logic here.
    Ok(array.clone())
}
