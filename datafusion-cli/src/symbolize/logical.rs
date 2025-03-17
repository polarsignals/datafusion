use crate::symbolize::debuginfo::{DebugInfoService, MockDebugInfoService};
use crate::symbolize::physical::SymbolizeJoinExec;
use crate::symbolize::udf::SYMBOLIZE_FUNCTION_NAME;
use async_trait::async_trait;
use datafusion::common::{DFSchemaRef, Result};
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::SessionState;
use datafusion::logical_expr::expr::ScalarFunction;
use datafusion::logical_expr::{
    Expr, Extension, LogicalPlan, Projection, UserDefinedLogicalNode,
    UserDefinedLogicalNodeCore,
};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{
    DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner,
};
use datafusion_common::tree_node::Transformed;
use datafusion_common::{DataFusionError, ScalarValue};
use std::fmt::{self, Debug};
use std::sync::Arc;

#[derive(Debug)]
pub struct SymbolizeQueryPlanner {}

#[async_trait]
impl QueryPlanner for SymbolizeQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // Teach the default physical planner how to plan TopK nodes.
        let physical_planner =
            DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(
                SymbolizeMarkerPlanner {
                    debug_service: Arc::new(MockDebugInfoService::new()),
                },
            )]);
        // Delegate most work of physical planning to the default physical planner
        physical_planner
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}

/// SymbolizeMarkerNode represents a logical marker that will be replaced
/// with a physical SymbolizeJoinExec during planning.
#[derive(PartialEq, PartialOrd, Eq, Hash)]
pub struct SymbolizeMarkerNode {
    /// Input plan
    input: LogicalPlan,
    /// Name of the locations column to symbolize.
    locations_field: String,
    /// Name of the build ID field in the structs to perform lookups with.
    build_id_field: String,
    /// Name of the address field in the structs to perform lookups with.
    address_field: String,
    /// Name of the lines field in the structs to symbolize.
    lines_field: String,
}

impl SymbolizeMarkerNode {
    pub fn new(
        input: LogicalPlan,
        locations_field: String,
        build_id_field: String,
        address_field: String,
        lines_field: String,
    ) -> Result<Self> {
        Ok(Self {
            input,
            locations_field,
            build_id_field,
            address_field,
            lines_field,
        })
    }
}

impl Debug for SymbolizeMarkerNode {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SymbolizeMarkerNode: list_column={}, build_id_field={}, address_field={}",
            self.locations_field, self.build_id_field, self.address_field,
        )
    }
}

impl UserDefinedLogicalNodeCore for SymbolizeMarkerNode {
    fn name(&self) -> &str {
        "SymbolizeMarker"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        // The schema is unchanged from the input.
        self.input.schema()
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SymbolizeMarker: locations_field={}, build_id_field={}, add_field={}, lines_field={}",
            self.locations_field,self.build_id_field, self.address_field, self.lines_field
        )
    }

    fn with_exprs_and_inputs(
        &self,
        _exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        assert_eq!(inputs.len(), 1, "input size inconsistent");
        let input = inputs.pop().unwrap();
        Self::new(
            input,
            self.locations_field.clone(),
            self.address_field.clone(),
            self.build_id_field.clone(),
            self.lines_field.clone(),
        )
    }

    fn supports_limit_pushdown(&self) -> bool {
        true
    }
}

// symbolize_column_names_from_fn extracts the locations column name first and
// the build_id, address, and lines column names. If only the locations column
// name is provided, this is where the default column names for the other
// columns are defined.
fn symbolize_column_names_from_fn(f: &ScalarFunction) -> Result<Vec<String>> {
    let Expr::Column(locations_col) = &f.args[0] else {
        return Err(DataFusionError::Plan(
            "Expected locations column as first argument to symbolize UDF".to_string(),
        ));
    };
    let locations_col_name = locations_col.name();
    if f.args.len() == 1 {
        return Ok(vec![
            locations_col_name.to_string(),
            "mapping_build_id".to_string(),
            "address".to_string(),
            "lines".to_string(),
        ]);
    }

    let mut result = vec![locations_col_name.to_string()];
    let Expr::Literal(ScalarValue::Utf8(Some(build_id_field))) = &f.args[1] else {
        return Err(DataFusionError::Plan(
            "Expected build id field name as second argument to symbolize UDF"
                .to_string(),
        ));
    };
    result.push(build_id_field.to_string());
    let Expr::Literal(ScalarValue::Utf8(Some(addr_field))) = &f.args[2] else {
        return Err(DataFusionError::Plan(
            "Expected address field name as third argument to symbolize UDF".to_string(),
        ));
    };
    result.push(addr_field.to_string());
    let Expr::Literal(ScalarValue::Utf8(Some(lines_field))) = &f.args[3] else {
        return Err(DataFusionError::Plan(
            "Expected lines field name as fourth argument to symbolize UDF".to_string(),
        ));
    };
    result.push(lines_field.to_string());
    Ok(result)
}

/// Optimizer rule that detects the symbolize marker UDF and converts it to a
/// SymbolizeMarkerNode.
#[derive(Debug)]
pub struct SymbolizeMarkerOptimizerRule {}

impl OptimizerRule for SymbolizeMarkerOptimizerRule {
    fn name(&self) -> &str {
        "SymbolizeMarkerOptimizerRule"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    /// rewrite searches for the Symbolize UDF in a projection and replaces it
    /// with a `SymbolizeMarkerNode` over this projection that will eventually
    /// be replaced with a `SymbolizeJoinExec`.
    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Projection(proj) = &plan else {
            return Ok(Transformed::no(plan));
        };

        // Skip rewriting if the projection's input is already a
        // SymbolizeMarkerNode.
        if let LogicalPlan::Extension(Extension { node }) = proj.input.as_ref() {
            if node
                .as_any()
                .downcast_ref::<SymbolizeMarkerNode>()
                .is_some()
            {
                return Ok(Transformed::no(plan));
            }
        }

        let Some(func) = proj.expr.iter().find_map(|expr| {
            let Expr::ScalarFunction(func) = expr else {
                return None;
            };
            (func.name() == SYMBOLIZE_FUNCTION_NAME).then_some(func)
        }) else {
            // No symbolize UDF found.
            return Ok(Transformed::no(plan));
        };

        let column_names = symbolize_column_names_from_fn(func)?;
        assert_eq!(column_names.len(), 4);

        // Create a logical plan extension with our marker node
        let marker_plan = LogicalPlan::Extension(Extension {
            node: Arc::new(SymbolizeMarkerNode::new(
                proj.input.as_ref().clone(),
                column_names[0].clone(),
                column_names[1].clone(),
                column_names[2].clone(),
                column_names[3].clone(),
            )?),
        });

        // We do not update the projection to remove the UDF since otherwise
        // this fails optimization invariants. However, the output type does not
        // change, so this is not a problem.
        Ok(Transformed::yes(LogicalPlan::Projection(
            Projection::try_new(proj.expr.clone(), Arc::new(marker_plan))?,
        )))
    }
}

pub struct SymbolizeMarkerPlanner {
    /// The debug info service to use
    debug_service: Arc<dyn DebugInfoService>,
}

impl SymbolizeMarkerPlanner {
    pub fn new(debug_service: Arc<dyn DebugInfoService>) -> Self {
        Self { debug_service }
    }
}

impl Debug for SymbolizeMarkerPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SymbolizeMarkerPlanner")
            // skip debug_service
            .finish()
    }
}

#[async_trait]
impl ExtensionPlanner for SymbolizeMarkerPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(marker) = node.as_any().downcast_ref::<SymbolizeMarkerNode>() else {
            return Ok(None);
        };

        if physical_inputs.len() != 1 {
            return Err(DataFusionError::Internal(
                "SymbolizeMarkerNode expects exactly one input".to_string(),
            ));
        }
        Ok(Some(Arc::new(SymbolizeJoinExec::try_new(
            physical_inputs[0].clone(),
            self.debug_service.clone(),
            marker.locations_field.clone(),
            marker.build_id_field.clone(),
            marker.address_field.clone(),
            marker.lines_field.clone(),
        )?)))
    }
}
