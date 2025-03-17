use datafusion::arrow::datatypes::DataType;
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};
use datafusion_common::DataFusionError;
use std::any::Any;
use std::fmt::Debug;

/// The name of the symbolize marker function
pub const SYMBOLIZE_FUNCTION_NAME: &str = "symbolize";

/// A custom implementation of ScalarUDFImpl for the symbolize marker function
#[derive(Debug)]
struct SymbolizeUdf {
    // The function signature with UserDefined type signature
    signature: Signature,
}

impl SymbolizeUdf {
    fn new() -> Self {
        // Use TypeSignature::UserDefined to bypass DataFusion's type checking
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SymbolizeUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        SYMBOLIZE_FUNCTION_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if arg_types.is_empty() {
            return Err(DataFusionError::Plan(
                "symbolize UDF requires at least one argument".to_string(),
            ));
        }
        if !(arg_types.len() == 1 || arg_types.len() == 3) {
            return Err(DataFusionError::Plan(
                format!(
                    "symbolize UDF requires one or three arguments locations column, <build_id field name>, <address field name> but found {}", arg_types.len(),
                )
            ));
        }
        let DataType::List(struct_field) = &arg_types[0] else {
            return Err(DataFusionError::Plan(format!(
                "symbolize UDF requires a list type as first argument but found {}",
                arg_types[0]
            )));
        };
        let DataType::Struct(_) = struct_field.data_type() else {
            return Err(DataFusionError::Plan(format!(
                "symbolize UDF requires a list of structs as first argument but found {}",
                struct_field.data_type()
            )));
        };

        // Just return the type of the first argument as the result type since
        // symbolization just replaces NULL lines. Further type checking is done
        // during optimization.
        Ok(arg_types[0].clone())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        // Even though this function is a marker for the optimizer to insert a
        // special join into the plan, this function is still retained in the
        // plan to simplify maintaining type invariants.
        Ok(args.args[0].clone())
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        // Bypass coerce_types.
        Ok(arg_types.to_vec())
    }
}

/// Create and return the symbolize UDF marker.
/// This UDF is just a marker for the optimizer to detect and doesn't actually perform
/// any computation - it's replaced during optimization.
pub fn create_symbolize_udf() -> ScalarUDF {
    // Create a new ScalarUDF from our custom implementation
    ScalarUDF::new_from_impl(SymbolizeUdf::new())
}

/// Register the symbolize UDF with a session context
pub fn register_symbolize_udf(ctx: &datafusion::execution::context::SessionContext) {
    let udf = create_symbolize_udf();
    ctx.register_udf(udf);
}
