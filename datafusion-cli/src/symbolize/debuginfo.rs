use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef};
use datafusion_common::Result;
use std::sync::Arc;
use arrow::util::data_gen::create_random_batch;

/// Trait for a service that can look up debug information for addresses
#[async_trait::async_trait]
pub trait DebugInfoService: Send + Sync {
    /// Look up debuginfo for a RecordBatch with two columns: build IDs and
    /// addresses. The result is a batch with one column of lists of lines
    /// structs. Each row in the input batch will have a corresponding resulting
    /// row. If no debuginfo was found for a (build ID, address) pair, the lines
    /// row will be NULL.
    async fn lookup_debuginfo(&self, batch: RecordBatch) -> Result<RecordBatch>;
}

/// A simple implementation of DebugInfoService that returns mock data.
pub struct MockDebugInfoService {
    /// Schema for the debug info results
    schema: SchemaRef,
}

impl Default for MockDebugInfoService {
    fn default() -> Self {
        Self::new()
    }
}

impl MockDebugInfoService {
    pub fn new() -> Self {
        let struct_fields = Fields::from(vec![
            Field::new("line", DataType::Int64, false),
            Field::new_dictionary("function_name", DataType::UInt32, DataType::Utf8, false),
            Field::new_dictionary("function_system_name", DataType::UInt32, DataType::Utf8, false),
            Field::new_dictionary("function_filename", DataType::UInt32, DataType::Utf8, false),
            Field::new("function_start_line", DataType::Int64, false),
        ]);

        Self {
            schema: Arc::new(Schema::new(vec![Field::new_list("lines", Field::new_struct("item", struct_fields, true), true)])),
        }
    }
}

#[async_trait::async_trait]
impl DebugInfoService for MockDebugInfoService {
    async fn lookup_debuginfo(&self, batch: RecordBatch) -> Result<RecordBatch> {
        Ok(create_random_batch(self.schema.clone(), batch.num_rows(), 0.0, 0.0)?)
    }
}

