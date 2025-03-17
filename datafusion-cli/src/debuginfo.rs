use arrow::array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow::util::data_gen::create_random_array;
use async_trait::async_trait;
use datafusion::arrow::datatypes::{DataType, Field, Fields, Schema};
use datafusion::common::DataFusionError;
use datafusion::datasource::memory::{DataSourceExec, MemorySourceConfig};
use datafusion::datasource::TableType;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;
use datafusion_common::create_array;
use std::any::Any;
use std::sync::Arc;

#[derive(Debug)]
pub struct CatalogProviderList {}

impl Default for CatalogProviderList {
    fn default() -> Self {
        Self::new()
    }
}

impl CatalogProviderList {
    pub fn new() -> Self {
        CatalogProviderList {}
    }
}

impl datafusion::catalog::CatalogProviderList for CatalogProviderList {
    fn as_any(&self) -> &dyn Any {
        self
    }

    // Register catalog not implemented since catalogs can't be created in our
    // system.
    fn register_catalog(
        &self,
        _: String,
        _: Arc<dyn datafusion::catalog::CatalogProvider>,
    ) -> Option<Arc<dyn datafusion::catalog::CatalogProvider>> {
        None
    }

    fn catalog_names(&self) -> Vec<String> {
        vec![]
    }

    fn catalog(
        &self,
        _name: &str,
    ) -> Option<Arc<dyn datafusion::catalog::CatalogProvider>> {
        Some(Arc::new(CatalogProvider {}))
    }
}

#[derive(Debug)]
pub struct CatalogProvider {}

impl datafusion::catalog::CatalogProvider for CatalogProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema_names(&self) -> Vec<String> {
        vec![]
    }

    fn schema(&self, _: &str) -> Option<Arc<dyn datafusion::catalog::SchemaProvider>> {
        Some(Arc::new(SchemaProvider {}))
    }

    fn register_schema(
        &self,
        _name: &str,
        _schema: Arc<dyn datafusion::catalog::SchemaProvider>,
    ) -> datafusion::error::Result<Option<Arc<dyn datafusion::catalog::SchemaProvider>>>
    {
        Err(DataFusionError::NotImplemented(
            "register_schema".to_string(),
        ))
    }

    fn deregister_schema(
        &self,
        _name: &str,
        _cascade: bool,
    ) -> datafusion::error::Result<Option<Arc<dyn datafusion::catalog::SchemaProvider>>>
    {
        Err(DataFusionError::NotImplemented(
            "deregister_schema".to_string(),
        ))
    }
}

const TABLE_NAME: &str = "debuginfo";

#[derive(Debug, Clone)]
pub struct SchemaProvider {}

#[async_trait]
impl datafusion::catalog::SchemaProvider for SchemaProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn table_names(&self) -> Vec<String> {
        vec![TABLE_NAME.to_string()]
    }

    async fn table(
        &self,
        name: &str,
    ) -> Result<Option<Arc<dyn datafusion::datasource::TableProvider>>, DataFusionError>
    {
        if name != TABLE_NAME {
            return Ok(None);
        }
        Ok(Some(Arc::new(TableProvider {})))
    }

    fn register_table(
        &self,
        _name: String,
        _table: Arc<dyn datafusion::datasource::TableProvider>,
    ) -> Result<Option<Arc<dyn datafusion::datasource::TableProvider>>, DataFusionError>
    {
        Err(DataFusionError::NotImplemented(
            "cannot register other tables".to_string(),
        ))
    }

    fn deregister_table(
        &self,
        __name: &str,
    ) -> Result<Option<Arc<dyn datafusion::datasource::TableProvider>>, DataFusionError>
    {
        Err(DataFusionError::NotImplemented(
            "cannot deregister tables".to_string(),
        ))
    }

    fn table_exist(&self, name: &str) -> bool {
        if name == TABLE_NAME {
            return true;
        }
        false
    }
}

#[derive(Debug)]
pub struct TableProvider {}

#[async_trait]
impl datafusion::datasource::TableProvider for TableProvider {
    fn as_any(&self) -> &dyn Any {
        unimplemented!()
    }

    fn schema(&self) -> Arc<Schema> {
        // TODO(asubiotto): This is a simplified schema to work with mock data.
        // The real schema includes run end encoding and dictionary types.
        Arc::new(Schema::new(vec![
            Field::new("mapping_build_id", DataType::Utf8, false),
            Field::new("address", DataType::UInt64, false),
            Field::new(
                "lines",
                DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::Struct(Fields::from(vec![
                        Field::new("line", DataType::Int64, false),
                        Field::new("function_name", DataType::Utf8, false),
                        Field::new("function_system_name", DataType::Utf8, false),
                        Field::new("function_filename", DataType::Utf8, false),
                        Field::new("function_start_line", DataType::Int64, false),
                    ])),
                    false,
                ))),
                true,
            ),
        ]))
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn datafusion::catalog::Session,
        projections: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        println!("scan called projections: {projections:?} filters: {filters:?} limit: {limit:?}");
        let schema = self.schema();
        let columns = schema
            .fields()
            .iter()
            .map(|field| match field.name().as_str() {
                "address" => create_array!(UInt64, vec![4385521; 10]) as ArrayRef,
                "mapping_build_id" => create_array!(
                    Utf8,
                    vec!["87987728412ffaff58e302177248f3fd6436d132"; 10]
                ) as ArrayRef,
                _ => create_random_array(field, 10, 0.0, 0.0).unwrap(),
            })
            .collect::<Vec<ArrayRef>>();

        let batch = RecordBatch::try_new_with_options(
            schema.clone(),
            columns,
            &RecordBatchOptions::new().with_match_field_names(false),
        )?;
        Ok(Arc::new(DataSourceExec::new(Arc::new(
            MemorySourceConfig::try_new(&[vec![batch]], schema, projections.cloned())?,
        ))))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion_common::Result<Vec<TableProviderFilterPushDown>> {
        // Inexact filtering because it's using it to perform large-grained file pruning.
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }
}

/*#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::SessionContext;

    #[tokio::test]
    async fn test_catalog_provider_list() -> Result<(), anyhow::Error> {
        let cpl = Arc::new(CatalogProviderList::new());
        let ctx = SessionContext::new();
        ctx.register_catalog_list(cpl);
        let result = ctx.sql("select * from debuginfo").await?.collect().await?;
        println!("{result:?}");
        Ok(())
    }
}
*/
