use std::any::Any;
use std::fmt::{self, Debug};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{ready, Context, Poll};

use crate::symbolize::debuginfo::DebugInfoService;
use arrow::array::{
    downcast_array, make_array, Array, ArrayData, ArrayRef, BooleanArray, ListArray,
    MutableArrayData, StructArray,
};
use arrow::compute::{filter, is_null, SlicesIterator};
use arrow::datatypes::{DataType, FieldRef, Schema, SchemaRef};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use datafusion::common::{Result, Statistics};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet,
};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, RecordBatchStream,
    SendableRecordBatchStream,
};
use datafusion_common::DataFusionError;
use futures::{FutureExt, Stream, StreamExt};

#[derive(Copy, Clone)]
struct ColumnIndices {
    locations: usize,
    // The following three indices are relative to the location struct fields.
    build_id: usize,
    address: usize,
    lines: usize,
}

/// A physical execution plan that symbolizes location structs by looking up
/// debug information for addresses and build IDs.
pub struct SymbolizeJoinExec {
    /// Input of locations to symbolize.
    input: Arc<dyn ExecutionPlan>,
    /// Debug info service to query for symbolization.
    debug_service: Arc<dyn DebugInfoService>,
    /// Column indices in the input schema for required columns.
    column_indices: ColumnIndices,
    /// Name of the (locations, build_id, address, lines) columns in the input.
    column_names: (String, String, String, String),
    properties: PlanProperties,
    metrics: ExecutionPlanMetricsSet,
}

impl SymbolizeJoinExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        debug_service: Arc<dyn DebugInfoService>,
        locations_column: String,
        build_id_field: String,
        address_field: String,
        lines_field: String,
    ) -> Result<Self> {
        let schema = input.schema();
        let (locations_idx, locations_field) =
            schema.fields.find(&locations_column).ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "Locations column '{}' not found in schema {:?}",
                    locations_column, schema.fields,
                ))
            })?;
        let DataType::List(struct_field) = locations_field.data_type() else {
            return Err(DataFusionError::Internal(format!(
                "Locations column '{}' is not a List, it is a {}",
                locations_column,
                locations_field.data_type(),
            )));
        };
        let DataType::Struct(struct_fields) = struct_field.data_type() else {
            return Err(DataFusionError::Internal(format!(
                "Locations column '{}' is not a Struct, it is a {}",
                locations_column,
                struct_field.data_type(),
            )));
        };
        let (build_id_idx, _) = struct_fields.find(&build_id_field).ok_or_else(|| {
            DataFusionError::Internal(format!(
                "Build ID field '{}' not found in struct fields {:?}",
                build_id_field, struct_fields,
            ))
        })?;
        let (address_idx, _) = struct_fields.find(&address_field).ok_or_else(|| {
            DataFusionError::Internal(format!(
                "Address field '{}' not found in struct fields {:?}",
                address_field, struct_fields,
            ))
        })?;
        let (lines_idx, _) = struct_fields.find(&lines_field).ok_or_else(|| {
            DataFusionError::Internal(format!(
                "Lines field '{}' not found in struct fields {:?}",
                lines_field, struct_fields,
            ))
        })?;

        let column_indices = ColumnIndices {
            locations: locations_idx,
            build_id: build_id_idx,
            address: address_idx,
            lines: lines_idx,
        };

        let input_properties = input.properties();
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema),
            input_properties.output_partitioning().clone(),
            input_properties.emission_type,
            input_properties.boundedness,
        );
        Ok(Self {
            input,
            debug_service,
            column_indices,
            column_names: (locations_column, build_id_field, address_field, lines_field),
            properties,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl Debug for SymbolizeJoinExec {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SymbolizeJoinExec")
            .field("locations_column", &self.column_names.0)
            .field("build_id_field", &self.column_names.1)
            .field("address_field", &self.column_names.2)
            .field("lines_field", &self.column_names.3)
            .finish()
    }
}

impl DisplayAs for SymbolizeJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "SymbolizeJoinExec: locations_column={}, build_id_field={}, addr_field={}, lines_field={}", 
                    self.column_names.0, self.column_names.1, self.column_names.2, self.column_names.3
                )
            }
            DisplayFormatType::TreeRender => {
                write!(
                    f,
                    "SymbolizeJoinExec: {}[{}.{}.{}]",
                    self.column_names.0,
                    self.column_names.1,
                    self.column_names.2,
                    self.column_names.3
                )
            }
        }
    }
}

impl ExecutionPlan for SymbolizeJoinExec {
    fn name(&self) -> &str {
        "SymbolizeJoinExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "SymbolizeJoinExec requires exactly one child".to_string(),
            ));
        }

        Ok(Arc::new(SymbolizeJoinExec::try_new(
            children[0].clone(),
            self.debug_service.clone(),
            self.column_names.0.clone(),
            self.column_names.1.clone(),
            self.column_names.2.clone(),
            self.column_names.3.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        Ok(Box::pin(SymbolizeJoinStream::new(
            self.input.execute(partition, context)?,
            self.debug_service.clone(),
            self.column_indices,
            self.input.schema().clone(),
            BaselineMetrics::new(&self.metrics, partition),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn statistics(&self) -> Result<Statistics> {
        // For simplicity, use the left input's statistics
        self.input.statistics()
    }

    fn supports_limit_pushdown(&self) -> bool {
        true
    }
}

enum ProcessingState {
    /// Ready to process the next batch.
    Ready,
    /// Currently issuing a debuginfo lookup.
    IssuingLookup {
        /// The batch being processed.
        batch: RecordBatch,
        /// The null mask for the lines list array. Note that this does not
        /// directly apply to batch.
        null_mask: BooleanArray,
        /// The future that is executing the lookup.
        lookup_future: Pin<Box<dyn Future<Output = Result<RecordBatch>> + Send>>,
    },
    /// End of stream.
    Done,
}

/// The stream that performs the actual symbolization operation
pub struct SymbolizeJoinStream {
    /// Input stream (unsymbolized locations).
    input: SendableRecordBatchStream,
    debuginfo_service: Arc<dyn DebugInfoService>,
    column_indices: ColumnIndices,
    lookup_schema: SchemaRef,
    /// The field reference for the location struct type. Helpful to construct
    /// join result.
    location_field: FieldRef,
    /// Output schema
    schema: SchemaRef,
    /// BaselineMetrics for the computation. Ideally, we'd use
    /// BuildProbeJoinMetrics, but that's hidden away in the crate where the
    /// joins are in datafusion.
    metrics: BaselineMetrics,
    /// Current state of batch processing.
    state: ProcessingState,
}

impl SymbolizeJoinStream {
    fn new(
        input: SendableRecordBatchStream,
        debuginfo_service: Arc<dyn DebugInfoService>,
        column_indices: ColumnIndices,
        schema: SchemaRef,
        metrics: BaselineMetrics,
    ) -> Self {
        let DataType::List(struct_field) =
            schema.field(column_indices.locations).data_type()
        else {
            panic!("Expected locations column to be a List");
        };
        let DataType::Struct(struct_fields) = struct_field.data_type() else {
            panic!("Expected locations list values to be a Struct");
        };
        // Schema for record batches used to perform debuginfo lookups.
        let lookup_schema = Arc::new(Schema::new(vec![
            struct_fields[column_indices.build_id].clone(),
            struct_fields[column_indices.address].clone(),
        ]));
        Self {
            input,
            debuginfo_service,
            column_indices,
            lookup_schema,
            location_field: struct_field.clone(),
            schema,
            metrics,
            state: ProcessingState::Ready,
        }
    }

    /// Extract unique address/build ID pairs from a batch. This applies a
    /// filter (which may or may not build a new array, see `filter` docs) so
    /// that only rows with NULL lines are symbolized. The null mask is also
    /// returned.
    fn extract_lookup_keys(
        &self,
        batch: &RecordBatch,
    ) -> Result<(RecordBatch, BooleanArray)> {
        let locations: ListArray =
            downcast_array(batch.column(self.column_indices.locations));
        let location_structs: StructArray = downcast_array(locations.values());
        let lines = location_structs.column(self.column_indices.lines);
        if lines.null_count() == 0 {
            return Ok((
                RecordBatch::new_empty(self.lookup_schema.clone()),
                BooleanArray::new_null(0),
            ));
        }
        let null_mask = is_null(lines)?;

        let lookup_build_ids = filter(
            location_structs.column(self.column_indices.build_id),
            &null_mask,
        )?;
        let lookup_addrs = filter(
            location_structs.column(self.column_indices.address),
            &null_mask,
        )?;

        Ok((
            RecordBatch::try_new(
                self.lookup_schema.clone(),
                vec![lookup_build_ids, lookup_addrs],
            )?,
            null_mask,
        ))
    }

    /// Join debug info results with the original batch.
    fn join_results(
        location_field: FieldRef,
        column_indices: ColumnIndices,
        batch: &RecordBatch,
        null_mask: &BooleanArray,
        debug_info: &RecordBatch,
    ) -> Result<RecordBatch> {
        let locations: ListArray = downcast_array(batch.column(column_indices.locations));
        let location_structs: StructArray = downcast_array(locations.values());
        let lines = location_structs.column(column_indices.lines);
        let symbolized_lines =
            zip_non_equal(null_mask, &debug_info.column(0).to_data(), &lines.to_data())?;

        let new_children = location_structs
            .columns()
            .iter()
            .enumerate()
            .map(|(i, child)| {
                if i != column_indices.lines {
                    return child.clone();
                }
                symbolized_lines.clone()
            })
            .collect::<Vec<_>>();

        let new_location_structs = Arc::new(StructArray::new(
            location_structs.fields().clone(),
            new_children,
            location_structs.nulls().cloned(),
        )) as ArrayRef;

        let new_locations = Arc::new(ListArray::new(
            location_field,
            locations.offsets().clone(),
            new_location_structs,
            locations.nulls().cloned(),
        )) as ArrayRef;
        let mut new_columns = batch.columns().to_vec();
        new_columns[column_indices.locations] = new_locations;

        Ok(RecordBatch::try_new(batch.schema().clone(), new_columns)?)
    }

    fn poll_next_impl(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<RecordBatch>>> {
        loop {
            match &mut self.state {
                ProcessingState::Done => return Poll::Ready(None),
                ProcessingState::Ready => {
                    let batch = match ready!(self.input.poll_next_unpin(cx)) {
                        Some(Ok(b)) => b,
                        Some(Err(e)) => return Poll::Ready(Some(Err(e))),
                        None => {
                            // Input is finished.
                            self.state = ProcessingState::Done;
                            return Poll::Ready(None);
                        }
                    };
                    let _timer = self.metrics.elapsed_compute().timer();
                    let (lookup_batch, null_mask) = match self.extract_lookup_keys(&batch)
                    {
                        Ok(b) => b,
                        Err(e) => return Poll::Ready(Some(Err(e))),
                    };

                    if lookup_batch.num_rows() == 0 {
                        // No debuginfo lookup necessary, skip straight to
                        // outputting result.
                        self.state = ProcessingState::Ready;
                        return Poll::Ready(Some(Ok(batch)));
                    }

                    let debuginfo_service = self.debuginfo_service.clone();
                    let lookup_future = Box::pin(async move {
                        debuginfo_service.lookup_debuginfo(lookup_batch).await
                    });

                    // Update state and continue the loop to issue the lookup
                    // directly.
                    self.state = ProcessingState::IssuingLookup {
                        batch,
                        null_mask,
                        lookup_future,
                    };
                    continue;
                }
                ProcessingState::IssuingLookup {
                    batch,
                    null_mask,
                    lookup_future,
                } => {
                    let lines_batch = match ready!(lookup_future.poll_unpin(cx)) {
                        Ok(b) => b,
                        Err(e) => {
                            // TODO(asubiotto): Lookup error, log and continue?
                            self.state = ProcessingState::Ready;
                            return Poll::Ready(Some(Err(e)));
                        }
                    };
                    let _timer = self.metrics.elapsed_compute().timer();
                    // Because of borrowing issues, join_results needs to be a
                    // function on the type rather than a method, otherwise this
                    // call requires cloning the batch and null_mask, which I
                    // think is unnecessary.
                    let result = SymbolizeJoinStream::join_results(
                        self.location_field.clone(),
                        self.column_indices,
                        batch,
                        null_mask,
                        &lines_batch,
                    );

                    self.state = ProcessingState::Ready;
                    return Poll::Ready(Some(result));
                }
            }
        }
    }
}

impl RecordBatchStream for SymbolizeJoinStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Stream for SymbolizeJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let result = self.poll_next_impl(cx);
        self.metrics.record_poll(result)
    }
}

/// Zips two arrays based on a mask. This function is inspired by
/// arrow_select::zip:
/// - `mask` is a BooleanArray of length N.
/// - `truthy` is an array (as ArrayData) with length equal to the number
///    of true values in `mask`.
/// - `falsy` is an array (as ArrayData) of length N.
///
/// For each index `i` in 0..N:
///   - if mask[i] is true, the output gets the next element from `truthy`,
///   - otherwise the output gets falsy[i].
///
/// Note: Prefer using arrow_select::zip for zipping two arrays of the same
/// length.
///
/// # Example:
/// Given:
///   - falsy:   [1, NULL, 2, 3, NULL]
///   - mask:    [false, true, false, false, true]
///   - truthy:  [4, 5]
///
/// The result is [1, 4, 2, 3, 5].
fn zip_non_equal(
    mask: &BooleanArray,
    truthy: &ArrayData,
    falsy: &ArrayData,
) -> Result<ArrayRef, ArrowError> {
    // Check that falsy array length equals the mask length.
    if mask.len() != falsy.len() {
        return Err(ArrowError::InvalidArgumentError(
            "mask and falsy array must have the same length".to_string(),
        ));
    }

    if mask.true_count() != truthy.len() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "number of true values in mask ({}) must equal the length of the truthy array ({})",
            mask.true_count(),
            truthy.len()
        )));
    }

    // Create a MutableArrayData that will be built from two sources:
    // index 0: falsy, index 1: truthy.
    // We set `use_nulls` to true if either source contains nulls.
    let use_nulls = falsy.null_count() > 0 || truthy.null_count() > 0;
    let mut mutable = MutableArrayData::new(vec![falsy, truthy], use_nulls, mask.len());

    let mut filled = 0;
    let mut truthy_index = 0;
    // SlicesIterator yields contiguous (start, end) segments where mask is true.
    for (start, end) in SlicesIterator::new(mask) {
        // Fill the gap [filled, start) with values from the falsy array.
        if start > filled {
            mutable.extend(0, filled, start);
        }
        // For the [start, end) segment (mask true), take that many elements from truthy.
        let segment_len = end - start;
        mutable.extend(1, truthy_index, truthy_index + segment_len);
        truthy_index += segment_len;
        filled = end;
    }
    // Fill any remaining positions with falsy values.
    if filled < mask.len() {
        mutable.extend(0, filled, mask.len());
    }

    // Freeze the mutable data into ArrayData and wrap it as an ArrayRef.
    Ok(make_array(mutable.freeze()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, BooleanArray, Int32Array};

    #[test]
    fn test_zip_non_equal() -> Result<(), ArrowError> {
        // [1, NULL, 3, 4, NULL]
        let falsy = Int32Array::from(vec![Some(1), None, Some(3), Some(4), None]);
        // [false, true, false, false, true]
        let mask = BooleanArray::from(vec![
            Some(false),
            Some(true),
            Some(false),
            Some(false),
            Some(true),
        ]);
        // [2, 5]
        let truthy = Int32Array::from(vec![Some(2), Some(5)]);

        // Expected result: [1, 2, 3, 4, 5]
        let expected = Arc::new(Int32Array::from(vec![
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
        ])) as ArrayRef;

        let result_array = zip_non_equal(&mask, &truthy.to_data(), &falsy.to_data())?;

        assert_eq!(&result_array, &expected);
        Ok(())
    }
}
