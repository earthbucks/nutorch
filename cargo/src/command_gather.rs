use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use tch::Kind;
use uuid::Uuid;

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;

/// torch gather
/// Usage:  <source-tensor comes through pipeline>  torch gather <dim:int> <index_tensor_id>
pub struct CommandGather;

impl PluginCommand for CommandGather {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch gather"
    }

    fn description(&self) -> &str {
        "Gather values along an axis using an index tensor \
         (like `x.gather(dim, index)` in PyTorch)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch gather")
            // source tensor id must arrive through the pipeline
            .input_output_types(vec![(Type::String, Type::String)])
            .required("dim", SyntaxShape::Int, "Dimension along which to gather")
            .required(
                "index_id",
                SyntaxShape::String,
                "ID of the index tensor (int64)",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![Example {
            description: "Gather columns 2,1,0 from each row (dim=1)",
            example: r#"
let src  = ([[10 11 12] [20 21 22]] | torch tensor)
let idx  = ([[2 1 0]   [0 0 2]]     | torch tensor --dtype int64)
$src | torch gather 1 $idx | torch value
"#
            .trim(),
            result: None,
        }]
    }

    fn run(
        &self,
        _plugin: &NutorchPlugin,
        _engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        //--------------------------------------------------------------
        // 1. get tensor-id from pipeline, dim and index-id from args
        //--------------------------------------------------------------
        let PipelineData::Value(source_id_val, _) = input else {
            return Err(LabeledError::new("Missing input").with_label(
                "Source tensor ID must be supplied via the pipeline",
                call.head,
            ));
        };
        let source_id = source_id_val.as_str().map(|s| s.to_string()).map_err(|_| {
            LabeledError::new("Invalid input")
                .with_label("Pipeline input must be a tensor ID (string)", call.head)
        })?;

        let dim = call
            .nth(0)
            .ok_or_else(|| {
                LabeledError::new("Missing dim")
                    .with_label("Dimension argument is required", call.head)
            })?
            .as_int()
            .map_err(|_| {
                LabeledError::new("Invalid dim")
                    .with_label("Dimension must be an integer", call.head)
            })?;

        let index_id = call
            .nth(1)
            .ok_or_else(|| {
                LabeledError::new("Missing index tensor")
                    .with_label("Index tensor ID argument is required", call.head)
            })?
            .as_str()
            .map(|s| s.to_string())
            .map_err(|_| {
                LabeledError::new("Invalid index tensor ID")
                    .with_label("Must be a string", call.head)
            })?;

        //--------------------------------------------------------------
        // 2. fetch tensors
        //--------------------------------------------------------------
        let mut reg = TENSOR_REGISTRY.lock().unwrap();
        let source = reg
            .get(&source_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found")
                    .with_label("Invalid source tensor ID", call.head)
            })?
            .shallow_clone();

        let mut index = reg
            .get(&index_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found")
                    .with_label("Invalid index tensor ID", call.head)
            })?
            .shallow_clone();

        // ensure int64
        if index.kind() != Kind::Int64 {
            index = index.to_kind(Kind::Int64);
        }

        //--------------------------------------------------------------
        // 3. validate shapes & index-range
        //--------------------------------------------------------------
        let src_shape = source.size();
        let idx_shape = index.size();
        let ndims = src_shape.len() as i64;

        // dim bounds
        if dim < 0 || dim >= ndims {
            return Err(LabeledError::new("Invalid dimension").with_label(
                format!("Dim {dim} out of bounds for tensor with {ndims} dims"),
                call.head,
            ));
        }

        // same rank
        if idx_shape.len() != src_shape.len() {
            return Err(LabeledError::new("Shape mismatch").with_label(
                format!(
                    "Index tensor rank {} differs from source rank {}",
                    idx_shape.len(),
                    src_shape.len()
                ),
                call.head,
            ));
        }

        // all dims except 'dim' must match exactly
        for (d, (&s, &i)) in src_shape.iter().zip(idx_shape.iter()).enumerate() {
            if d as i64 != dim && s != i {
                return Err(LabeledError::new("Shape mismatch").with_label(
                    format!("Size mismatch at dim {d}: source={s}, index={i}",),
                    call.head,
                ));
            }
        }

        // index values must be in [0, src_shape[dim])
        let max_idx = index.max().int64_value(&[]);
        let min_idx = index.min().int64_value(&[]);
        if min_idx < 0 || max_idx >= src_shape[dim as usize] {
            return Err(LabeledError::new("Index out of range").with_label(
                format!(
                    "Index values must be between 0 and {} (exclusive); found [{}, {}]",
                    src_shape[dim as usize] - 1,
                    min_idx,
                    max_idx
                ),
                call.head,
            ));
        }

        //--------------------------------------------------------------
        // 4. gather  (sparse_grad matches source tensor)
        //--------------------------------------------------------------
        let sparse_grad = source.is_sparse();
        let result_tensor = source.gather(dim, &index, sparse_grad);

        //--------------------------------------------------------------
        // 5. store & return
        //--------------------------------------------------------------
        let new_id = Uuid::new_v4().to_string();
        reg.insert(new_id.clone(), result_tensor);
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}
