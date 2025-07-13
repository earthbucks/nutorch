use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use uuid::Uuid;

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;

// torch squeeze  -----------------------------------------------------------
// Remove a dimension of size 1 from a tensor (like tensor.squeeze(dim) in PyTorch).
// The tensor **must** be supplied through the pipeline; the single positional
// argument is the dimension to squeeze.
// -------------------------------------------------------------------------
pub struct CommandSqueeze;

impl PluginCommand for CommandSqueeze {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch squeeze"
    }

    fn description(&self) -> &str {
        "Remove a dimension of size 1 from a tensor (similar to tensor.squeeze(dim) in PyTorch). \
         The tensor ID is taken from the pipeline; the dimension is a required argument."
    }

    fn signature(&self) -> Signature {
        Signature::build("torch squeeze")
            .input_output_types(vec![(Type::String, Type::String)]) // tensor id in, tensor id out
            .required(
                "dim",
                SyntaxShape::Int,
                "Dimension to squeeze (must have size 1)",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![Example {
            description: "Squeeze dimension 0 of a [1,2,3] tensor",
            example: r#"let t = (torch full [1,2,3] 1); $t | torch squeeze 0 | torch shape"#,
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
        // ------ tensor ID must come from the pipeline --------------------
        let PipelineData::Value(tensor_id_val, _) = input else {
            return Err(LabeledError::new("Unsupported input")
                .with_label("Only Value inputs are supported", call.head));
        };

        let tensor_id = tensor_id_val.as_str().map(|s| s.to_string()).map_err(|_| {
            LabeledError::new("Invalid input")
                .with_label("Pipeline input must be a tensor ID (string)", call.head)
        })?;

        // ------ dimension argument ---------------------------------------
        let dim_val = call.nth(0).ok_or_else(|| {
            LabeledError::new("Missing dimension")
                .with_label("A dimension argument is required", call.head)
        })?;

        let dim = dim_val.as_int().map_err(|_| {
            LabeledError::new("Invalid dimension")
                .with_label("Dimension must be an integer", call.head)
        })?;

        // ------ fetch tensor ---------------------------------------------
        let mut reg = TENSOR_REGISTRY.lock().unwrap();
        let tensor = reg
            .get(&tensor_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor ID", call.head)
            })?
            .shallow_clone();

        // ------ validate dimension ---------------------------------------
        let shape = tensor.size();
        let ndims = shape.len() as i64;
        if dim < 0 || dim >= ndims {
            return Err(LabeledError::new("Invalid dimension").with_label(
                format!("Dim {dim} out of bounds for tensor with {ndims} dims"),
                call.head,
            ));
        }
        if shape[dim as usize] != 1 {
            return Err(LabeledError::new("Cannot squeeze").with_label(
                format!("Dim {dim} has size {} (expected 1)", shape[dim as usize]),
                call.head,
            ));
        }

        // ------ perform squeeze ------------------------------------------
        let result_tensor = tensor.squeeze_dim(dim);

        // ------ store & return -------------------------------------------
        let new_id = Uuid::new_v4().to_string();
        reg.insert(new_id.clone(), result_tensor);
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}
