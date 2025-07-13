use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use uuid::Uuid;

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;


// torch unsqueeze -----------------------------------------------------------
// Insert a size-1 dimension at the given index (like tensor.unsqueeze(dim))
// Tensor ID must be supplied via the pipeline; one positional argument = dim.
// ---------------------------------------------------------------------------
pub struct CommandUnsqueeze;

impl PluginCommand for CommandUnsqueeze {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch unsqueeze"
    }

    fn description(&self) -> &str {
        "Insert a dimension of size 1 into a tensor (similar to tensor.unsqueeze(dim) in PyTorch)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch unsqueeze")
            // tensor id comes from pipeline, returns new tensor id
            .input_output_types(vec![(Type::String, Type::String)])
            .required(
                "dim",
                SyntaxShape::Int,
                "Dimension index at which to insert size-1 dimension",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Unsqueeze dim 0 of a [2,3] tensor -> shape [1,2,3]",
                example: r#"let t = (torch full [2,3] 1); $t | torch unsqueeze 0 | torch shape"#,
                result: None,
            },
            Example {
                description: "Unsqueeze dim 2 of a [2,3] tensor -> shape [2,3,1]",
                example: r#"let t = (torch full [2,3] 1); $t | torch unsqueeze 2 | torch shape"#,
                result: None,
            },
        ]
    }

    fn run(
        &self,
        _plugin: &NutorchPlugin,
        _engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        // ---- tensor ID must come from pipeline --------------------------
        let PipelineData::Value(tensor_id_val, _) = input else {
            return Err(LabeledError::new("Unsupported input")
                .with_label("Tensor ID must be supplied via the pipeline", call.head));
        };

        let tensor_id = tensor_id_val.as_str().map(|s| s.to_string()).map_err(|_| {
            LabeledError::new("Invalid input")
                .with_label("Pipeline input must be a tensor ID (string)", call.head)
        })?;

        // ---- parse dimension argument -----------------------------------
        let dim_val = call.nth(0).ok_or_else(|| {
            LabeledError::new("Missing dimension")
                .with_label("A dimension argument is required", call.head)
        })?;

        let dim = dim_val.as_int().map_err(|_| {
            LabeledError::new("Invalid dimension")
                .with_label("Dimension must be an integer", call.head)
        })?;

        // ---- fetch tensor ------------------------------------------------
        let mut reg = TENSOR_REGISTRY.lock().unwrap();
        let tensor = reg
            .get(&tensor_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor ID", call.head)
            })?
            .shallow_clone();

        // ---- validate dim ------------------------------------------------
        let ndims = tensor.size().len() as i64;
        // In PyTorch unsqueeze allows dim == ndims (insert at end)
        if dim < 0 || dim > ndims {
            return Err(LabeledError::new("Invalid dimension").with_label(
                format!("Dim {dim} out of bounds for tensor with {ndims} dims"),
                call.head,
            ));
        }

        // ---- perform unsqueeze ------------------------------------------
        let result_tensor = tensor.unsqueeze(dim);

        // ---- store & return ---------------------------------------------
        let new_id = Uuid::new_v4().to_string();
        reg.insert(new_id.clone(), result_tensor);
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}
