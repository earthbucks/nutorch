use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use tch::Tensor;
use uuid::Uuid;

use crate::get_kind_from_call;
use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;


pub struct CommandMean;

impl PluginCommand for CommandMean {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch mean"
    }

    fn description(&self) -> &str {
        "Compute the mean value of a tensor (similar to torch.mean single tensor mode)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch mean")
            .input_output_types(vec![(Type::String, Type::String)])
            .named(
                "dtype",
                SyntaxShape::String,
                "Data type of the tensor (default: 'float32')",
                None,
            )
            .named(
                "dim",
                SyntaxShape::Int,
                "Dimension along which to compute mean (default: over all elements)",
                None,
            )
            .named(
                "keepdim",
                SyntaxShape::Boolean,
                "Whether to keep the reduced dimension as size 1 (default: false)",
                None,
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Compute mean value over all elements of a tensor",
                example: "let t1 = (torch full 5 2 3); $t1 | torch mean | torch value",
                result: None,
            },
            Example {
                description: "Compute mean along a specific dimension with keepdim",
                example: "let t1 = (torch full 5 2 3); $t1 | torch mean --dim 1 --keepdim true | torch value",
                result: None,
            }
        ]
    }

    fn run(
        &self,
        _plugin: &NutorchPlugin,
        _engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        // Get tensor1 ID from input (pipeline)
        let input_value = input.into_value(call.head)?;
        let tensor1_id = input_value.as_str().map(|s| s.to_string()).map_err(|_| {
            LabeledError::new("Invalid input")
                .with_label("Unable to parse tensor1 ID from input", call.head)
        })?;

        // Look up tensor in registry
        let mut registry = TENSOR_REGISTRY.lock().unwrap();
        let tensor1 = registry
            .get(&tensor1_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor1 ID", call.head)
            })?
            .shallow_clone();

        let kind = get_kind_from_call(call)?;

        // Single tensor mode (mean over dimension or entire tensor)
        let dim_opt: Option<i64> = call.get_flag("dim")?;
        let keepdim = call.get_flag::<bool>("keepdim")?.unwrap_or(false);
        let result_tensor: Tensor = match dim_opt {
            Some(dim) => {
                let num_dims = tensor1.size().len() as i64;
                if dim < 0 || dim >= num_dims {
                    return Err(LabeledError::new("Invalid dimension").with_label(
                        format!(
                            "Dimension {dim} out of bounds for tensor with {num_dims} dimensions"
                        ),
                        call.head,
                    ));
                }
                // Use mean_dim for dimension-specific mean
                tensor1.mean_dim(dim, keepdim, kind)
            }
            None => tensor1.mean(kind), // Meanimum over all elements
        };

        // Store result in registry with new ID
        let new_id = Uuid::new_v4().to_string();
        registry.insert(new_id.clone(), result_tensor);
        // Return new ID wrapped in PipelineData
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}
