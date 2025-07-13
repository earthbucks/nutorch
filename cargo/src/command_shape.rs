
use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;

pub struct CommandShape;

impl PluginCommand for CommandShape {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch shape"
    }

    fn description(&self) -> &str {
        "Get the shape (dimensions) of a tensor as a list (similar to tensor.shape in PyTorch)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch shape")
            .input_output_types(vec![
                (Type::String, Type::List(Box::new(Type::Int))),
                (Type::Nothing, Type::List(Box::new(Type::Int))),
            ])
            .optional(
                "tensor_id",
                SyntaxShape::String,
                "ID of the tensor to get the shape of (if not using pipeline input)",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Get the shape of a tensor using pipeline input",
                example: "let t1 = (torch full [2, 3] 1); $t1 | torch shape",
                result: None,
            },
            Example {
                description: "Get the shape of a tensor using argument",
                example: "let t1 = (torch full [2, 3] 1); torch shape $t1",
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
        // Check for pipeline input
        let pipeline_input = match input {
            PipelineData::Empty => None,
            PipelineData::Value(val, _) => Some(val),
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or Value inputs are supported", call.head));
            }
        };

        // Check for positional argument input
        let arg_input = call.nth(0);

        // Validate that exactly one data source is provided
        let tensor_id: String = match (pipeline_input, arg_input) {
            (None, None) => {
                return Err(LabeledError::new("Missing input").with_label(
                    "Tensor ID must be provided via pipeline or as an argument",
                    call.head,
                ));
            }
            (Some(_), Some(_)) => {
                return Err(LabeledError::new("Conflicting input").with_label(
                    "Tensor ID cannot be provided both via pipeline and as an argument",
                    call.head,
                ));
            }
            (Some(input_val), None) => input_val.as_str().map(|s| s.to_string()).map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Pipeline input must be a tensor ID (string)", call.head)
            })?,
            (None, Some(arg_val)) => arg_val.as_str().map(|s| s.to_string()).map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Argument must be a tensor ID (string)", call.head)
            })?,
        };

        // Look up tensor in registry
        let registry = TENSOR_REGISTRY.lock().unwrap();
        let tensor = registry.get(&tensor_id).ok_or_else(|| {
            LabeledError::new("Tensor not found").with_label("Invalid tensor ID", call.head)
        })?;

        // Get the shape (dimensions) of the tensor
        let shape = tensor.size();
        let shape_values: Vec<Value> = shape
            .into_iter()
            .map(|dim| Value::int(dim, call.head))
            .collect();
        let shape_list = Value::list(shape_values, call.head);

        // Return the shape as a list directly (not stored in registry)
        Ok(PipelineData::Value(shape_list, None))
    }
}
