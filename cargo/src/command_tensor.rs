use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use uuid::Uuid;

use crate::add_grad_from_call;
use crate::get_device_from_call;
use crate::get_kind_from_call;
use crate::value_to_tensor;
use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;


// Command to convert Nushell data structure to tensor (tensor)
pub struct CommandTensor;

impl PluginCommand for CommandTensor {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch tensor"
    }

    fn description(&self) -> &str {
        "Convert a Nushell Value (nested list structure) to a tensor"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch tensor")
            .input_output_types(vec![(Type::Any, Type::String)])
            .optional(
                "data",
                SyntaxShape::Any,
                "Data to convert to a tensor (list or nested list)",
            )
            .named(
                "device",
                SyntaxShape::String,
                "Device to create the tensor on (default: 'cpu')",
                None,
            )
            .named(
                "dtype",
                SyntaxShape::String,
                "Data type of the tensor (default: 'float32')",
                None,
            )
            .named(
                "requires_grad",
                SyntaxShape::Boolean,
                "Whether the tensor requires gradient tracking for autograd (default: false)",
                None,
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Convert a 1D list to a tensor via pipeline",
                example: "[0.0, 1.0, 2.0, 3.0] | torch tensor",
                result: None,
            },
            Example {
                description: "Convert a 2D nested list to a tensor with specific device and dtype via pipeline",
                example: "[[0.0, 1.0], [2.0, 3.0]] | torch tensor --device cpu --dtype float64",
                result: None,
            },
            Example {
                description: "Convert a 1D list to a tensor via argument",
                example: "torch tensor [0.0, 1.0, 2.0, 3.0]",
                result: None,
            },
            Example {
                description: "Convert a 2D nested list to a tensor with specific device via argument",
                example: "torch tensor [[0.0, 1.0], [2.0, 3.0]] --device cpu",
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
        // Check for pipeline input
        let pipeline_input = match input {
            PipelineData::Empty => None,
            PipelineData::Value(val, _) => Some(val),
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Value or Empty input is supported", call.head));
            }
        };

        // Check for positional argument input
        let arg_input = call.nth(0);

        // Validate that exactly one data source is provided
        match (&pipeline_input, &arg_input) {
            (None, None) => {
                return Err(LabeledError::new("Missing input").with_label(
                    "Data must be provided via pipeline or as an argument",
                    call.head,
                ));
            }
            (Some(_), Some(_)) => {
                return Err(LabeledError::new("Conflicting input").with_label(
                    "Data cannot be provided both via pipeline and as an argument",
                    call.head,
                ));
            }
            (Some(input_val), None) => input_val,
            (None, Some(arg_val)) => arg_val,
        };

        let input_value = match (pipeline_input, arg_input) {
            (Some(input_val), None) => input_val,
            (None, Some(arg_val)) => arg_val,
            _ => unreachable!("Validation above ensures one source is provided"),
        };

        // Handle optional device argument
        let device = get_device_from_call(call)?;

        // Handle optional dtype argument
        let kind = get_kind_from_call(call)?;

        // Convert Nushell Value to tensor
        let mut tensor = value_to_tensor(&input_value, kind, device, call.head)?;

        // Handle optional requires_grad argument
        tensor = add_grad_from_call(call, tensor)?;

        let id = Uuid::new_v4().to_string();
        // Store in registry
        TENSOR_REGISTRY.lock().unwrap().insert(id.clone(), tensor);
        // Return the ID as a string to Nushell, wrapped in PipelineData
        Ok(PipelineData::Value(Value::string(id, call.head), None))
    }
}
