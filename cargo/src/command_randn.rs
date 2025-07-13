use nu_plugin::PluginCommand;
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use tch::Tensor;
use uuid::Uuid;

use crate::add_grad_from_call;
use crate::get_device_from_call;
use crate::get_kind_from_call;
use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;

pub struct CommandRandn;

impl PluginCommand for CommandRandn {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch randn"
    }

    fn description(&self) -> &str {
        "Generate a tensor filled with random numbers from a normal distribution (mean 0, std 1)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch randn")
            .rest(
                "dims",
                SyntaxShape::Int,
                "Dimensions of the tensor (e.g., 2 3 for a 2x3 tensor)",
            )
            .named(
                "device",
                SyntaxShape::String,
                "Device to create the tensor on ('cpu', 'cuda', 'mps', default: 'cpu')",
                None,
            )
            .named(
                "dtype",
                SyntaxShape::String,
                "Data type of the tensor ('float32', 'float64', 'int32', 'int64')",
                None,
            )
            .named(
                "requires_grad",
                SyntaxShape::Boolean,
                "Whether the tensor requires gradient tracking for autograd (default: false)",
                None,
            )
            .input_output_types(vec![(Type::Nothing, Type::String)])
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Generate a 2x3 tensor with random values from a normal distribution",
                example: "torch randn 2 3 | torch tovalue",
                result: None,
            },
            Example {
                description:
                    "Generate a 1D tensor of size 5 with a specific seed for reproducibility",
                example: "torch manual_seed 42; torch randn 5 | torch tovalue",
                result: None,
            },
        ]
    }

    fn run(
        &self,
        _plugin: &NutorchPlugin,
        _engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        _input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        // Get dimensions for the tensor shape
        let dims: Vec<i64> = call
            .rest(0)
            .map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Unable to parse dimensions", call.head)
            })?
            .into_iter()
            .map(|v: Value| v.as_int())
            .collect::<Result<Vec<i64>, _>>()?;
        if dims.is_empty() {
            return Err(LabeledError::new("Invalid input")
                .with_label("At least one dimension must be provided", call.head));
        }
        if dims.iter().any(|&d| d < 1) {
            return Err(LabeledError::new("Invalid input")
                .with_label("All dimensions must be positive", call.head));
        }

        // Handle optional device argument
        let device = get_device_from_call(call)?;

        // Handle optional dtype argument
        let kind = get_kind_from_call(call)?;

        // Create a random tensor using tch-rs
        let mut tensor = Tensor::randn(&dims, (kind, device));

        // Handle optional requires_grad argument
        tensor = add_grad_from_call(call, tensor)?;

        // Generate a unique ID for the tensor
        let id = Uuid::new_v4().to_string();
        // Store in registry
        TENSOR_REGISTRY.lock().unwrap().insert(id.clone(), tensor);
        // Return the ID as a string to Nushell, wrapped in PipelineData
        Ok(PipelineData::Value(Value::string(id, call.head), None))
    }
}
