use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use tch::Kind;
use tch::Tensor;
use uuid::Uuid;

use crate::add_grad_from_call;
use crate::get_device_from_call;
use crate::get_kind_from_call;
use crate::Number;
use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;

pub struct CommandFull;

impl PluginCommand for CommandFull {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch full"
    }

    fn description(&self) -> &str {
        "Create a tensor of specified shape filled with a given value (similar to torch.full)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch full")
            .required(
                "size",
                SyntaxShape::List(Box::new(SyntaxShape::Int)),
                "The shape of the tensor as a list of dimensions (e.g., [2, 3] for a 2x3 tensor)",
            )
            .required(
                "fill_value",
                SyntaxShape::Number,
                "The value to fill the tensor with",
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
                description: "Create a 1D tensor of length 5 filled with value 7",
                example: "torch full [5] 7 | torch value",
                result: None,
            },
            Example {
                description: "Create a 2x3 tensor filled with value 0.5 with float64 dtype on CPU",
                example: "torch full [2, 3] 0.5 --dtype float64 --device cpu | torch value",
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
        // Get the size (list of dimensions)
        let size_val = call.nth(0).unwrap();
        let dims: Vec<i64> = size_val
            .as_list()
            .map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Size must be a list of integers", call.head)
            })?
            .iter()
            .map(|v| v.as_int())
            .collect::<Result<Vec<i64>, _>>()?;
        if dims.is_empty() {
            return Err(LabeledError::new("Invalid input").with_label(
                "At least one dimension must be provided in size list",
                call.head,
            ));
        }
        if dims.iter().any(|&d| d < 1) {
            return Err(LabeledError::new("Invalid input")
                .with_label("All dimensions must be positive", call.head));
        }

        // Get the fill value (try as int first, then float)
        let fill_value_val = call.nth(1).unwrap();
        let fill_value_result = match fill_value_val.as_int() {
            Ok(int_val) => Ok(Number::Int(int_val)),
            Err(_) => fill_value_val.as_float().map(Number::Float).map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Fill value must be a number (integer or float)", call.head)
            }),
        };
        let fill_value = fill_value_result?;

        // Handle optional device argument using convenience method
        let device = get_device_from_call(call)?;

        // Handle optional dtype argument using convenience method
        let kind = get_kind_from_call(call)?;

        let mut tensor = match (fill_value, kind) {
            (Number::Int(i), Kind::Int | Kind::Int64) => {
                // Use integer-specific creation if tch-rs supports it directly
                // Since Tensor::full may expect f64, we pass as f64 but kind ensures it's stored as int
                Tensor::full(&dims, i, (kind, device))
            }
            (Number::Int(i), Kind::Float | Kind::Double) => {
                // Safe to cast int to float for float dtype
                Tensor::full(&dims, i, (kind, device))
            }
            (Number::Float(f), Kind::Float | Kind::Double) => {
                // Direct float usage
                Tensor::full(&dims, f, (kind, device))
            }
            _ => {
                return Err(LabeledError::new("Invalid dtype")
                    .with_label("Invalid data/dtype combo.", call.head));
            }
        };

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
