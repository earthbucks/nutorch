use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use tch::Tensor;
use uuid::Uuid;

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;

pub struct CommandCat;

impl PluginCommand for CommandCat {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch cat"
    }

    fn description(&self) -> &str {
        "Concatenate a sequence of tensors along a specified dimension (similar to torch.cat)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch cat")
            .input_output_types(vec![
                (Type::List(Box::new(Type::String)), Type::String),
                (Type::Nothing, Type::String),
            ])
            .optional(
                "tensor_ids",
                SyntaxShape::List(Box::new(SyntaxShape::String)),
                "List of tensor IDs to concatenate (if not using pipeline input)",
            )
            .named(
                "dim",
                SyntaxShape::Int,
                "Dimension along which to concatenate (default: 0)",
                None,
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Concatenate two 2x3 tensors along dimension 0 using pipeline input",
                example: "let t1 = (torch full 1 2 3); let t2 = (torch full 2 2 3); [$t1, $t2] | torch cat --dim 0 | torch value",
                result: None,
            },
            Example {
                description: "Concatenate three 2x3 tensors along dimension 1 using argument",
                example: "let t1 = (torch full 1 2 3); let t2 = (torch full 2 2 3); let t3 = (torch full 3 2 3); torch cat [$t1, $t2, $t3] --dim 1 | torch value",
                result: None,
            }
        ]
    }

    #[allow(clippy::too_many_lines)]
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
        let tensor_ids: Vec<String> = match (pipeline_input, arg_input) {
            (None, None) => {
                return Err(LabeledError::new("Missing input").with_label(
                    "Tensor IDs must be provided via pipeline or as an argument",
                    call.head,
                ));
            }
            (Some(_), Some(_)) => {
                return Err(LabeledError::new("Conflicting input").with_label(
                    "Tensor IDs cannot be provided both via pipeline and as an argument",
                    call.head,
                ));
            }
            (Some(input_val), None) => input_val
                .as_list()
                .map_err(|_| {
                    LabeledError::new("Invalid input")
                        .with_label("Pipeline input must be a list of tensor IDs", call.head)
                })?
                .iter()
                .map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Result<Vec<String>, _>>()?,
            (None, Some(arg_val)) => arg_val
                .as_list()
                .map_err(|_| {
                    LabeledError::new("Invalid input")
                        .with_label("Argument must be a list of tensor IDs", call.head)
                })?
                .iter()
                .map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Result<Vec<String>, _>>()?,
        };

        // Validate that at least two tensors are provided
        if tensor_ids.len() < 2 {
            return Err(LabeledError::new("Invalid input").with_label(
                "At least two tensor IDs must be provided for concatenation",
                call.head,
            ));
        }

        // Get the dimension to concatenate along (default to 0)
        let dim: i64 = match call.get_flag::<i64>("dim")? {
            Some(d) => {
                if d < 0 {
                    return Err(LabeledError::new("Invalid input")
                        .with_label("Dimension must be non-negative", call.head));
                }
                d
            }
            None => 0,
        };

        // Look up tensors in registry
        let mut registry = TENSOR_REGISTRY.lock().unwrap();
        let mut tensors: Vec<Tensor> = Vec::new();
        for id in &tensor_ids {
            match registry.get(id) {
                Some(tensor) => tensors.push(tensor.shallow_clone()),
                None => {
                    return Err(LabeledError::new("Tensor not found")
                        .with_label(format!("Invalid tensor ID: {}", id), call.head))
                }
            }
        }

        // Check if tensors have compatible shapes for concatenation
        if tensors.is_empty() {
            return Err(LabeledError::new("Invalid input")
                .with_label("No tensors provided for concatenation", call.head));
        }
        let first_shape = tensors[0].size();
        if first_shape.len() as i64 <= dim {
            return Err(LabeledError::new("Invalid dimension").with_label(
                format!(
                    "Dimension {} out of bounds for tensor with {} dimensions",
                    dim,
                    first_shape.len()
                ),
                call.head,
            ));
        }
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            let shape = tensor.size();
            if shape.len() != first_shape.len() {
                return Err(LabeledError::new("Shape mismatch").with_label(
                    format!(
                        "Tensor {} has different number of dimensions ({} vs {})",
                        i,
                        shape.len(),
                        first_shape.len()
                    ),
                    call.head,
                ));
            }
            for (d, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if d as i64 != dim && s1 != s2 {
                    return Err(LabeledError::new("Shape mismatch").with_label(
                        format!(
                            "Tensor {} has mismatched size in dimension {} ({} vs {})",
                            i, d, s2, s1
                        ),
                        call.head,
                    ));
                }
            }
        }

        // Create references to tensors for cat
        let tensor_refs: Vec<&Tensor> = tensors.iter().collect();

        // Perform concatenation using tch-rs
        let result_tensor = Tensor::cat(&tensor_refs, dim);

        // Store result in registry with new ID
        let new_id = Uuid::new_v4().to_string();
        registry.insert(new_id.clone(), result_tensor);
        // Return new ID wrapped in PipelineData
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}
