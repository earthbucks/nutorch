use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use uuid::Uuid;

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;

// torch detach  -------------------------------------------------------------
// Return a new tensor that shares storage with the original but is detached
// from the autograd graph (requires_grad = false).  Usage:
//
//     $x  | torch detach
//     torch detach $x
//
// The original tensor remains unchanged and can still track gradients; the
// returned tensor does not.
//
// ---------------------------------------------------------------------------
pub struct CommandDetach;

impl PluginCommand for CommandDetach {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch detach"
    }

    fn description(&self) -> &str {
        "Create a view of a tensor that does **not** track gradients \
         (like Tensor.detach() in PyTorch)."
    }

    fn signature(&self) -> Signature {
        Signature::build("torch detach")
            .input_output_types(vec![
                (Type::String, Type::String),  // ID via pipeline → ID
                (Type::Nothing, Type::String), // ID via arg      → ID
            ])
            .optional(
                "tensor_id",
                SyntaxShape::String,
                "ID of the tensor to detach (if not provided via pipeline)",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Detach a tensor received through the pipeline",
                example: r#"
let x = (torch randn [2 2] --requires_grad true)
$x | torch detach | torch requires_grad?
"#
                .trim(),
                result: None,
            },
            Example {
                description: "Detach via positional argument",
                example: r#"
let x = (torch randn [2] --requires_grad true)
torch detach $x | torch requires_grad?
"#
                .trim(),
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
        //------------------------------------------------------------------
        // 1. Collect tensor ID (pipeline xor arg)
        //------------------------------------------------------------------
        let piped = match input {
            PipelineData::Value(v, _) => Some(v),
            PipelineData::Empty => None,
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or Value inputs are supported", call.head))
            }
        };
        let arg0 = call.nth(0);

        match (&piped, &arg0) {
            (None, None) => {
                return Err(LabeledError::new("Missing input")
                    .with_label("Provide tensor ID via pipeline or argument", call.head))
            }
            (Some(_), Some(_)) => {
                return Err(LabeledError::new("Conflicting input").with_label(
                    "Provide tensor ID via pipeline OR argument, not both",
                    call.head,
                ))
            }
            _ => {}
        }

        let id_val = piped.or(arg0).unwrap();
        let tensor_id = id_val.as_str()?.to_string();

        //------------------------------------------------------------------
        // 2. Fetch tensor from registry
        //------------------------------------------------------------------
        let mut reg = TENSOR_REGISTRY.lock().unwrap();
        let t = reg
            .get(&tensor_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor ID", call.head)
            })?
            .shallow_clone();

        //------------------------------------------------------------------
        // 3. Detach and store result
        //------------------------------------------------------------------
        let detached = t.detach(); // no longer tracks gradients
        let new_id = Uuid::new_v4().to_string();
        reg.insert(new_id.clone(), detached);

        //------------------------------------------------------------------
        // 4. Return ID of detached tensor
        //------------------------------------------------------------------
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}
