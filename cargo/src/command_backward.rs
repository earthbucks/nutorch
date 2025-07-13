use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;

/// torch backward
/// Usage :   <loss-tensor-id> | torch backward
///        or torch backward <loss-tensor-id>
///
/// Calls Tensor::backward() on a scalar loss tensor so that gradients are
/// accumulated into all leaf tensors that have `requires_grad == true`.
///
/// The command returns the same tensor-id so the value can still be piped.
pub struct CommandBackward;

impl PluginCommand for CommandBackward {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch backward"
    }

    fn description(&self) -> &str {
        "Run back-propagation from a scalar loss tensor (loss.backward())."
    }

    fn signature(&self) -> Signature {
        Signature::build("torch backward")
            .input_output_types(vec![
                (Type::String, Type::String),  // tensor id via pipeline
                (Type::Nothing, Type::String), // tensor id via arg
            ])
            .optional(
                "loss_id",
                SyntaxShape::String,
                "ID of the scalar loss tensor (if not supplied by pipeline)",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Backward via pipeline",
                example: r#"
let w = (torch full [1] 2 --requires_grad true)
let loss = ($w | torch mean)
$loss | torch backward
"#
                .trim(),
                result: None,
            },
            Example {
                description: "Backward via argument",
                example: r#"
let w = (torch full [1] 2 --requires_grad true)
let loss = ($w | torch mean)
torch backward $loss
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
        // ───── collect loss tensor-id either from pipeline or arg ─────
        let piped: Option<Value> = match input {
            PipelineData::Value(v, _) => Some(v),
            PipelineData::Empty => None,
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or single Value inputs accepted", call.head))
            }
        };

        let arg0: Option<Value> = call.nth(0);

        match (&piped, &arg0) {
            (None, None) => {
                return Err(LabeledError::new("Missing input")
                    .with_label("Provide loss tensor ID via pipeline or argument", call.head))
            }
            (Some(_), Some(_)) => {
                return Err(LabeledError::new("Conflicting input").with_label(
                    "Provide loss tensor ID via pipeline OR argument, not both",
                    call.head,
                ))
            }
            _ => {}
        }

        let loss_id_val = piped.or(arg0).unwrap();
        let loss_id = loss_id_val.as_str()?.to_string();

        // ───── fetch the loss tensor ─────
        let reg = TENSOR_REGISTRY.lock().unwrap();
        let loss = reg
            .get(&loss_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found")
                    .with_label("Invalid loss tensor ID", call.head)
            })?
            .shallow_clone();

        // ───── ensure it is scalar (numel == 1)  (PyTorch expectation) ──
        if loss.numel() != 1 {
            return Err(LabeledError::new("Invalid loss tensor")
                .with_label("Backward currently supports only scalar losses", call.head));
        }

        // ───── run backward  (grad-mode ON by default) ─────
        loss.backward();

        // return the same id for convenience
        Ok(PipelineData::Value(Value::string(loss_id, call.head), None))
    }
}
