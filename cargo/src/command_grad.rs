use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use uuid::Uuid;

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;

//--------------------------------------------------------------------------
// torch grad
//
// Return the gradient tensor associated with a leaf tensor.
//
//    $param | torch grad
//    torch grad $param
//
// – If no grad exists, returns Nushell `null`.
// – If a grad exists, stores it in the registry and returns its UUID string.
//--------------------------------------------------------------------------
pub struct CommandGrad;

impl PluginCommand for CommandGrad {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch grad"
    }

    fn description(&self) -> &str {
        "Fetch the .grad of a tensor. Returns null if no gradient is defined."
    }

    fn signature(&self) -> Signature {
        Signature::build("torch grad")
            .input_output_types(vec![
                (Type::String, Type::String), // tensor id via pipeline → string (uuid) or null
                (Type::Nothing, Type::String), // tensor id as arg       → "
            ])
            .optional(
                "tensor_id",
                SyntaxShape::String,
                "Tensor ID (if not supplied through the pipeline)",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Inspect a gradient that exists",
                example: r#"
let w   = (torch full [1] 3 --requires_grad true)
let loss = ($w | torch mean)
$loss | torch backward
$w | torch grad | torch value        # shows 1-tensor gradient
"#
                .trim(),
                result: None,
            },
            Example {
                description: "Returns null when no grad defined",
                example: r#"
let w = (torch full [1] 5 --requires_grad true)
torch grad $w              # → null
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
        //---------------- obtain tensor ID ------------------------------
        let piped = match &input {
            PipelineData::Value(v, _) => Some(v.clone()),
            PipelineData::Empty => None,
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or Value pipeline inputs accepted", call.head))
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

        //---------------- fetch tensor & its grad -----------------------
        let mut reg = TENSOR_REGISTRY.lock().unwrap();
        let t = reg
            .get(&tensor_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor ID", call.head)
            })?
            .shallow_clone();

        let g = t.grad(); // always returns Tensor
        if !g.defined() {
            // Return null to Nushell
            return Ok(PipelineData::Value(Value::nothing(call.head), None));
        }

        //---------------- store grad in registry ------------------------
        let gid = Uuid::new_v4().to_string();
        reg.insert(gid.clone(), g);

        Ok(PipelineData::Value(Value::string(gid, call.head), None))
    }
}
