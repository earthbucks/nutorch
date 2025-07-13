use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use uuid::Uuid;

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;


// torch t  -----------------------------------------------------------------
// 2-D matrix transpose (like Tensor.t() in PyTorch / tch-rs).
//
//     $mat | torch t
//     torch t $mat
// --------------------------------------------------------------------------
pub struct CommandT;

impl PluginCommand for CommandT {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch t"
    }

    fn description(&self) -> &str {
        "Matrix transpose for 2-D tensors (equivalent to tensor.t() in PyTorch)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch t")
            .input_output_types(vec![
                (Type::String, Type::String),  // ID via pipe  → ID
                (Type::Nothing, Type::String), // ID via arg   → ID
            ])
            .optional(
                "tensor_id",
                SyntaxShape::String,
                "ID of the tensor to transpose (if not supplied by pipeline)",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Transpose a 2×3 matrix",
                example: r#"
let m = ([[1 2 3] [4 5 6]] | torch tensor)
$m | torch t | torch value   # → [[1 4] [2 5] [3 6]]
"#
                .trim(),
                result: None,
            },
            Example {
                description: "Error on non-2-D tensor",
                example: r#"
let v = ([1 2 3] | torch tensor)
torch t $v        # → error “Tensor must be 2-D”
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
        // 1. Obtain tensor ID (pipeline xor argument)
        //------------------------------------------------------------------
        let piped = match input {
            PipelineData::Value(v, _) => Some(v),
            PipelineData::Empty => None,
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or Value inputs supported", call.head))
            }
        };
        let arg0 = call.nth(0);

        match (&piped, &arg0) {
            (None, None) => {
                return Err(LabeledError::new("Missing input")
                    .with_label("Supply tensor ID via pipeline or argument", call.head))
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
        // 2. Fetch tensor and check dimensionality
        //------------------------------------------------------------------
        let mut reg = TENSOR_REGISTRY.lock().unwrap();
        let t = reg
            .get(&tensor_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor ID", call.head)
            })?
            .shallow_clone();

        if t.dim() != 2 {
            return Err(LabeledError::new("Invalid tensor dimension")
                .with_label(format!("Tensor must be 2-D, got {}-D", t.dim()), call.head));
        }

        //------------------------------------------------------------------
        // 3. Transpose and store
        //------------------------------------------------------------------
        let transposed = t.transpose(0, 1); // transpose(0,1)

        let new_id = Uuid::new_v4().to_string();
        reg.insert(new_id.clone(), transposed);

        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}
