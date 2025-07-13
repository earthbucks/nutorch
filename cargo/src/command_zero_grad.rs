use nu_plugin::PluginCommand;
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use tch::Tensor;

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;

// torch zero_grad  ---------------------------------------------------------
// Clear .grad buffers for one or several tensors.
//
//   [$id1 $id2] | torch zero_grad
//   torch zero_grad [$id1 $id2]
//
// Returns the list of IDs unchanged.
// --------------------------------------------------------------------------
pub struct CommandZeroGrad;

impl PluginCommand for CommandZeroGrad {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch zero_grad"
    }

    fn description(&self) -> &str {
        "Set the .grad field of the given tensor(s) to zero (like Tensor.zero_grad() in PyTorch)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch zero_grad")
            .input_output_types(vec![
                (Type::String, Type::String), // single ID via pipe
                (
                    Type::List(Box::new(Type::String)),
                    Type::List(Box::new(Type::String)),
                ), // list via pipe
                (Type::Nothing, Type::List(Box::new(Type::String))), // list as arg
            ])
            .optional(
                "tensors",
                SyntaxShape::List(Box::new(SyntaxShape::String)),
                "List of tensor IDs (if not given by pipeline)",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Clear gradients of a single tensor via pipeline",
                example: r#"
let w = (torch full [2,2] 1 --requires_grad true)
$w | torch zero_grad
"#
                .trim(),
                result: None,
            },
            Example {
                description: "Clear gradients of several tensors via argument",
                example: r#"
let w1 = (torch full [1] 1 --requires_grad true)
let w2 = (torch full [1] 2 --requires_grad true)
torch zero_grad [$w1, $w2]
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
        // 1. Collect tensor IDs either from pipeline or argument
        //------------------------------------------------------------------
        let piped_val: Option<Value> = match &input {
            PipelineData::Value(v, _) => Some(v.clone()),
            PipelineData::Empty => None,
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or Value inputs are supported", call.head))
            }
        };

        let arg_val: Option<Value> = call.nth(0);

        match (&piped_val, &arg_val) {
            (None, None) => {
                return Err(LabeledError::new("Missing input").with_label(
                    "Provide tensor IDs via pipeline or as an argument",
                    call.head,
                ));
            }
            (Some(_), Some(_)) => {
                return Err(LabeledError::new("Conflicting input").with_label(
                    "Provide tensor IDs via pipeline OR argument, not both",
                    call.head,
                ));
            }
            _ => {}
        }

        let list_val = piped_val.or(arg_val).unwrap();

        // Accept either single-ID string or list-of-IDs
        let ids: Vec<String> = if let Ok(lst) = list_val.as_list() {
            lst.iter()
                .map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![list_val.as_str().map(|s| s.to_string())?]
        };

        //------------------------------------------------------------------
        // 2. Fetch tensors and clear grads
        //------------------------------------------------------------------
        let reg = TENSOR_REGISTRY.lock().unwrap();
        tch::no_grad(|| {
            for id in &ids {
                let t = reg.get(id).ok_or_else(|| {
                    LabeledError::new("Tensor not found")
                        .with_label(format!("Invalid tensor ID: {id}"), call.head)
                })?;
                let mut tensor: Tensor = t.shallow_clone();
                tensor.zero_grad(); // in-place; ignore returned Result
            }
            Ok::<(), LabeledError>(()) // propagate any not-found error
        })?; // ? outside closure

        //------------------------------------------------------------------
        // 3. Return the same list of IDs
        //------------------------------------------------------------------
        let out_vals: Vec<Value> = ids
            .into_iter()
            .map(|id| Value::string(id, call.head))
            .collect();

        Ok(PipelineData::Value(Value::list(out_vals, call.head), None))
    }
}
