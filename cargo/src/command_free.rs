use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;

// torch free  ---------------------------------------------------------------
// Explicitly drop tensors from the global registry so their memory can be
// reclaimed.  Accepts IDs via pipeline OR as the first positional argument
// (string or list of strings) but not both.
//
//     $id  | torch free
//     [$id1 $id2] | torch free
//     torch free $id
//     torch free [$id1 $id2]
//
// Returns the list of IDs that were successfully removed.
// ---------------------------------------------------------------------------
pub struct CommandFree;

impl PluginCommand for CommandFree {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch free"
    }

    fn description(&self) -> &str {
        "Remove tensor(s) from the internal registry, freeing their memory \
         when no other references exist."
    }

    fn signature(&self) -> Signature {
        Signature::build("torch free")
            .input_output_types(vec![
                (Type::String, Type::List(Box::new(Type::String))),
                (
                    Type::List(Box::new(Type::String)),
                    Type::List(Box::new(Type::String)),
                ),
                (Type::Nothing, Type::List(Box::new(Type::String))),
            ])
            .optional(
                "tensor_ids",
                SyntaxShape::List(Box::new(SyntaxShape::String)),
                "Tensor ID or list of IDs to free (if not provided by pipeline)",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Free a single tensor via pipeline",
                example: r#"
let t = (torch full [1000] 1)
$t | torch free
"#
                .trim(),
                result: None,
            },
            Example {
                description: "Free several tensors in one call",
                example: r#"
let a = (torch randn [1000 1000])
let b = (torch randn [1000 1000])
torch free [$a $b]
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
        // ── gather IDs from pipeline or argument ───────────────────────
        let piped: Option<Value> = match &input {
            PipelineData::Value(v, _) => Some(v.clone()),
            PipelineData::Empty => None,
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or Value inputs accepted", call.head))
            }
        };
        let arg0: Option<Value> = call.nth(0);

        match (&piped, &arg0) {
            (None, None) => {
                return Err(LabeledError::new("Missing input")
                    .with_label("Provide tensor ID(s) via pipeline or argument", call.head))
            }
            (Some(_), Some(_)) => {
                return Err(LabeledError::new("Conflicting input")
                    .with_label("Provide IDs via pipeline OR argument, not both", call.head))
            }
            _ => {}
        }

        let ids_val = piped.or(arg0).unwrap();

        // accept single string or list-of-strings
        let ids: Vec<String> = if let Ok(list) = ids_val.as_list() {
            list.iter()
                .map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![ids_val.as_str()?.to_string()]
        };

        if ids.is_empty() {
            return Err(
                LabeledError::new("Empty list").with_label("No tensor IDs supplied", call.head)
            );
        }

        // ── remove from registry ───────────────────────────────────────
        let mut reg = TENSOR_REGISTRY.lock().unwrap();
        let mut freed: Vec<Value> = Vec::new();

        for id in ids {
            if reg.remove(&id).is_some() {
                // entry removed; push to return list
                freed.push(Value::string(id, call.head));
            } else {
                return Err(LabeledError::new("Tensor not found")
                    .with_label(format!("Invalid tensor ID: {id}"), call.head));
            }
        }

        // returning list of IDs that were freed
        Ok(PipelineData::Value(Value::list(freed, call.head), None))
    }
}
