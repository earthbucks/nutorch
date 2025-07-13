use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use tch::Tensor;
use uuid::Uuid;

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;


// torch stack  --------------------------------------------------------------
// Stack a list of tensors along a new dimension (like torch.stack in PyTorch)
//
//   [$t1 $t2] | torch stack --dim 0
//   torch stack [$t1 $t2] --dim 1
//
// All tensors must have identical shapes.
// ---------------------------------------------------------------------------
pub struct CommandStack;

impl PluginCommand for CommandStack {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch stack"
    }

    fn description(&self) -> &str {
        "Concatenate a sequence of tensors along a new axis (torch.stack)."
    }

    fn signature(&self) -> Signature {
        Signature::build("torch stack")
            .input_output_types(vec![
                (Type::List(Box::new(Type::String)), Type::String), // list via pipeline
                (Type::Nothing, Type::String),                      // list via arg
            ])
            .optional(
                "tensor_ids",
                SyntaxShape::List(Box::new(SyntaxShape::String)),
                "List of tensor IDs (if not provided via pipeline)",
            )
            .named(
                "dim",
                SyntaxShape::Int,
                "Dimension index at which to insert the new axis (default 0)",
                None,
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Stack two 2×3 tensors along dim 0",
                example: r#"
let x = ([[1 2 3] [4 5 6]] | torch tensor)
let y = ([[7 8 9] [1 1 1]] | torch tensor)
[$x $y] | torch stack --dim 0 | torch shape   # -> [2, 2, 3]
"#
                .trim(),
                result: None,
            },
            Example {
                description: "Stack the same tensors along dim 1",
                example: r#"
let x = ([[1 2 3] [4 5 6]] | torch tensor)
let y = ([[7 8 9] [1 1 1]] | torch tensor)
torch stack [$x $y] --dim 1 | torch shape    # -> [2, 2, 3]
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
        // 1. Collect list of IDs (pipeline xor argument)
        //------------------------------------------------------------------
        let piped = match input {
            PipelineData::Value(v, _) => Some(v),
            PipelineData::Empty => None,
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or Value pipeline inputs supported", call.head))
            }
        };
        let arg0 = call.nth(0);

        match (&piped, &arg0) {
            (None, None) => {
                return Err(LabeledError::new("Missing input")
                    .with_label("Provide tensor list via pipeline or argument", call.head))
            }
            (Some(_), Some(_)) => {
                return Err(LabeledError::new("Conflicting input").with_label(
                    "Provide tensor list via pipeline OR argument, not both",
                    call.head,
                ))
            }
            _ => {}
        }

        let list_val = piped.or(arg0).unwrap();

        // accept single-level list of strings
        let ids: Vec<String> = list_val
            .as_list()
            .map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Expected a list of tensor IDs", call.head)
            })?
            .iter()
            .map(|v| v.as_str().map(|s| s.to_string()))
            .collect::<Result<Vec<_>, _>>()?;

        if ids.is_empty() {
            return Err(
                LabeledError::new("Empty list").with_label("No tensor IDs supplied", call.head)
            );
        }

        //------------------------------------------------------------------
        // 2. Parse dim flag (default 0)
        //------------------------------------------------------------------
        let mut dim: i64 = call.get_flag("dim")?.unwrap_or(0);

        //------------------------------------------------------------------
        // 3. Fetch tensors & validate shape equality
        //------------------------------------------------------------------
        let mut reg = TENSOR_REGISTRY.lock().unwrap();
        let tensors: Vec<Tensor> = ids
            .iter()
            .map(|id| {
                reg.get(id)
                    .ok_or_else(|| {
                        LabeledError::new("Tensor not found")
                            .with_label(format!("Invalid tensor ID: {id}"), call.head)
                    })
                    .map(|t| t.shallow_clone())
            })
            .collect::<Result<_, _>>()?;

        // ensure identical shapes
        let first_shape = tensors[0].size();
        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.size() != first_shape {
                return Err(LabeledError::new("Shape mismatch").with_label(
                    format!(
                        "Tensor at index {i} has shape {:?}, expected {:?}",
                        t.size(),
                        first_shape
                    ),
                    call.head,
                ));
            }
        }

        //------------------------------------------------------------------
        // 4. Adjust dim (negative allowed) and stack
        //------------------------------------------------------------------
        let rank = first_shape.len() as i64;
        if dim < 0 {
            dim += rank + 1;
        }
        if dim < 0 || dim > rank {
            return Err(LabeledError::new("Invalid dim").with_label(
                format!("dim must be in [0, {}], got {}", rank, dim),
                call.head,
            ));
        }

        let result = Tensor::stack(&tensors, dim);

        //------------------------------------------------------------------
        // 5. Store result & return new ID
        //------------------------------------------------------------------
        let new_id = Uuid::new_v4().to_string();
        reg.insert(new_id.clone(), result);

        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}
