use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use tch::Kind;
use uuid::Uuid;

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;


// torch repeat_interleave ---------------------------------------------------
// Usage examples
//
//   $x | torch repeat_interleave 3                     # scalar repeat
//   $x | torch repeat_interleave $rep_tensor           # tensor repeat counts
//   $x | torch repeat_interleave 2 --dim 1
//   $x | torch repeat_interleave $rep_tensor --output-size 12
//
// Source tensor MUST be provided by pipeline.
// ---------------------------------------------------------------------------
pub struct CommandRepeatInterleave;

impl PluginCommand for CommandRepeatInterleave {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch repeat_interleave"
    }

    fn description(&self) -> &str {
        "Repeat elements of a tensor either a fixed number of times or 
        according to another tensor of repeat counts."
    }

    fn signature(&self) -> Signature {
        Signature::build("torch repeat_interleave")
            // pipeline input must be a tensor-id string, result is tensor-id
            .input_output_types(vec![(Type::String, Type::String)])
            .required(
                "repeats",
                SyntaxShape::Any,
                "Repeat factor (integer) or tensor ID",
            )
            .named(
                "dim",
                SyntaxShape::Int,
                "Dimension along which to repeat (default: flatten)",
                None,
            )
            .named(
                "output_size",
                SyntaxShape::Int,
                "Optional output size hint",
                None,
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Scalar repeat",
                example: r#"
let x = ([1 2 3] | torch tensor)
$x | torch repeat_interleave 2 | torch value   # -> [1 1 2 2 3 3]
"#
                .trim(),
                result: None,
            },
            Example {
                description: "Per-element repeat counts (tensor)",
                example: r#"
let x   = ([10 20 30] | torch tensor)
let rep = ([1 2 3]    | torch tensor --dtype int64)
$x | torch repeat_interleave $rep | torch value
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
        // 1. source tensor must come via pipeline
        //------------------------------------------------------------------
        let PipelineData::Value(src_val, _) = input else {
            return Err(LabeledError::new("Missing input")
                .with_label("Source tensor ID must be piped in", call.head));
        };
        let src_id = src_val.as_str()?.to_string();

        //------------------------------------------------------------------
        // 2. 'repeats' positional argument
        //------------------------------------------------------------------
        let repeats_val = call
            .nth(0)
            .ok_or_else(|| {
                LabeledError::new("Missing repeats")
                    .with_label("Provide repeat count (int) or tensor ID", call.head)
            })?
            .clone();

        //------------------------------------------------------------------
        // 3. optional named flags
        //------------------------------------------------------------------
        let dim_opt: Option<i64> = call.get_flag("dim")?;
        let osize_opt: Option<i64> = call.get_flag("output_size")?;

        //------------------------------------------------------------------
        // 4. fetch source tensor, registry lock
        //------------------------------------------------------------------
        let mut reg = TENSOR_REGISTRY.lock().unwrap();
        let src = reg
            .get(&src_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found")
                    .with_label("Invalid source tensor ID", call.head)
            })?
            .shallow_clone();

        //------------------------------------------------------------------
        // 5. branch: repeats is int OR tensor-id
        //------------------------------------------------------------------
        let result = if let Ok(rep_int) = repeats_val.as_int() {
            if rep_int <= 0 {
                return Err(LabeledError::new("Invalid repeats")
                    .with_label("Repeat count must be > 0", call.head));
            }
            src.repeat_interleave_self_int(rep_int, dim_opt, osize_opt)
        } else {
            let rep_id = repeats_val.as_str()?.to_string();
            let rep_t = reg
                .get(&rep_id)
                .ok_or_else(|| {
                    LabeledError::new("Tensor not found")
                        .with_label("Invalid repeats tensor ID", call.head)
                })?
                .shallow_clone();

            let rep_t = if rep_t.kind() == Kind::Int64 {
                rep_t
            } else {
                rep_t.to_kind(Kind::Int64)
            };

            src.repeat_interleave_self_tensor(&rep_t, dim_opt, osize_opt)
        };

        //------------------------------------------------------------------
        // 6. store result & return ID
        //------------------------------------------------------------------
        let new_id = Uuid::new_v4().to_string();
        reg.insert(new_id.clone(), result);
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}
