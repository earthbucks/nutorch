use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use uuid::Uuid;

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;

// torch reshape  -----------------------------------------------------------
// Reshape a tensor to a new shape.
//   $tensor | torch reshape [dim0 dim1 ...]
//
// • The source tensor **must** be supplied through the pipeline.
// • The first positional argument is a Nushell list of integers that becomes
//   the new shape.  `-1` is allowed once to let PyTorch infer that dimension.
// --------------------------------------------------------------------------
pub struct CommandReshape;

impl PluginCommand for CommandReshape {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str { "torch reshape" }

    fn description(&self) -> &str {
        "Return a tensor with the same data but a new shape (wraps Tensor::reshape)."
    }

    fn signature(&self) -> Signature {
        Signature::build("torch reshape")
            .input_output_types(vec![(Type::String, Type::String)])   // tensor id in/out
            .required(
                "shape",
                SyntaxShape::List(Box::new(SyntaxShape::Int)),
                "Target shape, supplied as a list of ints (may contain one -1)",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Reshape a length-6 vector to 2×3",
                example: r#"
let v = ([1 2 3 4 5 6] | torch tensor)
$v | torch reshape [2 3] | torch shape         # → [2, 3]
"#.trim(),
                result: None,
            },
            Example {
                description: "Use -1 to infer one dimension",
                example: r#"
let v = ([1 2 3 4 5 6] | torch tensor)
$v | torch reshape [3 -1] | torch shape        # → [3, 2]
"#.trim(),
                result: None,
            },
        ]
    }

    fn run(
        &self,
        _plugin : &NutorchPlugin,
        _engine : &nu_plugin::EngineInterface,
        call    : &nu_plugin::EvaluatedCall,
        input   : PipelineData,
    ) -> Result<PipelineData, LabeledError>
    {
        // ── source tensor must come through pipeline ───────────────────
        let PipelineData::Value(tid_val, _) = input else {
            return Err(LabeledError::new("Missing input")
                .with_label("Tensor ID must be piped into torch reshape", call.head));
        };
        let src_id = tid_val.as_str()?.to_string();

        // ── required shape list argument ───────────────────────────────
        let shape_val = call.nth(0).ok_or_else(|| {
            LabeledError::new("Missing shape")
                .with_label("First argument must be the target shape list", call.head)
        })?;

        let shape_list = shape_val.as_list().map_err(|_| {
            LabeledError::new("Invalid shape")
                .with_label("Shape must be a list of integers", call.head)
        })?;

        let mut shape: Vec<i64> = Vec::with_capacity(shape_list.len());
        for (i, dim_val) in shape_list.iter().enumerate() {
            let dim = dim_val.as_int().map_err(|_| {
                LabeledError::new("Invalid shape element")
                    .with_label(format!("Shape element at index {i} is not an int"), call.head)
            })?;
            shape.push(dim as i64);
        }

        // ── fetch tensor ───────────────────────────────────────────────
        let mut reg = TENSOR_REGISTRY.lock().unwrap();
        let src = reg.get(&src_id).ok_or_else(|| {
            LabeledError::new("Tensor not found")
                .with_label("Invalid source tensor ID", call.head)
        })?.shallow_clone();

        // ── reshape (tch will error if incompatible) ───────────────────
        let result = src.reshape(&shape);

        // ── store & return ─────────────────────────────────────────────
        let new_id = Uuid::new_v4().to_string();
        reg.insert(new_id.clone(), result);
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}
