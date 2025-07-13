use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use uuid::Uuid;

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;

pub struct CommandMm;

impl PluginCommand for CommandMm {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch mm"
    }

    fn description(&self) -> &str {
        "Matrix multiply two 2-D tensors (like torch.mm)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch mm")
            // tensor id(s) may come from pipeline or args
            .input_output_types(vec![
                (Type::String, Type::String),  // single ID via pipe
                (Type::Nothing, Type::String), // both IDs via args
            ])
            .optional(
                "tensor1_id",
                SyntaxShape::String,
                "First tensor ID (if not piped)",
            )
            .optional("tensor2_id", SyntaxShape::String, "Second tensor ID")
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Pipeline first tensor, argument second tensor",
                example: r#"
let a = ([[1 2] [3 4]] | torch tensor)      # 2×2
let b = ([[5] [6]]     | torch tensor)      # 2×1
$a | torch mm $b | torch value              # → [[17] [39]]
"#
                .trim(),
                result: None,
            },
            Example {
                description: "Both tensors as arguments",
                example: r#"
let a = ([[1 2] [3 4]] | torch tensor)
let b = ([[5] [6]]     | torch tensor)
torch mm $a $b | torch value
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
        // -------- Collect exactly two tensor IDs --------------------------
        let mut ids: Vec<String> = Vec::new();

        // pipeline contribution
        if let PipelineData::Value(v, _) = input {
            if !v.is_nothing() {
                ids.push(v.as_str().map(|s| s.to_string()).map_err(|_| {
                    LabeledError::new("Invalid input")
                        .with_label("Pipeline input must be a tensor ID (string)", call.head)
                })?);
            }
        }

        // positional args (max two)
        for i in 0..2 {
            if let Some(arg) = call.nth(i) {
                ids.push(arg.as_str()?.to_string());
            }
        }

        if ids.len() != 2 {
            return Err(LabeledError::new("Invalid input count").with_label(
                "Exactly two tensor IDs are required (pipeline+arg or two args)",
                call.head,
            ));
        }
        let (id_a, id_b) = (ids.remove(0), ids.remove(0));

        // -------- Fetch tensors -------------------------------------------
        let mut reg = TENSOR_REGISTRY.lock().unwrap();
        let a = reg
            .get(&id_a)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found")
                    .with_label("Invalid first tensor ID", call.head)
            })?
            .shallow_clone();

        let b = reg
            .get(&id_b)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found")
                    .with_label("Invalid second tensor ID", call.head)
            })?
            .shallow_clone();

        // -------- Validate shapes (must be 2-D and inner dims equal) -------
        let sa = a.size();
        let sb = b.size();
        if sa.len() != 2 {
            return Err(LabeledError::new("Invalid tensor dimension").with_label(
                format!("First tensor must be 2-D, got {}-D", sa.len()),
                call.head,
            ));
        }
        if sb.len() != 2 {
            return Err(LabeledError::new("Invalid tensor dimension").with_label(
                format!("Second tensor must be 2-D, got {}-D", sb.len()),
                call.head,
            ));
        }
        if sa[1] != sb[0] {
            return Err(LabeledError::new("Incompatible dimensions").with_label(
                format!(
                    "Cannot multiply {}×{} with {}×{}",
                    sa[0], sa[1], sb[0], sb[1]
                ),
                call.head,
            ));
        }

        // -------- Compute mm ----------------------------------------------
        let result = a.mm(&b);

        // -------- Store & return ------------------------------------------
        let new_id = Uuid::new_v4().to_string();
        reg.insert(new_id.clone(), result);
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}
