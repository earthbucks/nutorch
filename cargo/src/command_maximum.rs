use nu_plugin::{PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
};
use uuid::Uuid;

use crate::NutorchPlugin;
use crate::TENSOR_REGISTRY;

// torch maximum  -----------------------------------------------------------
//  1) [$t1 $t2] | torch maximum              (both IDs piped as a list)
//  2)  $t1      | torch maximum $t2          (first ID piped, second as arg)
//  3)  torch maximum $t1 $t2                 (no pipeline, two args – kept for b-compat)
// --------------------------------------------------------------------------
pub struct CommandMaximum;

impl PluginCommand for CommandMaximum {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch maximum"
    }

    fn description(&self) -> &str {
        "Element-wise maximum of two tensors with broadcasting (like torch.maximum)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch maximum")
            .input_output_types(vec![
                (Type::String, Type::String),                       // single id via pipe
                (Type::List(Box::new(Type::String)), Type::String), // list via pipe
                (Type::Nothing, Type::String),                      // all by args
            ])
            .optional(
                "tensor1_id",
                SyntaxShape::String,
                "ID of 1st tensor (if not piped)",
            )
            .optional(
                "tensor2_id",
                SyntaxShape::String,
                "ID of 2nd tensor (or 1st if one piped)",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Both tensor IDs in a list via pipeline",
                example: r#"
let a = (torch full [2,3] 1)
let b = (torch full [2,3] 2)
[$a $b] | torch maximum | torch value
"#
                .trim(),
                result: None,
            },
            Example {
                description: "First ID piped, second as argument",
                example: r#"
let a = (torch full [2,3] 1)
let b = (torch full [2,3] 2)
$a | torch maximum $b | torch value
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
        // collect exactly two tensor IDs  (pipeline list / pipeline single /
        // positional args)  –– same logic as before
        //------------------------------------------------------------------
        let mut ids: Vec<String> = Vec::new();

        match input {
            PipelineData::Empty => {}
            PipelineData::Value(v, _) => {
                if let Ok(list) = v.as_list() {
                    for itm in list {
                        ids.push(itm.as_str()?.to_string());
                    }
                } else {
                    ids.push(v.as_str()?.to_string());
                }
            }
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or Value inputs are supported", call.head))
            }
        }

        for i in 0..2 {
            if let Some(arg) = call.nth(i) {
                ids.push(arg.as_str()?.to_string());
            }
        }

        if ids.len() != 2 {
            return Err(LabeledError::new("Invalid input count")
                .with_label("Provide exactly two tensor IDs", call.head));
        }
        let (id1, id2) = (ids.remove(0), ids.remove(0));

        //------------------------------------------------------------------
        // fetch tensors
        //------------------------------------------------------------------
        let mut reg = TENSOR_REGISTRY.lock().unwrap();
        let t1 = reg
            .get(&id1)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found")
                    .with_label("Invalid first tensor ID", call.head)
            })?
            .shallow_clone();
        let t2 = reg
            .get(&id2)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found")
                    .with_label("Invalid second tensor ID", call.head)
            })?
            .shallow_clone();

        //------------------------------------------------------------------
        // broadcast-compatibility check
        //------------------------------------------------------------------
        #[allow(clippy::items_after_statements)]
        fn broadcast_ok(a: &[i64], b: &[i64]) -> bool {
            let mut ia = a.len() as isize - 1;
            let mut ib = b.len() as isize - 1;
            while ia >= 0 || ib >= 0 {
                let sa = if ia >= 0 { a[ia as usize] } else { 1 };
                let sb = if ib >= 0 { b[ib as usize] } else { 1 };
                if sa != sb && sa != 1 && sb != 1 {
                    return false;
                }
                ia -= 1;
                ib -= 1;
            }
            true
        }

        let shape1 = t1.size();
        let shape2 = t2.size();
        if !broadcast_ok(&shape1, &shape2) {
            return Err(LabeledError::new("Shape mismatch").with_label(
                format!(
                    "Tensors cannot be broadcast together: {:?} vs {:?}",
                    shape1, shape2
                ),
                call.head,
            ));
        }

        //------------------------------------------------------------------
        // compute maximum
        //------------------------------------------------------------------
        let result_tensor = t1.maximum(&t2);

        //------------------------------------------------------------------
        // store & return
        //------------------------------------------------------------------
        let new_id = Uuid::new_v4().to_string();
        reg.insert(new_id.clone(), result_tensor);
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}
