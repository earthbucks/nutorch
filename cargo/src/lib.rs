use lazy_static::lazy_static;
use nu_plugin::{Plugin, PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, Span, SyntaxShape, Type, Value,
};
use std::collections::HashMap;
use std::sync::Mutex;
use tch::{Device, Kind, Tensor};
use uuid::Uuid;
mod command_add;
mod command_arange;
mod command_argmax;
mod command_backward;
mod command_cat;
mod command_detach;
mod command_div;
mod command_exp;
mod command_free;
mod command_full;
mod command_gather;
mod command_grad;
mod command_linspace;
mod command_log_softmax;
mod command_max;
mod command_maximum;
mod command_mean;
mod command_mm;
mod command_mul;
pub use command_add::CommandAdd;
pub use command_arange::CommandArange;
pub use command_argmax::CommandArgmax;
pub use command_backward::CommandBackward;
pub use command_cat::CommandCat;
pub use command_detach::CommandDetach;
pub use command_div::CommandDiv;
pub use command_exp::CommandExp;
pub use command_free::CommandFree;
pub use command_full::CommandFull;
pub use command_gather::CommandGather;
pub use command_grad::CommandGrad;
pub use command_linspace::CommandLinspace;
pub use command_log_softmax::CommandLogSoftmax;
pub use command_max::CommandMax;
pub use command_maximum::CommandMaximum;
pub use command_mean::CommandMean;
pub use command_mm::CommandMm;
pub use command_mul::CommandMul;

// Global registry to store tensors by ID (thread-safe)
lazy_static! {
    pub static ref TENSOR_REGISTRY: Mutex<HashMap<String, Tensor>> = Mutex::new(HashMap::new());
}

pub struct NutorchPlugin;

impl Plugin for NutorchPlugin {
    fn commands(&self) -> Vec<Box<dyn PluginCommand<Plugin = Self>>> {
        vec![
            // Top-level Torch command
            Box::new(CommandTorch),
            // Configuration and other global commands
            Box::new(CommandManualSeed),
            Box::new(CommandDevices),
            // Tensor operations
            // Moved
            Box::new(CommandAdd),
            Box::new(CommandArange),
            Box::new(CommandArgmax),
            Box::new(CommandBackward),
            Box::new(CommandCat),
            Box::new(CommandDetach),
            Box::new(CommandDiv),
            Box::new(CommandExp),
            Box::new(CommandFree),
            Box::new(CommandFull),
            Box::new(CommandGather),
            Box::new(CommandGrad),
            Box::new(CommandLinspace),
            Box::new(CommandLogSoftmax),
            Box::new(CommandMax),
            Box::new(CommandMaximum),
            Box::new(CommandMean),
            Box::new(CommandMm),
            Box::new(CommandMul),
            // Not yet moved
            Box::new(CommandNeg),
            Box::new(CommandRandn),
            Box::new(CommandRepeat),
            Box::new(CommandRepeatInterleave),
            Box::new(CommandReshape),
            Box::new(CommandSgdStep),
            Box::new(CommandShape),
            Box::new(CommandSin),
            Box::new(CommandSqueeze),
            Box::new(CommandStack),
            Box::new(CommandSub),
            Box::new(CommandT),
            Box::new(CommandTensor),
            Box::new(CommandUnsqueeze),
            Box::new(CommandValue),
            Box::new(CommandZeroGrad),
        ]
    }

    fn version(&self) -> std::string::String {
        "0.1.3".to_string()
    }
}

// New top-level Torch command
struct CommandTorch;

impl PluginCommand for CommandTorch {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch").category(Category::Custom("torch".into()))
    }

    fn description(&self) -> &str {
        "The entry point for the Nutorch plugin, providing access to tensor operations and utilities"
    }

    fn examples(&self) -> Vec<Example> {
        vec![Example {
            description: "Run the torch command to test the plugin".into(),
            example: "torch".into(),
            result: Some(Value::string(
                "Welcome to Nutorch. Type `torch --help` for more information.",
                nu_protocol::Span::unknown(),
            )),
        }]
    }

    fn run(
        &self,
        _plugin: &NutorchPlugin,
        _engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        _input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        Ok(PipelineData::Value(
            Value::string(
                "Welcome to Nutorch. Type `torch --help` for more information.",
                call.head,
            ),
            None,
        ))
    }
}

// torch neg  ---------------------------------------------------------------
// Negate a tensor (element-wise) :  y = -x
// Accept the tensor ID either from the pipeline or as a single argument.
// -------------------------------------------------------------------------
struct CommandNeg;

impl PluginCommand for CommandNeg {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch neg"
    }

    fn description(&self) -> &str {
        "Return the element-wise negative of a tensor (like –tensor in PyTorch)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch neg")
            .input_output_types(vec![
                (Type::String, Type::String), // pipeline-in
                (Type::Nothing, Type::String),
            ]) // arg-in
            .optional(
                "tensor_id",
                SyntaxShape::String,
                "ID of the tensor to negate (if not provided via pipeline)",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Negate a tensor supplied by pipeline",
                example: "let t = (torch full [2,3] 1); $t | torch neg | torch value",
                result: None,
            },
            Example {
                description: "Negate a tensor supplied as argument",
                example: "let t = (torch full [2,3] 1); torch neg $t | torch value",
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
        // -------- figure out where the tensor ID comes from ----------------
        let piped = match input {
            PipelineData::Empty => None,
            PipelineData::Value(v, _span) => Some(v),
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or single Value inputs are supported", call.head))
            }
        };
        let arg0 = call.nth(0);

        let tensor_id = match (piped, arg0) {
            (Some(_), Some(_)) => {
                return Err(LabeledError::new("Conflicting input").with_label(
                    "Provide tensor ID either via pipeline OR argument, not both",
                    call.head,
                ))
            }
            (None, None) => {
                return Err(LabeledError::new("Missing input").with_label(
                    "Tensor ID must be supplied via pipeline or argument",
                    call.head,
                ))
            }
            (Some(v), None) => v.as_str().map(|s| s.to_string()).map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Pipeline input must be a tensor ID (string)", call.head)
            })?,
            (None, Some(a)) => a.as_str().map(|s| s.to_string()).map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Argument must be a tensor ID (string)", call.head)
            })?,
        };

        // -------- fetch tensor from registry -------------------------------
        let mut reg = TENSOR_REGISTRY.lock().unwrap();
        let tensor = reg
            .get(&tensor_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor ID", call.head)
            })?
            .shallow_clone();

        // -------- perform negation -----------------------------------------
        let result_tensor = -tensor; // std::ops::Neg is implemented for tch::Tensor

        // -------- store & return -------------------------------------------
        let new_id = Uuid::new_v4().to_string();
        reg.insert(new_id.clone(), result_tensor);
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}

struct CommandShape;

impl PluginCommand for CommandShape {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch shape"
    }

    fn description(&self) -> &str {
        "Get the shape (dimensions) of a tensor as a list (similar to tensor.shape in PyTorch)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch shape")
            .input_output_types(vec![
                (Type::String, Type::List(Box::new(Type::Int))),
                (Type::Nothing, Type::List(Box::new(Type::Int))),
            ])
            .optional(
                "tensor_id",
                SyntaxShape::String,
                "ID of the tensor to get the shape of (if not using pipeline input)",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Get the shape of a tensor using pipeline input",
                example: "let t1 = (torch full [2, 3] 1); $t1 | torch shape",
                result: None,
            },
            Example {
                description: "Get the shape of a tensor using argument",
                example: "let t1 = (torch full [2, 3] 1); torch shape $t1",
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
        // Check for pipeline input
        let pipeline_input = match input {
            PipelineData::Empty => None,
            PipelineData::Value(val, _) => Some(val),
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or Value inputs are supported", call.head));
            }
        };

        // Check for positional argument input
        let arg_input = call.nth(0);

        // Validate that exactly one data source is provided
        let tensor_id: String = match (pipeline_input, arg_input) {
            (None, None) => {
                return Err(LabeledError::new("Missing input").with_label(
                    "Tensor ID must be provided via pipeline or as an argument",
                    call.head,
                ));
            }
            (Some(_), Some(_)) => {
                return Err(LabeledError::new("Conflicting input").with_label(
                    "Tensor ID cannot be provided both via pipeline and as an argument",
                    call.head,
                ));
            }
            (Some(input_val), None) => input_val.as_str().map(|s| s.to_string()).map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Pipeline input must be a tensor ID (string)", call.head)
            })?,
            (None, Some(arg_val)) => arg_val.as_str().map(|s| s.to_string()).map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Argument must be a tensor ID (string)", call.head)
            })?,
        };

        // Look up tensor in registry
        let registry = TENSOR_REGISTRY.lock().unwrap();
        let tensor = registry.get(&tensor_id).ok_or_else(|| {
            LabeledError::new("Tensor not found").with_label("Invalid tensor ID", call.head)
        })?;

        // Get the shape (dimensions) of the tensor
        let shape = tensor.size();
        let shape_values: Vec<Value> = shape
            .into_iter()
            .map(|dim| Value::int(dim, call.head))
            .collect();
        let shape_list = Value::list(shape_values, call.head);

        // Return the shape as a list directly (not stored in registry)
        Ok(PipelineData::Value(shape_list, None))
    }
}


struct CommandSub;

impl PluginCommand for CommandSub {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch sub"
    }

    fn description(&self) -> &str {
        "Compute the element-wise difference of two tensors with broadcasting (similar to torch.sub or - operator)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch sub")
            .input_output_types(vec![(Type::String, Type::String), (Type::Nothing, Type::String)])
            .optional("tensor1_id", SyntaxShape::String, "ID of the first tensor (if not using pipeline input)")
            .optional("tensor2_id", SyntaxShape::String, "ID of the second tensor")
            .named(
                "alpha",
                SyntaxShape::String,
                "ID of a tensor to use as a multiplier for the second tensor before subtraction (default: tensor with value 1.0)",
                None,
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Subtract two tensors using pipeline and argument",
                example: "let t1 = (torch full 5 2 3); let t2 = (torch full 2 2 3); $t1 | torch sub $t2 | torch value",
                result: None,
            },
            Example {
                description: "Subtract two tensors using arguments only",
                example: "let t1 = (torch full 5 2 3); let t2 = (torch full 2 2 3); torch sub $t1 $t2 | torch value",
                result: None,
            },
            Example {
                description: "Subtract two tensors with alpha scaling tensor on the second tensor",
                example: "let t1 = (torch full 5 2 3); let t2 = (torch full 2 2 3); let alpha = (torch full 0.5 1); $t1 | torch sub $t2 --alpha $alpha | torch value",
                result: None,
            }
        ]
    }

    fn run(
        &self,
        _plugin: &NutorchPlugin,
        _engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        // Check for pipeline input
        let pipeline_input = match input {
            PipelineData::Empty => None,
            PipelineData::Value(val, _) => Some(val),
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or Value inputs are supported", call.head));
            }
        };

        // Check for positional argument inputs
        let arg1 = call.nth(0);
        let arg2 = call.nth(1);

        // Validate the number of tensor inputs (exactly 2 tensors required for subtraction)
        let pipeline_count = if pipeline_input.is_some() { 1 } else { 0 };
        let arg_count = if arg1.is_some() { 1 } else { 0 } + if arg2.is_some() { 1 } else { 0 };
        let total_count = pipeline_count + arg_count;

        if total_count != 2 {
            return Err(LabeledError::new("Invalid input count").with_label(
                "Exactly two tensors must be provided via pipeline and/or arguments",
                call.head,
            ));
        }

        // Determine the two tensor IDs based on input sources
        let (tensor1_id, tensor2_id) = match (pipeline_input, arg1, arg2) {
            (Some(input_val), Some(arg_val), None) => {
                let input_id = input_val.as_str().map(|s| s.to_string()).map_err(|_| {
                    LabeledError::new("Invalid input")
                        .with_label("Unable to parse tensor ID from pipeline input", call.head)
                })?;
                let arg_id = arg_val.as_str().map(|s| s.to_string()).map_err(|_| {
                    LabeledError::new("Invalid input")
                        .with_label("Unable to parse tensor ID from argument", call.head)
                })?;
                (input_id, arg_id)
            }
            (None, Some(arg1_val), Some(arg2_val)) => {
                let arg1_id = arg1_val.as_str().map(|s| s.to_string()).map_err(|_| {
                    LabeledError::new("Invalid input")
                        .with_label("Unable to parse first tensor ID from argument", call.head)
                })?;
                let arg2_id = arg2_val.as_str().map(|s| s.to_string()).map_err(|_| {
                    LabeledError::new("Invalid input")
                        .with_label("Unable to parse second tensor ID from argument", call.head)
                })?;
                (arg1_id, arg2_id)
            }
            _ => {
                return Err(LabeledError::new("Invalid input configuration").with_label(
                    "Must provide exactly two tensors via pipeline and/or arguments",
                    call.head,
                ));
            }
        };

        // Look up tensors in registry
        let mut registry = TENSOR_REGISTRY.lock().unwrap();
        let tensor1 = registry
            .get(&tensor1_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found")
                    .with_label("Invalid first tensor ID", call.head)
            })?
            .shallow_clone();
        let tensor2 = registry
            .get(&tensor2_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found")
                    .with_label("Invalid second tensor ID", call.head)
            })?
            .shallow_clone();

        // Handle optional alpha argument (as a tensor ID)
        let result_tensor = match call.get_flag::<String>("alpha")? {
            Some(alpha_id) => {
                let alpha_tensor = registry
                    .get(&alpha_id)
                    .ok_or_else(|| {
                        LabeledError::new("Tensor not found")
                            .with_label("Invalid alpha tensor ID", call.head)
                    })?
                    .shallow_clone();
                tensor1 - (alpha_tensor * tensor2)
            }
            None => {
                // No alpha scaling, just subtract the two tensors
                tensor1 - tensor2
            }
        };

        // Store result in registry with new ID
        let new_id = Uuid::new_v4().to_string();
        registry.insert(new_id.clone(), result_tensor);
        // Return new ID wrapped in PipelineData
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}

struct CommandManualSeed;

impl PluginCommand for CommandManualSeed {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch manual_seed"
    }

    fn description(&self) -> &str {
        "Set the random seed for PyTorch operations to ensure reproducibility"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch manual_seed")
            .required(
                "seed",
                SyntaxShape::Int,
                "The seed value for the random number generator",
            )
            .input_output_types(vec![(Type::Nothing, Type::Nothing)])
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![Example {
            description: "Set the random seed to 42 for reproducibility",
            example: "torch manual_seed 42",
            result: None,
        }]
    }

    fn run(
        &self,
        _plugin: &NutorchPlugin,
        _engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        _input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        // Get the seed value from the first argument
        let seed: i64 = call.nth(0).unwrap().as_int()?;
        // Set the random seed using tch-rs
        tch::manual_seed(seed);
        // Return nothing (Type::Nothing) as the operation modifies global state
        Ok(PipelineData::Empty)
    }
}

struct CommandRandn;

impl PluginCommand for CommandRandn {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch randn"
    }

    fn description(&self) -> &str {
        "Generate a tensor filled with random numbers from a normal distribution (mean 0, std 1)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch randn")
            .rest(
                "dims",
                SyntaxShape::Int,
                "Dimensions of the tensor (e.g., 2 3 for a 2x3 tensor)",
            )
            .named(
                "device",
                SyntaxShape::String,
                "Device to create the tensor on ('cpu', 'cuda', 'mps', default: 'cpu')",
                None,
            )
            .named(
                "dtype",
                SyntaxShape::String,
                "Data type of the tensor ('float32', 'float64', 'int32', 'int64')",
                None,
            )
            .named(
                "requires_grad",
                SyntaxShape::Boolean,
                "Whether the tensor requires gradient tracking for autograd (default: false)",
                None,
            )
            .input_output_types(vec![(Type::Nothing, Type::String)])
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Generate a 2x3 tensor with random values from a normal distribution",
                example: "torch randn 2 3 | torch tovalue",
                result: None,
            },
            Example {
                description:
                    "Generate a 1D tensor of size 5 with a specific seed for reproducibility",
                example: "torch manual_seed 42; torch randn 5 | torch tovalue",
                result: None,
            },
        ]
    }

    fn run(
        &self,
        _plugin: &NutorchPlugin,
        _engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        _input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        // Get dimensions for the tensor shape
        let dims: Vec<i64> = call
            .rest(0)
            .map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Unable to parse dimensions", call.head)
            })?
            .into_iter()
            .map(|v: Value| v.as_int())
            .collect::<Result<Vec<i64>, _>>()?;
        if dims.is_empty() {
            return Err(LabeledError::new("Invalid input")
                .with_label("At least one dimension must be provided", call.head));
        }
        if dims.iter().any(|&d| d < 1) {
            return Err(LabeledError::new("Invalid input")
                .with_label("All dimensions must be positive", call.head));
        }

        // Handle optional device argument
        let device = get_device_from_call(call)?;

        // Handle optional dtype argument
        let kind = get_kind_from_call(call)?;

        // Create a random tensor using tch-rs
        let mut tensor = Tensor::randn(&dims, (kind, device));

        // Handle optional requires_grad argument
        tensor = add_grad_from_call(call, tensor)?;

        // Generate a unique ID for the tensor
        let id = Uuid::new_v4().to_string();
        // Store in registry
        TENSOR_REGISTRY.lock().unwrap().insert(id.clone(), tensor);
        // Return the ID as a string to Nushell, wrapped in PipelineData
        Ok(PipelineData::Value(Value::string(id, call.head), None))
    }
}

// torch sgd_step  -----------------------------------------------------------
// Performs *in-place* SGD update:  p -= lr * p.grad  (and zeroes the grad).
// Accept a list of tensor-IDs either from the pipeline **or** as the first
// positional argument (but not both).  Returns the *same* list of IDs.
//
// Example usage
//     [$w1 $w2] | torch sgd_step --lr 0.05
//     torch sgd_step [$w1 $w2] --lr 0.05
// ---------------------------------------------------------------------------

struct CommandSgdStep;

impl PluginCommand for CommandSgdStep {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch sgd_step"
    }

    fn description(&self) -> &str {
        "Vanilla stochastic-gradient-descene step: p -= lr * p.grad (in-place) \
         and p.grad is zeroed."
    }

    fn signature(&self) -> Signature {
        Signature::build("torch sgd_step")
            // list of ids in  -> list of ids out
            .input_output_types(vec![
                (
                    Type::List(Box::new(Type::String)),
                    Type::List(Box::new(Type::String)),
                ),
                (Type::Nothing, Type::List(Box::new(Type::String))),
            ])
            .optional(
                "params",
                SyntaxShape::List(Box::new(SyntaxShape::String)),
                "List of parameter tensor IDs (if not supplied by pipeline)",
            )
            .named(
                "lr",
                SyntaxShape::Float,
                "Learning-rate (default 0.01)",
                None,
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "SGD step with parameter list piped in",
                example: r#"
let w1 = (torch full [2,2] 1)      # pretend w1.grad is already populated
let w2 = (torch full [2,2] 2)
[$w1, $w2] | torch sgd_step --lr 0.1
"#
                .trim(),
                result: None,
            },
            Example {
                description: "SGD step with parameter list as argument",
                example: r#"
let w1 = (torch full [2,2] 1)
let w2 = (torch full [2,2] 2)
torch sgd_step [$w1, $w2] --lr 0.05
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
        //--------------------------------------------------------------
        // 1. Collect parameter IDs
        //--------------------------------------------------------------
        let list_from_pipe: Option<Value> = match &input {
            PipelineData::Value(v, _) => Some(v.clone()),
            PipelineData::Empty => None,
            _ => {
                return Err(LabeledError::new("Unsupported input").with_label(
                    "Only Value or Empty pipeline inputs are supported",
                    call.head,
                ))
            }
        };

        let list_from_arg: Option<Value> = call.nth(0);

        match (&list_from_pipe, &list_from_arg) {
            (None, None) => {
                return Err(LabeledError::new("Missing input")
                    .with_label("Provide parameter list via pipeline or argument", call.head));
            }
            (Some(_), Some(_)) => {
                return Err(LabeledError::new("Conflicting input").with_label(
                    "Provide parameter list via pipeline OR argument, not both",
                    call.head,
                ));
            }
            _ => {}
        };

        let list_val = list_from_pipe.or(list_from_arg).unwrap();

        let param_ids: Vec<String> = list_val
            .as_list()
            .map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Parameter list must be a list of tensor IDs", call.head)
            })?
            .iter()
            .map(|v| v.as_str().map(|s| s.to_string()))
            .collect::<Result<Vec<String>, _>>()?;

        if param_ids.is_empty() {
            return Err(
                LabeledError::new("Invalid input").with_label("Parameter list is empty", call.head)
            );
        }

        //--------------------------------------------------------------
        // 2. Learning-rate flag (default 0.01)
        //--------------------------------------------------------------
        let lr: f64 = call.get_flag("lr")?.unwrap_or(0.01);

        //--------------------------------------------------------------
        // 3. Fetch tensors and perform in-place update under no_grad
        //--------------------------------------------------------------
        let registry = TENSOR_REGISTRY.lock().unwrap();
        let mut tensors: Vec<Tensor> = Vec::new();
        for id in &param_ids {
            match registry.get(id) {
                Some(tensor) => tensors.push(tensor.shallow_clone()),
                None => {
                    return Err(LabeledError::new("Tensor not found")
                        .with_label(format!("Invalid tensor ID: {}", id), call.head))
                }
            }
        }
        // disable grad-mode for the duration of the update
        tch::no_grad(|| {
            for mut p in tensors {
                let g = p.grad();
                if g.defined() {
                    let before_ptr = p.data_ptr();
                    let r = p.f_sub_(&(g * lr)).unwrap();
                    assert_eq!(before_ptr, r.data_ptr()); // same memory
                }
            }
        });

        //--------------------------------------------------------------
        // 4. Return the (still the same) list of parameter IDs
        //--------------------------------------------------------------
        let out_vals: Vec<Value> = param_ids
            .iter()
            .map(|id| Value::string(id, call.head))
            .collect();
        Ok(PipelineData::Value(Value::list(out_vals, call.head), None))
    }
}

// torch zero_grad  ---------------------------------------------------------
// Clear .grad buffers for one or several tensors.
//
//   [$id1 $id2] | torch zero_grad
//   torch zero_grad [$id1 $id2]
//
// Returns the list of IDs unchanged.
// --------------------------------------------------------------------------
struct CommandZeroGrad;

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

pub enum Number {
    Int(i64),
    Float(f64),
}

// Devices command to list available devices
struct CommandDevices;

impl PluginCommand for CommandDevices {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch devices"
    }

    fn description(&self) -> &str {
        "List some available devices. Additional devices may be available, but unlisted here."
    }

    fn signature(&self) -> Signature {
        Signature::build("torch devices")
            .input_output_types(vec![(Type::Nothing, Type::List(Box::new(Type::String)))])
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![Example {
            description: "List available devices for tensor operations",
            example: "torch devices",
            result: None,
        }]
    }

    fn run(
        &self,
        _plugin: &NutorchPlugin,
        _engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        _input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        let span = call.head;
        let mut devices = vec![Value::string("cpu", span)];

        // Check for CUDA availability
        if tch::Cuda::is_available() {
            devices.push(Value::string("cuda", span));
        }

        // TODO: This doesn't actually work. But when tch-rs enables this feature, we can use it.
        // // Check for MPS (Metal Performance Shaders) availability on macOS
        // if tch::Mps::is_available() {
        //     devices.push(Value::string("mps", span));
        // }

        Ok(PipelineData::Value(Value::list(devices, span), None))
    }
}

// Repeat command to replicate a tensor into a multidimensional structure
struct CommandRepeat;

impl PluginCommand for CommandRepeat {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch repeat"
    }

    fn description(&self) -> &str {
        "Repeat a tensor along specified dimensions to create a multidimensional tensor"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch repeat")
            .rest(
                "sizes",
                SyntaxShape::Int,
                "Number of times to repeat along each dimension",
            )
            .input_output_types(vec![(Type::String, Type::String)])
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Repeat a tensor 3 times along the first dimension",
                example: "torch linspace 0.0 1.0 4 | torch repeat 3 | torch value",
                result: None,
            },
            Example {
                description: "Repeat a tensor 2 times along first dim and 2 times along second dim (creates new dim if needed)",
                example: "torch linspace 0.0 1.0 4 | torch repeat 2 2 | torch value",
                result: None,
            }
        ]
    }

    fn run(
        &self,
        _plugin: &NutorchPlugin,
        _engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        // Get tensor ID from input
        let input_value = input.into_value(call.head)?;
        let tensor_id = input_value.as_str()?;
        // Get repeat sizes (rest arguments)
        let sizes: Vec<i64> = call
            .rest(0)
            .map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Unable to parse repeat sizes", call.head)
            })?
            .into_iter()
            .map(|v: Value| v.as_int())
            .collect::<Result<Vec<i64>, _>>()?;
        if sizes.is_empty() {
            return Err(LabeledError::new("Invalid input")
                .with_label("At least one repeat size must be provided", call.head));
        }
        if sizes.iter().any(|&n| n < 1) {
            return Err(LabeledError::new("Invalid input")
                .with_label("All repeat sizes must be at least 1", call.head));
        }
        // Look up tensor in registry
        let mut registry = TENSOR_REGISTRY.lock().unwrap();
        let tensor = registry
            .get(tensor_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor ID", call.head)
            })?
            .shallow_clone();
        // Get tensor dimensions
        let dims = tensor.size();
        // Adjust tensor dimensions to match the length of sizes by unsqueezing if necessary
        let mut working_tensor = tensor;
        let target_dims = sizes.len();
        let current_dims = dims.len();
        if target_dims > current_dims {
            // Add leading singleton dimensions to match sizes length
            for _ in 0..(target_dims - current_dims) {
                working_tensor = working_tensor.unsqueeze(0);
            }
        }
        // Now repeat_dims can be directly set to sizes (or padded with 1s if sizes is shorter)
        let final_dims = working_tensor.size();
        let mut repeat_dims = vec![1; final_dims.len()];
        for (i, &size) in sizes.iter().enumerate() {
            repeat_dims[i] = size;
        }
        // Apply repeat operation
        let result_tensor = working_tensor.repeat(&repeat_dims);
        // Store result in registry with new ID
        let new_id = Uuid::new_v4().to_string();
        registry.insert(new_id.clone(), result_tensor);
        // Return new ID wrapped in PipelineData
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}

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
struct CommandRepeatInterleave;

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

// torch squeeze  -----------------------------------------------------------
// Remove a dimension of size 1 from a tensor (like tensor.squeeze(dim) in PyTorch).
// The tensor **must** be supplied through the pipeline; the single positional
// argument is the dimension to squeeze.
// -------------------------------------------------------------------------
struct CommandSqueeze;

impl PluginCommand for CommandSqueeze {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch squeeze"
    }

    fn description(&self) -> &str {
        "Remove a dimension of size 1 from a tensor (similar to tensor.squeeze(dim) in PyTorch). \
         The tensor ID is taken from the pipeline; the dimension is a required argument."
    }

    fn signature(&self) -> Signature {
        Signature::build("torch squeeze")
            .input_output_types(vec![(Type::String, Type::String)]) // tensor id in, tensor id out
            .required(
                "dim",
                SyntaxShape::Int,
                "Dimension to squeeze (must have size 1)",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![Example {
            description: "Squeeze dimension 0 of a [1,2,3] tensor",
            example: r#"let t = (torch full [1,2,3] 1); $t | torch squeeze 0 | torch shape"#,
            result: None,
        }]
    }

    fn run(
        &self,
        _plugin: &NutorchPlugin,
        _engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        // ------ tensor ID must come from the pipeline --------------------
        let PipelineData::Value(tensor_id_val, _) = input else {
            return Err(LabeledError::new("Unsupported input")
                .with_label("Only Value inputs are supported", call.head));
        };

        let tensor_id = tensor_id_val.as_str().map(|s| s.to_string()).map_err(|_| {
            LabeledError::new("Invalid input")
                .with_label("Pipeline input must be a tensor ID (string)", call.head)
        })?;

        // ------ dimension argument ---------------------------------------
        let dim_val = call.nth(0).ok_or_else(|| {
            LabeledError::new("Missing dimension")
                .with_label("A dimension argument is required", call.head)
        })?;

        let dim = dim_val.as_int().map_err(|_| {
            LabeledError::new("Invalid dimension")
                .with_label("Dimension must be an integer", call.head)
        })?;

        // ------ fetch tensor ---------------------------------------------
        let mut reg = TENSOR_REGISTRY.lock().unwrap();
        let tensor = reg
            .get(&tensor_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor ID", call.head)
            })?
            .shallow_clone();

        // ------ validate dimension ---------------------------------------
        let shape = tensor.size();
        let ndims = shape.len() as i64;
        if dim < 0 || dim >= ndims {
            return Err(LabeledError::new("Invalid dimension").with_label(
                format!("Dim {dim} out of bounds for tensor with {ndims} dims"),
                call.head,
            ));
        }
        if shape[dim as usize] != 1 {
            return Err(LabeledError::new("Cannot squeeze").with_label(
                format!("Dim {dim} has size {} (expected 1)", shape[dim as usize]),
                call.head,
            ));
        }

        // ------ perform squeeze ------------------------------------------
        let result_tensor = tensor.squeeze_dim(dim);

        // ------ store & return -------------------------------------------
        let new_id = Uuid::new_v4().to_string();
        reg.insert(new_id.clone(), result_tensor);
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}

// torch unsqueeze -----------------------------------------------------------
// Insert a size-1 dimension at the given index (like tensor.unsqueeze(dim))
// Tensor ID must be supplied via the pipeline; one positional argument = dim.
// ---------------------------------------------------------------------------
struct CommandUnsqueeze;

impl PluginCommand for CommandUnsqueeze {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch unsqueeze"
    }

    fn description(&self) -> &str {
        "Insert a dimension of size 1 into a tensor (similar to tensor.unsqueeze(dim) in PyTorch)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch unsqueeze")
            // tensor id comes from pipeline, returns new tensor id
            .input_output_types(vec![(Type::String, Type::String)])
            .required(
                "dim",
                SyntaxShape::Int,
                "Dimension index at which to insert size-1 dimension",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Unsqueeze dim 0 of a [2,3] tensor -> shape [1,2,3]",
                example: r#"let t = (torch full [2,3] 1); $t | torch unsqueeze 0 | torch shape"#,
                result: None,
            },
            Example {
                description: "Unsqueeze dim 2 of a [2,3] tensor -> shape [2,3,1]",
                example: r#"let t = (torch full [2,3] 1); $t | torch unsqueeze 2 | torch shape"#,
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
        // ---- tensor ID must come from pipeline --------------------------
        let PipelineData::Value(tensor_id_val, _) = input else {
            return Err(LabeledError::new("Unsupported input")
                .with_label("Tensor ID must be supplied via the pipeline", call.head));
        };

        let tensor_id = tensor_id_val.as_str().map(|s| s.to_string()).map_err(|_| {
            LabeledError::new("Invalid input")
                .with_label("Pipeline input must be a tensor ID (string)", call.head)
        })?;

        // ---- parse dimension argument -----------------------------------
        let dim_val = call.nth(0).ok_or_else(|| {
            LabeledError::new("Missing dimension")
                .with_label("A dimension argument is required", call.head)
        })?;

        let dim = dim_val.as_int().map_err(|_| {
            LabeledError::new("Invalid dimension")
                .with_label("Dimension must be an integer", call.head)
        })?;

        // ---- fetch tensor ------------------------------------------------
        let mut reg = TENSOR_REGISTRY.lock().unwrap();
        let tensor = reg
            .get(&tensor_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor ID", call.head)
            })?
            .shallow_clone();

        // ---- validate dim ------------------------------------------------
        let ndims = tensor.size().len() as i64;
        // In PyTorch unsqueeze allows dim == ndims (insert at end)
        if dim < 0 || dim > ndims {
            return Err(LabeledError::new("Invalid dimension").with_label(
                format!("Dim {dim} out of bounds for tensor with {ndims} dims"),
                call.head,
            ));
        }

        // ---- perform unsqueeze ------------------------------------------
        let result_tensor = tensor.unsqueeze(dim);

        // ---- store & return ---------------------------------------------
        let new_id = Uuid::new_v4().to_string();
        reg.insert(new_id.clone(), result_tensor);
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}


// Sin command to apply sine to a tensor
struct CommandSin;

impl PluginCommand for CommandSin {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch sin"
    }

    fn description(&self) -> &str {
        "Apply sin function element-wise to a tensor"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch sin").category(Category::Custom("torch".into()))
    }

    fn run(
        &self,
        _plugin: &NutorchPlugin,
        _engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        // Get tensor ID from input
        let input_value = input.into_value(call.head)?;
        let tensor_id = input_value.as_str()?;
        // Look up tensor in registry
        let mut registry = TENSOR_REGISTRY.lock().unwrap();
        let tensor = registry
            .get(tensor_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor ID", call.head)
            })?
            .shallow_clone();
        // Apply sine operation
        let result_tensor = tensor.sin();
        // Store result in registry with new ID
        let new_id = Uuid::new_v4().to_string();
        registry.insert(new_id.clone(), result_tensor);
        // Return new ID wrapped in PipelineData
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}

// torch stack  --------------------------------------------------------------
// Stack a list of tensors along a new dimension (like torch.stack in PyTorch)
//
//   [$t1 $t2] | torch stack --dim 0
//   torch stack [$t1 $t2] --dim 1
//
// All tensors must have identical shapes.
// ---------------------------------------------------------------------------
struct CommandStack;

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


// torch reshape  -----------------------------------------------------------
// Reshape a tensor to a new shape.
//   $tensor | torch reshape [dim0 dim1 ...]
//
// • The source tensor **must** be supplied through the pipeline.
// • The first positional argument is a Nushell list of integers that becomes
//   the new shape.  `-1` is allowed once to let PyTorch infer that dimension.
// --------------------------------------------------------------------------
struct CommandReshape;

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


// Command to convert tensor to Nushell data structure (value)
struct CommandValue;

impl PluginCommand for CommandValue {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch value"
    }

    fn description(&self) -> &str {
        "Convert a tensor to a Nushell Value (nested list structure)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch value")
            .input_output_types(vec![(Type::String, Type::Any)])
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Convert a 1D tensor to a Nushell Value (list)",
                example: "torch linspace 0.0 1.0 4 | torch value",
                result: None,
            },
            Example {
                description: "Convert a 2D or higher dimensional tensor to nested Values",
                example: "torch linspace 0.0 1.0 4 | torch repeat 2 2 | torch value",
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
        // Get tensor ID from input
        let input_value = input.into_value(call.head)?;
        let tensor_id = input_value.as_str()?;
        // Look up tensor in registry
        let registry = TENSOR_REGISTRY.lock().unwrap();
        let tensor = registry.get(tensor_id).ok_or_else(|| {
            LabeledError::new("Tensor not found").with_label("Invalid tensor ID", call.head)
        })?;
        // Ensure tensor is on CPU before accessing data
        let tensor = tensor.to_device(Device::Cpu);
        // Convert tensor to Nushell Value with support for arbitrary dimensions
        let span = call.head;
        let value = tensor_to_value(&tensor, span)?;
        Ok(PipelineData::Value(value, None))
    }
}

// Command to convert Nushell data structure to tensor (tensor)
struct CommandTensor;

impl PluginCommand for CommandTensor {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch tensor"
    }

    fn description(&self) -> &str {
        "Convert a Nushell Value (nested list structure) to a tensor"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch tensor")
            .input_output_types(vec![(Type::Any, Type::String)])
            .optional(
                "data",
                SyntaxShape::Any,
                "Data to convert to a tensor (list or nested list)",
            )
            .named(
                "device",
                SyntaxShape::String,
                "Device to create the tensor on (default: 'cpu')",
                None,
            )
            .named(
                "dtype",
                SyntaxShape::String,
                "Data type of the tensor (default: 'float32')",
                None,
            )
            .named(
                "requires_grad",
                SyntaxShape::Boolean,
                "Whether the tensor requires gradient tracking for autograd (default: false)",
                None,
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Convert a 1D list to a tensor via pipeline",
                example: "[0.0, 1.0, 2.0, 3.0] | torch tensor",
                result: None,
            },
            Example {
                description: "Convert a 2D nested list to a tensor with specific device and dtype via pipeline",
                example: "[[0.0, 1.0], [2.0, 3.0]] | torch tensor --device cpu --dtype float64",
                result: None,
            },
            Example {
                description: "Convert a 1D list to a tensor via argument",
                example: "torch tensor [0.0, 1.0, 2.0, 3.0]",
                result: None,
            },
            Example {
                description: "Convert a 2D nested list to a tensor with specific device via argument",
                example: "torch tensor [[0.0, 1.0], [2.0, 3.0]] --device cpu",
                result: None,
            }
        ]
    }

    fn run(
        &self,
        _plugin: &NutorchPlugin,
        _engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        // Check for pipeline input
        let pipeline_input = match input {
            PipelineData::Empty => None,
            PipelineData::Value(val, _) => Some(val),
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Value or Empty input is supported", call.head));
            }
        };

        // Check for positional argument input
        let arg_input = call.nth(0);

        // Validate that exactly one data source is provided
        match (&pipeline_input, &arg_input) {
            (None, None) => {
                return Err(LabeledError::new("Missing input").with_label(
                    "Data must be provided via pipeline or as an argument",
                    call.head,
                ));
            }
            (Some(_), Some(_)) => {
                return Err(LabeledError::new("Conflicting input").with_label(
                    "Data cannot be provided both via pipeline and as an argument",
                    call.head,
                ));
            }
            (Some(input_val), None) => input_val,
            (None, Some(arg_val)) => arg_val,
        };

        let input_value = match (pipeline_input, arg_input) {
            (Some(input_val), None) => input_val,
            (None, Some(arg_val)) => arg_val,
            _ => unreachable!("Validation above ensures one source is provided"),
        };

        // Handle optional device argument
        let device = get_device_from_call(call)?;

        // Handle optional dtype argument
        let kind = get_kind_from_call(call)?;

        // Convert Nushell Value to tensor
        let mut tensor = value_to_tensor(&input_value, kind, device, call.head)?;

        // Handle optional requires_grad argument
        tensor = add_grad_from_call(call, tensor)?;

        let id = Uuid::new_v4().to_string();
        // Store in registry
        TENSOR_REGISTRY.lock().unwrap().insert(id.clone(), tensor);
        // Return the ID as a string to Nushell, wrapped in PipelineData
        Ok(PipelineData::Value(Value::string(id, call.head), None))
    }
}

// torch t  -----------------------------------------------------------------
// 2-D matrix transpose (like Tensor.t() in PyTorch / tch-rs).
//
//     $mat | torch t
//     torch t $mat
// --------------------------------------------------------------------------
struct CommandT;

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

pub fn add_grad_from_call(
    call: &nu_plugin::EvaluatedCall,
    mut tensor: Tensor,
) -> Result<Tensor, LabeledError> {
    let requires_grad = call.get_flag::<bool>("requires_grad")?.unwrap_or(false);
    if requires_grad {
        tensor = tensor.set_requires_grad(true);
    }
    Ok(tensor)
}

pub fn get_device_from_call(call: &nu_plugin::EvaluatedCall) -> Result<Device, LabeledError> {
    match call.get_flag::<String>("device")? {
        Some(device_str) => match device_str.as_str() {
            "cpu" => Ok(Device::Cpu),
            "cuda" => Ok(Device::Cuda(0)),
            "mps" => Ok(Device::Mps),
            _ if device_str.starts_with("cuda:") => {
                // Handle specific CUDA device like "cuda:0", "cuda:1", etc.
                if let Some(num) = device_str[5..].parse::<usize>().ok() {
                    Ok(Device::Cuda(num))
                } else {
                    Err(LabeledError::new("Invalid CUDA device")
                        .with_label("Invalid CUDA device", call.head))
                }
            }
            _ => Err(LabeledError::new("Invalid device").with_label("Invalid device", call.head)),
        },
        None => Ok(Device::Cpu), // Default to CPU if not specified
    }
}

pub fn get_kind_from_call(call: &nu_plugin::EvaluatedCall) -> Result<Kind, LabeledError> {
    match call.get_flag::<String>("dtype")? {
        Some(dtype_str) => match dtype_str.as_str() {
            "float32" | "float" => Ok(Kind::Float),
            "float64" | "double" => Ok(Kind::Double),
            "int32" | "int" => Ok(Kind::Int),
            "int64" | "long" => Ok(Kind::Int64),
            _ => Err(LabeledError::new("Invalid dtype").with_label(
                "Data type must be 'float32', 'float64', 'int32', or 'int64'",
                call.head,
            )),
        },
        None => Ok(Kind::Float), // Default to float32 if not specified
    }
}

// Helper function to recursively convert a tensor to a nested Nushell Value
pub fn tensor_to_value(tensor: &Tensor, span: Span) -> Result<Value, LabeledError> {
    let dims = tensor.size();
    let kind = tensor.kind();

    if dims.is_empty() {
        // Scalar tensor (0D)
        let value = match kind {
            Kind::Int | Kind::Int8 | Kind::Int16 | Kind::Int64 | Kind::Uint8 => {
                let int_val = tensor.int64_value(&[]);
                Value::int(int_val, span)
            }
            Kind::Float | Kind::Double | Kind::Half => {
                let float_val = tensor.double_value(&[]);
                Value::float(float_val, span)
            }
            _ => {
                return Err(LabeledError::new("Unsupported tensor type")
                    .with_label(format!("Cannot convert tensor of type {kind:?}"), span))
            }
        };
        return Ok(value);
    }

    if dims.len() == 1 {
        // 1D tensor to list
        let size = dims[0] as usize;
        let list: Vec<Value> = match kind {
            Kind::Int | Kind::Int8 | Kind::Int16 | Kind::Int64 | Kind::Uint8 => {
                let mut data: Vec<i64> = Vec::with_capacity(size);
                for i in 0..size as i64 {
                    data.push(tensor.get(i).int64_value(&[]));
                }
                data.into_iter().map(|v| Value::int(v, span)).collect()
            }
            Kind::Float | Kind::Double | Kind::Half => {
                let mut data: Vec<f64> = Vec::with_capacity(size);
                for i in 0..size as i64 {
                    data.push(tensor.get(i).double_value(&[]));
                }
                data.into_iter().map(|v| Value::float(v, span)).collect()
            }
            _ => {
                return Err(LabeledError::new("Unsupported tensor type")
                    .with_label(format!("Cannot convert tensor of type {kind:?}"), span))
            }
        };
        return Ok(Value::list(list, span));
    }

    // For higher dimensions, create nested lists recursively
    let first_dim_size = dims[0] as usize;
    let mut nested_data: Vec<Value> = Vec::with_capacity(first_dim_size);
    for i in 0..first_dim_size as i64 {
        // Get a subtensor by indexing along the first dimension
        let subtensor = tensor.get(i);
        // Recursively convert the subtensor to a Value
        let nested_value = tensor_to_value(&subtensor, span)?;
        nested_data.push(nested_value);
    }
    Ok(Value::list(nested_data, span))
}

// Helper function to recursively convert a Nushell Value to a tensor
pub fn value_to_tensor(
    value: &Value,
    kind: Kind,
    device: Device,
    span: Span,
) -> Result<Tensor, LabeledError> {
    match value {
        Value::List { vals, .. } => {
            if vals.is_empty() {
                return Err(
                    LabeledError::new("Invalid input").with_label("List cannot be empty", span)
                );
            }
            // Check if the first element is a list (nested structure)
            if let Some(first_val) = vals.first() {
                if matches!(first_val, Value::List { .. }) {
                    // Nested list: recursively convert each sublist to a tensor and stack them
                    let subtensors: Result<Vec<Tensor>, LabeledError> = vals
                        .iter()
                        .map(|v| value_to_tensor(v, kind, device, span))
                        .collect();
                    let subtensors = subtensors?;
                    // Stack tensors along a new dimension (dim 0)
                    return Ok(Tensor::stack(&subtensors, 0)
                        .to_kind(kind)
                        .to_device(device));
                }
            }
            // Flat list: convert to 1D tensor
            // Check if all elements are integers to decide initial tensor type
            let all_ints = vals.iter().all(|v| matches!(v, Value::Int { .. }));
            if all_ints {
                let data: Result<Vec<i64>, LabeledError> = vals
                    .iter()
                    .map(|v| {
                        v.as_int().map_err(|_| {
                            LabeledError::new("Invalid input")
                                .with_label("Expected integer value", span)
                        })
                    })
                    .collect();
                let data = data?;
                // Create 1D tensor from integer data
                Ok(Tensor::from_slice(&data).to_kind(kind).to_device(device))
            } else {
                let data: Result<Vec<f64>, LabeledError> = vals
                    .iter()
                    .map(|v| {
                        v.as_float().map_err(|_| {
                            LabeledError::new("Invalid input")
                                .with_label("Expected numeric value", span)
                        })
                    })
                    .collect();
                let data = data?;
                // Create 1D tensor from float data
                Ok(Tensor::from_slice(&data).to_kind(kind).to_device(device))
            }
        }
        Value::Float { val, .. } => {
            // Single float value (scalar)
            Ok(Tensor::from(*val).to_kind(kind).to_device(device))
        }
        Value::Int { val, .. } => {
            // Single int value (scalar)
            Ok(Tensor::from(*val).to_kind(kind).to_device(device))
        }
        _ => Err(LabeledError::new("Invalid input").with_label(
            "Input must be a number or a list (nested for higher dimensions)",
            span,
        )),
    }
}
