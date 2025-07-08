use lazy_static::lazy_static;
use nu_plugin::{serve_plugin, Plugin, PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, Span, SyntaxShape, Type, Value,
};
use std::collections::HashMap;
use std::sync::Mutex;
use tch::{Device, Kind, Tensor};
use uuid::Uuid;

// Global registry to store tensors by ID (thread-safe)
lazy_static! {
    static ref TENSOR_REGISTRY: Mutex<HashMap<String, Tensor>> = Mutex::new(HashMap::new());
}

struct NutorchPlugin;

impl Plugin for NutorchPlugin {
    fn commands(&self) -> Vec<Box<dyn PluginCommand<Plugin = Self>>> {
        vec![
            // Top-level Torch command
            Box::new(CommandTorch),
            // Configuration and other global commands
            Box::new(CommandManualSeed),
            Box::new(CommandDevices),
            // Tensor operations
            Box::new(CommandAdd),
            Box::new(CommandArange),
            Box::new(CommandBackward),
            Box::new(CommandCat),
            Box::new(CommandDetach),
            Box::new(CommandDiv),
            Box::new(CommandExp),
            Box::new(CommandStack),
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
            Box::new(CommandNeg),
            Box::new(CommandRandn),
            Box::new(CommandRepeat),
            Box::new(CommandSgdStep),
            Box::new(CommandShape),
            Box::new(CommandSin),
            Box::new(CommandSqueeze),
            Box::new(CommandSub),
            Box::new(CommandT),
            Box::new(CommandTensor),
            Box::new(CommandUnsqueeze),
            Box::new(CommandValue),
            Box::new(CommandZeroGrad),
        ]
    }

    fn version(&self) -> std::string::String {
        "0.0.1".to_string()
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

struct CommandMul;

impl PluginCommand for CommandMul {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch mul"
    }

    fn description(&self) -> &str {
        "Compute the element-wise product of two tensors with broadcasting (similar to torch.mul or * operator)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch mul")
            .input_output_types(vec![(Type::String, Type::String), (Type::Nothing, Type::String)])
            .optional("tensor1_id", SyntaxShape::String, "ID of the first tensor (if not using pipeline input)")
            .optional("tensor2_id", SyntaxShape::String, "ID of the second tensor")
            .named(
                "alpha",
                SyntaxShape::String,
                "ID of a tensor to use as a multiplier for the second tensor before multiplication (default: tensor with value 1.0)",
                None,
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Multiply two tensors using pipeline and argument",
                example: "let t1 = (torch full 2 2 3); let t2 = (torch full 3 2 3); $t1 | torch mul $t2 | torch value",
                result: None,
            },
            Example {
                description: "Multiply two tensors using arguments only",
                example: "let t1 = (torch full 2 2 3); let t2 = (torch full 3 2 3); torch mul $t1 $t2 | torch value",
                result: None,
            },
            Example {
                description: "Multiply two tensors with alpha scaling tensor on the second tensor",
                example: "let t1 = (torch full 2 2 3); let t2 = (torch full 3 2 3); let alpha = (torch full 0.5 1); $t1 | torch mul $t2 --alpha $alpha | torch value",
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

        // Validate the number of tensor inputs (exactly 2 tensors required for multiplication)
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
                tensor1 * (alpha_tensor * tensor2)
            }
            None => {
                // No alpha scaling, just multiply the two tensors
                tensor1 * tensor2
            }
        };

        // Store result in registry with new ID
        let new_id = Uuid::new_v4().to_string();
        registry.insert(new_id.clone(), result_tensor);
        // Return new ID wrapped in PipelineData
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}

struct CommandDiv;

impl PluginCommand for CommandDiv {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch div"
    }

    fn description(&self) -> &str {
        "Compute the element-wise quotient of two tensors with broadcasting (similar to torch.div or / operator)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch div")
            .input_output_types(vec![(Type::String, Type::String), (Type::Nothing, Type::String)])
            .optional("tensor1_id", SyntaxShape::String, "ID of the first tensor (if not using pipeline input)")
            .optional("tensor2_id", SyntaxShape::String, "ID of the second tensor")
            .named(
                "alpha",
                SyntaxShape::String,
                "ID of a tensor to use as a multiplier for the second tensor before division (default: tensor with value 1.0)",
                None,
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Divide two tensors using pipeline and argument",
                example: "let t1 = (torch full 6 2 3); let t2 = (torch full 2 2 3); $t1 | torch div $t2 | torch value",
                result: None,
            },
            Example {
                description: "Divide two tensors using arguments only",
                example: "let t1 = (torch full 6 2 3); let t2 = (torch full 2 2 3); torch div $t1 $t2 | torch value",
                result: None,
            },
            Example {
                description: "Divide two tensors with alpha scaling tensor on the second tensor",
                example: "let t1 = (torch full 6 2 3); let t2 = (torch full 2 2 3); let alpha = (torch full 0.5 1); $t1 | torch div $t2 --alpha $alpha | torch value",
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

        // Validate the number of tensor inputs (exactly 2 tensors required for division)
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
                tensor1 / (alpha_tensor * tensor2)
            }
            None => {
                // No alpha scaling, just divide the two tensors
                tensor1 / tensor2
            }
        };

        // Store result in registry with new ID
        let new_id = Uuid::new_v4().to_string();
        registry.insert(new_id.clone(), result_tensor);
        // Return new ID wrapped in PipelineData
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}

struct CommandAdd;

impl PluginCommand for CommandAdd {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch add"
    }

    fn description(&self) -> &str {
        "Compute the element-wise sum of two tensors with broadcasting (similar to torch.add or + operator)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch add")
            .input_output_types(vec![(Type::String, Type::String), (Type::Nothing, Type::String)])
            .optional("tensor1_id", SyntaxShape::String, "ID of the first tensor (if not using pipeline input)")
            .optional("tensor2_id", SyntaxShape::String, "ID of the second tensor")
            .named(
                "alpha",
                SyntaxShape::String,
                "ID of a tensor to use as a multiplier for the second tensor before addition (default: tensor with value 1.0)",
                None,
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Add two tensors using pipeline and argument",
                example: "let t1 = (torch full 1 2 3); let t2 = (torch full 2 2 3); $t1 | torch add $t2 | torch value",
                result: None,
            },
            Example {
                description: "Add two tensors using arguments only",
                example: "let t1 = (torch full 1 2 3); let t2 = (torch full 2 2 3); torch add $t1 $t2 | torch value",
                result: None,
            },
            Example {
                description: "Add two tensors with alpha scaling tensor on the second tensor",
                example: "let t1 = (torch full 1 2 3); let t2 = (torch full 2 2 3); let alpha = (torch full 0.5 1); $t1 | torch add $t2 --alpha $alpha | torch value",
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

        // Validate the number of tensor inputs (exactly 2 tensors required for addition)
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
                tensor1 + (alpha_tensor * tensor2)
            }
            None => {
                // Default to a scalar tensor with value 1.0
                tensor1 + tensor2
            }
        };

        // Store result in registry with new ID
        let new_id = Uuid::new_v4().to_string();
        registry.insert(new_id.clone(), result_tensor);
        // Return new ID wrapped in PipelineData
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
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

struct CommandCat;

impl PluginCommand for CommandCat {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch cat"
    }

    fn description(&self) -> &str {
        "Concatenate a sequence of tensors along a specified dimension (similar to torch.cat)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch cat")
            .input_output_types(vec![
                (Type::List(Box::new(Type::String)), Type::String),
                (Type::Nothing, Type::String),
            ])
            .optional(
                "tensor_ids",
                SyntaxShape::List(Box::new(SyntaxShape::String)),
                "List of tensor IDs to concatenate (if not using pipeline input)",
            )
            .named(
                "dim",
                SyntaxShape::Int,
                "Dimension along which to concatenate (default: 0)",
                None,
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Concatenate two 2x3 tensors along dimension 0 using pipeline input",
                example: "let t1 = (torch full 1 2 3); let t2 = (torch full 2 2 3); [$t1, $t2] | torch cat --dim 0 | torch value",
                result: None,
            },
            Example {
                description: "Concatenate three 2x3 tensors along dimension 1 using argument",
                example: "let t1 = (torch full 1 2 3); let t2 = (torch full 2 2 3); let t3 = (torch full 3 2 3); torch cat [$t1, $t2, $t3] --dim 1 | torch value",
                result: None,
            }
        ]
    }

    #[allow(clippy::too_many_lines)]
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
        let tensor_ids: Vec<String> = match (pipeline_input, arg_input) {
            (None, None) => {
                return Err(LabeledError::new("Missing input").with_label(
                    "Tensor IDs must be provided via pipeline or as an argument",
                    call.head,
                ));
            }
            (Some(_), Some(_)) => {
                return Err(LabeledError::new("Conflicting input").with_label(
                    "Tensor IDs cannot be provided both via pipeline and as an argument",
                    call.head,
                ));
            }
            (Some(input_val), None) => input_val
                .as_list()
                .map_err(|_| {
                    LabeledError::new("Invalid input")
                        .with_label("Pipeline input must be a list of tensor IDs", call.head)
                })?
                .iter()
                .map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Result<Vec<String>, _>>()?,
            (None, Some(arg_val)) => arg_val
                .as_list()
                .map_err(|_| {
                    LabeledError::new("Invalid input")
                        .with_label("Argument must be a list of tensor IDs", call.head)
                })?
                .iter()
                .map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Result<Vec<String>, _>>()?,
        };

        // Validate that at least two tensors are provided
        if tensor_ids.len() < 2 {
            return Err(LabeledError::new("Invalid input").with_label(
                "At least two tensor IDs must be provided for concatenation",
                call.head,
            ));
        }

        // Get the dimension to concatenate along (default to 0)
        let dim: i64 = match call.get_flag::<i64>("dim")? {
            Some(d) => {
                if d < 0 {
                    return Err(LabeledError::new("Invalid input")
                        .with_label("Dimension must be non-negative", call.head));
                }
                d
            }
            None => 0,
        };

        // Look up tensors in registry
        let mut registry = TENSOR_REGISTRY.lock().unwrap();
        let mut tensors: Vec<Tensor> = Vec::new();
        for id in &tensor_ids {
            match registry.get(id) {
                Some(tensor) => tensors.push(tensor.shallow_clone()),
                None => {
                    return Err(LabeledError::new("Tensor not found")
                        .with_label(format!("Invalid tensor ID: {}", id), call.head))
                }
            }
        }

        // Check if tensors have compatible shapes for concatenation
        if tensors.is_empty() {
            return Err(LabeledError::new("Invalid input")
                .with_label("No tensors provided for concatenation", call.head));
        }
        let first_shape = tensors[0].size();
        if first_shape.len() as i64 <= dim {
            return Err(LabeledError::new("Invalid dimension").with_label(
                format!(
                    "Dimension {} out of bounds for tensor with {} dimensions",
                    dim,
                    first_shape.len()
                ),
                call.head,
            ));
        }
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            let shape = tensor.size();
            if shape.len() != first_shape.len() {
                return Err(LabeledError::new("Shape mismatch").with_label(
                    format!(
                        "Tensor {} has different number of dimensions ({} vs {})",
                        i,
                        shape.len(),
                        first_shape.len()
                    ),
                    call.head,
                ));
            }
            for (d, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if d as i64 != dim && s1 != s2 {
                    return Err(LabeledError::new("Shape mismatch").with_label(
                        format!(
                            "Tensor {} has mismatched size in dimension {} ({} vs {})",
                            i, d, s2, s1
                        ),
                        call.head,
                    ));
                }
            }
        }

        // Create references to tensors for cat
        let tensor_refs: Vec<&Tensor> = tensors.iter().collect();

        // Perform concatenation using tch-rs
        let result_tensor = Tensor::cat(&tensor_refs, dim);

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

// torch arange  -------------------------------------------------------------
// Create a 1-D tensor with evenly spaced values.
//   torch arange end
//   torch arange start end
//   torch arange start end step
//
// Optional flags handled by helpers already available:
//   --dtype <kind>   --device <cpu|cuda:N>   --requires_grad
// ---------------------------------------------------------------------------
struct CommandArange;

impl PluginCommand for CommandArange {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch arange"
    }

    fn description(&self) -> &str {
        "Return a 1-D tensor with values in [start, end) and the given step \
         (like torch.arange in PyTorch)."
    }

    fn signature(&self) -> Signature {
        Signature::build("torch arange")
            .required(
                "end_or_start",
                SyntaxShape::Number,
                "If only one number is given, it is `end`; if two or three are given, it is `start`",
            )
            .optional(
                "end",
                SyntaxShape::Number,
                "End value (exclusive) if start supplied",
            )
            .optional(
                "step",
                SyntaxShape::Number,
                "Step (default 1) if start and end supplied",
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
            // we already declared global flags for dtype/device/grad earlier,
            // so they are parsed by the helper functions; no need to repeat.
            .input_output_types(vec![(Type::Nothing, Type::String)])
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "arange(5)  -> 0 1 2 3 4",
                example: "torch arange 5 | torch value",
                result: None,
            },
            Example {
                description: "arange(2, 7)  -> 2 3 4 5 6",
                example: "torch arange 2 7 | torch value",
                result: None,
            },
            Example {
                description: "arange(1, 5, 0.5)  -> 1 1.5 … 4.5 (float)",
                example: "torch arange 1 5 0.5 --dtype float | torch value",
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
        //------------------------------------------------------------------
        // 1. parse positional numbers
        //------------------------------------------------------------------
        let argc = call.positional.iter().count();
        if !(1..=3).contains(&argc) {
            return Err(LabeledError::new("Invalid arange usage")
                .with_label("Require 1, 2 or 3 numeric arguments", call.head));
        }

        // helper to convert a Value to f64
        let to_f64 = |v: &Value| -> Result<f64, LabeledError> {
            if let Ok(i) = v.as_int() {
                Ok(i as f64)
            } else if let Ok(f) = v.as_float() {
                Ok(f)
            } else {
                Err(LabeledError::new("Expected number")
                    .with_label("Argument must be int or float", v.span()))
            }
        };

        let arg0 = call.nth(0).unwrap(); // safe: argc>=1
        let a0 = to_f64(&arg0)?; // end OR start

        let (start, end, step) = match argc {
            1 => (0.0, a0, 1.0),
            2 => {
                let arg1 = call.nth(1).unwrap();
                (a0, to_f64(&arg1)?, 1.0)
            }
            3 => {
                let arg1 = call.nth(1).unwrap();
                let arg2 = call.nth(2).unwrap();
                (a0, to_f64(&arg1)?, to_f64(&arg2)?)
            }
            _ => unreachable!(),
        };

        if step == 0.0 {
            return Err(LabeledError::new("Step cannot be zero").with_label("step", call.head));
        }

        //------------------------------------------------------------------
        // 2. dtype, device, requires_grad flags
        //------------------------------------------------------------------
        let device = get_device_from_call(call)?;
        let kind = get_kind_from_call(call)?;

        //------------------------------------------------------------------
        // 3. build tensor with tch-rs
        //------------------------------------------------------------------
        let options = (kind, device);
        let mut t = if (start.fract() == 0.0) && (end.fract() == 0.0) && (step.fract() == 0.0) {
            // integer path
            let s = start as i64;
            let e = end as i64;
            let k = step as i64;
            match argc {
                1 => Tensor::arange(e, options),
                2 => Tensor::arange_start(s, e, options),
                _ => Tensor::arange_start_step(s, e, k, options),
            }
        } else {
            // floating path
            match argc {
                1 => Tensor::arange(end, options),
                2 => Tensor::arange_start(start, end, options),
                _ => Tensor::arange_start_step(start, end, step, options),
            }
        };

        // handle --requires_grad
        t = add_grad_from_call(call, t)?;

        //------------------------------------------------------------------
        // 4. store tensor & return id
        //------------------------------------------------------------------
        let id = Uuid::new_v4().to_string();
        TENSOR_REGISTRY.lock().unwrap().insert(id.clone(), t);

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

/// torch backward
/// Usage :   <loss-tensor-id> | torch backward
///        or torch backward <loss-tensor-id>
///
/// Calls Tensor::backward() on a scalar loss tensor so that gradients are
/// accumulated into all leaf tensors that have `requires_grad == true`.
///
/// The command returns the same tensor-id so the value can still be piped.
struct CommandBackward;

impl PluginCommand for CommandBackward {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch backward"
    }

    fn description(&self) -> &str {
        "Run back-propagation from a scalar loss tensor (loss.backward())."
    }

    fn signature(&self) -> Signature {
        Signature::build("torch backward")
            .input_output_types(vec![
                (Type::String, Type::String),  // tensor id via pipeline
                (Type::Nothing, Type::String), // tensor id via arg
            ])
            .optional(
                "loss_id",
                SyntaxShape::String,
                "ID of the scalar loss tensor (if not supplied by pipeline)",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Backward via pipeline",
                example: r#"
let w = (torch full [1] 2 --requires_grad true)
let loss = ($w | torch mean)
$loss | torch backward
"#
                .trim(),
                result: None,
            },
            Example {
                description: "Backward via argument",
                example: r#"
let w = (torch full [1] 2 --requires_grad true)
let loss = ($w | torch mean)
torch backward $loss
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
        // ───── collect loss tensor-id either from pipeline or arg ─────
        let piped: Option<Value> = match input {
            PipelineData::Value(v, _) => Some(v),
            PipelineData::Empty => None,
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or single Value inputs accepted", call.head))
            }
        };

        let arg0: Option<Value> = call.nth(0);

        match (&piped, &arg0) {
            (None, None) => {
                return Err(LabeledError::new("Missing input")
                    .with_label("Provide loss tensor ID via pipeline or argument", call.head))
            }
            (Some(_), Some(_)) => {
                return Err(LabeledError::new("Conflicting input").with_label(
                    "Provide loss tensor ID via pipeline OR argument, not both",
                    call.head,
                ))
            }
            _ => {}
        }

        let loss_id_val = piped.or(arg0).unwrap();
        let loss_id = loss_id_val.as_str()?.to_string();

        // ───── fetch the loss tensor ─────
        let reg = TENSOR_REGISTRY.lock().unwrap();
        let loss = reg
            .get(&loss_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found")
                    .with_label("Invalid loss tensor ID", call.head)
            })?
            .shallow_clone();

        // ───── ensure it is scalar (numel == 1)  (PyTorch expectation) ──
        if loss.numel() != 1 {
            return Err(LabeledError::new("Invalid loss tensor")
                .with_label("Backward currently supports only scalar losses", call.head));
        }

        // ───── run backward  (grad-mode ON by default) ─────
        loss.backward();

        // return the same id for convenience
        Ok(PipelineData::Value(Value::string(loss_id, call.head), None))
    }
}

enum Number {
    Int(i64),
    Float(f64),
}

struct CommandFull;

impl PluginCommand for CommandFull {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch full"
    }

    fn description(&self) -> &str {
        "Create a tensor of specified shape filled with a given value (similar to torch.full)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch full")
            .required(
                "size",
                SyntaxShape::List(Box::new(SyntaxShape::Int)),
                "The shape of the tensor as a list of dimensions (e.g., [2, 3] for a 2x3 tensor)",
            )
            .required(
                "fill_value",
                SyntaxShape::Number,
                "The value to fill the tensor with",
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
                description: "Create a 1D tensor of length 5 filled with value 7",
                example: "torch full [5] 7 | torch value",
                result: None,
            },
            Example {
                description: "Create a 2x3 tensor filled with value 0.5 with float64 dtype on CPU",
                example: "torch full [2, 3] 0.5 --dtype float64 --device cpu | torch value",
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
        // Get the size (list of dimensions)
        let size_val = call.nth(0).unwrap();
        let dims: Vec<i64> = size_val
            .as_list()
            .map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Size must be a list of integers", call.head)
            })?
            .iter()
            .map(|v| v.as_int())
            .collect::<Result<Vec<i64>, _>>()?;
        if dims.is_empty() {
            return Err(LabeledError::new("Invalid input").with_label(
                "At least one dimension must be provided in size list",
                call.head,
            ));
        }
        if dims.iter().any(|&d| d < 1) {
            return Err(LabeledError::new("Invalid input")
                .with_label("All dimensions must be positive", call.head));
        }

        // Get the fill value (try as int first, then float)
        let fill_value_val = call.nth(1).unwrap();
        let fill_value_result = match fill_value_val.as_int() {
            Ok(int_val) => Ok(Number::Int(int_val)),
            Err(_) => fill_value_val.as_float().map(Number::Float).map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Fill value must be a number (integer or float)", call.head)
            }),
        };
        let fill_value = fill_value_result?;

        // Handle optional device argument using convenience method
        let device = get_device_from_call(call)?;

        // Handle optional dtype argument using convenience method
        let kind = get_kind_from_call(call)?;

        let mut tensor = match (fill_value, kind) {
            (Number::Int(i), Kind::Int | Kind::Int64) => {
                // Use integer-specific creation if tch-rs supports it directly
                // Since Tensor::full may expect f64, we pass as f64 but kind ensures it's stored as int
                Tensor::full(&dims, i, (kind, device))
            }
            (Number::Int(i), Kind::Float | Kind::Double) => {
                // Safe to cast int to float for float dtype
                Tensor::full(&dims, i, (kind, device))
            }
            (Number::Float(f), Kind::Float | Kind::Double) => {
                // Direct float usage
                Tensor::full(&dims, f, (kind, device))
            }
            _ => {
                return Err(LabeledError::new("Invalid dtype")
                    .with_label("Invalid data/dtype combo.", call.head));
            }
        };

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

// Linspace command to create a tensor
struct CommandLinspace;

impl PluginCommand for CommandLinspace {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch linspace"
    }

    fn description(&self) -> &str {
        "Create a 1D tensor with linearly spaced values"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch linspace")
            .required("start", SyntaxShape::Float, "Start value")
            .required("end", SyntaxShape::Float, "End value")
            .required("steps", SyntaxShape::Int, "Number of steps")
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
        vec![Example {
            description: "Create a tensor from 0.0 to 1.0 with 4 steps",
            example: "torch linspace 0.0 1.0 4",
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
        let start: f64 = call.nth(0).unwrap().as_float()?;
        let end: f64 = call.nth(1).unwrap().as_float()?;
        let steps: i64 = call.nth(2).unwrap().as_int()?;

        // Handle optional device argument
        let device = get_device_from_call(call)?;

        // Handle optional dtype argument
        let kind = get_kind_from_call(call)?;

        // Create a PyTorch tensor using tch-rs
        let mut tensor = Tensor::linspace(start, end, steps, (kind, device));

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

struct CommandMm;

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

struct CommandLogSoftmax;

impl PluginCommand for CommandLogSoftmax {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch log_softmax"
    }

    fn description(&self) -> &str {
        "Compute the log-softmax of a tensor along a specified dimension (similar to torch.log_softmax)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch log_softmax")
            // tensor id may come from pipeline or from a single argument
            .input_output_types(vec![
                (Type::String, Type::String),  // pipeline-in
                (Type::Nothing, Type::String), // arg-in
            ])
            .optional(
                "tensor_id",
                SyntaxShape::String,
                "ID of the tensor (if not supplied by pipeline)",
            )
            .named(
                "dim",
                SyntaxShape::Int,
                "Dimension along which to compute log-softmax (default: last dimension)",
                None,
            )
            .named(
                "dtype",
                SyntaxShape::String,
                "Data type of the output tensor (default: inherits input dtype)",
                None,
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Compute log-softmax over the last dimension (pipeline input)",
                example: "let t = (torch linspace 0 5 6 | torch repeat 2 1); $t | torch log_softmax | torch value",
                result: None,
            },
            Example {
                description: "Compute log-softmax along dim 1 (argument input)",
                example: "let t = (torch linspace 0 5 6 | torch repeat 2 1); torch log_softmax $t --dim 1 | torch value",
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
        // -------------------------------------------------------------
        // Fetch tensor id: either from pipeline or from first argument
        // -------------------------------------------------------------
        let piped = match input {
            PipelineData::Empty => None,
            PipelineData::Value(v, _) => Some(v),
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or single Value inputs are supported", call.head))
            }
        };
        let arg0 = call.nth(0);

        let tensor_id = match (piped, arg0) {
            (Some(_), Some(_)) => {
                return Err(LabeledError::new("Conflicting input").with_label(
                    "Provide tensor ID via pipeline OR argument, not both",
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

        // -------------------- fetch tensor ---------------------------
        let mut registry = TENSOR_REGISTRY.lock().unwrap();
        let tensor = registry
            .get(&tensor_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor ID", call.head)
            })?
            .shallow_clone();

        // -------------------- dtype flag -----------------------------
        let kind = get_kind_from_call(call)?;

        // -------------------- dim flag -------------------------------
        let dim = match call.get_flag::<i64>("dim")? {
            Some(d) => {
                let n = tensor.size().len() as i64;
                if d < 0 || d >= n {
                    return Err(LabeledError::new("Invalid dimension").with_label(
                        format!("Dimension {d} out of bounds for tensor with {n} dimensions"),
                        call.head,
                    ));
                }
                d
            }
            None => (tensor.size().len() as i64) - 1,
        };

        // ------------------- compute --------------------------------
        let result_tensor = tensor.log_softmax(dim, kind);

        // ------------------- store & return --------------------------
        let new_id = Uuid::new_v4().to_string();
        registry.insert(new_id.clone(), result_tensor);
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

/// torch gather
/// Usage:  <source-tensor comes through pipeline>  torch gather <dim:int> <index_tensor_id>
struct CommandGather;

impl PluginCommand for CommandGather {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch gather"
    }

    fn description(&self) -> &str {
        "Gather values along an axis using an index tensor \
         (like `x.gather(dim, index)` in PyTorch)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch gather")
            // source tensor id must arrive through the pipeline
            .input_output_types(vec![(Type::String, Type::String)])
            .required("dim", SyntaxShape::Int, "Dimension along which to gather")
            .required(
                "index_id",
                SyntaxShape::String,
                "ID of the index tensor (int64)",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![Example {
            description: "Gather columns 2,1,0 from each row (dim=1)",
            example: r#"
let src  = ([[10 11 12] [20 21 22]] | torch tensor)
let idx  = ([[2 1 0]   [0 0 2]]     | torch tensor --dtype int64)
$src | torch gather 1 $idx | torch value
"#
            .trim(),
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
        //--------------------------------------------------------------
        // 1. get tensor-id from pipeline, dim and index-id from args
        //--------------------------------------------------------------
        let PipelineData::Value(source_id_val, _) = input else {
            return Err(LabeledError::new("Missing input").with_label(
                "Source tensor ID must be supplied via the pipeline",
                call.head,
            ));
        };
        let source_id = source_id_val.as_str().map(|s| s.to_string()).map_err(|_| {
            LabeledError::new("Invalid input")
                .with_label("Pipeline input must be a tensor ID (string)", call.head)
        })?;

        let dim = call
            .nth(0)
            .ok_or_else(|| {
                LabeledError::new("Missing dim")
                    .with_label("Dimension argument is required", call.head)
            })?
            .as_int()
            .map_err(|_| {
                LabeledError::new("Invalid dim")
                    .with_label("Dimension must be an integer", call.head)
            })?;

        let index_id = call
            .nth(1)
            .ok_or_else(|| {
                LabeledError::new("Missing index tensor")
                    .with_label("Index tensor ID argument is required", call.head)
            })?
            .as_str()
            .map(|s| s.to_string())
            .map_err(|_| {
                LabeledError::new("Invalid index tensor ID")
                    .with_label("Must be a string", call.head)
            })?;

        //--------------------------------------------------------------
        // 2. fetch tensors
        //--------------------------------------------------------------
        let mut reg = TENSOR_REGISTRY.lock().unwrap();
        let source = reg
            .get(&source_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found")
                    .with_label("Invalid source tensor ID", call.head)
            })?
            .shallow_clone();

        let mut index = reg
            .get(&index_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found")
                    .with_label("Invalid index tensor ID", call.head)
            })?
            .shallow_clone();

        // ensure int64
        if index.kind() != Kind::Int64 {
            index = index.to_kind(Kind::Int64);
        }

        //--------------------------------------------------------------
        // 3. validate shapes & index-range
        //--------------------------------------------------------------
        let src_shape = source.size();
        let idx_shape = index.size();
        let ndims = src_shape.len() as i64;

        // dim bounds
        if dim < 0 || dim >= ndims {
            return Err(LabeledError::new("Invalid dimension").with_label(
                format!("Dim {dim} out of bounds for tensor with {ndims} dims"),
                call.head,
            ));
        }

        // same rank
        if idx_shape.len() != src_shape.len() {
            return Err(LabeledError::new("Shape mismatch").with_label(
                format!(
                    "Index tensor rank {} differs from source rank {}",
                    idx_shape.len(),
                    src_shape.len()
                ),
                call.head,
            ));
        }

        // all dims except 'dim' must match exactly
        for (d, (&s, &i)) in src_shape.iter().zip(idx_shape.iter()).enumerate() {
            if d as i64 != dim && s != i {
                return Err(LabeledError::new("Shape mismatch").with_label(
                    format!("Size mismatch at dim {d}: source={s}, index={i}",),
                    call.head,
                ));
            }
        }

        // index values must be in [0, src_shape[dim])
        let max_idx = index.max().int64_value(&[]);
        let min_idx = index.min().int64_value(&[]);
        if min_idx < 0 || max_idx >= src_shape[dim as usize] {
            return Err(LabeledError::new("Index out of range").with_label(
                format!(
                    "Index values must be between 0 and {} (exclusive); found [{}, {}]",
                    src_shape[dim as usize] - 1,
                    min_idx,
                    max_idx
                ),
                call.head,
            ));
        }

        //--------------------------------------------------------------
        // 4. gather  (sparse_grad matches source tensor)
        //--------------------------------------------------------------
        let sparse_grad = source.is_sparse();
        let result_tensor = source.gather(dim, &index, sparse_grad);

        //--------------------------------------------------------------
        // 5. store & return
        //--------------------------------------------------------------
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

// Exp command to apply exp to a tensor
struct CommandExp;

impl PluginCommand for CommandExp {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch exp"
    }

    fn description(&self) -> &str {
        "Apply exp function element-wise to a tensor"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch exp").category(Category::Custom("torch".into()))
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
        // Apply expe operation
        let result_tensor = tensor.exp();
        // Store result in registry with new ID
        let new_id = Uuid::new_v4().to_string();
        registry.insert(new_id.clone(), result_tensor);
        // Return new ID wrapped in PipelineData
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}

// torch maximum  -----------------------------------------------------------
//  1) [$t1 $t2] | torch maximum              (both IDs piped as a list)
//  2)  $t1      | torch maximum $t2          (first ID piped, second as arg)
//  3)  torch maximum $t1 $t2                 (no pipeline, two args – kept for b-compat)
// --------------------------------------------------------------------------
struct CommandMaximum;

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

    fn name(&self) -> &str { "torch stack" }

    fn description(&self) -> &str {
        "Concatenate a sequence of tensors along a new axis (torch.stack)."
    }

    fn signature(&self) -> Signature {
        Signature::build("torch stack")
            .input_output_types(vec![
                (Type::List(Box::new(Type::String)),
                 Type::String),                   // list via pipeline
                (Type::Nothing, Type::String)     // list via arg
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
"#.trim(),
                result: None,
            },
            Example {
                description: "Stack the same tensors along dim 1",
                example: r#"
let x = ([[1 2 3] [4 5 6]] | torch tensor)
let y = ([[7 8 9] [1 1 1]] | torch tensor)
torch stack [$x $y] --dim 1 | torch shape    # -> [2, 2, 3]
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
        //------------------------------------------------------------------
        // 1. Collect list of IDs (pipeline xor argument)
        //------------------------------------------------------------------
        let piped = match input {
            PipelineData::Value(v, _) => Some(v),
            PipelineData::Empty       => None,
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or Value pipeline inputs supported", call.head))
            }
        };
        let arg0 = call.nth(0);

        match (&piped, &arg0) {
            (None, None) =>
                return Err(LabeledError::new("Missing input")
                    .with_label("Provide tensor list via pipeline or argument", call.head)),
            (Some(_), Some(_)) =>
                return Err(LabeledError::new("Conflicting input")
                    .with_label("Provide tensor list via pipeline OR argument, not both", call.head)),
            _ => {}
        }

        let list_val = piped.or(arg0).unwrap();

        // accept single-level list of strings
        let ids: Vec<String> = list_val.as_list().map_err(|_|{
                LabeledError::new("Invalid input")
                    .with_label("Expected a list of tensor IDs", call.head)
            })?
            .iter()
            .map(|v| v.as_str().map(|s| s.to_string()))
            .collect::<Result<Vec<_>, _>>()?;

        if ids.is_empty() {
            return Err(
                LabeledError::new("Empty list")
                    .with_label("No tensor IDs supplied", call.head)
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
        let tensors: Vec<Tensor> = ids.iter()
            .map(|id| {
                reg.get(id).ok_or_else(|| {
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
                return Err(
                    LabeledError::new("Shape mismatch")
                        .with_label(format!("Tensor at index {i} has shape {:?}, expected {:?}", t.size(), first_shape), call.head)
                );
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
            return Err(
                LabeledError::new("Invalid dim")
                    .with_label(format!("dim must be in [0, {}], got {}", rank, dim), call.head)
            );
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

struct CommandMean;

impl PluginCommand for CommandMean {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch mean"
    }

    fn description(&self) -> &str {
        "Compute the mean value of a tensor (similar to torch.mean single tensor mode)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch mean")
            .input_output_types(vec![(Type::String, Type::String)])
            .named(
                "dtype",
                SyntaxShape::String,
                "Data type of the tensor (default: 'float32')",
                None,
            )
            .named(
                "dim",
                SyntaxShape::Int,
                "Dimension along which to compute mean (default: over all elements)",
                None,
            )
            .named(
                "keepdim",
                SyntaxShape::Boolean,
                "Whether to keep the reduced dimension as size 1 (default: false)",
                None,
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Compute mean value over all elements of a tensor",
                example: "let t1 = (torch full 5 2 3); $t1 | torch mean | torch value",
                result: None,
            },
            Example {
                description: "Compute mean along a specific dimension with keepdim",
                example: "let t1 = (torch full 5 2 3); $t1 | torch mean --dim 1 --keepdim true | torch value",
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
        // Get tensor1 ID from input (pipeline)
        let input_value = input.into_value(call.head)?;
        let tensor1_id = input_value.as_str().map(|s| s.to_string()).map_err(|_| {
            LabeledError::new("Invalid input")
                .with_label("Unable to parse tensor1 ID from input", call.head)
        })?;

        // Look up tensor in registry
        let mut registry = TENSOR_REGISTRY.lock().unwrap();
        let tensor1 = registry
            .get(&tensor1_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor1 ID", call.head)
            })?
            .shallow_clone();

        let kind = get_kind_from_call(call)?;

        // Single tensor mode (mean over dimension or entire tensor)
        let dim_opt: Option<i64> = call.get_flag("dim")?;
        let keepdim = call.get_flag::<bool>("keepdim")?.unwrap_or(false);
        let result_tensor: Tensor = match dim_opt {
            Some(dim) => {
                let num_dims = tensor1.size().len() as i64;
                if dim < 0 || dim >= num_dims {
                    return Err(LabeledError::new("Invalid dimension").with_label(
                        format!(
                            "Dimension {dim} out of bounds for tensor with {num_dims} dimensions"
                        ),
                        call.head,
                    ));
                }
                // Use mean_dim for dimension-specific mean
                tensor1.mean_dim(dim, keepdim, kind)
            }
            None => tensor1.mean(kind), // Meanimum over all elements
        };

        // Store result in registry with new ID
        let new_id = Uuid::new_v4().to_string();
        registry.insert(new_id.clone(), result_tensor);
        // Return new ID wrapped in PipelineData
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}

struct CommandMax;

impl PluginCommand for CommandMax {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch max"
    }

    fn description(&self) -> &str {
        "Compute the maximum value of a tensor (similar to torch.max single tensor mode)"
    }

    fn signature(&self) -> Signature {
        Signature::build("torch max")
            .input_output_types(vec![(Type::String, Type::String)])
            .named(
                "dim",
                SyntaxShape::Int,
                "Dimension along which to compute maximum (default: over all elements)",
                None,
            )
            .named(
                "keepdim",
                SyntaxShape::Boolean,
                "Whether to keep the reduced dimension as size 1 (default: false)",
                None,
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Compute maximum value over all elements of a tensor",
                example: "let t1 = (torch full 5 2 3); $t1 | torch max | torch value",
                result: None,
            },
            Example {
                description: "Compute maximum along a specific dimension with keepdim",
                example: "let t1 = (torch full 5 2 3); $t1 | torch max --dim 1 --keepdim true | torch value",
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
        // Get tensor1 ID from input (pipeline)
        let input_value = input.into_value(call.head)?;
        let tensor1_id = input_value.as_str().map(|s| s.to_string()).map_err(|_| {
            LabeledError::new("Invalid input")
                .with_label("Unable to parse tensor1 ID from input", call.head)
        })?;

        // Look up tensor in registry
        let mut registry = TENSOR_REGISTRY.lock().unwrap();
        let tensor1 = registry
            .get(&tensor1_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor1 ID", call.head)
            })?
            .shallow_clone();

        // Single tensor mode (maximum over dimension or entire tensor)
        let dim_opt: Option<i64> = call.get_flag("dim")?;
        let keepdim = call.get_flag::<bool>("keepdim")?.unwrap_or(false);
        let result_tensor = match dim_opt {
            Some(dim) => {
                let num_dims = tensor1.size().len() as i64;
                if dim < 0 || dim >= num_dims {
                    return Err(LabeledError::new("Invalid dimension").with_label(
                        format!(
                            "Dimension {dim} out of bounds for tensor with {num_dims} dimensions"
                        ),
                        call.head,
                    ));
                }
                // Use max_dim for dimension-specific maximum
                let (values, _indices) = tensor1.max_dim(dim, keepdim);
                values
            }
            None => tensor1.max(), // Maximum over all elements
        };

        // Store result in registry with new ID
        let new_id = Uuid::new_v4().to_string();
        registry.insert(new_id.clone(), result_tensor);
        // Return new ID wrapped in PipelineData
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

// torch detach  -------------------------------------------------------------
// Return a new tensor that shares storage with the original but is detached
// from the autograd graph (requires_grad = false).  Usage:
//
//     $x  | torch detach
//     torch detach $x
//
// The original tensor remains unchanged and can still track gradients; the
// returned tensor does not.
//
// ---------------------------------------------------------------------------
struct CommandDetach;

impl PluginCommand for CommandDetach {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "torch detach"
    }

    fn description(&self) -> &str {
        "Create a view of a tensor that does **not** track gradients \
         (like Tensor.detach() in PyTorch)."
    }

    fn signature(&self) -> Signature {
        Signature::build("torch detach")
            .input_output_types(vec![
                (Type::String, Type::String),  // ID via pipeline → ID
                (Type::Nothing, Type::String), // ID via arg      → ID
            ])
            .optional(
                "tensor_id",
                SyntaxShape::String,
                "ID of the tensor to detach (if not provided via pipeline)",
            )
            .category(Category::Custom("torch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Detach a tensor received through the pipeline",
                example: r#"
let x = (torch randn [2 2] --requires_grad true)
$x | torch detach | torch requires_grad?
"#
                .trim(),
                result: None,
            },
            Example {
                description: "Detach via positional argument",
                example: r#"
let x = (torch randn [2] --requires_grad true)
torch detach $x | torch requires_grad?
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
        // 1. Collect tensor ID (pipeline xor arg)
        //------------------------------------------------------------------
        let piped = match input {
            PipelineData::Value(v, _) => Some(v),
            PipelineData::Empty => None,
            _ => {
                return Err(LabeledError::new("Unsupported input")
                    .with_label("Only Empty or Value inputs are supported", call.head))
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

        //------------------------------------------------------------------
        // 2. Fetch tensor from registry
        //------------------------------------------------------------------
        let mut reg = TENSOR_REGISTRY.lock().unwrap();
        let t = reg
            .get(&tensor_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor ID", call.head)
            })?
            .shallow_clone();

        //------------------------------------------------------------------
        // 3. Detach and store result
        //------------------------------------------------------------------
        let detached = t.detach(); // no longer tracks gradients
        let new_id = Uuid::new_v4().to_string();
        reg.insert(new_id.clone(), detached);

        //------------------------------------------------------------------
        // 4. Return ID of detached tensor
        //------------------------------------------------------------------
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}

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
struct CommandFree;

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
struct CommandGrad;

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

fn add_grad_from_call(
    call: &nu_plugin::EvaluatedCall,
    mut tensor: Tensor,
) -> Result<Tensor, LabeledError> {
    let requires_grad = call.get_flag::<bool>("requires_grad")?.unwrap_or(false);
    if requires_grad {
        tensor = tensor.set_requires_grad(true);
    }
    Ok(tensor)
}

fn get_device_from_call(call: &nu_plugin::EvaluatedCall) -> Result<Device, LabeledError> {
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

fn get_kind_from_call(call: &nu_plugin::EvaluatedCall) -> Result<Kind, LabeledError> {
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
fn tensor_to_value(tensor: &Tensor, span: Span) -> Result<Value, LabeledError> {
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
fn value_to_tensor(
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

fn main() {
    serve_plugin(&NutorchPlugin, nu_plugin::MsgPackSerializer);
}
