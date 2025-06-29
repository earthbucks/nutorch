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
            // Top-level Nutorch command
            Box::new(CommandNutorch),
            // Individual commands for tensor operations and utilities
            Box::new(CommandCat),
            Box::new(CommandDevices),
            Box::new(CommandFull),
            Box::new(CommandLinspace),
            Box::new(CommandManualSeed),
            Box::new(CommandMm),
            Box::new(CommandRandn),
            Box::new(CommandRepeat),
            Box::new(CommandSin),
            Box::new(CommandTensor),
            Box::new(CommandValue),
        ]
    }

    fn version(&self) -> std::string::String {
        "0.0.1".to_string()
    }
}

// New top-level Nutorch command
struct CommandNutorch;

impl PluginCommand for CommandNutorch {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "nutorch"
    }

    fn signature(&self) -> Signature {
        Signature::build("nutorch").category(Category::Custom("nutorch".into()))
    }

    fn description(&self) -> &str {
        "The entry point for the Nutorch plugin, providing access to tensor operations and utilities"
    }

    fn examples(&self) -> Vec<Example> {
        vec![Example {
            description: "Run the nutorch command to test the plugin".into(),
            example: "nutorch".into(),
            result: Some(Value::string(
                "Welcome to Nutorch. Type `nutorch --help` for more information.",
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
                "Welcome to Nutorch. Type `nutorch --help` for more information.",
                call.head,
            ),
            None,
        ))
    }
}

struct CommandCat;

impl PluginCommand for CommandCat {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "nutorch cat"
    }

    fn description(&self) -> &str {
        "Concatenate a sequence of tensors along a specified dimension (similar to torch.cat)"
    }

    fn signature(&self) -> Signature {
        Signature::build("nutorch cat")
            .rest(
                "tensor_ids",
                SyntaxShape::String,
                "IDs of the tensors to concatenate",
            )
            .named(
                "dim",
                SyntaxShape::Int,
                "Dimension along which to concatenate (default: 0)",
                None,
            )
            .input_output_types(vec![(Type::Nothing, Type::String)])
            .category(Category::Custom("nutorch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Concatenate two 2x3 tensors along dimension 0",
                example: "let t1 = (nutorch full 1 2 3); let t2 = (nutorch full 2 2 3); nutorch cat $t1 $t2 --dim 0 | nutorch value",
                result: None,
            },
            Example {
                description: "Concatenate three 2x3 tensors along dimension 1",
                example: "let t1 = (nutorch full 1 2 3); let t2 = (nutorch full 2 2 3); let t3 = (nutorch full 3 2 3); nutorch cat $t1 $t2 $t3 --dim 1 | nutorch value",
                result: None,
            }
        ]
    }

    fn run(
        &self,
        _plugin: &NutorchPlugin,
        _engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        _input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        // Get tensor IDs to concatenate
        let tensor_ids: Vec<String> = call
            .rest(0)
            .map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Unable to parse tensor IDs", call.head)
            })?
            .into_iter()
            .map(|v: Value| v.as_str().map(|s| s.to_string()))
            .collect::<Result<Vec<String>, _>>()?;
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
        "nutorch manual_seed"
    }

    fn description(&self) -> &str {
        "Set the random seed for PyTorch operations to ensure reproducibility"
    }

    fn signature(&self) -> Signature {
        Signature::build("nutorch manual_seed")
            .required(
                "seed",
                SyntaxShape::Int,
                "The seed value for the random number generator",
            )
            .input_output_types(vec![(Type::Nothing, Type::Nothing)])
            .category(Category::Custom("nutorch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![Example {
            description: "Set the random seed to 42 for reproducibility",
            example: "nutorch manual_seed 42",
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
        "nutorch randn"
    }

    fn description(&self) -> &str {
        "Generate a tensor filled with random numbers from a normal distribution (mean 0, std 1)"
    }

    fn signature(&self) -> Signature {
        Signature::build("nutorch randn")
            .rest("dims", SyntaxShape::Int, "Dimensions of the tensor (e.g., 2 3 for a 2x3 tensor)")
            .named(
                "device",
                SyntaxShape::String,
                "Device to create the tensor on ('cpu', 'cuda', 'mps', default: 'cpu')",
                None,
            )
            .named(
                "dtype",
                SyntaxShape::String,
                "Data type of the tensor ('float32', 'float64', 'int32', 'int64', default: 'float32')",
                None,
            )
            .input_output_types(vec![(Type::Nothing, Type::String)])
            .category(Category::Custom("nutorch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Generate a 2x3 tensor with random values from a normal distribution",
                example: "nutorch randn 2 3 | nutorch tovalue",
                result: None,
            },
            Example {
                description:
                    "Generate a 1D tensor of size 5 with a specific seed for reproducibility",
                example: "nutorch manual_seed 42; nutorch randn 5 | nutorch tovalue",
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
        let tensor = Tensor::randn(&dims, (kind, device));
        // Generate a unique ID for the tensor
        let id = Uuid::new_v4().to_string();
        // Store in registry
        TENSOR_REGISTRY.lock().unwrap().insert(id.clone(), tensor);
        // Return the ID as a string to Nushell, wrapped in PipelineData
        Ok(PipelineData::Value(Value::string(id, call.head), None))
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
        "nutorch full"
    }

    fn description(&self) -> &str {
        "Create a tensor of specified shape filled with a given value (similar to torch.full)"
    }

    fn signature(&self) -> Signature {
        Signature::build("nutorch full")
            .required("value", SyntaxShape::Number, "The value to fill the tensor with")
            .rest("dims", SyntaxShape::Int, "Dimensions of the tensor (e.g., 2 3 for a 2x3 tensor)")
            .named(
                "device",
                SyntaxShape::String,
                "Device to create the tensor on ('cpu', 'cuda', 'mps', default: 'cpu')",
                None,
            )
            .named(
                "dtype",
                SyntaxShape::String,
                "Data type of the tensor ('float32', 'float64', 'int32', 'int64', default: 'float32')",
                None,
            )
            .input_output_types(vec![(Type::Nothing, Type::String)])
            .category(Category::Custom("nutorch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Create a 1D tensor of length 5 filled with value 7",
                example: "nutorch full 7 5 | nutorch value",
                result: None,
            },
            Example {
                description: "Create a 2x3 tensor filled with value 0.5 with float64 dtype on CPU",
                example: "nutorch full 0.5 2 3 --dtype float64 --device cpu | nutorch value",
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
        // Get the fill value (try as int first, then float)
        let fill_value_val = call.nth(0).unwrap();
        let fill_value_result = match fill_value_val.as_int() {
            Ok(int_val) => Ok(Number::Int(int_val)),
            Err(_) => fill_value_val.as_float().map(Number::Float).map_err(|_| {
                LabeledError::new("Invalid input")
                    .with_label("Value must be a number (integer or float)", call.head)
            }),
        };
        let fill_value = fill_value_result?;

        // Get dimensions for the tensor shape
        let dims: Vec<i64> = call
            .rest(1)
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

        // Handle optional device argument using convenience method
        let device = get_device_from_call(call)?;

        // Handle optional dtype argument using convenience method
        let kind = get_kind_from_call(call)?;

        let tensor = match (fill_value, kind) {
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
        "nutorch devices"
    }

    fn description(&self) -> &str {
        "List some available devices. Additional devices may be available, but unlisted here."
    }

    fn signature(&self) -> Signature {
        Signature::build("nutorch devices")
            .input_output_types(vec![(Type::Nothing, Type::List(Box::new(Type::String)))])
            .category(Category::Custom("nutorch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![Example {
            description: "List available devices for tensor operations",
            example: "nutorch devices",
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
        "nutorch linspace"
    }

    fn description(&self) -> &str {
        "Create a 1D tensor with linearly spaced values"
    }

    fn signature(&self) -> Signature {
        Signature::build("nutorch linspace")
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
            .category(Category::Custom("nutorch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![Example {
            description: "Create a tensor from 0.0 to 1.0 with 4 steps",
            example: "nutorch linspace 0.0 1.0 4",
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
        let tensor = Tensor::linspace(start, end, steps, (kind, device));
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
        "nutorch repeat"
    }

    fn description(&self) -> &str {
        "Repeat a tensor along specified dimensions to create a multidimensional tensor"
    }

    fn signature(&self) -> Signature {
        Signature::build("nutorch repeat")
            .rest(
                "sizes",
                SyntaxShape::Int,
                "Number of times to repeat along each dimension",
            )
            .input_output_types(vec![(Type::String, Type::String)])
            .category(Category::Custom("nutorch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Repeat a tensor 3 times along the first dimension",
                example: "nutorch linspace 0.0 1.0 4 | nutorch repeat 3 | nutorch value",
                result: None,
            },
            Example {
                description: "Repeat a tensor 2 times along first dim and 2 times along second dim (creates new dim if needed)",
                example: "nutorch linspace 0.0 1.0 4 | nutorch repeat 2 2 | nutorch value",
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
        "nutorch mm"
    }

    fn description(&self) -> &str {
        "Perform matrix multiplication of two 2D tensors (similar to torch.mm)"
    }

    fn signature(&self) -> Signature {
        Signature::build("nutorch mm")
            .required(
                "tensor1_id",
                SyntaxShape::String,
                "ID of the first tensor for matrix multiplication",
            )
            .required(
                "tensor2_id",
                SyntaxShape::String,
                "ID of the second tensor for matrix multiplication",
            )
            .input_output_types(vec![(Type::Nothing, Type::String)])
            .category(Category::Custom("nutorch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Perform matrix multiplication between two tensors",
                example: "let t1 = (nutorch linspace 0 5 6 | nutorch repeat 2); let t2 = (nutorch linspace 0 2 3 | nutorch repeat 2); nutorch mm $t1 $t2 | nutorch value",
                result: None,
            }
        ]
    }

    fn run(
        &self,
        _plugin: &NutorchPlugin,
        _engine: &nu_plugin::EngineInterface,
        call: &nu_plugin::EvaluatedCall,
        _input: PipelineData,
    ) -> Result<PipelineData, LabeledError> {
        // Get tensor1 ID from first required argument
        let tensor1_id_opt = call.nth(0).unwrap();
        let tensor1_id = tensor1_id_opt.as_str()?;
        // Get tensor2 ID from second required argument
        let tensor2_id_opt = call.nth(1).unwrap();
        let tensor2_id = tensor2_id_opt.as_str()?;

        // Look up tensors in registry
        let mut registry = TENSOR_REGISTRY.lock().unwrap();
        let tensor1 = registry
            .get(tensor1_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor1 ID", call.head)
            })?
            .shallow_clone();
        let tensor2 = registry
            .get(tensor2_id)
            .ok_or_else(|| {
                LabeledError::new("Tensor not found").with_label("Invalid tensor2 ID", call.head)
            })?
            .shallow_clone();

        // Check if tensors are 2D
        let dims1 = tensor1.size();
        let dims2 = tensor2.size();
        if dims1.len() != 2 {
            return Err(LabeledError::new("Invalid tensor dimension").with_label(
                format!("First tensor must be 2D, got {}D", dims1.len()),
                call.head,
            ));
        }
        if dims2.len() != 2 {
            return Err(LabeledError::new("Invalid tensor dimension").with_label(
                format!("Second tensor must be 2D, got {}D", dims2.len()),
                call.head,
            ));
        }
        // Check if matrix multiplication is possible (columns of first == rows of second)
        if dims1[1] != dims2[0] {
            return Err(LabeledError::new("Incompatible dimensions").with_label(
                format!(
                    "Cannot multiply {}x{} with {}x{}",
                    dims1[0], dims1[1], dims2[0], dims2[1]
                ),
                call.head,
            ));
        }

        // Perform matrix multiplication
        let result_tensor = tensor1.mm(&tensor2);
        // Store result in registry with new ID
        let new_id = Uuid::new_v4().to_string();
        registry.insert(new_id.clone(), result_tensor);
        // Return new ID wrapped in PipelineData
        Ok(PipelineData::Value(Value::string(new_id, call.head), None))
    }
}

// Sin command to apply sine to a tensor
struct CommandSin;

impl PluginCommand for CommandSin {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "nutorch sin"
    }

    fn description(&self) -> &str {
        "Apply sine function element-wise to a tensor"
    }

    fn signature(&self) -> Signature {
        Signature::build("nutorch sin").category(Category::Custom("nutorch".into()))
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

// Command to convert tensor to Nushell data structure (value)
struct CommandValue;

impl PluginCommand for CommandValue {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "nutorch value"
    }

    fn description(&self) -> &str {
        "Convert a tensor to a Nushell Value (nested list structure)"
    }

    fn signature(&self) -> Signature {
        Signature::build("nutorch value")
            .input_output_types(vec![(Type::String, Type::Any)])
            .category(Category::Custom("nutorch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Convert a 1D tensor to a Nushell Value (list)",
                example: "nutorch linspace 0.0 1.0 4 | nutorch value",
                result: None,
            },
            Example {
                description: "Convert a 2D or higher dimensional tensor to nested Values",
                example: "nutorch linspace 0.0 1.0 4 | nutorch repeat 2 2 | nutorch value",
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
        "nutorch tensor"
    }

    fn description(&self) -> &str {
        "Convert a Nushell Value (nested list structure) to a tensor"
    }

    fn signature(&self) -> Signature {
        Signature::build("nutorch tensor")
            .input_output_types(vec![(Type::Any, Type::String)])
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
            .category(Category::Custom("nutorch".into()))
    }

    fn examples(&self) -> Vec<Example> {
        vec![
            Example {
                description: "Convert a 1D list to a tensor",
                example: "[0.0, 1.0, 2.0, 3.0] | nutorch tensor",
                result: None,
            },
            Example {
                description: "Convert a 2D nested list to a tensor with specific device and dtype",
                example: "[[0.0, 1.0], [2.0, 3.0]] | nutorch tensor --device cpu --dtype float64",
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
        let input_value = input.into_value(call.head)?;

        // Handle optional device argument
        let device = get_device_from_call(call)?;

        // Handle optional dtype argument
        let kind = get_kind_from_call(call)?;

        // Convert Nushell Value to tensor
        let mut tensor = value_to_tensor(&input_value, kind, device, call.head)?;
        // Generate a unique ID for the tensor

        // Handle optional requires_grad argument
        tensor = add_grad_from_call(call, tensor)?;

        let id = Uuid::new_v4().to_string();
        // Store in registry
        TENSOR_REGISTRY.lock().unwrap().insert(id.clone(), tensor);
        // Return the ID as a string to Nushell, wrapped in PipelineData
        Ok(PipelineData::Value(Value::string(id, call.head), None))
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
                    .with_label(format!("Cannot convert tensor of type {:?}", kind), span))
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
                    .with_label(format!("Cannot convert tensor of type {:?}", kind), span))
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
    serve_plugin(&NutorchPlugin, nu_plugin::MsgPackSerializer)
}
