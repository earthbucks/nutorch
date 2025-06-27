use lazy_static::lazy_static;
use nu_plugin::{serve_plugin, Plugin, PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Type, Value,
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
            Box::new(Nutorch), // New top-level command
            Box::new(Devices),
            Box::new(Linspace),
            Box::new(Sin),
            Box::new(Display),
        ]
    }

    fn version(&self) -> std::string::String {
        "0.0.1".to_string()
    }
}

// New top-level Nutorch command
struct Nutorch;

impl PluginCommand for Nutorch {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "nutorch"
    }

    fn signature(&self) -> Signature {
        Signature::build("nutorch").category(Category::Custom("nutorch".into()))
    }

    fn description(&self) -> &str {
        "A simple command to test the nutorch plugin"
    }

    fn examples(&self) -> Vec<Example> {
        vec![Example {
            description: "Run the nutorch command to test the plugin".into(),
            example: "nutorch".into(),
            result: Some(Value::string("Hello, World!", nu_protocol::Span::unknown())),
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
            Value::string("Hello, World!", call.head),
            None,
        ))
    }
}

// Devices command to list available devices
struct Devices;

impl PluginCommand for Devices {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "nutorch devices"
    }

    fn description(&self) -> &str {
        "List available devices for tensor operations"
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

        // // Check for MPS (Metal Performance Shaders) availability on macOS
        // if tch::Mps::is_available() {
        //     devices.push(Value::string("mps", span));
        // }

        Ok(PipelineData::Value(Value::list(devices, span), None))
    }
}

// Linspace command to create a tensor
struct Linspace;

impl PluginCommand for Linspace {
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
                "Device to create the tensor on (efault: 'cpu')",
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
        let device_str_opt = call
            .get_flag::<String>("device")
            .unwrap_or_else(|_| Some("cpu".to_string()));
        let device_str: String = match device_str_opt {
            Some(s) => s,
            None => "cpu".to_string(),
        };
        let device = match device_str.as_str() {
            "cpu" => Device::Cpu,
            "cuda" => Device::Cuda(0),
            "mps" => Device::Mps,
            // "mps" if tch::Mps::is_available() => Device::Mps,
            _ if device_str.starts_with("cuda:") => {
                // Handle specific CUDA device like "cuda:0", "cuda:1", etc.
                if let Some(num) = device_str[5..].parse::<usize>().ok() {
                    Device::Cuda(num)
                } else {
                    return Err(LabeledError::new("Invalid CUDA device")
                        .with_label("Device must be 'cpu', 'cuda', or 'mps'", call.head));
                }
            }
            _ => {
                return Err(LabeledError::new("Invalid device")
                    .with_label("Device must be 'cpu' or 'cuda'", call.head));
            }
        };

        // Handle optional dtype argument
        let dtype_str_opt = call
            .get_flag::<String>("dtype")
            .unwrap_or_else(|_| Some("float32".to_string()));
        let dtype_str: String = match dtype_str_opt {
            Some(s) => s,
            None => "float32".to_string(),
        };
        let kind = match dtype_str.as_str() {
            "float32" => Kind::Float,
            "float64" => Kind::Double,
            "int32" => Kind::Int,
            "int64" => Kind::Int64,
            _ => {
                return Err(LabeledError::new("Invalid dtype")
                    .with_label("Dtype must be 'float32', 'float64', 'int32', or 'int64'", call.head));
            }
        };

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

// Sin command to apply sine to a tensor
struct Sin;

impl PluginCommand for Sin {
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

// Display command to convert tensor to Nushell data structure for output
struct Display;

impl PluginCommand for Display {
    type Plugin = NutorchPlugin;

    fn name(&self) -> &str {
        "nutorch display"
    }

    fn description(&self) -> &str {
        "Display a tensor as a Nushell list or table"
    }

    fn signature(&self) -> Signature {
        Signature::build("nutorch display").category(Category::Custom("nutorch".into()))
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
        // Convert tensor to Nushell Value (simplified for 1D/2D tensors)
        let dims = tensor.size();
        let span = call.head;
        if dims.len() == 1 {
            // 1D tensor to list
            let size = dims[0] as usize;
            let mut data: Vec<f64> = Vec::with_capacity(size);
            for i in 0..size as i64 {
                data.push(tensor.get(i).double_value(&[]));
            }
            let list = data.into_iter().map(|v| Value::float(v, span)).collect();
            Ok(PipelineData::Value(Value::list(list, span), None))
        } else if dims.len() == 2 {
            // 2D tensor to list of lists
            // 2D tensor to list of lists
            let rows = dims[0] as usize;
            let cols = dims[1] as usize;
            let mut data = Vec::with_capacity(rows);
            for i in 0..rows as i64 {
                let mut row = Vec::with_capacity(cols);
                for j in 0..cols as i64 {
                    row.push(tensor.get(i).get(j).double_value(&[]));
                }
                data.push(row);
            }
            let list = data
                .into_iter()
                .map(|row| {
                    let row_list = row.into_iter().map(|v| Value::float(v, span)).collect();
                    Value::list(row_list, span)
                })
                .collect();
            Ok(PipelineData::Value(Value::list(list, span), None))
        } else {
            Err(LabeledError::new("Unsupported dimension")
                .with_label("Only 1D and 2D tensors supported for display", span))
        }
    }
}

fn main() {
    serve_plugin(&NutorchPlugin, nu_plugin::MsgPackSerializer)
}
