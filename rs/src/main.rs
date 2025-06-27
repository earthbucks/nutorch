use lazy_static::lazy_static;
use nu_plugin::{serve_plugin, Plugin, PluginCommand};
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, SyntaxShape, Value,
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
        vec![Box::new(Linspace), Box::new(Sin), Box::new(Display)]
    }

    fn version(&self) -> std::string::String {
        "0.0.1".to_string()
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
        let start: f64 = call.get_flag_value("start").unwrap().as_float()?;
        let end: f64 = call.get_flag_value("end").unwrap().as_float()?;
        let steps: i64 = call.get_flag_value("steps").unwrap().as_int()?;
        if steps < 2 {
            return Err(LabeledError::new("Invalid input")
                .with_label("Steps must be at least 2", call.head));
        }
        // Create a PyTorch tensor using tch-rs
        let tensor = Tensor::linspace(start, end, steps, (Kind::Float, Device::Cpu));
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
