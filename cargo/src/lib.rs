use lazy_static::lazy_static;
use nu_plugin::{Plugin, PluginCommand};
use nu_protocol::{LabeledError, Span, Value};
use std::collections::HashMap;
use std::sync::Mutex;
use tch::{Device, Kind, Tensor};

mod command_add;
mod command_arange;
mod command_argmax;
mod command_backward;
mod command_cat;
mod command_detach;
mod command_devices;
mod command_div;
mod command_exp;
mod command_free;
mod command_full;
mod command_gather;
mod command_grad;
mod command_linspace;
mod command_log_softmax;
mod command_manual_seed;
mod command_max;
mod command_maximum;
mod command_mean;
mod command_mm;
mod command_mul;
mod command_neg;
mod command_randn;
mod command_repeat;
mod command_repeat_interleave;
mod command_reshape;
mod command_sgd_step;
mod command_shape;
mod command_sin;
mod command_softmax;
mod command_squeeze;
mod command_stack;
mod command_sub;
mod command_t;
mod command_tensor;
mod command_torch;
mod command_unsqueeze;
mod command_value;
mod command_zero_grad;

pub use command_add::CommandAdd;
pub use command_arange::CommandArange;
pub use command_argmax::CommandArgmax;
pub use command_backward::CommandBackward;
pub use command_cat::CommandCat;
pub use command_detach::CommandDetach;
pub use command_devices::CommandDevices;
pub use command_div::CommandDiv;
pub use command_exp::CommandExp;
pub use command_free::CommandFree;
pub use command_full::CommandFull;
pub use command_gather::CommandGather;
pub use command_grad::CommandGrad;
pub use command_linspace::CommandLinspace;
pub use command_log_softmax::CommandLogSoftmax;
pub use command_manual_seed::CommandManualSeed;
pub use command_max::CommandMax;
pub use command_maximum::CommandMaximum;
pub use command_mean::CommandMean;
pub use command_mm::CommandMm;
pub use command_mul::CommandMul;
pub use command_neg::CommandNeg;
pub use command_randn::CommandRandn;
pub use command_repeat::CommandRepeat;
pub use command_repeat_interleave::CommandRepeatInterleave;
pub use command_reshape::CommandReshape;
pub use command_sgd_step::CommandSgdStep;
pub use command_shape::CommandShape;
pub use command_sin::CommandSin;
pub use command_softmax::CommandSoftmax;
pub use command_squeeze::CommandSqueeze;
pub use command_stack::CommandStack;
pub use command_sub::CommandSub;
pub use command_t::CommandT;
pub use command_tensor::CommandTensor;
pub use command_torch::CommandTorch;
pub use command_unsqueeze::CommandUnsqueeze;
pub use command_value::CommandValue;
pub use command_zero_grad::CommandZeroGrad;

// Global registry to store tensors by ID (thread-safe)
lazy_static! {
    pub static ref TENSOR_REGISTRY: Mutex<HashMap<String, Tensor>> = Mutex::new(HashMap::new());
}

pub struct NutorchPlugin;

impl Plugin for NutorchPlugin {
    fn commands(&self) -> Vec<Box<dyn PluginCommand<Plugin = Self>>> {
        vec![
            Box::new(CommandAdd),
            Box::new(CommandArange),
            Box::new(CommandArgmax),
            Box::new(CommandBackward),
            Box::new(CommandCat),
            Box::new(CommandDetach),
            Box::new(CommandDevices),
            Box::new(CommandDiv),
            Box::new(CommandExp),
            Box::new(CommandFree),
            Box::new(CommandFull),
            Box::new(CommandGather),
            Box::new(CommandGrad),
            Box::new(CommandLinspace),
            Box::new(CommandLogSoftmax),
            Box::new(CommandManualSeed),
            Box::new(CommandMax),
            Box::new(CommandMaximum),
            Box::new(CommandMean),
            Box::new(CommandMm),
            Box::new(CommandMul),
            Box::new(CommandNeg),
            Box::new(CommandRandn),
            Box::new(CommandRepeat),
            Box::new(CommandRepeatInterleave),
            Box::new(CommandReshape),
            Box::new(CommandSgdStep),
            Box::new(CommandShape),
            Box::new(CommandSin),
            Box::new(CommandSoftmax),
            Box::new(CommandSqueeze),
            Box::new(CommandStack),
            Box::new(CommandSub),
            Box::new(CommandT),
            Box::new(CommandTensor),
            Box::new(CommandTorch),
            Box::new(CommandUnsqueeze),
            Box::new(CommandValue),
            Box::new(CommandZeroGrad),
        ]
    }

    fn version(&self) -> std::string::String {
        "0.1.3".to_string()
    }
}

pub enum Number {
    Int(i64),
    Float(f64),
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
                        v.as_float()
                            .or_else(|_| v.as_int().map(|i| i as f64))
                            .map_err(|_| {
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
