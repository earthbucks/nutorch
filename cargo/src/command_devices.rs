use nu_plugin::PluginCommand;
use nu_protocol::{
    Category, Example, LabeledError, PipelineData, Signature, Type, Value,
};

use crate::NutorchPlugin;

// Devices command to list available devices
pub struct CommandDevices;

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
