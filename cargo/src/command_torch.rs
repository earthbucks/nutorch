use nu_plugin::PluginCommand;
use nu_protocol::{Category, Example, LabeledError, PipelineData, Signature, Value};

use crate::NutorchPlugin;

// New top-level Torch command
pub struct CommandTorch;

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
