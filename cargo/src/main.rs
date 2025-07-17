use nu_plugin::serve_plugin;
pub use nutorch::NutorchPlugin;

fn main() {
    serve_plugin(&NutorchPlugin, nu_plugin::MsgPackSerializer);
}
