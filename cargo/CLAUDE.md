# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nutorch is a Nushell plugin that wraps tch-rs (PyTorch's Rust bindings) to provide PyTorch tensor operations directly in the Nushell command line. It's essentially "PyTorch for Nushell instead of Python."

## Architecture

### Core Components

- **Plugin Framework**: Built as a Nushell plugin using `nu-plugin` crate
- **Tensor Registry**: Global thread-safe HashMap (`TENSOR_REGISTRY`) that stores tensors by UUID for cross-command persistence
- **Command Structure**: Each PyTorch operation is implemented as a separate command module (e.g., `command_add.rs`, `command_tensor.rs`)
- **Data Conversion**: Bidirectional conversion between Nushell Values (nested lists) and PyTorch tensors

### Key Files

- `src/lib.rs`: Main plugin implementation, tensor registry, and utility functions
- `src/main.rs`: Plugin entry point using `serve_plugin`
- `src/command_*.rs`: Individual command implementations for tensor operations
- `test/test_*.nu`: Nushell test files for each command

### Data Flow

1. Commands receive Nushell Values (lists, numbers)
2. Convert to PyTorch tensors using `value_to_tensor()`
3. Store tensors in `TENSOR_REGISTRY` with UUIDs
4. Return tensor IDs as strings to Nushell
5. Convert tensors back to Nushell Values using `tensor_to_value()` when needed

## Development Commands

### Building
```bash
cargo build --release
```

### Testing
Tests are written in Nushell and located in the `test/` directory:

```bash
cd test
pnpm install
# In Nushell:
use node_modules/test.nu
test run-tests
```

### Plugin Installation
After building, register the plugin with Nushell:
```bash
nu -c "plugin add target/release/nu_plugin_torch"
nu -c "plugin use torch"
```

## Command Implementation Pattern

Each command follows this structure:
- Implements `PluginCommand` trait
- Defines signature with parameters (`device`, `dtype`, `requires_grad` flags)
- Handles input from pipeline or arguments
- Uses utility functions from `lib.rs` for device/dtype parsing
- Stores results in `TENSOR_REGISTRY` and returns UUID strings

Common flags across commands:
- `--device`: Target device (cpu, cuda, mps, cuda:N)
- `--dtype`: Data type (float32, float64, int32, int64)
- `--requires_grad`: Enable gradient computation

## Tensor Lifecycle

- Tensors are created by commands like `torch tensor`
- Stored in global registry with UUID keys
- Referenced by UUID strings in subsequent operations
- Memory managed by PyTorch's reference counting
- Use `torch free` to explicitly remove tensors from registry