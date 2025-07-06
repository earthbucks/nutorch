# Nutorch

Nutorch is a [Nushell](https://github.com/nushell/nushell) plugin that wraps
[tch-rs](https://github.com/LaurentMazare/tch-rs), which itself is a wrapper for
libtorch, the C++ backend of [PyTorch](https://pytorch.org/).

In other words, **Nutorch is the same thing as PyTorch**, but in Nushell instead
of Python.

## Why?

Because Nushell is a shell, not just a programming language, this makes it
possible to operate on tensors on your GPU directly from the command line,
making Nutorch one of the most convenient ways to do data analysis if you spend
a lot of time in the terminal.

## Garbage Collection

After installing the plugin, you may want to lengthen the garbage collection
interval in your nushell settings:

```nu
$env.config.plugin_gc = {
  plugins: {
    nutorch: {
      stop_after: 10min
    }
  }
}
```

## TODO

- [x] torch.manual_seed
- [x] tensor.linspace
- [x] torch.randn
- [x] torch.mm
- [x] torch.full
- [x] torch.tensor
- [x] torch.mul
- [x] torch.add
- [x] torch.sub
- [x] torch.div
- [x] torch.neg
- [x] torch.gather
- [x] torch.squeeze
- [x] torch.unsqueeze
- [ ] fix broadcasting logic for add
- [ ] fix broadcasting logic for sub
- [ ] fix broadcasting logic for mul
- [ ] fix broadcasting logic for div
- [ ] torch... everything else
- [x] add autograd setting to torch.tensor
- [x] add autograd setting to torch.randn
- [x] add autograd setting to torch.full
- [x] add autograd setting to torch.mm
- [x] add autograd setting to torch.linspace

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.
