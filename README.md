# Nutorch

**Data analysis on the command line.**

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
- [ ] torch.add
- [x] torch.sub
- [x] torch.div
- [ ] torch.gather
- [ ] torch.squeeze
- [ ] torch.unsqueeze
- [ ] torch... everything else
- [x] add autograd setting to torch.tensor
- [x] add autograd setting to torch.randn
- [x] add autograd setting to torch.full
- [x] add autograd setting to torch.mm
- [x] add autograd setting to torch.linspace

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.
