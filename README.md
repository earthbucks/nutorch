# Nutorch

<img src="./raw-images/nutorch-3d.png" width="200" height="200" alt="Nutorch" />

**GPU-powered AI / ML on the command line.**

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file
for details.
