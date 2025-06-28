# Nutorch

https://github.com/user-attachments/assets/3d1c72e7-ab8b-4053-b215-097cfdff626c

**GPU-powered machine learning on the command line.**

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
