# Beautiful

Beautiful (`beautiful.nu`) is a Nushell module that provieds convenience methods
for generating [catpuccin](https://catppuccin.com/)-themed plots for
[Termplot](https://termplot.com) and
[Plotly.js](https://plotly.com/javascript/).

Beautiful is specifically designed to work with [Nutorch](https://nutorch.com)
to make it easy to create beautiful plots of machine learning data analysis
results.

## Installation

To install, first, install [Nushell](https://www.nushell.sh/), and
[pnpm](https://pnpm.io/).

Then, run the following command:

```nu
pnpm add beautiful.nu
```

This will install the `beautiful.nu` module in your local project folder.

Now, you can load beautiful from your shell like this:

```nu
source node_modules/beautiful.nu
```

## Usage

If `x` and `y` are two vectors of the same length, you can plot them like this:

```nu
[{x: $x y: $y}] | beautiful scatter
```

## TODO

Support:

- [x] scatter plots
- [x] contour plots
- [x] line plots
- [ ] all other types of plots

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.
