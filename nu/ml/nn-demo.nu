plugin use torch
source ~/dev/termplot/nu/beautiful.nu

# Set random seed for reproducibility
torch manual_seed 42
# torch manual_seed ( 42 * 2 )

def generate_data [
  --n_samples: int = 300 # Number of samples to generate
  --centers: int = 3 # Number of cluster centers
  --cluster_std: float = 0.7 # Standard deviation of clusters
  --skew_factor: float = 0.3 # Skew factor for data distribution
] {
  let n_samples_per_class: int = ($n_samples // $centers)
  mut X_list: list<string> = [] # nutorch tensors have string ids
  mut y_list: list<string> = [] # nutorch tensors have string ids

  # let blob_centers = [([0.0 0.0] | torch tensor) ([3.0 0.0] | torch tensor) ([1.5 2.5] | torch tensor)]
  let blob_centers: list<string> = [
    (torch tensor [0.0 0.0])
    (torch tensor [3.0 0.0])
    (torch tensor [1.5 2.5])
  ]

  for i in (seq 0 ($centers - 1)) {
    mut points: string = (torch randn $n_samples_per_class 2) | torch mul (torch tensor $cluster_std) | torch add ($blob_centers | get $i)
    if $i == 1 or $i == 2 {
      let skew_matrix: string = (torch tensor [[1.0 ($skew_factor * ($i - 1))] [($skew_factor * ($i - 1)) 1.0]])
      $points = (torch mm $points $skew_matrix)
    }
    let labels: string = torch full [$n_samples_per_class] $i --dtype 'int64'
    $X_list = $X_list | append $points
    $y_list = $y_list | append $labels
  }

  let X: string = $X_list | torch cat --dim 0
  let y: string = $y_list | torch cat --dim 0

  {X: $X y: $y}
}

# Call with named arguments (flags)
let res = (generate_data --n_samples 30 --centers 3 --cluster_std 0.7 --skew_factor 0.3)
let X: string = $res.X
let y: string = $res.y
{
  x: ($X | torch value | each {|xy| $xy | get 0 }) # Extract first column as x values
  y: ($X | torch value | each {|xy| $xy | get 1 }) # Extract second column as y values
  marker: {
    color: ($y | torch div ($y | torch max) | torch value) # Normalize y values for color
    colorscale: (beautiful colorscale 3)
    # colorscale: [
    #   [
    #     0
    #     "#a6e3a1"
    #   ]
    #   [
    #     0.15384615384615385
    #     "#94e2d5"
    #   ]
    #   [
    #     0.3076923076923077
    #     "#89dceb"
    #   ]
    #   [
    #     0.46153846153846156
    #     "#74c7ec"
    #   ]
    #   [
    #     0.6153846153846154
    #     "#89b4fa"
    #   ]
    #   [
    #     0.7692307692307693
    #     "#b4befe"
    #   ]
    #   [
    #     0.9230769230769231
    #     "#f5e0dc"
    #   ]
    #   [
    #     1.076923076923077
    #     "#f2cdcd"
    #   ]
    #   [
    #     1.2307692307692308
    #     "#f5c2e7"
    #   ]
    #   [
    #     1.3846153846153846
    #     "#cba6f7"
    #   ]
    #   [
    #     1.5384615384615385
    #     "#f38ba8"
    #   ]
    #   [
    #     1.6923076923076925
    #     "#eba0ac"
    #   ]
    #   [
    #     1.8461538461538463
    #     "#fab387"
    #   ]
    #   [
    #     2
    #     "#f9e2af"
    #   ]
    # ]
    # colorscale: [
    #   [0.000, "rgb(68, 1, 84)"],
    #   [0.111, "rgb(72, 40, 120)"],
    #   [0.222, "rgb(62, 74, 137)"],
    #   [0.333, "rgb(49, 104, 142)"],
    #   [0.444, "rgb(38, 130, 142)"],
    #   [0.556, "rgb(31, 158, 137)"],
    #   [0.667, "rgb(53, 183, 121)"],
    #   [0.778, "rgb(109, 205, 89)"],
    #   [0.889, "rgb(180, 222, 44)"],
    #   [1.000, "rgb(253, 231, 37)"]
    # ]
    # colorscale: [
    #   [
    #     0.000
    #     "rgb(166, 227, 161)"
    #   ]
    #   [
    #     0.153
    #     "rgb(148, 226, 213)"
    #   ]
    #   [
    #     0.307
    #     "rgb(137, 220, 235)"
    #   ]
    #   [
    #     0.461
    #     "rgb(116, 199, 236)"
    #   ]
    #   [
    #     0.615
    #     "rgb(137, 180, 250)"
    #   ]
    #   [
    #     0.769
    #     "rgb(180, 190, 254)"
    #   ]
    #   [
    #     0.923
    #     "rgb(245, 224, 220)"
    #   ]
    #   [
    #     1.000
    #     "rgb(242, 205, 205)"
    #   ]
    # ]
  }
# } | beautiful scatter | to json
} | beautiful scatter | to json | termplot
# asdf. asdf. asdf.
