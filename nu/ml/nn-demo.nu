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
  x: ($X | torch value | each { |xy| $xy | get 0 }) # Extract first column as x values
  y: ($X | torch value | each { |xy| $xy | get 1 }) # Extract second column as y values
  # marker: {
  #   color: ($y | torch value)
  # }
} | beautiful scatter | to json | termplot
# for i in (seq 0 (($X_value | length) - 1)) {
#   let x = $X_value | get $i
#   print ($x | into string)
# }
# print ($X | torch value) # Print the generated data
# print ($y | torch value) # Print the generated labels
