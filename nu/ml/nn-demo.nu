plugin use torch

# Set random seed for reproducibility
torch manual_seed 42

def generate_data [
  --n_samples: int = 300 # Number of samples to generate
  --centers: int = 3 # Number of cluster centers
  --cluster_std: float = 0.7 # Standard deviation of clusters
  --skew_factor: float = 0.3 # Skew factor for data distribution
] {
  # Your logic here (currently returns 5 as placeholder)
  let n_samples_per_class: int = ($n_samples // $centers)
  let X_list: list<string> = [] # nutorch tensors have string ids
  let y_list: list<string> = [] # nutorch tensors have string ids

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
    # points = torch mm ((points 
    # points: torch.Tensor = (
    #     torch.randn(n_samples_per_class, 2) * cluster_std + blob_centers[i]
    # )
  }

  return 5
}

# Call with named arguments (flags)
let res = (generate_data --n_samples 300 --centers 3 --cluster_std 0.7 --skew_factor 0.3)
print $res

# Call with some defaults
let res2 = (generate_data --n_samples 200)
print $res2 # Uses defaults for centers=3, cluster_std=0.7, skew_factor=0.3
