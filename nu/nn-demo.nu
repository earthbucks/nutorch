plugin use torch

# Set random seed for reproducibility
torch manual_seed 42

def generate_data [n_samples: int = 300, centers: int = 3, cluster_std: float = 0.7, skew_factor: float = 0.3] {
  return 5
}

let res = generate_data 300 3 0.7 0.3
print $res
