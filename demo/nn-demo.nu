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
]: [nothing -> record<X: string, y: string>] {
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
      let center = ($blob_centers | get $i)
      let skew = (
        torch tensor [
          [1.0 ($skew_factor * ($i - 1))]
          [($skew_factor * ($i - 1)) 1.0]
        ]
      )

      $points = (
        $points
        | torch sub $center # pts - center
        | torch mm $skew # * skew
        | torch add $center # + center
      )
    }
    let labels: string = torch full [$n_samples_per_class] $i --dtype 'int64'
    $X_list = $X_list | append $points
    $y_list = $y_list | append $labels
  }

  let X: string = $X_list | torch cat --dim 0
  let y: string = $y_list | torch cat --dim 0

  {X: $X y: $y}
}

def cross_entropy_loss [
  --logits: string # tensor id of model outputs
  --targets: string # tensor id of target labels
]: [nothing -> string] {
  let logp = $logits | torch log_softmax --dim 1
  # print $"logp: ($logp | torch mean | torch value)"
  let loss = $logp | torch gather 1 ($targets | torch unsqueeze 1) | torch squeeze 1 | torch mean | torch neg
  $loss
}

def model_init [
  --input_size: int = 2 # Number of input features
  --hidden_size: int = 20 # Number of hidden units
  --output_size: int = 3 # Number of output classes
]: [nothing -> record<w1: string, b1: string, w2: string, b2: string>] {
  {
    w1: (torch randn $hidden_size $input_size --requires_grad true)
    b1: (torch randn $hidden_size --requires_grad true)
    w2: (torch randn $output_size $hidden_size --requires_grad true)
    b2: (torch randn $output_size --requires_grad true)
  }
}

def model_get_parameters [
  --model: record<w1: string, b1: string, w2: string, b2: string>
]: [nothing -> list<string>] {
  [
    $model.w1
    $model.b1
    $model.w2
    $model.b2
  ]
}

def model_forward_pass [
  --model: record<w1: string, b1: string, w2: string, b2: string>
]: [string -> string] {
  # input tensor id -> output tensor id
  torch mm ($model.w1 | torch t) # Matrix multiplication with input and first layer weights
  | torch add $model.b1 # Add bias for first layer
  | torch maximum ([0.0] | torch tensor) # ReLU activation
  | torch mm ($model.w2 | torch t) # Matrix multiplication with second layer weights
  | torch add $model.b2 # Add bias for second layer
}

def train [
  --model: record<w1: string, b1: string, w2: string, b2: string>
  --X: string # Input tensor id
  --y: string # Target tensor id
  --epochs: int = 1000
  --lr: float = 0.1
  --record_every: int = 100
]: [nothing -> record<model: record<w1: string, b1: string, w2: string, b2: string>, losses: list<number>, steps: list<number>>] {
  mut losses: list<number> = []
  mut steps: list<number> = []
  let ps = model_get_parameters --model $model

  for epoch in (seq 0 ($epochs - 1)) {
    # forward and loss
    let logits = $X | model_forward_pass --model $model
    let loss = cross_entropy_loss --logits $logits --targets $y

    # zero existing grads, back-prop, SGD upadate
    for p in $ps {
      $p | torch zero_grad
    }
    $loss | torch backward
    torch sgd_step $ps --lr $lr

    if ($epoch + 1) mod $record_every == 0 {
      $losses = $losses | append ($loss | torch value)
      $steps = $steps | append ($epoch + 1)
      print $"epoch: ($epoch + 1)/($epochs), loss: (($loss | torch value | math round --precision 4))"
    }
  }

  return {
    model: $model
    losses: $losses
    steps: $steps
  }
}

def plot_raw_data [res: record<X: string, y: string>] {
  # Call with named arguments (flags)
  let X: string = $res.X
  let y: string = $res.y
  let X_value = $X | torch value
  let y_value = $y | torch value
  [
    {
      x: ($X_value | enumerate | each {|xy| if (($y_value | get $xy.index) == 0) { $xy.item.0 } })
      y: ($X_value | enumerate | each {|xy| if ($y_value | get $xy.index) == 0 { $xy.item.1 } })
    }
    {
      x: ($X_value | enumerate | each {|xy| if (($y_value | get $xy.index) == 1) { $xy.item.0 } })
      y: ($X_value | enumerate | each {|xy| if ($y_value | get $xy.index) == 1 { $xy.item.1 } })
    }
    {
      x: ($X_value | enumerate | each {|xy| if (($y_value | get $xy.index) == 2) { $xy.item.0 } })
      y: ($X_value | enumerate | each {|xy| if ($y_value | get $xy.index) == 2 { $xy.item.1 } })
    }
  ] | beautiful scatter | to json | termplot
}

def plot_loss [
  --losses: list<number> # list of loss values
  --steps: list<number> # list of steps (epochs) corresponding to losses
] {
  # Call with named arguments (flags)
  [
    {
      x: $steps
      y: $losses
    }
  ] | beautiful lines | to json | termplot
}

let raw_data = generate_data --n_samples 300 --centers 3 --cluster_std 0.7 --skew_factor 0.3
plot_raw_data $raw_data

let net = model_init --input_size 2 --hidden_size 20 --output_size 3
let model_res = train --model $net --X $raw_data.X --y $raw_data.y --epochs 3000 --lr 0.1 --record_every 100
plot_loss --losses $model_res.losses --steps $model_res.steps
