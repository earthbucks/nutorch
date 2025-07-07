plugin use torch

# Function to print SUCCESS in green with an uncolored message
def print_success [message: string] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/backward - " + $message)
}

# Function to print FAILURE in red with an uncolored message
def print_failure [message: string] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/backward - " + $message)
}

let p = (torch full [5] 1 --requires_grad true)
let loss = ($p | torch sin | torch mean)
$p | torch zero_grad # make sure grad starts undefined
$loss | torch backward # populate grad

# a gradient-aware op should now work: do one step and value must change
let before = ($p | torch value | get 0)
[$p] | torch sgd_step --lr 0.2
let after = ($p | torch value | get 0)

if ($after != $before) {
  print_success "Gradient produced and parameter changed (before $before → after $after)"
} else {
  print_failure "Parameter unchanged; backward may have failed"
  error make {msg: "backward-grad-defined failed"}
}
