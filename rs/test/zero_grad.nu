
plugin use torch

# Function to print SUCCESS in green with an uncolored message
def print_success [message: string] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/zero_grad - " + $message)
}

# Function to print FAILURE in red with an uncolored message
def print_failure [message: string] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/zero_grad - " + $message)
}

let p = (torch full [1] 5 --requires_grad true)
let loss = ($p | torch mean)
$loss | torch backward                # create grad
[$p] | torch zero_grad                # clear grad

# run sgd_step with giant lr; value should NOT change (grad == 0)
let before = ($p | torch value | get 0)
[$p] | torch sgd_step --lr 10
let after = ($p | torch value | get 0)

if ($before == $after) {
    print_success "Gradients cleared, parameter unchanged"
} else {
    print_failure "Parameter changed ($before → $after) — zero_grad failed"
    error make {msg:"zero_grad-pipeline failed"}
}
