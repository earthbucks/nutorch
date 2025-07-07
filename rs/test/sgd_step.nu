plugin use torch

# Function to print SUCCESS in green with an uncolored message
def print_success [message: string] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/sgd_step - " + $message)
}

# Function to print FAILURE in red with an uncolored message
def print_failure [message: string] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/sgd_step - " + $message)
}

# 1. parameter with grad
let w = (torch full [1] 5 --requires_grad true)

# 2. clear old grads (there are none yet, but mimics real loop)
[$w] | torch zero_grad

# 3. forward pass + scalar loss:  loss = mean( sin(w) )
let loss = ($w | torch sin | torch mean)

# 4. backward
$loss | torch backward

# 5. SGD update (lr = 0.1)
[$w] | torch sgd_step --lr 0.1

# 6. inspect new parameter value (should be < 5)
let result = $w | torch value | get 0
if ($result < 5) {
  print_success "SGD step test passed: Parameter updated successfully"
} else {
  print_failure "SGD step test failed: Expected value < 5, got $result"
  error make {msg: "SGD step test failed: Expected value < 5, got $result"}
}
