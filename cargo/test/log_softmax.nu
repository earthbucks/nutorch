plugin use torch

# Function to print SUCCESS in green with an uncolored message
def print_success [message: string] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/log_softmax - " + $message)
}

# Function to print FAILURE in red with an uncolored message
def print_failure [message: string] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/log_softmax - " + $message)
}

let result1 = torch tensor ([1 2 3]) | torch log_softmax --dim 0 | torch value | get 0 | math round
if ($result1 == -2) {
  print_success "Log softmax test passed"
} else {
  print_failure "Log softmax test failed: Expected -2, got $result1"
  error make {msg: "Log softmax test failed: Expected -2, got $result1"}
}

let result2 = torch log_softmax ([1 2 3] | torch tensor) --dim 0 | torch value | get 0 | math round
if ($result2 == -2) {
  print_success "Log softmax with tensors test passed"
} else {
  print_failure "Log softmax with tensors test failed: Expected -2, got $result2"
  error make {msg: "Log softmax with tensors test failed: Expected -2, got $result2"}
}
