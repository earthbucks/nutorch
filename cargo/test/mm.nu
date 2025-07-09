# test-mm.nu - Test script for torch mm command (matrix multiplication)
plugin use torch

# Function to print SUCCESS in green with an uncolored message
def print_success [message: string] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/mm - " + $message)
}

# Function to print FAILURE in red with an uncolored message
def print_failure [message: string] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/mm - " + $message)
}

let result1 = [[1] [2]] | torch tensor | torch mm ([[1 2]] | torch tensor) | torch value
if (($result1 | get 0 | get 0) == 1) {
  print_success "Matrix multiplication test passed"
} else {
  print_failure "Matrix multiplication test failed: Expected [[5]], got $result1"
  error make {msg: "Matrix multiplication test failed: Expected [[5]], got $result1"}
}

let result2 = torch mm ([[1 2]] | torch tensor) ([[1] [2]] | torch tensor) | torch value
if (($result2 | get 0 | get 0) == 5) {
  print_success "Matrix multiplication with tensors test passed"
} else {
  print_failure "Matrix multiplication with tensors test failed: Expected [[5]], got $result2"
  error make {msg: "Matrix multiplication with tensors test failed: Expected [[5]], got $result2"}
}
