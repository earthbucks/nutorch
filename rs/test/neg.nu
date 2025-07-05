plugin use torch

# Function to print SUCCESS in green with an uncolored message
def print_success [message: string] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/neg - " + $message)
}

# Function to print FAILURE in red with an uncolored message
def print_failure [message: string] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/neg - " + $message)
}

let result1 = ([1] | torch tensor) | torch neg | torch value | get 0
if ($result1 == -1) {
  print_success "Negation test passed"
} else {
  print_failure "Negation test failed: Expected -1, got $result1"
  error make {msg: "Negation test failed: Expected -1, got $result1"}
}

let result2 = torch neg ([1] | torch tensor) | torch value | get 0
if ($result2 == -1) {
  print_success "Negation with tensors test passed"
} else {
  print_failure "Negation with tensors test failed: Expected -1, got $result2"
  error make {msg: "Negation with tensors test failed: Expected -1, got $result2"}
}
