plugin use torch

# Function to print SUCCESS in green with an uncolored message
def print_success [message: string] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test-tensor - " + $message)
}

# Function to print FAILURE in red with an uncolored message
def print_failure [message: string] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test-tensor - " + $message)
}

let result1 = ([1] | torch tensor) | torch sub ([2] | torch tensor) | torch value | get 0
if ($result1 == -1) {
  print_success "Subtraction test passed"
} else {
  print_failure "Subtraction test failed: Expected -1, got $result1"
  error make {msg: "Subtraction test failed: Expected -1, got $result1"}
}

let result2 = torch sub ([1] | torch tensor) ([2] | torch tensor) | torch value | get 0
if ($result2 == -1) {
  print_success "Subtraction with tensors test passed"
} else {
  print_failure "Subtraction with tensors test failed: Expected -1, got $result2"
  error make {msg: "Subtraction with tensors test failed: Expected -1, got $result2"}
}

let result3 = ([1 2 3] | torch tensor) | torch sub ([4 5 6] | torch tensor) | torch value | get 2
if ($result3 == -3) {
  print_success "Subtraction with list input test passed"
} else {
  print_failure "Subtraction with list input test failed: Expected -3, got $result3"
  error make {msg: "Subtraction with list input test failed: Expected -3, got $result3"}
}


