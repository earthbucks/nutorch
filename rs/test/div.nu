plugin use torch

# Function to print SUCCESS in green with an uncolored message
def print_success [message: string] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test-tensor - " + $message)
}

# Function to print FAILURE in red with an uncolored message
def print_failure [message: string] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test-tensor - " + $message)
}

let result1 = ([4] | torch tensor) | torch div ([2] | torch tensor) | torch value | get 0
if ($result1 == 2) {
  print_success "Division test passed"
} else {
  print_failure "Division test failed: Expected 2, got $result1"
  error make {msg: "Division test failed: Expected 2, got $result1"}
}

let result2 = torch div ([4] | torch tensor) ([2] | torch tensor) | torch value | get 0
if ($result2 == 2) {
  print_success "Division with tensors test passed"
} else {
  print_failure "Division with tensors test failed: Expected 2, got $result2"
  error make {msg: "Division with tensors test failed: Expected 2, got $result2"}
}

let result3 = ([1 2 9] | torch tensor) | torch div ([4 5 3] | torch tensor) | torch value | get 2
if ($result3 == 3) {
  print_success "Division with list input test passed"
} else {
  print_failure "Division with list input test failed: Expected 9, got $result3"
  error make {msg: "Division with list input test failed: Expected 9, got $result3"}
}


