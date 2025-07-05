plugin use torch

# Function to print SUCCESS in green with an uncolored message
def print_success [message: string] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test-tensor - " + $message)
}

# Function to print FAILURE in red with an uncolored message
def print_failure [message: string] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test-tensor - " + $message)
}

let result1 = ([1] | torch tensor) | torch mul ([2] | torch tensor) | torch value | get 0
if ($result1 == 2) {
  print_success "Multiplication test passed"
} else {
  print_failure "Multiplication test failed: Expected 3, got $result1"
  error make {msg: "Multiplication test failed: Expected 3, got $result1"}
}

let result2 = torch mul ([1] | torch tensor) ([2] | torch tensor) | torch value | get 0
if ($result2 == 2) {
  print_success "Multiplication with tensors test passed"
} else {
  print_failure "Multiplication with tensors test failed: Expected 3, got $result2"
  error make {msg: "Multiplication with tensors test failed: Expected 3, got $result2"}
}

let result3 = ([1 2 3] | torch tensor) | torch mul ([4 5 6] | torch tensor) | torch value | get 2
if ($result3 == 18) {
  print_success "Multiplication with list input test passed"
} else {
  print_failure "Multiplication with list input test failed: Expected 9, got $result3"
  error make {msg: "Multiplication with list input test failed: Expected 9, got $result3"}
}


