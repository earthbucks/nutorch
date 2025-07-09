plugin use torch

# Function to print SUCCESS in green with an uncolored message
def print_success [message: string] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/add - " + $message)
}

# Function to print FAILURE in red with an uncolored message
def print_failure [message: string] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/add - " + $message)
}

let result1 = ([1] | torch tensor) | torch add ([2] | torch tensor) | torch value | get 0
if ($result1 == 3) {
  print_success "Addition test passed"
} else {
  print_failure "Addition test failed: Expected 3, got $result1"
  error make {msg: "Addition test failed: Expected 3, got $result1"}
}

let result2 = torch add ([1] | torch tensor) ([2] | torch tensor) | torch value | get 0
if ($result2 == 3) {
  print_success "Addition with tensors test passed"
} else {
  print_failure "Addition with tensors test failed: Expected 3, got $result2"
  error make {msg: "Addition with tensors test failed: Expected 3, got $result2"}
}

let result3 = ([1 2 3] | torch tensor) | torch add ([4 5 6] | torch tensor) | torch value | get 2
if ($result3 == 9) {
  print_success "Addition with list input test passed"
} else {
  print_failure "Addition with list input test failed: Expected 9, got $result3"
  error make {msg: "Addition with list input test failed: Expected 9, got $result3"}
}
