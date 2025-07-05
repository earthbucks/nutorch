# test-maximum.nu - Test script for torch maximum command (element-wise maximum)
def print_success [message: string] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/maximum - " + $message)
}

# Function to print FAILURE in red with an uncolored message
def print_failure [message: string] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/maximum - " + $message)
}

let result1 = (torch full [2,3] 1) | torch maximum (torch full [2,3] 2) | torch value | get 0 | get 0
if ($result1 == 2) {
  print_success "Maximum test passed"
} else {
  print_failure "Maximum test failed: Expected 2, got $result1"
  error make {msg: "Maximum test failed: Expected 2, got $result1"}
}

let result2 = torch maximum (torch full [2,3] 1) (torch full [2,3] 2) | torch value | get 0 | get 0
if ($result2 == 2) {
  print_success "Maximum with tensors test passed"
} else {
  print_failure "Maximum with tensors test failed: Expected 2, got $result2"
  error make {msg: "Maximum with tensors test failed: Expected 2, got $result2"}
}
