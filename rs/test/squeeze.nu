plugin use torch

# Function to print SUCCESS in green with an uncolored message
def print_success [message: string] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/squeeze - " + $message)
}

# Function to print FAILURE in red with an uncolored message
def print_failure [message: string] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/squeeze - " + $message)
}

let result1 = (torch full [1 2 3] 1 | torch squeeze 0) | torch shape | get 0
if ($result1 == 2) {
  print_success "Squeeze test passed"
} else {
  print_failure "Squeeze test failed: Expected [3], got $result1"
  error make {msg: "Squeeze test failed: Expected [3], got $result1"}
}
