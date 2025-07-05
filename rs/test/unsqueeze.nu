plugin use torch

# Function to print SUCCESS in green with an uncolored message
def print_success [message: string] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/unsqueeze - " + $message)
}

# Function to print FAILURE in red with an uncolored message
def print_failure [message: string] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/unsqueeze - " + $message)
}

let result1 = (torch full [2 3] 1 | torch unsqueeze 0) | torch shape | get 0
if ($result1 == 1) {
  print_success "Unsqueeze test passed"
} else {
  print_failure "Unsqueeze test failed: Expected [3], got $result1"
  error make {msg: "Unsqueeze test failed: Expected [3], got $result1"}
}
