plugin use torch

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def print_success [msg] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/t - " + $msg)
}
def print_failure [msg] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/t - " + $msg)
}

let result1 = [[1 2] [3 4]] | torch tensor | torch t | torch value
if ($result1 == [[1 3] [2 4]]) {
  print_success "Transpose test passed"
} else {
  print_failure "Transpose test failed: Expected [[1 3] [2 4]], got $result1"
  error make {msg: "Transpose test failed: Expected [[1 3] [2 4]], got $result1"}
}
