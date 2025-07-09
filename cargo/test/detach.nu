plugin use torch

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def print_success [msg] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/detach - " + $msg)
}
def print_failure [msg] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/detach - " + $msg)
}

let tensor = [1 2] | torch tensor
let detached_tensor = $tensor | torch detach
if (($tensor | torch value) == ($detached_tensor | torch value) and ($tensor != $detached_tensor)) {
  print_success "Detach test passed"
} else {
  print_failure "Detach test failed: Expected detached tensor to have same value as original tensor"
  error make {msg: "Detach test failed: Expected detached tensor to have same value as original tensor"}
}

# let result1 = [[1 2] [3 4]] | torch tensor | torch t | torch value
# if ($result1 == [[1 3] [2 4]]) {
#   print_success "Transpose test passed"
# } else {
#   print_failure "Transpose test failed: Expected [[1 3] [2 4]], got $result1"
#   error make {msg: "Transpose test failed: Expected [[1 3] [2 4]], got $result1"}
# }
