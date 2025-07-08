plugin use torch

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def print_success [msg] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/arange - " + $msg)
}
def print_failure [msg] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/arange - " + $msg)
}

############################################################
# Test 1 : arange 5  ->  [0 1 2 3 4]
############################################################
let r1 = (torch arange 5 | torch value)
if ($r1 == [0 1 2 3 4]) {
    print_success "arange 5 produced correct tensor"
} else {
    print_failure "expected [0 1 2 3 4] but got ($r1)"
    error make {msg:"arange 5 failed"}
}
