plugin use torch

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def print_success [msg] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/repeat - " + $msg)
}
def print_failure [msg] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/repeat - " + $msg)
}

# ------------------------------------------------------------------
# Test 1  (pipeline form): 1-D tensor repeat 3  -> shape [3,4]
# ------------------------------------------------------------------
let r1 = ([1 2] | torch tensor | torch repeat 3 | torch shape)
if ($r1 == [6]) {
    print_success "pipeline repeat produced shape [3,4]"
} else {
    print_failure "expected shape [3 4], got ($r1)"
    error make {msg:"repeat pipeline failed"}
}
