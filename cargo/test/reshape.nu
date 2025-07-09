plugin use torch

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def print_success [msg] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/reshape - " + $msg)
}
def print_failure [msg] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/reshape - " + $msg)
}

# ---------------------------------------------------------------------------
# Test 1 : 1×6 → 2×3
# ---------------------------------------------------------------------------
let v = ([1 2 3 4 5 6] | torch tensor)
let s1 = ($v | torch reshape [2 3] | torch shape)
if ($s1 == [2 3]) {
  print_success "vector to 2×3"
} else {
  print_failure "expected [2 3] got ($s1)"; error make {msg: "reshape 2x3"}
}

# ---------------------------------------------------------------------------
# Test 2 : infer dimension with -1
# ---------------------------------------------------------------------------
let s2 = ($v | torch reshape [3 -1] | torch shape)
if ($s2 == [3 2]) {
  print_success "-1 inference"
} else {
  print_failure "expected [3 2] got ($s2)"; error make {msg: "reshape -1"}
}

# ---------------------------------------------------------------------------
# Test 3 : reshape 2×3 matrix to flat 6
# ---------------------------------------------------------------------------
let m = ([[1 2 3] [4 5 6]] | torch tensor)
let s3 = ($m | torch reshape [6] | torch shape)
if ($s3 == [6]) {
  print_success "matrix to flat"
} else {
  print_failure "expected [6] got ($s3)"; error make {msg: "reshape flat"}
}
