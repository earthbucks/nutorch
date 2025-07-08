plugin use torch

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def print_success [msg] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/argmax - " + $msg)
}
def print_failure [msg] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/argmax - " + $msg)
}

###############################################################################
# Test 1 : flattened argmax
###############################################################################
let v = ([1 3 2] | torch tensor) # max at index 1
let idx1 = ($v | torch argmax | torch value)
if ($idx1 == 1) {
  print_success "flatten argmax"
} else {
  print_failure "expected 1 got ($idx1)"; error make {msg: "argmax flat"}
}

###############################################################################
# Test 2 : argmax dim 1 keepdim true
###############################################################################
let m = ([[1 5] [7 0]] | torch tensor)
let out_id = (torch argmax $m --dim 1 --keepdim true)
let res2 = ($out_id | torch value)
let exp2 = [[1] [0]] # shape [2,1]
if ($res2 == $exp2) {
  print_success "dim+keepdim"
} else {
  print_failure "expected ($exp2) got ($res2)"; error make {msg: "argmax dim"}
}
