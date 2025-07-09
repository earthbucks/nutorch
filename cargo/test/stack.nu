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

###############################################################################
# prepare two identical-shape tensors
###############################################################################
let t1 = ([[1 2] [3 4]] | torch tensor)
let t2 = ([[5 6] [7 8]] | torch tensor)

###############################################################################
# Test 1: stack along dim 0   → shape [2,2,2]
###############################################################################
let res0 = ([$t1 $t2] | torch stack --dim 0 | torch value)
let exp0 = [[[1 2] [3 4]]
            [[5 6] [7 8]]]

if ($res0 == $exp0) {
    print_success "stack dim 0 produced correct tensor"
} else {
    print_failure "stack dim 0 expected ($exp0) but got ($res0)"
    error make {msg:"stack dim 0 failed"}
}

###############################################################################
# Test 2: stack along dim 1   → shape [2,2,2]
###############################################################################
let res1 = (torch stack [$t1 $t2] --dim 1 | torch value)
let exp1 = [[[1 2] [5 6]]
            [[3 4] [7 8]]]

if ($res1 == $exp1) {
    print_success "stack dim 1 produced correct tensor"
} else {
    print_failure "stack dim 1 expected ($exp1) but got ($res1)"
    error make {msg:"stack dim 1 failed"}
}
