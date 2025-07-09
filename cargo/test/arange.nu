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

############################################################
# Test 2 : arange 2 7  ->  [2 3 4 5 6]
############################################################
let r2 = (torch arange 2 7 | torch value)
if ($r2 == [2 3 4 5 6]) {
    print_success "arange 2 7 produced correct tensor"
} else {
    print_failure "expected [2 3 4 5 6] but got ($r2)"
    error make {msg:"arange 2 7 failed"}
}

############################################################
# Test 3 : arange 1 5 0.5 (float)  ->  [1 1.5 2 2.5 3 3.5 4 4.5]
############################################################
let r3 = (torch arange 1 5 0.5 --dtype float32 | torch value)
let expected3 = [1 1.5 2 2.5 3 3.5 4 4.5]

if ($r3 == $expected3) {
    print_success "arange 1 5 0.5 produced correct tensor"
} else {
    print_failure "expected ($expected3) but got ($r3)"
    error make {msg:"arange 1 5 0.5 failed"}
}
