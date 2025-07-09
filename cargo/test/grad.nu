plugin use torch

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def print_success [msg] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/grad - " + $msg)
}
def print_failure [msg] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/grad - " + $msg)
}

# ---------------------------------------------------------------------------
# 1. grad exists after backward
# ---------------------------------------------------------------------------
let w = (torch full [1] 2 --requires_grad true)
[$w] | torch zero_grad              # clear grads
let loss = ($w | torch mean)        # dummy scalar loss
$loss | torch backward

let gid = ($w | torch grad)
if ($gid != null) {
    print_success "grad present after backward"
} else {
    print_failure "grad expected but null"
    error make {msg: "grad present test failed"}
}

# ---------------------------------------------------------------------------
# 2. grad is null when backward not yet run
# ---------------------------------------------------------------------------
let v = (torch full [1] 7 --requires_grad true)
let gnull = (torch grad $v)
if ($gnull == null) {
    print_success "grad correctly null before backward"
} else {
    print_failure "grad unexpectedly defined"
    error make {msg: "grad null test failed"}
}

# ---------------------------------------------------------------------------
# 3. derivative check  :  d/dx sin(x)  ==  cos(x)
# ---------------------------------------------------------------------------
let xval = 0.5
let x    = (torch full [1] $xval --requires_grad true)

[$x] | torch zero_grad
let loss2 = ($x | torch sin | torch mean)
$loss2 | torch backward

let g_id   = ($x | torch grad)
if ($g_id == null) {
    print_failure "sin derivative test: grad is null"
    error make {msg: "sin derivative test failed (null grad)"}
}

let grad_val = ($g_id | torch value | get 0)
let expected = ($xval | math cos)

# allow tiny numerical tolerance
let diff = (if ($grad_val > $expected) { $grad_val - $expected } else { $expected - $grad_val })
if ($diff < 1e-6) {
    print_success "sin derivative test passed (grad=$grad_val  expected=$expected)"
} else {
    print_failure "sin derivative test failed (grad=$grad_val  expected=$expected)"
    error make {msg: "sin derivative mismatch"}
}
