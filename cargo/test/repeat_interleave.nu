plugin use torch

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def print_success [msg] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/repeat_interleave - " + $msg)
}
def print_failure [msg] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/repeat_interleave - " + $msg)
}

# ---------------------------------------------------------------------------
# Test 1: scalar repeats   ($x via pipe, repeats=2)
# ---------------------------------------------------------------------------
let x = ([1 2 3] | torch tensor)
let r1 = ($x | torch repeat_interleave 2 | torch value)
if ($r1 == [1 1 2 2 3 3]) {
  print_success "scalar repeat"
} else {
  print_failure "scalar repeat wrong ($r1)"; error make {msg: "ri scalar"}
}

# ---------------------------------------------------------------------------
# Test 2: tensor repeats   ($x via pipe, repeats tensor as arg)
# ---------------------------------------------------------------------------
let rep = ([1 2 3] | torch tensor --dtype int64)
let r2 = ($x | torch repeat_interleave $rep | torch value)
let exp2 = [1 2 2 3 3 3]
if ($r2 == $exp2) {
  print_success "tensor repeat"
} else {
  print_failure "tensor repeat wrong ($r2)"; error make {msg: "ri tensor"}
}

# ---------------------------------------------------------------------------
# Test 3: scalar repeats with dim flag (2-D tensor)
# ---------------------------------------------------------------------------
let m = ([[1 2] [3 4]] | torch tensor)
let r3 = ($m | torch repeat_interleave 2 --dim 0 | torch shape)
if ($r3 == [4 2]) {
  print_success "dim flag repeat"
} else {
  print_failure "dim flag wrong shape ($r3)"; error make {msg: "ri dim"}
}
