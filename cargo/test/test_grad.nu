use std assert
use std/testing *

@test
def "Test grad 1" [] {
  let w = (torch full [1] 2 --requires_grad true)
  [$w] | torch zero_grad # clear grads
  let loss = ($w | torch mean) # dummy scalar loss
  $loss | torch backward

  let gid = ($w | torch grad)
  assert ($gid != null)
}

@test
def "Test grad 2" [] {
  let v = (torch full [1] 7 --requires_grad true)
  let gnull = (torch grad $v)
  assert ($gnull == null)
}

@test
def "Test grad 3" [] {
  let xval = 0.5
  let x = (torch full [1] $xval --requires_grad true)

  [$x] | torch zero_grad
  let loss2 = ($x | torch sin | torch mean)
  $loss2 | torch backward

  let g_id = ($x | torch grad)
  assert ($g_id != null)
}

@test
def "Test grad 4" [] {
  let xval = 0.5
  let x = (torch full [1] $xval --requires_grad true)

  [$x] | torch zero_grad
  let loss2 = ($x | torch sin | torch mean)
  $loss2 | torch backward

  let g_id = ($x | torch grad)

  let grad_val = ($g_id | torch value | get 0)
  let expected = ($xval | math cos)

  # allow tiny numerical tolerance
  let diff = (if ($grad_val > $expected) { $grad_val - $expected } else { $expected - $grad_val })
  assert ($diff < 1e-6)
}
