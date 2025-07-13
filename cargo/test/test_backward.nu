use std assert
use std/testing *

@test
def "Backward test 1" [] {
  let p = (torch full [5] 1 --requires_grad true)
  let loss = ($p | torch sin | torch mean)
  $p | torch zero_grad # make sure grad starts undefined
  $loss | torch backward # populate grad

  # a gradient-aware op should now work: do one step and value must change
  let before = ($p | torch value | get 0)
  [$p] | torch sgd_step --lr 0.2
  let after = ($p | torch value | get 0)

  assert ($after != $before)
}

@test
def "Backward test 2" [] {
  let t = (torch full [2 2] 1 --requires_grad true) # NOT scalar
  let result = (try { $t | torch backward } catch { "err" })

  assert ($result == "err")
}
