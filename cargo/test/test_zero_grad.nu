use std assert
use std/testing *

@test
def "Test zero grad" [] {
  let p = (torch full [1] 5 --requires_grad true)
  let loss = ($p | torch mean)
  $loss | torch backward # create grad
  [$p] | torch zero_grad # clear grad

  # run sgd_step with giant lr; value should NOT change (grad == 0)
  let before = ($p | torch value | get 0)
  [$p] | torch sgd_step --lr 10
  let after = ($p | torch value | get 0)

  assert ($before == $after)
}
