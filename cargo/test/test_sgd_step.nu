use std assert
use std/testing *

@test
def "Test sgd_step" [] {
  # 1. parameter with grad
  let w = (torch full [1] 5 --requires_grad true)

  # 2. clear old grads (there are none yet, but mimics real loop)
  [$w] | torch zero_grad

  # 3. forward pass + scalar loss:  loss = mean( sin(w) )
  let loss = ($w | torch sin | torch mean)

  # 4. backward
  $loss | torch backward

  # 5. SGD update (lr = 0.1)
  [$w] | torch sgd_step --lr 0.1

  # 6. inspect new parameter value (should be < 5)
  let result = $w | torch value | get 0
  assert ($result < 5)
}
