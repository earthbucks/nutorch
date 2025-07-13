use std assert
use std/testing *

@test
def "Log softmax 1" [] {
  let input_data = $in
  let result1 = torch tensor ([1 2 3]) | torch log_softmax --dim 0 | torch value | get 0 | math round
  assert ($result1 == -2)
}

@test
def "Log softmax 2" [] {
  let input_data = $in
  let result2 = torch log_softmax ([1 2 3] | torch tensor) --dim 0 | torch value | get 0 | math round
  assert ($result2 == -2)
}
