use std assert
use std/testing *

@test
def "Div test 1" [] {
  let input_data = $in
  let result1 = ([4] | torch tensor) | torch div ([2] | torch tensor) | torch value | get 0
  assert ($result1 == 2)
}

@test
def "Div test 2" [] {
  let input_data = $in
  let result2 = torch div ([4] | torch tensor) ([2] | torch tensor) | torch value | get 0
  assert ($result2 == 2)
}

@test
def "Div test 3" [] {
  let input_data = $in
  let result3 = ([1 2 9] | torch tensor) | torch div ([4 5 3] | torch tensor) | torch value | get 2
  assert ($result3 == 3)
}
