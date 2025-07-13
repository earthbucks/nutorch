use std assert
use std/testing *

@test
def "Mul test 1" [] {
  let input_data = $in
  let result1 = ([1] | torch tensor) | torch mul ([2] | torch tensor) | torch value | get 0
  assert ($result1 == 2)
}

@test
def "Mul test 2" [] {
  let input_data = $in
  let result2 = torch mul ([1] | torch tensor) ([2] | torch tensor) | torch value | get 0
  assert ($result2 == 2)
}

@test
def "Mul test 3" [] {
  let input_data = $in
  let result3 = ([1 2 3] | torch tensor) | torch mul ([4 5 6] | torch tensor) | torch value | get 2
  assert ($result3 == 18)
}
