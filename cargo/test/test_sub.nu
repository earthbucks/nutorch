use std assert
use std/testing *

@test
def "Subtraction test" [] {
  let input_data = $in
  let result1 = ([1] | torch tensor) | torch sub ([2] | torch tensor) | torch value | get 0
  assert ($result1 == -1)
}

@test
def "Subtraction test 2" [] {
  let input_data = $in
  let result2 = torch sub ([1] | torch tensor) ([2] | torch tensor) | torch value | get 0
  assert ($result2 == -1)
}

@test
def "Subtraction test 3" [] {
  let input_data = $in
  let result3 = ([1 2 3] | torch tensor) | torch sub ([4 5 6] | torch tensor) | torch value | get 2
  assert ($result3 == -3)
}
