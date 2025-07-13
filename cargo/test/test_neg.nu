use std assert
use std/testing *

@test
def "Test neg 1" [] {
  let input_data = $in
  let result1 = ([1] | torch tensor) | torch neg | torch value | get 0
  assert ($result1 == -1)
}

@test
def "Test neg 2" [] {
  let input_data = $in
  let result2 = torch neg ([1] | torch tensor) | torch value | get 0
  assert ($result2 == -1)
}
