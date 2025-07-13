use std assert
use std/testing *

@test
def "Test add 1" [] {
  let input_dat = $in
  let result1 = ([1] | torch tensor) | torch add ([2] | torch tensor) | torch value | get 0
  assert ($result1 == 3)
}

@test
def "Test add 2" [] {
  let input_dat = $in
  let result2 = torch add ([1] | torch tensor) ([2] | torch tensor) | torch value | get 0
  assert ($result2 == 3)
}

@test
def "Test add 3" [] {
  let input_dat = $in
  let result3 = ([1 2 3] | torch tensor) | torch add ([4 5 6] | torch tensor) | torch value | get 2
  assert ($result3 == 9)
}
