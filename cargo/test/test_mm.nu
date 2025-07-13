use std assert
use std/testing *

@test
def "MM test 1" [] {
  let input_data = $in
  let result1 = [[1] [2]] | torch tensor | torch mm ([[1 2]] | torch tensor) | torch value
  assert (($result1 | get 0 | get 0) == 1)
}

@test
def "MM test 2" [] {
  let input_data = $in
  let result2 = torch mm ([[1 2]] | torch tensor) ([[1] [2]] | torch tensor) | torch value
  assert (($result2 | get 0 | get 0) == 5)
}
