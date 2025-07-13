use std assert
use std/testing *

@test
def "Transpose test" [] {
  let result = [[1 2] [3 4]] | torch tensor | torch t | torch value
  assert ($result == [[1 3] [2 4]])
}
