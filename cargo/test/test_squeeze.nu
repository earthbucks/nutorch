use std assert
use std/testing *

@test
def "Test squeeze" [] {
  let input_data = $in
  let result1 = (torch full [1 2 3] 1 | torch squeeze 0) | torch shape | get 0
  assert ($result1 == 2)
}
