use std assert
use std/testing *

@test
def "Test unsqueeze with pipeline input" [] {
  let input_data = $in
  let result = (torch full [2 3] 1 | torch unsqueeze 0) | torch shape | get 0
  assert ($result == 1)
}
