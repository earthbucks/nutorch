use std assert
use std/testing *

@test
def "Test detach" [] {
  let input_data = $in
  let tensor = [1 2] | torch tensor
  let detached_tensor = $tensor | torch detach
  assert (($tensor | torch value) == ($detached_tensor | torch value) and ($tensor != $detached_tensor))
}
