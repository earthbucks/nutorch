use std assert
use std/testing *

@test
def "Test repeat" [] {
  let r1 = ([1 2] | torch tensor | torch repeat 3 | torch shape)
  assert ($r1 == [6])
}
