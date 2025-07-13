use std assert
use std/testing *

@test
def "Test repeat interleave 1" [] {
  let x = ([1 2 3] | torch tensor)
  let r1 = ($x | torch repeat_interleave 2 | torch value)
  assert ($r1 == [1 1 2 2 3 3])
}

@test
def "Test repeat interleave 2" [] {
  let x = ([1 2 3] | torch tensor)
  let rep = ([1 2 3] | torch tensor --dtype int64)
  let r2 = ($x | torch repeat_interleave $rep | torch value)
  let exp2 = [1 2 2 3 3 3]
  assert ($r2 == $exp2)
}

@test
def "Test repeat interleave 3" [] {
  let m = ([[1 2] [3 4]] | torch tensor)
  let r3 = ($m | torch repeat_interleave 2 --dim 0 | torch shape)
  assert ($r3 == [4 2])
}
