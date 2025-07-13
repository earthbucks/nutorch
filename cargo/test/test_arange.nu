use std assert
use std/testing *

@test
def "Arange test 1" [] {
  let r1 = (torch arange 5 | torch value)
  assert ($r1 == [0 1 2 3 4])
}

@test
def "Arange test 2" [] {
  let r2 = (torch arange 2 7 | torch value)
  assert ($r2 == [2 3 4 5 6])
}

@test
def "Arange test 3" [] {
  let r3 = (torch arange 1 5 0.5 --dtype float32 | torch value)
  let expected3 = [1 1.5 2 2.5 3 3.5 4 4.5]
  assert ($r3 == $expected3)
}
