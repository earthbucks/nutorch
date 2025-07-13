use std assert
use std/testing *

@test
def "Stack test 1" [] {
  let input_data = $in
  let t1 = ([[1 2] [3 4]] | torch tensor)
  let t2 = ([[5 6] [7 8]] | torch tensor)

  let res0 = ([$t1 $t2] | torch stack --dim 0 | torch value)
  let exp0 = [
    [[1 2] [3 4]]
    [[5 6] [7 8]]
  ]

  assert ($res0 == $exp0)
}

@test
def "Stack test 2" [] {
  let input_data = $in
  let t1 = ([[1 2] [3 4]] | torch tensor)
  let t2 = ([[5 6] [7 8]] | torch tensor)

  let res1 = (torch stack [$t1 $t2] --dim 1 | torch value)
  let exp1 = [
    [[1 2] [5 6]]
    [[3 4] [7 8]]
  ]

  assert ($res1 == $exp1)
}
