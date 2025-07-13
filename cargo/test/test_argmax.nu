use std assert
use std/testing *

@test
def "Test argmax 1" [] {
  let v = ([1 3 2] | torch tensor) # max at index 1
  let idx1 = ($v | torch argmax | torch value)
  assert ($idx1 == 1)
}

@test
def "Test argmax 2" [] {
  let m = ([[1 5] [7 0]] | torch tensor)
  let out_id = (torch argmax $m --dim 1 --keepdim true)
  let res2 = ($out_id | torch value)
  let exp2 = [[1] [0]] # shape [2,1]
  assert ($res2 == $exp2)
}
