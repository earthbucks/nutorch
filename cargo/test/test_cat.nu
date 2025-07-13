use std assert
use std/testing *

@test
def "Cat test 1" [] {
  # Test 1: Concatenate two 2x3 tensors along dimension 0 (rows)
  let t1 = (torch full [2 3] 1) # Shape: [2, 3], filled with 1
  $t1 | torch value # Should show [[1, 1, 1], [1, 1, 1]]
  let t2 = (torch full [2 3] 2) # Shape: [2, 3], filled with 2
  $t2 | torch value # Should show [[2, 2, 2], [2, 2, 2]]
  let result1 = (torch cat [$t1 $t2] --dim 0 | torch value)
  assert ($result1 == [[1 1 1] [1 1 1] [2 2 2] [2 2 2]])
}

@test
def "Cat test 2" [] {
  # Test 2: Concatenate two 2x3 tensors along dimension 1 (columns)
  let t1 = (torch full [2 3] 1) # Shape: [2, 3], filled with 1
  $t1 | torch value # Should show [[1, 1, 1], [1, 1, 1]]
  let t2 = (torch full [2 3] 2) # Shape: [2, 3], filled with 2
  $t2 | torch value # Should show [[2, 2, 2], [2, 2, 2]]
  let t3 = (torch full [2 3] 3) # Shape: [2, 3], filled with 3
  $t3 | torch value # Should show [[3, 3, 3], [3, 3, 3]]
  let result2 = (torch cat [$t1 $t3] --dim 1 | torch value)
  assert ($result2 == [[1 1 1 3 3 3] [1 1 1 3 3 3]])
}

@test
def "Cat test 3" [] {
  # Test 3: Error case - incompatible shapes, expect failure as success
  let t1 = (torch full [2 3] 1) # Shape: [2, 3], filled with 1
  $t1 | torch value # Should show [[1, 1, 1], [1, 1, 1]]
  let t2 = (torch full [2 3] 2) # Shape: [2, 3], filled with 2
  $t2 | torch value # Should show [[2, 2, 2], [2, 2, 2]]
  let t3 = (torch full [2 3] 3) # Shape: [2, 3], filled with 3
  $t3 | torch value # Should show [[3, 3, 3], [3, 3, 3]]
  let t4 = (torch full [2 2] 4) # Shape: [2, 2], filled with 4
  let error_result = try {
    torch cat [$t1 $t4] --dim 0 | torch value
    assert false # expected error
  } catch {
    # If an error occurs, test passes
  }
}
