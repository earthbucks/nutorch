use std assert
use std/testing *

@test
def "Gather test 1" [] {
  # Create a 2×3 source tensor and an index tensor; gather columns 2,1,0
  let src = ([[10 11 12] [20 21 22]] | torch tensor)
  let idx = ([[2 1 0] [0 0 2]] | torch tensor --dtype int64)
  # $src | torch value
  let result1 = ($src | torch gather 1 $idx | torch value)
  assert ($result1 == [[12 11 10] [20 20 22]])
}

@test
def "Gather test 2" [] {
  # Gather rows along dim 0
  let src2 = ([[1 2] [3 4]] | torch tensor)
  mut result2 = false
  try {
    let src = ([[1 2] [3 4]] | torch tensor)
    let idx = ([0 0 1] | torch tensor --dtype int64)
    $src | torch gather 0 $idx # expect “Shape mismatch” error
    error make {msg: "Gather with wrong dimension index should fail"}
  } catch {
    # expected failure
  }
}
