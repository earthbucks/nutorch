# test-maximum.nu - Test script for torch maximum command (element-wise maximum)
use std assert
use std/testing *

@test
def "Test pipeline input for maximum" [] {
  let input_data = $in
  let result1 = (torch full [2 3] 1) | torch maximum (torch full [2 3] 2) | torch value | get 0 | get 0
}

@test
def "Test two arguments and no pipeline for maximum" [] {
  let input_data = $in
  (torch maximum (torch full [2 3] 1) (torch full [2 3] 2) | torch value | get 0 | get 0)
}

@test
def "test-incompatible-shapes-expect-error" [] {
  let input_data = $in
  try {
    let result = (torch full [2] 1) | torch maximum (torch full [2 3] 2) | torch value
    error make {msg: "Expected error for incompatible shapes, but got result: $result3"}
  } catch {
    # good, failure expected
  }
}
