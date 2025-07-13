use std assert
use std/testing *

@test
def "Test tensor creation" [] {
  let input_data = $in
  let res = ([1.0 2.0 3.0] | torch tensor)
  assert ($res | describe | str contains "string")
}

@test
def "Convert a 1d list to a tensor via argument" [] {
  let input_data = $in
  let res = (torch tensor [1.0 2.0 3.0])
  assert ($res | describe | str contains "string")
}

@test
def "Error case of no input provided" [] {
  let input_data = $in
  try {
    torch tensor
    error make {msg: "Expected error from no input"}
  } catch {
    # expected
  }
}

@test
def "Expect an error if pipeline and argument both provided" [] {
  let input_data = $in
  try {
    let res = ([1 2 3] | torch tensor [1.0 2.0 3.0])
    error make {msg: "Expected error if pipeline and argument both provided"}
  } catch {
    # expected
  }
}
