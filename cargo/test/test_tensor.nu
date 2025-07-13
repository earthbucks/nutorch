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

# # Test 4: Error case - both pipeline and argument provided
# let result4: bool = try {
#   let res = ([1 2 3] | torch tensor [1.0 2.0 3.0])
#   print_failure "Expected an error for conflicting input, but no error occurred"
#   false
# } catch {
#   print_success "Expected error occurred for conflicting input"
#   true
# }

# if not $result4 {
#   error make {msg: "Test 4 failed"}
# }

# if ($result1 and $result2 and $result3 and $result4) {
#   print_success "All tests passed successfully!"
# } else {
#   print_failure "Some tests failed. Please check the output above for details."
# }
