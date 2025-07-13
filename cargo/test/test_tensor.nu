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

# # Test 2: Convert a 1D list to a tensor via argument
# let result2 = try {
#   let res = (torch tensor [1.0 2.0 3.0])
#   if ($res | describe | str contains "string") {
#     print_success "Tensor created via argument"
#     true
#   } else {
#     print_failure "Expected a tensor ID string, got unexpected output"
#     error make {msg: "Test 2 failed: Did not return a tensor ID"}
#     false
#   }
# } catch {
#   print_failure "Unexpected error while creating tensor via argument"
#   error make {msg: "Test 2 failed: Unexpected error"}
# }

# # Test 3: Error case - no input provided
# let result3 = try {
#   torch tensor
#   print_failure "Expected an error for missing input, but no error occurred"
#   error make {msg: "Test 3 failed: No error for missing input"}
#   false
# } catch {
#   print_success "Expected error occurred for missing input"
#   true
# }

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
