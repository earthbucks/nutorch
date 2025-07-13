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

# def run-tests [] {
#   print "Running tests..."

#   # Find all test functions (those with "test" attribute)
#   let test_commands = (
#     scope commands
#       | where { |cmd| $cmd.attributes | any { |attr| $attr.name == "test" } }
#       | get name
#   )

#   mut results = []

#   for test in $test_commands {
#     print $"Running test: ($test)"

#     let result = do ($test)
#     # let re = (try {
#     #   do ($test)   # run the test function
#     #   "pass"
#     # } catch {|e|
#     #   "fail"
#     # })

#     $results = ($results | append {
#       name: $test
#       status: $status
#     })
#   }

#   print "Tests completed"
#   $results   # return table
# }

# run-tests | print
