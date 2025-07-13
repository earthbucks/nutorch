# test-maximum.nu - Test script for torch maximum command (element-wise maximum)
use std assert
use std/testing *

def print_success [message: string] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/maximum - " + $message)
}

def print_failure [message: string] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/maximum - " + $message)
}

@test
def "test-pipeline-tensor-argument-tensor" [] {
  let result1 = (torch full [2 3] 1) | torch maximum (torch full [2 3] 2) | torch value | get 0 | get 0
  if ($result1 == 2) {
    print_success "Maximum test passed"
  } else {
    print_failure "Maximum test failed: Expected 2, got $result1"
    error make {msg: "Maximum test failed: Expected 2, got $result1"}
  }
}

# @test
# def "test-two-arguments" [] {
#   let result2 = torch maximum (torch full [2 3] 1) (torch full [2 3] 2) | torch value | get 0 | get 0
#   if ($result2 == 2) {
#     print_success "Maximum with tensors test passed"
#   } else {
#     print_failure "Maximum with tensors test failed: Expected 2, got $result2"
#     error make {msg: "Maximum with tensors test failed: Expected 2, got $result2"}
#   }
# }

# @test
# def "test-incompatible-shapes-expect-error" [] {
#   try {
#     let result3 = (torch full [2] 1) | torch maximum (torch full [2 3] 2) | torch value
#     print_failure "Maximum with incompatible shapes test failed: Expected an error, but got result: $result3"
#     error make {msg: "Expected error for incompatible shapes, but got result: $result3"}
#   } catch {
#     print_success "Maximum with incompatible shapes test passed: Caught expected error"
#   }
# }

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
