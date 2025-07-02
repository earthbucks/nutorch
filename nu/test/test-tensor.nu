plugin use torch

# Test 1: Convert a 1D list to a tensor via pipeline
print "Running Test 1: Convert a 1D list to a tensor via pipeline"
let result1 = try {
  let res = ([1.0 2.0 3.0] | torch tensor)
  if ($res | describe | str contains "string") {
    print "Test 1: SUCCESS - Tensor created via pipeline"
    true
  } else {
    print "Test 1: FAILURE - Expected a tensor ID string, got unexpected output"
    error make {msg: "Test 1 failed: Did not return a tensor ID"}
    false
  }
} catch {
  print "Test 1: FAILURE - Unexpected error while creating tensor via pipeline"
  error make {msg: "Test 1 failed: Unexpected error"}
}

# Test 2: Convert a 1D list to a tensor via argument
print "Running Test 2: Convert a 1D list to a tensor via argument"
let result2 = try {
  let res = (torch tensor [1.0 2.0 3.0])
  if ($res | describe | str contains "string") {
    print "Test 2: SUCCESS - Tensor created via argument"
    true
  } else {
    print "Test 2: FAILURE - Expected a tensor ID string, got unexpected output"
    error make {msg: "Test 2 failed: Did not return a tensor ID"}
    false
  }
} catch {
  print "Test 2: FAILURE - Unexpected error while creating tensor via argument"
  error make {msg: "Test 2 failed: Unexpected error"}
}

# Test 3: Error case - no input provided
print "Running Test 3: Error case - no input provided"
let result3 = try {
  torch tensor
  print "Test 3: FAILURE - Expected an error for missing input, but no error occurred"
  error make {msg: "Test 3 failed: No error for missing input"}
  false
} catch {
  print "Test 3: SUCCESS - Expected error occurred for missing input"
  true
}

# Test 4: Error case - both pipeline and argument provided
print "Running Test 4: Error case - both pipeline and argument provided"
let result4 = try {
  [1.0 2.0 3.0] | torch tensor [4.0 5.0 6.0]
  print "Test 4: FAILURE - Expected an error for conflicting input, but no error occurred"
  error make {msg: "Test 4 failed: No error for conflicting input"}
  false
} catch {
  print "Test 4: SUCCESS - Expected error occurred for conflicting input"
  true
}

print "All tests completed. If no FAILURE messages, all tests passed!"
