# test-maximum.nu - Test script for torch maximum command (element-wise maximum)

# Helper function to compare two nested lists for strict equality
def compare_nested_lists [actual: list expected: list] {
  if ($actual | length) != ($expected | length) {
    return false
  }
  for i in 0..(($actual | length) - 1) {
    let actual_val = ($actual | get $i)
    let expected_val = ($expected | get $i)
    if ($actual_val | describe | str contains "list") and ($expected_val | describe | str contains "list") {
      if not (compare_nested_lists $actual_val $expected_val) {
        return false
      }
    } else {
      if $actual_val != $expected_val {
        return false
      }
    }
  }
  true
}

# Test 1: Compute element-wise maximum between two tensors of same shape
print "Running Test 1: Element-wise maximum between two tensors of same shape"
let t1 = (torch full -1 2 3) # Shape: [2, 3], filled with -1
$t1 | torch value # Should show [[-1, -1, -1], [-1, -1, -1]]
let t2 = (torch full 0 2 3) # Shape: [2, 3], filled with 0
$t2 | torch value # Should show [[0, 0, 0], [0, 0, 0]]
let result1 = (torch maximum $t2 $t1 | torch value)
$result1 # Expected shape [2, 3]: [[0, 0, 0], [0, 0, 0]]
let expected1 = [[0 0 0] [0 0 0]]
if (compare_nested_lists $result1 $expected1) {
  print "Test 1: SUCCESS - Element-wise maximum computed correctly"
} else {
  print "Test 1: FAILURE - Element-wise maximum result does not match expected output"
  print "Actual: $result1"
  print "Expected: $expected1"
  exit 1 # Exit with error code to indicate failure for automation
}

# Test 2: Compute element-wise maximum with broadcasting (scalar vs tensor)
print "Running Test 2: Element-wise maximum with broadcasting (scalar vs tensor)"
let t3 = (torch full 0 1) # Shape: [], scalar tensor filled with 0
$t3 | torch value # Should show 0
let t4 = (torch linspace -2 2 5) # Shape: [5], values from -2 to 2
$t4 | torch value # Should show [-2, -1, 0, 1, 2] (approx)
let result2 = (torch maximum $t3 $t4 | torch value)
$result2 # Expected shape [5]: [0, 0, 0, 1, 2]
let expected2 = [0 0 0 1 2]
if (compare_nested_lists $result2 $expected2) {
  print "Test 2: SUCCESS - Element-wise maximum with broadcasting computed correctly"
} else {
  print "Test 2: FAILURE - Element-wise maximum with broadcasting result does not match expected output"
  print "Actual: $result2"
  print "Expected: $expected2"
  exit 1 # Exit with error code to indicate failure for automation
}

print "All tests passed successfully!"
