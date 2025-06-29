# Test script for nutorch cat command

# Test 1: Concatenate two 2x3 tensors along dimension 0 (rows)
print "Running Test 1: Concatenate two 2x3 tensors along dimension 0 (rows)"
let t1 = (nutorch full 1 2 3)  # Shape: [2, 3], filled with 1
$t1 | nutorch value  # Should show [[1, 1, 1], [1, 1, 1]]
let t2 = (nutorch full 2 2 3)  # Shape: [2, 3], filled with 2
$t2 | nutorch value  # Should show [[2, 2, 2], [2, 2, 2]]
let result1 = (nutorch cat $t1 $t2 --dim 0 | nutorch value)
$result1  # Expected shape [4, 3]: [[1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]]
print "Test 1: SUCCESS - Concatenation along dim 0 completed (manual check required for values)"

# Test 2: Concatenate two 2x3 tensors along dimension 1 (columns)
print "Running Test 2: Concatenate two 2x3 tensors along dimension 1 (columns)"
let t3 = (nutorch full 3 2 3)  # Shape: [2, 3], filled with 3
$t3 | nutorch value  # Should show [[3, 3, 3], [3, 3, 3]]
let result2 = (nutorch cat $t1 $t3 --dim 1 | nutorch value)
$result2  # Expected shape [2, 6]: [[1, 1, 1, 3, 3, 3], [1, 1, 1, 3, 3, 3]]
print "Test 2: SUCCESS - Concatenation along dim 1 completed (manual check required for values)"

# Test 3: Error case - incompatible shapes, expect failure as success
print "Running Test 3: Concatenate incompatible shapes (expect failure)"
let t4 = (nutorch full 4 2 2)  # Shape: [2, 2], filled with 4
let error_result = try {
    nutorch cat $t1 $t4 --dim 0 | nutorch value
    false  # If no error occurs, test fails
} catch {
    true  # If an error occurs, test passes
}
if $error_result {
    print "Test 3: SUCCESS - Expected failure occurred due to shape mismatch"
} else {
    print "Test 3: FAILURE - Expected failure did not occur, concatenation succeeded unexpectedly"
}
