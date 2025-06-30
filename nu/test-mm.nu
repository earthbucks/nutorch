# test-mm.nu - Test script for torch mm command (matrix multiplication)

# Helper function to compare two nested lists for strict equality
def compare_nested_lists [actual: list, expected: list] {
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

# Test 1: Matrix multiplication of 2x3 and 3x3 matrices with integers
print "Running Test 1: 2x3 * 3x3 = 2x3 matrix multiplication (integers)"

# Create a 1D tensor of length 3 with integer values
let base1 = (torch linspace 0 2 3 --dtype int32)  # Shape: [3], values [0, 1, 2]
# print "Base1 tensor:"
# $base1 | torch value

# Use repeat to create a 2x3 matrix (repeat 2 times along a new leading dimension)
let t1 = ($base1 | torch repeat 2 1)  # Shape: [2, 3]
# print "Tensor 1 (2x3):"
# $t1 | torch value  # Expected: [[0, 1, 2], [0, 1, 2]]

# Create a 1D tensor of length 3 with integer values
let base2 = (torch linspace 1 3 3 --dtype int32)  # Shape: [3], values [1, 2, 3]
# print "Base2 tensor:"
# $base2 | torch value

# Use repeat to create a 3x3 matrix (repeat 3 times along a new leading dimension)
let t2 = ($base2 | torch repeat 3 1)  # Shape: [3, 3]
# print "Tensor 2 (3x3):"
# $t2 | torch value  # Expected: [[1, 2, 3], [1, 2, 3], [1, 2, 3]]

# Perform matrix multiplication: (2x3) * (3x3) = (2x3)
let result = (torch mm $t1 $t2 | torch value)
# print "Result of 2x3 * 3x3:"
# $result

# Expected result (shape [2, 3]):
let expected = [[3, 6, 9], [3, 6, 9]]
# print "Expected result:"
# $expected

# Check if result matches expected (strict equality for integers)
if (compare_nested_lists $result $expected) {
    print "Test 1: SUCCESS - Matrix multiplication result matches expected output"
} else {
    print "Test 1: FAILURE - Matrix multiplication result does not match expected output"
    print "Actual: $result"
    print "Expected: $expected"
}

# Test 2: Matrix multiplication of 3x2 and 2x4 matrices with integers
print "Running Test 2: 3x2 * 2x4 = 3x4 matrix multiplication (integers)"

# Create a 1D tensor of length 2 with integer values
let base3 = (torch linspace 1 2 2 --dtype int32)  # Shape: [2], values [1, 2]
# print "Base3 tensor:"
# $base3 | torch value

# Use repeat to create a 3x2 matrix (repeat 3 times along a new leading dimension)
let t3 = ($base3 | torch repeat 3 1)  # Shape: [3, 2]
# print "Tensor 3 (3x2):"
# $t3 | torch value  # Expected: [[1, 2], [1, 2], [1, 2]]

# Create a 1D tensor of length 4 with integer values
let base4 = (torch linspace 0 3 4 --dtype int32)  # Shape: [4], values [0, 1, 2, 3]
# print "Base4 tensor:"
# $base4 | torch value

# Use repeat to create a 2x4 matrix (repeat 2 times along a new leading dimension)
let t4 = ($base4 | torch repeat 2 1)  # Shape: [2, 4]
# print "Tensor 4 (2x4):"
# $t4 | torch value  # Expected: [[0, 1, 2, 3], [0, 1, 2, 3]]

# Perform matrix multiplication: (3x2) * (2x4) = (3x4)
let result2 = (torch mm $t3 $t4 | torch value)
# print "Result of 3x2 * 2x4:"
# $result2

# Expected result (shape [3, 4]):
let expected2 = [[0, 3, 6, 9], [0, 3, 6, 9], [0, 3, 6, 9]]
# print "Expected result:"
# $expected2

# Check if result matches expected (strict equality for integers)
if (compare_nested_lists $result2 $expected2) {
    print "Test 2: SUCCESS - Matrix multiplication result matches expected output"
} else {
    print "Test 2: FAILURE - Matrix multiplication result does not match expected output"
    print "Actual: $result2"
    print "Expected: $expected2"
}
