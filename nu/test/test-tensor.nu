plugin use torch

# Test 1: Convert a 1D list to a tensor via pipeline
let result1 = ([1.0, 2.0, 3.0] | torch tensor | torch value)
$result1  # Expected: [1, 2, 3]
print "Test 1: SUCCESS if result is [1, 2, 3]"

# Test 2: Convert a 1D list to a tensor via argument
let result2 = (torch tensor [1.0, 2.0, 3.0] | torch value)
$result2  # Expected: [1, 2, 3]
print "Test 2: SUCCESS if result is [1, 2, 3]"

# Test 3: Error case - no input provided
torch tensor
# Expected: Error "Missing input"

# Test 4: Error case - both pipeline and argument provided
[1.0, 2.0, 3.0] | torch tensor [4.0, 5.0, 6.0]
# Expected: Error "Conflicting input"
