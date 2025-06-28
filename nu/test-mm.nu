# Create a 1D tensor of length 3
let base1 = (nutorch linspace 0 2 3)  # Shape: [3], values approx [0.0, 1.0, 2.0]
$base1 | nutorch value  # Should show [0.0, 1.0, 2.0]

# Use repeat to create a 2x3 matrix (repeat 2 times along a new leading dimension)
let t1 = ($base1 | nutorch repeat 2 1)  # Shape: [2, 3]
$t1 | nutorch value  # Expected: [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]

# Create a 1D tensor of length 3
let base2 = (nutorch linspace 1 3 3)  # Shape: [3], values approx [1.0, 2.0, 3.0]
$base2 | nutorch value  # Should show [1.0, 2.0, 3.0]

# Use repeat to create a 3x3 matrix (repeat 3 times along a new leading dimension)
let t2 = ($base2 | nutorch repeat 3 1)  # Shape: [3, 3]
$t2 | nutorch value  # Expected: [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]

# Perform matrix multiplication: (2x3) * (3x3) = (2x3)
let result = (nutorch mm $t1 $t2 | nutorch value)
$result  # Display the result

# Let's compute the expected result manually to verify:

# - `t1` (shape `[2, 3]`): `[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]`
# - `t2` (shape `[3, 3]`): `[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]`
# - Matrix multiplication `(2x3) * (3x3)`:
#   - For row 0 of `t1` and columns of `t2`:
#     - `[0.0, 1.0, 2.0] * [1.0, 1.0, 1.0]` (first column) = `0.0*1.0 + 1.0*1.0 + 2.0*1.0 = 3.0`
#     - `[0.0, 1.0, 2.0] * [2.0, 2.0, 2.0]` (second column) = `0.0*2.0 + 1.0*2.0 + 2.0*2.0 = 6.0`
#     - `[0.0, 1.0, 2.0] * [3.0, 3.0, 3.0]` (third column) = `0.0*3.0 + 1.0*3.0 + 2.0*3.0 = 9.0`
#     - So, row 0 of result: `[3.0, 6.0, 9.0]`
#   - For row 1 of `t1` (same as row 0 in this case), the result is identical:
#     - Row 1 of result: `[3.0, 6.0, 9.0]`
# - Expected output (shape `[2, 3]`):
#   ```
#   [[3.0, 6.0, 9.0],
#    [3.0, 6.0, 9.0]]
#   ```
