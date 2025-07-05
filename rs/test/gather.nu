plugin use torch

# Function to print SUCCESS in green with an uncolored message
def print_success [message: string] {
  print ((ansi green) + "SUCCESS" + (ansi reset) + " - test/gather - " + $message)
}

# Function to print FAILURE in red with an uncolored message
def print_failure [message: string] {
  print ((ansi red) + "FAILURE" + (ansi reset) + " - test/gather - " + $message)
}

# Create a 2×3 source tensor and an index tensor; gather columns 2,1,0
let src = ([[10 11 12] [20 21 22]] | torch tensor)
let idx = ([[2 1 0] [0 0 2]] | torch tensor --dtype int64)
# $src | torch value
let result1 = ($src | torch gather 1 $idx | torch value)
if ($result1 == [[12 11 10] [20 20 22]]) {
  print_success "Gather test passed"
} else {
  print_failure "Gather test failed: Expected [[12, 11, 10], [20, 20, 22]], got $result1"
  error make {msg: "Gather test failed: Expected [[12, 11, 10], [20, 20, 22]], got $result1"}
}
# → [[12, 11, 10], [20, 20, 22]]

# Gather rows along dim 0
let src2 = ([[1 2] [3 4]] | torch tensor)
mut result2 = false
try {
  let src = ([[1 2] [3 4]] | torch tensor)
  let idx = ([0 0 1] | torch tensor --dtype int64)
  $src | torch gather 0 $idx # expect “Shape mismatch” error
  $result2 = false
  print_failure "Gather with wrong dimension index should fail"
  error make {msg: "Gather with wrong dimension index should fail"}
} catch {
  print_success "Gather with wrong dimension index failed as expected"
}
$result2 = true
