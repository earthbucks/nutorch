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
let src  = ([[10 11 12] [20 21 22]] | torch tensor)
let idx  = ([[2 1 0]   [0 0 2]]     | torch tensor --dtype int64)
# $src | torch value
$src | torch gather 1 $idx | torch value
# → [[12, 11, 10], [20, 20, 22]]

# Gather rows along dim 0
let src2 = ([[1 2] [3 4]] | torch tensor)
let idx2 = ([0 0 1]       | torch tensor --dtype int64)
print ($src2 | torch gather 0 $idx2 | torch value)
# → [[1, 2], [1, 2], [3, 4]]
