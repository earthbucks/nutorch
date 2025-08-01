use std assert
use std/testing *

@test
def "Softmax 1" [] {
  let input_data = $in
  let result1 = torch tensor ([1 2 3]) | torch softmax --dim 0 | torch value
  # softmax([1,2,3]) = [exp(1), exp(2), exp(3)] / (exp(1) + exp(2) + exp(3))
  # Should sum to 1.0
  let sum = ($result1 | math sum)
  assert ((($sum - 1.0) | math abs) < 0.0001)
}

@test
def "Softmax 2" [] {
  let input_data = $in
  let result2 = torch softmax ([1 2 3] | torch tensor) --dim 0 | torch value
  # Should sum to 1.0
  let sum = ($result2 | math sum)
  assert ((($sum - 1.0) | math abs) < 0.0001)
}

@test
def "Softmax preserves shape" [] {
  let input_data = $in
  let original = torch tensor ([[1 2] [3 4]])
  let result = $original | torch softmax --dim 1 | torch value
  # Should have same shape as input: 2x2
  assert (($result | length) == 2)
  assert (($result | get 0 | length) == 2)
  assert (($result | get 1 | length) == 2)
}

