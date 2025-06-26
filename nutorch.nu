# nutorch module for tensor operations

export module nutorch {
  # Generate a 1D tensor with linearly spaced values (similar to torch.linspace)
  export def linspace [
    start: float # Start value of the sequence
    end: float # End value of the sequence
    steps: int # Number of points in the sequence
  ] {
    if $steps < 2 {
      error make {msg: "Steps must be at least 2"}
    }
    let step_size = ($end - $start) / ($steps - 1)
    seq $start $step_size $end | take $steps
  }

  export def pi [] {
    let PI = 3.14159265358979323846
    $PI
  }

  export def e [] {
    let E = 2.71828182845904523536
    $E
  }

  # apply sine function element-wise to a tensor
  export def sin [] {
    let input = $in # get input from pipeline
    if ($input | describe | str contains "list") {
      $input | each {|elem|
        if ($elem | describe | str contains "list") {
          $elem | each {|sub_elem|
            if ($sub_elem | describe | str contains "list") {
              $sub_elem | each {|val| $val | math sin }
            } else {
              $sub_elem | math sin
            }
          }
        } else {
          $elem | math sin
        }
      }
    } else {
      error make {msg: "input must be a tensor (list)"}
    }
  }
}
