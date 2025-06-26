
# nutorch module for tensor operations
export module nutorch {
    # Generate a tensor (1D or multi-dimensional)
    export def generate_tensor [
        --start: float = 0.0,  # Start value for linear data (1D only)
        --end: float = 1.0,    # End value for linear data (1D only)
        --step: float = 0.1,   # Step increment for linear data (1D only)
        ...dims: int           # Dimensions of the tensor (e.g., 3 for 1D, 2 2 for 2D)
    ] {
        if ($dims | length) == 0 {
            error make {msg: "At least one dimension must be specified"}
        }

        if ($dims | length) == 1 {
            # 1D tensor: generate linear data from start to end with step
            let size = ($end - $start) / $step | math ceil
            seq $start $step $end | take ($size + 1)
        } else {
            # Multi-dimensional tensor: generate nested lists with linear data
            let total_size = $dims | reduce -f 1 { |it, acc| $acc * $it }
            let flat_data = seq $start $step ($start + $step * ($total_size - 1)) | take $total_size
            build_nd_tensor $flat_data $dims
        }
    }

    # Helper function to build a multi-dimensional tensor from flat data
    def build_nd_tensor [flat_data: list, dims: list] {
        if ($dims | length) == 1 {
            return ($flat_data | take $dims.0)
        }

        let chunk_size = $dims | skip 1 | reduce -f 1 { |it, acc| $acc * $it }
        let sub_dims = $dims | skip 1
        mut result = []
        mut idx = 0

        for _ in 0..($dims.0 - 1) {
            let sub_data = $flat_data | skip $idx | take $chunk_size
            let sub_tensor = build_nd_tensor $sub_data $sub_dims
            $result = ($result | append $sub_tensor)
            $idx = $idx + $chunk_size
        }
        $result
    }
}
