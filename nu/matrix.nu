# matrix.nu - Convert nested list structures into a table format for better visualization

def matrix [] {
    let input = $in  # Capture pipeline input

    if ($input | describe | str contains "list") {
        let first_elem = ($input | first)
        if ($first_elem | describe | str contains "list") {
            # 2D structure (list of lists), display directly as table
            $input | table -i false
        } else {
            # 1D structure (single list), treat as a column vector
            $input | wrap value | table -i false
        }
    } else {
        # Not a list, just return the input as-is
        $input
    }
}
