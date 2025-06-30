def to_row [] {
    let data = $in  # Capture input from pipeline
    mut my_record = {}
    for i in 0..( ($data | length) - 1) {
        let item = $data | get $i
        let name = $i | into string
        $my_record = $my_record | merge { $name: $item }
    }
    $my_record
}

def to_matrix [] {
    let data = $in  # Capture input from pipeline
    
    # Check if input is empty
    if ($data | is-empty) {
        return []
    }
    
    # Initialize table to first row
    mut my_table = []
    $my_table = ($data | get 0 | to_row)

    # Then merge every row starting with the second row
    let num_rows = $data | length
    for i in 1..($num_rows - 1) {
        let row = $data | get $i
        let my_record = $row | to_row
        $my_table = $my_table | append $my_record
    }
    $my_table
}
