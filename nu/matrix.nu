def record_from_list [data: list] {
  mut my_record = {}
  for i in 0..( ($data | length) - 1) {
    let item = $data | get $i
    let name = $i | into string
    $my_record = $my_record | merge { $name: $item }
  }
  $my_record
}

def table_from_list [data: list<list>] {
  mut my_table = [[]; []]
  let num_rows = $data | length
  let num_cols = $data | get 0 | length
  for i in 0..($num_rows - 1) {
    let row = $data | get $i
    mut my_record = {}
    for j in 0..($num_cols - 1) {
      let col_name = $j | into string
      $my_record = $my_record | merge { $col_name: ( $row | get $j ) }
    }
    $my_table = $my_table | append $my_record
  }
  $my_table
}

