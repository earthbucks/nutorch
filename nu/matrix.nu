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
  # initialize table to first row
  mut my_table = [[]; []]
  $my_table = record_from_list ($data | get 0)

  # then merge every row starting with the second row
  let num_rows = $data | length
  for i in 1..($num_rows - 1) {
    let row = $data | get $i
    let my_record = record_from_list $row
    $my_table = $my_table | append $my_record
  }
  $my_table
}

