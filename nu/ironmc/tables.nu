# Vertex-table generation (Plan D5 CLI wrappers).

use common.nu [ROOT DATASET table-path]

# Build one vertex table. Defaults are the certified production settings.
export def build [
    grid: int              # q/k grid size n (table covers n³ points)
    --rank: int = 800      # global SVD budget (1600 = untruncated reference)
    --precision: string = "f32"
    --max-gib: int         # override the memory guard (deliberate decision!)
] {
    let out = (table-path $grid $rank $precision)
    cd $ROOT
    if $max_gib == null {
        cargo run --release -p rmc-ab-initio -- table --grid $grid --total-rank $rank --precision $precision --out $out
    } else {
        cargo run --release -p rmc-ab-initio -- table --grid $grid --total-rank $rank --precision $precision --out $out --max-gib $max_gib
    }
    $out
}

# Build every missing production ladder table (20³..50³ at rank 800/f32).
export def ladder [--rank: int = 800] {
    [20 32 40 50] | each {|n|
        let out = (table-path $n $rank)
        let existed = ($out | path exists)
        if not $existed { build $n --rank $rank }
        {grid: $n, table: $out, built: (not $existed)}
    }
}

# The untruncated reference table for rank guards (rank 1600).
export def reference [grid: int = 20] {
    build $grid --rank 1600
}

# List built table artifacts with sizes.
export def list [] {
    ls $DATASET | where name =~ '\.rmcvtab$' | select name size modified
}
