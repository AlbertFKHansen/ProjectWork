#!/bin/bash

cd Thumbnails/Dataset

# Define the two top-level directories and their respective patterns and grid sizes
declare -A PATTERNS
PATTERNS["rot"]="$(seq 0 5 355)"
PATTERNS["temp"]="$(seq 2000 100 6500)"

declare -A GRID_SIZES
GRID_SIZES["rot"]="9x8"
GRID_SIZES["temp"]="10x5"

# Loop through both top-level directories
for top_dir in rot temp; do
    echo "== Entering top-level directory: $top_dir =="

    cd "$top_dir" || continue

    # Process each subdirectory
    for dir in */ ; do
        cd "$dir" || continue

        echo "→ Processing subdirectory: $dir"

        images=()
        for i in ${PATTERNS[$top_dir]}; do
            img=$(printf "*_%s.png" "$i")
            match=( $img )
            if [[ -f "${match[0]}" ]]; then
                images+=("${match[0]}")
            fi
        done

        expected_count=$(echo "${PATTERNS[$top_dir]}" | wc -w)
        if [[ ${#images[@]} -eq $expected_count ]]; then
            dirname=$(basename "$PWD")
            outfile="../../compacts/all_${top_dir}_${dirname}.png"
            echo "    Creating grid image: $outfile"

            montage "${images[@]}" -tile ${GRID_SIZES[$top_dir]} -geometry +0+0 "$outfile"
        fi

        cd ..
    done

    cd ..
done

cd ../..
