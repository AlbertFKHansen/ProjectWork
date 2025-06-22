#!/bin/bash

# Set this to where your PNG plots are stored
PLOT_DIR="gif_dir"

# Set this to where you want to save the final GIFs
OUT_DIR="gifs"

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR/temp"
mkdir -p "$OUT_DIR/rot"

# ImageMagick options
DELAY=10
LOOP=0

for mode in rot temp; do
  for obj_dir in "$PLOT_DIR/$mode"/*/; do
    obj_name=$(basename "$obj_dir")
    echo "Processing $mode for object: $obj_name"

    # Build list of PNGs, sorted
    img_list=($(ls "$obj_dir"/*.png | sort -V))
    if [ ${#img_list[@]} -eq 0 ]; then
      echo "  No PNG files found in $obj_dir, skipping."
      continue
    fi

    # Output gif path
    output_gif="$OUT_DIR/$mode/$obj_name.gif"

    # Create GIF
    convert -delay "$DELAY" -loop "$LOOP" "${img_list[@]}" "$output_gif"
    echo "  Saved: $output_gif"
  done
done