#!/usr/bin/env bash
# The image captioning set, from a built synthetic dataset (build_synthetic.sh) in SOURCE: per split its
# Parquet shards in OUT/data.
set -euo pipefail
SOURCE="${1:-synthetic}"
OUT="${2:-image-captioning}"
for split in train val test; do
    uv run create_image_captioning.py --dataset-dir "$SOURCE" --split "$split" --output-dir "$OUT"
done
