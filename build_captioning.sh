#!/usr/bin/env bash
# The image captioning set, from a built synthetic dataset (build_synthetic.sh) in SOURCE: per split its
# Parquet shards in OUT/data; then the counts and the card.
set -euo pipefail
SOURCE="${1:-synthetic}"
OUT="${2:-captioning}"
for split in train val test; do
    uv run create_captioning.py --dataset-dir "$SOURCE" --split "$split" --output-dir "$OUT"
done
uv run describe_captioning.py --dataset-dir "$OUT" --card dataset_cards/synthetic-floorplans-captioning.md
