#!/usr/bin/env bash
# The entity captioning set, from a built synthetic dataset (build_synthetic.sh) in SOURCE: per split its
# Parquet shards in OUT/data.
set -euo pipefail
SOURCE="${1:-synthetic}"
OUT="${2:-entity-captioning}"
for split in train val test; do
    uv run create_entity_captioning.py --dataset-dir "$SOURCE" --split "$split" --output-dir "$OUT"
done
