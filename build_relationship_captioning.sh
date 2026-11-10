#!/usr/bin/env bash
# The relationship captioning set, from a predicted dataset (build_predicted.sh) in SOURCE, or any built like
# a synthetic one: per split its Parquet shards in OUT/data; then the counts and the card.
set -euo pipefail
SOURCE="${1:-predicted}"
OUT="${2:-relationship-captioning}"
for split in train val test; do
    uv run create_relationship_captioning.py --dataset-dir "$SOURCE" --split "$split" --output-dir "$OUT"
done
uv run describe_captioning.py --dataset-dir "$OUT" --card dataset_cards/synthetic-floorplans-relationship-captioning.md
