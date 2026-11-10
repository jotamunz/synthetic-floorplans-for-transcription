#!/usr/bin/env bash
# The predicted dataset, from a transcribing model's output in EMBEDDINGS, a folder per part (val, test,
# train-00 …) with its records.jsonl: each part's records laid out as a built synthetic dataset's,
# OUT/{part}_labels.jsonl, then drawn from them, OUT/{part}_images.npy.
set -euo pipefail
EMBEDDINGS="${1:?usage: build_predicted.sh EMBEDDINGS [OUT]}"
OUT="${2:-predicted}"
mkdir -p "$OUT"
found=0
for records in "$EMBEDDINGS"/*/records.jsonl; do
    [ -f "$records" ] || continue
    part="$(basename "$(dirname "$records")")"
    cp "$records" "$OUT/${part}_labels.jsonl"
    uv run draw_records.py --dataset-dir "$OUT" --split "$part"
    found=1
done
if [ "$found" = 0 ]; then
    echo "No {part}/records.jsonl in $EMBEDDINGS" >&2
    exit 1
fi
