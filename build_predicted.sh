#!/usr/bin/env bash
# The predicted dataset, from a transcribing model's output in SOURCE, a folder per part (val, test,
# train-00 …) with its predicted_records.jsonl: each part's records laid out as a built synthetic dataset's,
# OUT/{part}_labels.jsonl, then drawn from them, OUT/{part}_images.npy.
set -euo pipefail
SOURCE="${1:?usage: build_predicted.sh SOURCE [OUT]}"
OUT="${2:-predicted}"
mkdir -p "$OUT"
for records in "$SOURCE"/*/predicted_records.jsonl; do
    part="$(basename "$(dirname "$records")")"
    cp "$records" "$OUT/${part}_labels.jsonl"
    uv run draw_records.py --dataset-dir "$OUT" --split "$part"
done
