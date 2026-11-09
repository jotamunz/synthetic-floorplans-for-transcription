"""A captioning dataset folder's `metadata.json`, counting the plans, the nodes and relationships of their
records and the questions and answers of each split, and its card copied in as `README.md`. It reads the
shards as they were written, so it describes what a build actually holds.

    uv run describe_captioning.py --dataset-dir ./captioning --card dataset_cards/synthetic-floorplans-captioning.md
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

#: How many rows are read at a time, without their images.
BATCH = 1024


def describe(shards: list) -> dict:
    """The counts of a split, from its Parquet shards."""
    seeds, counts, asked, answers = [], Counter(), Counter(), set()
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=BATCH, columns=["id", "record", "conversations"]):
            for row in batch.to_pylist():
                seeds.append(int(row["id"]))
                counts.update(node["class"] for node in row["record"]["nodes"])
                counts.update(f"relationship: {r['type']}" for r in row["record"].get("relationships", []))
                for turn in row["conversations"]:
                    if turn["from"] == "gpt":
                        answers.add(turn["value"])
                        continue
                    entity = re.search(r"\b([ER])(\d+)\b", turn["value"])
                    if not entity:
                        asked["plan"] += 1
                    elif entity[1] == "E":
                        asked["node"] += 1
                    else:
                        asked[f"relationship: {row['record']['relationships'][int(entity[2])]['type']}"] += 1
    return {"plans": len(seeds), "seeds": [min(seeds), max(seeds)], "shards": len(shards),
            "counts": dict(sorted(counts.items())), "questions": dict(sorted(asked.items())),
            "distinct answers": len(answers)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe a captioning dataset folder.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    args = parser.parse_args()
    splits = {}
    for shard in sorted((args.dataset_dir / "data").glob("*.parquet")):
        splits.setdefault(shard.name.split("-")[0], []).append(shard)
    described = {}
    for name, shards in splits.items():
        described[name] = describe(shards)
        print(name, described[name]["plans"], flush=True)
    (args.dataset_dir / "metadata.json").write_text(json.dumps({
        "format": "llava-conversations-v1",
        "source": "records of synthetic.py of synthetic-floorplans-for-transcription, one per row",
        "images": "512 x 512 grayscale PNG, an Image feature in the row, drawn from the record by draw.py",
        "frame": "pixels of the drawing, x from the left, y from the top",
        "splits": described,
    }, indent=2), encoding="utf-8")
    (args.dataset_dir / "README.md").write_text(args.card.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
