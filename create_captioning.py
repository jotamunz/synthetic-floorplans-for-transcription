"""The captioning set of a split, in the conversation format of LLaVA-Pretrain
(https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) and stored as a Hugging Face dataset: per
plan its id, its image (an `Image` feature, the PNG inside the row) and one conversation asking for a
summary of the image, answered by the count of every class in its record. The question is one of
LLaVA-Pretrain's own, taken in turn by seed.

The images and records are the ones `draw_records.py` wrote. Each part of the split (`train-00` …, or
the split itself if it is whole) becomes one Parquet shard, `data/{split}-0000k-of-0000n.parquet`, in
seed order, with the dataset's features in its schema, so that `load_dataset` and the dataset viewer
read the images as images. A plan's id is its seed, so it can be found in the source split.

    uv run create_captioning.py --dataset-dir ./synthetic --split val --output-dir ./captioning
    uv run create_captioning.py --dataset-dir ./synthetic --split train --output-dir ./captioning
"""

import argparse
import io
import json
import multiprocessing
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features, Image as ImageFeature, List, Value
from PIL import Image

#: The questions LLaVA-Pretrain opens its conversations with, each phrase with the image before and after it.
PROMPTS = json.loads((Path(__file__).parent / "prompts" / "captioning.json").read_text(encoding="utf-8"))

#: The classes a caption counts, in the order it names them: (class, singular, plural).
CLASSES = [("outside wall", "outside wall", "outside walls"), ("inside wall", "inside wall", "inside walls"),
           ("door", "door", "doors"), ("outside door", "outside door", "outside doors"),
           ("window", "window", "windows")]

FEATURES = Features({"id": Value("string"), "image": ImageFeature(),
                     "conversations": List({"from": Value("string"), "value": Value("string")})})
SCHEMA = FEATURES.arrow_schema

#: Rows per row group, as `datasets` writes image datasets.
ROW_GROUP = 100
#: How many plans are read and encoded at a time.
BLOCK = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="The LLaVA captioning set of a split.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("synthetic"))
    parser.add_argument("--split", required=True, help="A split, or the name its parts share: train for train-00 ….")
    parser.add_argument("--output-dir", type=Path, default=Path("captioning"))
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    return parser.parse_args()


def prompt(seed: int) -> str:
    """The question for a plan: the prompts in turn, by seed, so that consecutive plans get the next one."""
    return PROMPTS[seed % len(PROMPTS)]


def caption(record: dict) -> str:
    """'a floorplan with 9 outside walls, 7 inside walls, 6 doors, 1 outside door and 8 windows.'"""
    counts = Counter(node["class"] for node in record["nodes"])
    named = [f"{counts[kind]} {one if counts[kind] == 1 else many}" if counts[kind] else f"no {many}"
             for kind, one, many in CLASSES]
    return f"a floorplan with {', '.join(named[:-1])} and {named[-1]}."


def png(image: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(image).save(out, format="PNG", optimize=True)
    return out.getvalue()


def row(record: dict, image: bytes) -> dict:
    name = f"{record['seed']:07d}"
    return {"id": name, "image": {"bytes": image, "path": f"{name}.png"},
            "conversations": [{"from": "human", "value": prompt(record["seed"])},
                              {"from": "gpt", "value": caption(record)}]}


def parts(dataset_dir: Path, split: str) -> list:
    whole = dataset_dir / f"{split}_labels.jsonl"
    found = [whole] if whole.exists() else sorted(dataset_dir.glob(f"{split}-*_labels.jsonl"))
    if not found:
        raise FileNotFoundError(f"No {split}_labels.jsonl or {split}-*_labels.jsonl in {dataset_dir}")
    return [labels.name[: -len("_labels.jsonl")] for labels in found]


def main() -> None:
    args = parse_args()
    names = parts(args.dataset_dir, args.split)
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # The split's shards from an earlier run go, as their number may differ.
    for stale in data_dir.glob(f"{args.split}-*-of-*.parquet"):
        stale.unlink()
    with multiprocessing.Pool(args.workers) as pool:
        for index, part in enumerate(names):
            images = np.load(args.dataset_dir / f"{part}_images.npy", mmap_mode="r")
            with open(args.dataset_dir / f"{part}_labels.jsonl", encoding="utf-8") as lines:
                records = [json.loads(line) for line in lines]
            if len(records) != len(images):
                raise ValueError(f"{part}: {len(records)} records but {len(images)} images")
            shard = data_dir / f"{args.split}-{index:05d}-of-{len(names):05d}.parquet"
            # Written under a partial name first, so that a shard under its own name is always whole.
            partial = shard.with_suffix(".parquet.part")
            with pq.ParquetWriter(partial, SCHEMA) as writer:
                # Read from the memmap a block at a time, so that a part of any size fits in memory.
                for start in range(0, len(records), BLOCK):
                    block = records[start:start + BLOCK]
                    encoded = pool.map(png, [np.asarray(images[at]) for at in range(start, start + len(block))], chunksize=32)
                    rows = [row(record, image) for record, image in zip(block, encoded)]
                    writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), row_group_size=ROW_GROUP)
            partial.replace(shard)
            print(f"{part}: {len(records)} plans into {shard.name}", flush=True)


if __name__ == "__main__":
    main()
