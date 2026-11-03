"""What the captioning sets share: their configs, their arguments, and writing a split of rows.

The images and records are the ones `draw_records.py` wrote. Each part of the split (`train-00` …, or
the split itself if it is whole) becomes one Parquet shard, `data/{split}-0000k-of-0000n.parquet`, in
seed order, with the dataset's features in its schema, so that `load_dataset` and the dataset viewer
read the images as images. A plan's id is its seed, so it can be found in the source split.
"""

import argparse
import io
import json
import multiprocessing
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

#: Rows per row group, as `datasets` writes image datasets.
ROW_GROUP = 100
#: How many plans are read and encoded at a time.
BLOCK = 4096


def config(name: str) -> dict:
    """A config of `captioning_config/`, by its file name."""
    return json.loads((Path(__file__).parent / "captioning_config" / name).read_text(encoding="utf-8"))


def parse_args(description: str, output_dir: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-dir", type=Path, default=Path("synthetic"))
    parser.add_argument("--split", required=True, help="A split, or the name its parts share: train for train-00 ….")
    parser.add_argument("--output-dir", type=Path, default=Path(output_dir))
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    return parser.parse_args()


def png(image: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(image).save(out, format="PNG", optimize=True)
    return out.getvalue()


def parts(dataset_dir: Path, split: str) -> list:
    whole = dataset_dir / f"{split}_labels.jsonl"
    found = [whole] if whole.exists() else sorted(dataset_dir.glob(f"{split}-*_labels.jsonl"))
    if not found:
        raise FileNotFoundError(f"No {split}_labels.jsonl or {split}-*_labels.jsonl in {dataset_dir}")
    return [path.name[: -len("_labels.jsonl")] for path in found]


def write_split(args: argparse.Namespace, schema: pa.Schema, row: Callable[[dict, bytes], dict]) -> None:
    """The split's rows, `row` of each record and its image as a PNG, into a shard per part."""
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
            with pq.ParquetWriter(partial, schema) as writer:
                # Read from the memmap a block at a time, so that a part of any size fits in memory.
                for start in range(0, len(records), BLOCK):
                    block = records[start:start + BLOCK]
                    encoded = pool.map(png, [np.asarray(images[at]) for at in range(start, start + len(block))], chunksize=32)
                    rows = [row(record, image) for record, image in zip(block, encoded)]
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema), row_group_size=ROW_GROUP)
            partial.replace(shard)
            print(f"{part}: {len(records)} plans into {shard.name}", flush=True)
