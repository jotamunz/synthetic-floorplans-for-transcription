---
pretty_name: Synthetic Floor Plans Captioning
license: cc-by-4.0
language:
  - en
size_categories:
  - 100K<n<1M
task_categories:
  - image-to-text
tags:
  - floor-plans
  - architecture
  - captioning
  - synthetic
  - llava
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*
      - split: validation
        path: data/val-*
      - split: test
        path: data/test-*
---

# Synthetic Floor Plans Captioning

Generated floor plan drawings, each with a caption saying what the plan holds: how many walls, doors,
windows and rooms, which way it is shaped, and where its entrance is. The set is meant for aligning a
vision encoder with a language model, and follows the conversation format of
[LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).

The plans are fresh ones from the generator of
[synthetic-floorplans-for-transcription](https://huggingface.co/datasets/benikm91/synthetic-floorplans-for-transcription),
and each row holds its plan's record. Every caption is computed from that record, the same record the
drawing is rendered from, so nothing in a caption is estimated or annotated by hand. Its companion,
[synthetic-floorplans-entity-captioning](https://huggingface.co/datasets/jotamunz/synthetic-floorplans-entity-captioning),
asks about the single nodes and relationships of the same plans.

```python
from datasets import load_dataset

plans = load_dataset("jotamunz/synthetic-floorplans-captioning", split="train")
plans[3]["image"]          # a PIL image
plans[3]["record"]         # the nodes and relationships it is drawn from
plans[3]["conversations"]  # one question and its caption
```

## Data format

| field | type | description |
|---|---|---|
| `id` | string | The plan's seed, padded to 7 digits. |
| `image` | Image | A 512 × 512 grayscale PNG, black on white, drawn from the record alone. |
| `record` | struct | The plan's nodes and relationships, described [below](#the-record). |
| `conversations` | list of {`from`, `value`} | One human question and one `gpt` answer. |

```json
{"id": "0000003",
 "image": "<512 × 512 PNG>",
 "record": {"nodes": ["…"], "relationships": ["…"]},
 "conversations": [
   {"from": "human", "value": "<image>\nGive a brief description of the image."},
   {"from": "gpt", "value": "a tall layout comprising a few doors, multiple exterior walls, a handful of internal walls, the front door at the right side and some outer windows."}]}
```

LLaVA-Pretrain ships its images in a zip and names each in its `image` field. Here the image is in the
row, so the dataset viewer shows it and nothing needs unpacking. For training code that expects the
original layout, write each image to `{id}.png` and set `image` to that name.

## The record

Every row holds the record its plan was generated with, and its drawing and answers follow from that
record alone.

```json
{"nodes": [{"class": "outside wall", "points": [[124.0, 90.0], [220.0, 90.0]]}, "…"],
 "relationships": [{"type": "joined", "nodes": [0, 1]}, "…"]}
```

- A **node** has a class (`outside wall`, `inside wall`, `door`, `outside door` or `window`) and the two
  ends of its segment, in pixels of the image, x from the left and y from the top.
- A **relationship** has a type and the two nodes it links, as positions in `nodes`. `joined` links two
  walls that share an end, and `in` links an opening to the wall it stands in, opening first.

The format is the one of
[synthetic-floorplans-for-transcription](https://huggingface.co/datasets/benikm91/synthetic-floorplans-for-transcription),
whose card describes it in full.

## Questions

Each question is one of the 22 that LLaVA-Pretrain asks: 11 phrasings, each with `<image>` before and
after it. The plan with seed `s` gets question `s mod 22`, so all of them are asked equally often.

## Captions

A caption opens with what the image is, adds a verb, and then lists the plan's contents in a shuffled
order. All word choices come from a random generator seeded by the plan, so a plan always gets the same
caption. Captions average about 29 words.

**Counts** are given in words, not exact numbers, because exact counting is left to question answering.
The ranges are close across classes, so that a word means about the same amount everywhere. The shares
for walls and windows were measured on 2,000 training plans. Outside wall counts are always even,
because a rectilinear outline turns an even number of times.

| counted | "a few", "a handful of" | "several", "multiple", "a number of", "some" | "many", "a lot of", "numerous", … |
|---|---|---|---|
| outside walls | 4–6: 9% | 8–10: 22% | 12+: 69% |
| inside walls | 2–6: 26% | 7–10: 39% | 11+: 32% |
| windows | 2–4: 27% | 5–8: 61% | 9+: 12% |
| doors | 2–3: 10% | 4–7: 58% | 8+: 28% |
| rooms | 2–4: 13% | 5–8: 59% | 9+: 28% |

A count of one reads "a" or "a single", and a count of zero reads "no".

**Walls** are counted the way a viewer sees them, as straight runs. The record ends a wall wherever
another meets it, so one straight facade can be several record walls; a caption counts it as one.

**Doors or rooms:** half the captions count the inside doors and half count the rooms. A plan has one
room more than it has inside doors, and balconies count as rooms. A plan without inside doors is
always "a single room", and its wall counts are left out.

**Orientation** is named only when it is evident, from the box around the outside walls. If the longer
side is at most 1.1 times the shorter, the plan is square (19% of plans). If it is at least 1.5 times
the shorter, the plan is wide (18%) or tall (10%). The remaining 53% name no orientation.

**The entrance** is named but not counted, since every plan has exactly one. Its side is the way it
faces: "the front door at the right side".

**Placement:** 30% of the time, windows and door counts also say which walls they stand in: "several
windows in the exterior walls".

**Synonyms:** each class has several names, such as "outside wall", "exterior wall", "external wall"
and "perimeter wall". An evaluation should map them back to their class.

## Splits

| split | plans | seeds | files |
|---|---|---|---|
| train | 200,000 | 3,000,000 – 3,199,999 | `data/train-0000k-of-00004.parquet` |
| validation | 5,000 | 4,000,000 – 4,004,999 | `data/val-00000-of-00001.parquet` |
| test | 5,000 | 5,000,000 – 5,004,999 | `data/test-00000-of-00001.parquet` |

A plan's `id` is its seed. The seeds are apart from those of
[synthetic-floorplans-for-transcription](https://huggingface.co/datasets/benikm91/synthetic-floorplans-for-transcription),
so a model trained on that dataset has not seen these plans.

## Limitations

- The plans are generated, and all their walls are horizontal or vertical. Real floor plans are more
  varied, and the drawings carry no text, dimensions or furniture.
- The drawing does not show what a room is used for, so captions never name room types.
- Count words are coarse, and neighbouring counts on either side of a range get different words.
- Captions describe the plan as a whole. They give no positions apart from the entrance's side.

## Regenerating

The set is built with the `instruction-tuning` branch of
[synthetic-floorplans-for-transcription](https://github.com/jotamunz/synthetic-floorplans-for-transcription/tree/instruction-tuning):
the plans from their seeds, then the captions. A seed always gives the same plan. Questions and words
are in `captioning_config/vocabulary.json`.

```bash
uv run create_synthetic_records.py --output-dir ./plans --split val --first-seed 4000000 --count 5000
uv run create_synthetic_records.py --output-dir ./plans --split test --first-seed 5000000 --count 5000
for part in 0 1 2 3; do
    uv run create_synthetic_records.py --output-dir ./plans --split "train-0$part" --first-seed $((3000000 + part * 50000)) --count 50000
done
for split in val test train-00 train-01 train-02 train-03; do
    uv run draw_records.py --dataset-dir ./plans --split "$split"
done
./build_captioning.sh ./plans ./captioning
```

## License

CC BY 4.0, as synthetic-floorplans-for-transcription.
