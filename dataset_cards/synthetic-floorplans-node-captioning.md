---
pretty_name: Synthetic Floor Plans Node Captioning
license: cc-by-4.0
language:
  - en
size_categories:
  - 100K<n<1M
task_categories:
  - image-text-to-text
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

# Synthetic Floor Plans Node Captioning

Generated floor plans with multi-turn conversations about single nodes of their records: what a labelled
wall segment, door or window is and where it lies. The set is meant for aligning the node embeddings of a
record decoder with a language model, and uses LLaVA's conversation format with a `<record>` token in
place of `<image>`.

The plans are fresh ones from the generator of
[synthetic-floorplans-for-transcription](https://huggingface.co/datasets/benikm91/synthetic-floorplans-for-transcription).
Two companions use the same plans:
[synthetic-floorplans-relationship-captioning](https://huggingface.co/datasets/jotamunz/synthetic-floorplans-relationship-captioning)
asks about the relationships between nodes, and
[synthetic-floorplans-captioning](https://huggingface.co/datasets/jotamunz/synthetic-floorplans-captioning)
captions each plan as a whole.

## Intended use

This set is meant to be built from **predicted records**, the output of a model transcribing a drawing,
and not from the true synthetic records. A decoder writes its nodes and their embeddings in the same
order, so `E0` in a question is the decoder's first node and its first embedding.

The dataset itself does not depend on where a record came from. A row's image, record and conversation
are all built from one record, the image drawn from it, so they agree with each other and work the same
whether that record is predicted or true.

## Usage

```python
from datasets import load_dataset

plans = load_dataset("jotamunz/synthetic-floorplans-node-captioning", split="train")
plans[3]["record"]         # the nodes the questions name, and their relationships
plans[3]["conversations"]  # ten questions and their answers
```

## Data format

| field | type | description |
|---|---|---|
| `id` | string | The plan's seed, padded to 7 digits. |
| `image` | Image | A 512 × 512 grayscale PNG, black on white, drawn from the record alone. |
| `record` | struct | The plan's whole record, its nodes and relationships, described [below](#the-record). |
| `conversations` | list of {`from`, `value`} | Ten questions, each followed by its answer. |

```json
{"id": "4000000",
 "image": "<512 × 512 PNG>",
 "record": {"nodes": ["…"], "relationships": ["…"]},
 "conversations": [
   {"from": "human", "value": "Describe E16.\n<record>"},
   {"from": "gpt", "value": "a vertical external wall segment at the left of the apartment layout"},
   {"from": "human", "value": "What does E41 represent?"},
   {"from": "gpt", "value": "a horizontal window opening at the bottom left of the floorplan"},
   {"from": "human", "value": "What is E7?"},
   {"from": "gpt", "value": "a horizontal external wall segment at the top right of the apartment layout"}]}
```

`E{i}` is node `i` of the row's `record`. The labels are not stored, because the record's order defines
them: a decoder that writes nodes and their embeddings in that order can match `E{i}` directly to its
text and its embedding. `<record>` is a placeholder for the training code to fill: **the rows hold the
record itself but no embeddings**.

## The record

Every row holds the whole record it was built from, the same as its companions do, and its drawing and
answers follow from that record alone. The questions here are only about its nodes.

```json
{"nodes": [{"class": "outside wall", "points": [[124.0, 90.0], [220.0, 90.0]]}, "…"],
 "relationships": [{"type": "joined", "nodes": [0, 1]}, "…"]}
```

- A **node** has a class (`outside wall`, `inside wall`, `door`, `outside door` or `window`) and the two
  ends of its segment, in pixels of the image, x from the left and y from the top.
- A **relationship** has a type and the two nodes it links, as positions in `nodes`. `joined` links two
  wall segments that share an end, and `in` links an opening to the wall segment it stands in, opening
  first.

The format is the one of
[synthetic-floorplans-for-transcription](https://huggingface.co/datasets/benikm91/synthetic-floorplans-for-transcription),
whose card describes it in full.

## Conversations

Each conversation asks about 10 nodes, picked at random from the plan's nodes; a plan with fewer, such as
a single room, is asked about all of them. No node is asked about twice in a conversation. The picks and
all word choices come from a random generator seeded by the plan, so a plan always gets the same
conversation.

The questions are "What is E14?", "Describe E14." and "What does E14 represent?". `<record>` appears
once, in the first question, before or after the text, and later questions leave it out, as LLaVA does
with `<image>`.

## Answers

An answer gives which way the node runs, its class, and the sides of the plan it is clearly on:

> a horizontal window at the top left of the floor plan

A side is named only when it is clear. Around the middle of the plan's outline lies a central band,
reaching 10% of the plan's size to each side, along each axis. A node is on a side only if all of it is
past the band, so a node crossing the middle has no side on that axis. Across true nodes, 47% name both
axes, 49% name one, and 4% name neither:

| clear sides | example |
|---|---|
| both | a vertical inside wall segment at the top right of the floorplan |
| one | a horizontal door at the top of the architectural plan |
| none | a horizontal inside wall segment in the apartment layout |

**Walls are named as segments.** A wall node is one piece of the record, and the record ends a wall
wherever another meets it, so one straight wall in the drawing can be several segments. The image
captioning set counts walls the way the drawing shows them, as straight runs, so across these sets a
"wall" is a straight run and a "wall segment" is one node.

Classes and the plan are each named with synonyms, such as "outside wall segment", "exterior wall
segment" and "perimeter wall segment", or "floor plan", "layout" and "plan drawing". An evaluation
should map them back to their class.

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

- Answers say what the record says, not what the original plan holds. In a set built from predictions,
  an error of the prediction shows in the image and the answers alike, so they stay consistent with
  each other but can differ from the original plan. A tilted segment, which only a prediction holds,
  is drawn tilted and named by the way it spans further.
- The shares quoted on this card were measured on true records. Predicted records can differ.
- A straight wall in the drawing can be several wall segments. In predicted records, where a wall is
  split depends on the prediction.
- Positions in answers are coarse. Exact coordinates are only in `record`.
- Every question is about something that exists, so the set has no negative examples.
- The plans are generated, and all their walls are horizontal or vertical.

## Regenerating

The set is built with the `instruction-tuning` branch of
[synthetic-floorplans-for-transcription](https://github.com/jotamunz/synthetic-floorplans-for-transcription/tree/instruction-tuning):
the plans from their seeds, a transcribing model's records of them, drawn anew, then the
conversations. A seed always gives the same plan. Questions and
words are in `captioning_config/entity_vocabulary.json`, and class names in
`captioning_config/vocabulary.json`.

```bash
uv run create_synthetic_records.py --output-dir ./plans --split val --first-seed 4000000 --count 5000
uv run create_synthetic_records.py --output-dir ./plans --split test --first-seed 5000000 --count 5000
for part in 0 1 2 3; do
    uv run create_synthetic_records.py --output-dir ./plans --split "train-0$part" --first-seed $((3000000 + part * 50000)) --count 50000
done
for split in val test train-00 train-01 train-02 train-03; do
    uv run draw_records.py --dataset-dir ./plans --split "$split"
done
```

The model transcribes the drawings and writes its records to a folder per part, `val/`, `train-00/` …,
each with a `records.jsonl` of one plan per line, its `seed`, `nodes` and `relationships`, in the
format of the plans' own. `build_predicted.sh` lays them out as the plans are and draws them, and the
set is built from them:

```bash
./build_predicted.sh ./predictions ./predicted
./build_node_captioning.sh ./predicted ./node-captioning
```

Built from `./plans` instead, the set holds the true records.

## License

CC BY 4.0, as synthetic-floorplans-for-transcription.
