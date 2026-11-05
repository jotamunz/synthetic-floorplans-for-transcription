---
pretty_name: Synthetic Floor Plans Entity Captioning
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

# Synthetic Floor Plans Entity Captioning

Generated floor plans with multi-turn conversations about single entities of their records: what a
labelled wall, door or window is and where it lies, and which two nodes a relationship links. The set
is meant for aligning the per-entity embeddings of a record decoder with a language model, and uses
LLaVA's conversation format with a `<record>` token in place of `<image>`.

The plans are fresh ones from the generator of
[synthetic-floorplans-for-transcription](https://huggingface.co/datasets/benikm91/synthetic-floorplans-for-transcription).
Its companion,
[synthetic-floorplans-captioning](https://huggingface.co/datasets/jotamunz/synthetic-floorplans-captioning),
captions the same plans as a whole.

## Intended use

This set is meant to be built from **predicted records**, the output of a model transcribing a drawing,
and not from the true synthetic records. A decoder writes its entities and their embeddings in the same
order, so `E0` in a question is the decoder's first node and its first embedding.

The dataset itself does not depend on where a record came from. A row's image, record and conversation
are all built from one record, so they agree with each other and work the same whether that record is
predicted or true.

## Usage

```python
from datasets import load_dataset

plans = load_dataset("jotamunz/synthetic-floorplans-entity-captioning", split="train")
plans[3]["record"]         # the nodes and relationships the questions name
plans[3]["conversations"]  # ten questions and their answers
```

## Data format

| field | type | description |
|---|---|---|
| `id` | string | The plan's seed, padded to 7 digits. |
| `image` | Image | A 512 × 512 grayscale PNG, black on white, drawn from the record alone. |
| `record` | struct | The plan's nodes and relationships, described [below](#the-record). |
| `conversations` | list of {`from`, `value`} | Ten questions, each followed by its answer. |

```json
{"id": "0000003",
 "image": "<512 × 512 PNG>",
 "record": {"nodes": ["…"], "relationships": ["…"]},
 "conversations": [
   {"from": "human", "value": "<record>\nWhat is R2?"},
   {"from": "gpt", "value": "E14 and E0 are connected"},
   {"from": "human", "value": "What is E16?"},
   {"from": "gpt", "value": "a vertical exterior wall at the top right of the floorplan"},
   {"from": "human", "value": "What does E23 represent?"},
   {"from": "gpt", "value": "a vertical outer window at the bottom left of the architectural plan"}]}
```

`E{i}` is node `i` of the row's `record`, and `R{j}` is relationship `j`. The labels are not stored,
because the record's order defines them: a decoder that writes entities and their embeddings in that
order can match `E{i}` directly to its text and its embedding. `<record>` is a placeholder for the
training code to fill: **the rows hold the record itself but no embeddings**.

## The record

Every row holds the record it was built from, and its drawing and answers follow from that record
alone.

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

## Conversations

Each conversation asks about 10 entities, picked at random from all of the plan's entities. A true
record has about 42 nodes and 60 relationships on average, so about 41% of the questions are about nodes, 45%
about `joined` relationships and 13% about `in` relationships. No entity is asked about twice in a
conversation. The picks and all word choices come from a random generator seeded by the plan, so a plan
always gets the same conversation.

The questions are "What is E14?", "Describe E14." and "What does E14 represent?". `<record>` appears
once, in the first question, before or after the text, and later questions leave it out, as LLaVA does
with `<image>`.

## Answers

### Nodes

An answer gives which way the node runs, its class, and the sides of the plan it is clearly on:

> a horizontal window at the top left of the floor plan

A side is named only when it is clear. Around the middle of the plan's outline lies a central band,
reaching 10% of the plan's size to each side, along each axis. A node is on a side only if all of it is
past the band, so a node crossing the middle has no side on that axis. Across true nodes, 47% name both
axes, 49% name one, and 4% name neither:

| clear sides | example |
|---|---|
| both | a vertical inside wall at the top right of the floorplan |
| one | a horizontal door at the top of the architectural plan |
| none | a horizontal inside wall in the apartment layout |

Classes and the plan are each named with synonyms, such as "outside wall", "exterior wall", "external
wall" and "perimeter wall", or "floor plan", "layout" and "plan drawing". An evaluation should map them
back to their class.

### Relationships

An answer names the two nodes a relationship links. Which order the nodes appear in varies:

| type | meaning | forms |
|---|---|---|
| `joined` | Two walls share an end. | E3 is joined to E7 · E7 is linked to E3 · E3 and E7 are connected |
| `in` | An opening stands in a wall. | E41 is in E3 · E41 sits in E3 · E3 contains E41 |

For `in`, the sentence always keeps its meaning: the opening is in the wall, and the wall contains the
opening.

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
  each other but can differ from the original plan.
- The shares quoted on this card were measured on true records. Predicted records can differ.
- Entities are the record's pieces. A straight wall in the drawing can be several nodes, because the
  record ends a wall wherever another meets it, so two `joined` walls can look like one.
- Positions in answers are coarse. Exact coordinates are only in `record`.
- A relationship's answer names labels only, and does not say what the two nodes are.
- Every question is about something that exists, so the set has no negative examples.
- The plans are generated, and all their walls are horizontal or vertical.

## Regenerating

The set is built with the `instruction-tuning` branch of
[synthetic-floorplans-for-transcription](https://github.com/jotamunz/synthetic-floorplans-for-transcription/tree/instruction-tuning):
the plans from their seeds, then the conversations. A seed always gives the same plan. Questions and
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
./build_entity_captioning.sh ./plans ./entity-captioning
```

To build from predicted records, write them in the same format to `{split}_labels.jsonl`, one plan per
line with its `seed`, `nodes` and `relationships`, in place of the true ones. Then run the last two
steps: `draw_records.py` renders the images from them, and the build writes the rows.

## License

CC BY 4.0, as synthetic-floorplans-for-transcription.
