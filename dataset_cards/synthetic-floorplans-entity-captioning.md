---
pretty_name: Synthetic Floor Plans Entity Captioning
license: cc-by-4.0
language:
  - en
size_categories:
  - 100K<n<1M
task_categories:
  - image-text-to-text
source_datasets:
  - benikm91/synthetic-floorplans-for-transcription
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

The plans are those of
[synthetic-floorplans-for-transcription](https://huggingface.co/datasets/benikm91/synthetic-floorplans-for-transcription),
and every answer is computed from the plan's record. Its companion,
[synthetic-floorplans-captioning](https://huggingface.co/datasets/jotamunz/synthetic-floorplans-captioning),
captions the same plans as a whole.

```python
from datasets import load_dataset

plans = load_dataset("jotamunz/synthetic-floorplans-entity-captioning", split="train")
plans[3]["labels"]         # ["E0", "E1", …, "R0", "R1", …]
plans[3]["conversations"]  # ten questions and their answers
```

## Data format

| field | type | description |
|---|---|---|
| `id` | string | The plan's seed, padded to 7 digits. |
| `image` | Image | A 512 × 512 grayscale PNG, black on white, with the same pixels as the source dataset. |
| `labels` | list of strings | A label for every entity of the record: `E0`, `E1`, … then `R0`, `R1`, … |
| `conversations` | list of {`from`, `value`} | Ten questions, each followed by its answer. |

```json
{"id": "0000003",
 "image": "<512 × 512 PNG>",
 "labels": ["E0", "E1", "…", "E27", "R0", "R1", "…", "R36"],
 "conversations": [
   {"from": "human", "value": "<record>\nWhat is R2?"},
   {"from": "gpt", "value": "E14 and E0 are connected"},
   {"from": "human", "value": "What is E16?"},
   {"from": "gpt", "value": "a vertical exterior wall at the top right of the floorplan"},
   {"from": "human", "value": "What does E23 represent?"},
   {"from": "gpt", "value": "a vertical outer window at the bottom left of the architectural plan"}]}
```

**The rows do not hold the record or any embeddings.** `<record>` is a placeholder for the training
code to fill. `E{i}` is node `i` of the plan's record and `R{j}` is relationship `j`, in the record's
order. The record is found in the source dataset by the plan's seed, which is its `id`.

## Conversations

Each conversation asks about 10 entities, picked at random from all of the plan's labels. A plan has
about 42 nodes and 60 relationships on average, so about 41% of the questions are about nodes, 45%
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
past the band, so a node crossing the middle has no side on that axis. Across all nodes, 47% name both
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
| train | 450,000 | 0 – 449,999 | `data/train-0000k-of-00009.parquet`, one per source part |
| validation | 5,000 | 1,000,000 – 1,004,999 | `data/val-00000-of-00001.parquet` |
| test | 5,000 | 2,000,000 – 2,004,999 | `data/test-00000-of-00001.parquet` |

A plan's record is on line `seed − first seed` of its part in the source dataset. Shard `k` of the
training split holds source part `train-0k`, which starts at seed 50,000 k.

## Limitations

- Answers are computed from ground-truth records. A model reading predicted records will meet missing,
  extra or shifted entities that this set does not show, and its labels will not match these.
- Entities are the record's pieces. A straight wall in the drawing can be several nodes, because the
  record ends a wall wherever another meets it, so two `joined` walls can look like one.
- Positions are coarse, and exact coordinates are not given.
- A relationship's answer names labels only, and does not say what the two nodes are.
- Every question is about something that exists, so the set has no negative examples.
- The plans are generated, and all their walls are horizontal or vertical.

## Regenerating

The set is built from the source dataset with the `instruction-tuning` branch of
[synthetic-floorplans-for-transcription](https://github.com/jotamunz/synthetic-floorplans-for-transcription/tree/instruction-tuning).
Questions and words are in `captioning_config/entity_vocabulary.json`, and class names in
`captioning_config/vocabulary.json`.

```bash
./build_synthetic.sh ./synthetic
./build_entity_captioning.sh ./synthetic ./entity-captioning
```

## License

CC BY 4.0, as the source dataset.
