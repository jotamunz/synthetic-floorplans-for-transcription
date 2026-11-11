---
pretty_name: Synthetic Floor Plans Relationship Captioning
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

# Synthetic Floor Plans Relationship Captioning

Generated floor plans with multi-turn conversations about single relationships of their records: which
two nodes a labelled relationship links, and how. The set is meant for aligning the relationship
embeddings of a record decoder with a language model, and uses LLaVA's conversation format with a
`<record>` token in place of `<image>`.

The plans are fresh ones from the generator of
[synthetic-floorplans-for-transcription](https://huggingface.co/datasets/benikm91/synthetic-floorplans-for-transcription).
Two companions use the same plans:
[synthetic-floorplans-node-captioning](https://huggingface.co/datasets/jotamunz/synthetic-floorplans-node-captioning)
asks about the nodes themselves, and
[synthetic-floorplans-captioning](https://huggingface.co/datasets/jotamunz/synthetic-floorplans-captioning)
captions each plan as a whole.

## Intended use

This set is meant to be built from **predicted records**, the output of a model transcribing a drawing,
and not from the true synthetic records. A decoder writes its relationships and their embeddings in the
same order, so `R0` in a question is the decoder's first relationship and its first embedding.

The dataset itself does not depend on where a record came from. A row's image, record and conversation
are all built from one record, the image drawn from it, so they agree with each other and work the same
whether that record is predicted or true.

## Usage

```python
from datasets import load_dataset

plans = load_dataset("jotamunz/synthetic-floorplans-relationship-captioning", split="train")
plans[3]["record"]         # the nodes and relationships the questions and answers name
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
   {"from": "human", "value": "What is R16?\n<record>"},
   {"from": "gpt", "value": "E5 and E27 are connected"},
   {"from": "human", "value": "What does R60 represent?"},
   {"from": "gpt", "value": "E11 holds E41"},
   {"from": "human", "value": "What does R7 represent?"},
   {"from": "gpt", "value": "E29 and E2 are joined"}]}
```

`R{j}` is relationship `j` of the row's `record`, and `E{i}` in an answer is node `i`. The labels are not
stored, because the record's order defines them: a decoder that writes relationships and their
embeddings in that order can match `R{j}` directly to its text and its embedding. `<record>` is a
placeholder for the training code to fill: **the rows hold the record itself but no embeddings**.

## The record

Every row holds the whole record it was built from: the relationships the questions are about, and the
nodes their answers name. Its drawing and answers follow from that record alone.

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

Each conversation asks about 10 relationships, picked at random from the plan's relationships; a plan
with fewer, such as a single room, is asked about all of them. A true record has about 60 relationships on
average, and on the validation split 78% of the questions are about `joined` relationships and 22% about
`in`. No relationship is asked about twice in a conversation. The picks and all word choices come from a
random generator seeded by the plan, so a plan always gets the same conversation.

The questions are "What is R14?", "Describe R14." and "What does R14 represent?". `<record>` appears
once, in the first question, before or after the text, and later questions leave it out, as LLaVA does
with `<image>`.

## Answers

An answer names the two nodes a relationship links. Which order the nodes appear in varies:

| type | meaning | forms |
|---|---|---|
| `joined` | Two wall segments share an end. | E3 is joined to E7 · E7 is linked to E3 · E3 and E7 are connected |
| `in` | An opening stands in a wall segment. | E41 is in E3 · E41 sits in E3 · E3 contains E41 |

For `joined`, both orders say the same thing, so the nodes appear in either order. For `in`, the
sentence always keeps its meaning: the opening is in the wall segment, and the wall segment contains the
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
  each other but can differ from the original plan. A tilted segment, which only a prediction holds,
  is drawn tilted.
- The shares quoted on this card were measured on true records. Predicted records can differ, and a
  model that writes no relationships gives this set nothing to ask about.
- An answer names node labels only, and does not say what the two nodes are; the node captioning set
  covers that.
- Two `joined` wall segments can look like one straight wall in the drawing.
- Every question is about something that exists, so the set has no negative examples.
- The plans are generated, and all their walls are horizontal or vertical.

## Regenerating

The set is built with the `instruction-tuning` branch of
[synthetic-floorplans-for-transcription](https://github.com/jotamunz/synthetic-floorplans-for-transcription/tree/instruction-tuning):
the plans from their seeds, a transcribing model's records of them, drawn anew, then the
conversations. A seed always gives the same plan. Questions and
words are in `captioning_config/entity_vocabulary.json`.

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
each with a `predicted_records.jsonl` of one plan per line, its `seed`, `nodes` and `relationships`, in the
format of the plans' own. `build_predicted.sh` lays them out as the plans are and draws them, and the
set is built from them:

```bash
./build_predicted.sh ./predictions ./predicted
./build_relationship_captioning.sh ./predicted ./relationship-captioning
```

Built from `./plans` instead, the set holds the true records.

## License

CC BY 4.0, as synthetic-floorplans-for-transcription.
