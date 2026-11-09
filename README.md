# synthetic-floorplans-for-transcription

This repo generates synthetic floor plans and builds four datasets from them:

| dataset | what a row holds | used for |
|---|---|---|
| [Transcription](#transcription) | a drawing and its record | training a model to read a drawing into a record |
| [Captioning](#captioning) | a drawing and a caption of the whole plan | aligning a vision encoder with a language model |
| [Node captioning](#node-captioning) | a drawing, its nodes and captions of single nodes | aligning a record decoder's node embeddings with a language model |
| [Relationship captioning](#relationship-captioning) | a drawing, its record and captions of single relationships | aligning a record decoder's relationship embeddings with a language model |

Every plan is generated from its seed, so a seed always gives the same plan, record and drawing. A
record lists the plan's walls, doors and windows, which walls are joined and which wall each opening
stands in. A plan keeps the rules a plan must -- walls horizontal or vertical, exactly one wall between
two rooms, every room reachable and opening off the living room, windows only in outside walls -- and
its drawing is drawn from the record alone.

## Datasets

### Transcription

Per plan, the drawing a model reads and the record it should write: 450,000 training plans in nine
parts, 5,000 each for validation and test, images at 512 px. What a plan holds and how it is made is in
[its card](dataset_cards/synthetic.md), which each build copies into its folder as `README.md`.

```
./build_synthetic.sh ./synthetic
```

Generating is what takes the time, about 75,000 plans an hour on ten cores; drawing is a few minutes
for all of them. The steps on their own:

```
uv run create_synthetic_records.py --output-dir ./synthetic --split val --first-seed 1000000 --count 5000
uv run draw_records.py --dataset-dir ./synthetic --split val                # --size 256 for smaller images
uv run preview_records.py --dataset-dir ./synthetic --split train-00 --output ./synthetic/train_examples_8x16.png
uv run describe_records.py --dataset-dir ./synthetic --card dataset_cards/synthetic.md
```

- `synthetic.py` generates a plan from its seed, checks it and writes its record;
- `draw.py` draws a record;

#### Browse the plans

Sample plans by seed in a web UI, each drawn from its record with the record over it; the rooms it
was laid out from can be shown too.

```
uv run view_synthetic.py
```

#### Compare with ResPlan

How the plans compare with the real ones of [ResPlan](https://github.com/m-agour/ResPlan): outline
corners, and per kind of room how many doors it has.

```
./download_resplan.sh
uv run compare_resplan.py --resplan ./source/ResPlan.pkl
```

### Captioning

The same plans in the conversation format of LLaVA-Pretrain: per plan its image, one of LLaVA-Pretrain's
own questions, and a caption of how many walls, doors, windows and rooms it has, which way it is shaped
and where its entrance is. See [its card](dataset_cards/synthetic-floorplans-captioning.md).

> a tall layout comprising a few doors, multiple exterior walls, a handful of internal walls, the front
> door at the right side and some outer windows.

Built from a built transcription dataset:

```
./build_captioning.sh ./synthetic ./captioning
```

or a split at a time (`--split train` reads `train-00` … in order):

```
uv run create_captioning.py --dataset-dir ./synthetic --split val --output-dir ./captioning
```

Its questions and words are in `captioning_config/vocabulary.json`.

### Node captioning

The same plans, each with the nodes of its record and a conversation of questions about ten of them,
picked at random by seed. The record's order names them `E0`, `E1`, …, and `<record>` stands in the first
question only. A node is answered by which way it runs, its class and the sides of the plan it is
clearly on. See [its card](dataset_cards/synthetic-floorplans-node-captioning.md).

> What is E7? → a horizontal external wall segment at the top right of the apartment layout

```
./build_node_captioning.sh ./synthetic ./node-captioning
```

### Relationship captioning

The same plans, each with its whole record and a conversation of questions about ten of its
relationships, picked at random by seed. The record's order names them `R0`, `R1`, …, and a relationship
is answered by the two nodes it links, in either order. See
[its card](dataset_cards/synthetic-floorplans-relationship-captioning.md).

> What is R16? → E5 and E27 are connected

```
./build_relationship_captioning.sh ./synthetic ./relationship-captioning
```

Both are built from a built transcription dataset, or a split at a time:

```
uv run create_node_captioning.py --dataset-dir ./synthetic --split val --output-dir ./node-captioning
uv run create_relationship_captioning.py --dataset-dir ./synthetic --split val --output-dir ./relationship-captioning
```

Their questions and words are in `captioning_config/entity_vocabulary.json`, and the node captioning
set's class names in `captioning_config/vocabulary.json`, shared with image captioning.