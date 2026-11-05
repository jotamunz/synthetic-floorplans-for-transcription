# synthetic-floorplans-for-transcription

This repo generates synthetic floor plans and builds three datasets from them:

| dataset | what a row holds | used for |
|---|---|---|
| [Transcription](#transcription) | a drawing and its record | training a model to read a drawing into a record |
| [Captioning](#captioning) | a drawing and a caption of the whole plan | aligning a vision encoder with a language model |
| [Entity captioning](#entity-captioning) | a drawing and captions of single entities | aligning a record decoder's embeddings with a language model |

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

### Entity captioning

The same plans, each with its record and a conversation of questions about ten of its entities, picked
at random by seed. The record's order names them: `E0`, `E1`, … are its nodes and `R0`, `R1`, … its
relationships. `<record>` stands in the first question only. See
[its card](dataset_cards/synthetic-floorplans-entity-captioning.md).

A node is answered by which way it runs, its class and the sides of the plan it is clearly on, a
relationship by the two nodes it links, in either order:

> What is E16? → a vertical exterior wall at the top right of the floorplan
>
> What is R2? → E14 and E0 are connected

Built from a built transcription dataset:

```
./build_entity_captioning.sh ./synthetic ./entity-captioning
```

or a split at a time:

```
uv run create_entity_captioning.py --dataset-dir ./synthetic --split val --output-dir ./entity-captioning
```

Its questions and words are in `captioning_config/entity_vocabulary.json`, and its class names in
`captioning_config/vocabulary.json`, shared with image captioning.