"""The entity captioning set of a split, in the conversation format of LLaVA and stored as a Hugging Face
dataset: per plan its id, its image (an `Image` feature, the PNG inside the row), the labels of its
entities and one conversation of questions about some of them, each answered by the entity's class.

An entity is a node of the plan's record, labelled `E0`, `E1`, … in the record's order. A conversation
asks about `QUESTIONS` of them, picked at random from the plan's seed, in the order they are picked;
the record is named once, by `<record>` in the first question, as LLaVA names its image once. The
questions are in `captioning_config/entity_vocabulary.json`.

The splits are read and written as `captioning_utils.py` does for every captioning set.

    uv run create_entity_captioning.py --dataset-dir ./synthetic --split val --output-dir ./entity-captioning
    uv run create_entity_captioning.py --dataset-dir ./synthetic --split train --output-dir ./entity-captioning
"""

import random

from datasets import Features, Image as ImageFeature, List, Value

from captioning_utils import config, parse_args, write_split

#: The questions, each with a place for the label it asks about and with the record before or after it.
VOCABULARY = config("entity_vocabulary.json")

#: How many of a plan's entities its conversation asks about; a plan with fewer is asked about all.
QUESTIONS = 10

#: What names the record in a question.
RECORD = "<record>"

FEATURES = Features({"id": Value("string"), "image": ImageFeature(), "labels": List(Value("string")),
                     "conversations": List({"from": Value("string"), "value": Value("string")})})
SCHEMA = FEATURES.arrow_schema


def labels(record: dict) -> list:
    """'E0', 'E1', …: a label for every node of the record, in its order."""
    return [f"E{at}" for at in range(len(record["nodes"]))]


def question(rng: random.Random, label: str, first: bool) -> str:
    """A question about `label`; only the first of a conversation names the record."""
    text = rng.choice(VOCABULARY["prompts"]).format(label=label)
    return text if first else text.replace(RECORD, "").strip()


def conversation(record: dict) -> list:
    """Questions about some of the plan's entities, each answered by its class."""
    rng = random.Random(record["seed"])
    named = labels(record)
    turns = []
    for at in rng.sample(range(len(named)), min(QUESTIONS, len(named))):
        turns += [{"from": "human", "value": question(rng, named[at], first=not turns)},
                  {"from": "gpt", "value": record["nodes"][at]["class"]}]
    return turns


def row(record: dict, image: bytes) -> dict:
    name = f"{record['seed']:07d}"
    return {"id": name, "image": {"bytes": image, "path": f"{name}.png"}, "labels": labels(record),
            "conversations": conversation(record)}


def main() -> None:
    write_split(parse_args("The LLaVA entity captioning set of a split.", "entity-captioning"), SCHEMA, row)


if __name__ == "__main__":
    main()
