"""The node captioning set of a split, in the conversation format of LLaVA and stored as a Hugging Face
dataset: per plan its id, its image (an `Image` feature, the PNG inside the row), its record and one
conversation of questions about some of its nodes, each answered by the node's class, which way it runs
and the sides of the plan it is clearly on.

The nodes are labelled `E0`, `E1`, … in the record's order. A conversation asks about `QUESTIONS` of them,
picked at random from the plan's seed, in the order they are picked; the record is named once, by
`<record>` in the first question. The questions and the words of the answers are in
`captioning_config/entity_vocabulary.json`, which the relationship captioning set shares.

The splits are read and written as `captioning_utils.py` does for every captioning set.

    uv run create_node_captioning.py --dataset-dir ./synthetic --split val --output-dir ./node-captioning
    uv run create_node_captioning.py --dataset-dir ./synthetic --split train --output-dir ./node-captioning
"""

import random

import numpy as np
from datasets import Features, Image as ImageFeature, List, Value

from captioning_utils import RECORD_FEATURE, config, entity_conversation, parse_args, write_split

#: The questions, each with a place for the label it asks about and with the record before or after it,
#: the templates of an answer with and without a position, and its words: a list per direction, row and
#: column (see `place`), what it calls the plan, and the names of the classes it names differently.
VOCABULARY = config("entity_vocabulary.json")

#: What each class is called: the names the image captioning set uses, except where the entity config
#: names a class differently. A wall here is one piece of the record, so walls are named as segments; the
#: image captioning set counts walls as the drawing shows them, as straight runs.
CLASSES = {**config("vocabulary.json")["classes"], **VOCABULARY.get("classes", {})}

#: How many of a plan's nodes its conversation asks about; a plan with fewer is asked about all.
QUESTIONS = 10

#: How far the central band reaches from the plan's middle along each axis, as a share of the plan's size
#: along it. A node is on a side only if all of it is past the band, so one crossing the middle is not.
CLEAR = 0.1

FEATURES = Features({"id": Value("string"), "image": ImageFeature(), "record": RECORD_FEATURE,
                     "conversations": List({"from": Value("string"), "value": Value("string")})})
SCHEMA = FEATURES.arrow_schema


def labels(record: dict) -> list:
    """'E0', 'E1', …: a label for every node of the record, in its order."""
    return [f"E{at}" for at in range(len(record["nodes"]))]


def bounds(record: dict) -> tuple:
    """The lowest and highest x and y of the plan's outside walls."""
    points = np.array([point for node in record["nodes"] if node["class"] == "outside wall" for point in node["points"]])
    return points.min(axis=0), points.max(axis=0)


def side(start: float, end: float, low: float, high: float) -> int:
    """Which side of the plan a stretch from `start` to `end` is clearly on, along one axis running from
    `low` to `high`: 1 before the central band, 2 after it, 0 when any of it is in the band."""
    band = CLEAR * (high - low)
    centre = (low + high) / 2
    if max(start, end) < centre - band:
        return 1
    if min(start, end) > centre + band:
        return 2
    return 0


def place(node: dict, low: np.ndarray, high: np.ndarray) -> tuple:
    """Which way a node runs and where it clearly lies, the indices of their words in the vocabulary:
    direction 0 horizontal, 1 vertical; row 0 not clear, 1 top, 2 bottom; column 0 not clear, 1 left,
    2 right. The image's y runs downwards, so the top is where y is smaller."""
    (x0, y0), (x1, y1) = node["points"]
    return int(y0 != y1), side(y0, y1, low[1], high[1]), side(x0, x1, low[0], high[0])


def answer(rng: random.Random, node: dict, low: np.ndarray, high: np.ndarray) -> str:
    """'a horizontal window at the top left of the floor plan', 'a vertical doorway at the bottom of the
    layout', 'a horizontal partition wall segment in the floorplan' …: only the sides it is clearly on are
    named."""
    direction, row_at, column = place(node, low, high)
    position = " ".join(f"{rng.choice(VOCABULARY['rows'][row_at])} {rng.choice(VOCABULARY['columns'][column])}".split())
    template = VOCABULARY["answer"] if position else VOCABULARY["unplaced_answer"]
    return template.format(direction=rng.choice(VOCABULARY["directions"][direction]),
                           kind=rng.choice(CLASSES[node["class"]]["singular"]), position=position,
                           plan=rng.choice(VOCABULARY["plans"]))


def conversation(record: dict) -> list:
    """Questions about some of the plan's nodes, each answered by what it is and where."""
    rng = random.Random(record["seed"])
    low, high = bounds(record)
    return entity_conversation(rng, VOCABULARY["prompts"], labels(record), QUESTIONS,
                               lambda at: answer(rng, record["nodes"][at], low, high))


def main() -> None:
    write_split(parse_args("The LLaVA node captioning set of a split.", "node-captioning"), SCHEMA, conversation)


if __name__ == "__main__":
    main()
