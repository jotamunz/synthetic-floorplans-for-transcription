"""The entity captioning set of a split, in the conversation format of LLaVA and stored as a Hugging Face
dataset: per plan its id, its image (an `Image` feature, the PNG inside the row), its record and one
conversation of questions about some of its entities. A node is answered by its class, which way it
runs and the sides of the plan it is clearly on; a relationship by the two nodes it links.

An entity is a node or a relationship of the plan's record: nodes are labelled `E0`, `E1`, … and
relationships `R0`, `R1`, …, each in the record's order. A conversation asks about `QUESTIONS` of them,
picked at random from the plan's seed, in the order they are picked; the record is named once, by
`<record>` in the first question, as LLaVA names its image once. The questions and the words of the
answers are in `captioning_config/entity_vocabulary.json`.

The splits are read and written as `captioning_utils.py` does for every captioning set.

    uv run create_entity_captioning.py --dataset-dir ./synthetic --split val --output-dir ./entity-captioning
    uv run create_entity_captioning.py --dataset-dir ./synthetic --split train --output-dir ./entity-captioning
"""

import random

import numpy as np
from datasets import Features, Image as ImageFeature, List, Value

from captioning_utils import RECORD_FEATURE, config, parse_args, stored_record, write_split

#: The questions, each with a place for the label it asks about and with the record before or after it,
#: the templates of a node's answer with and without a position, and its words: a list per direction, row
#: and column (see `place`), and what it calls the plan; and per type of relationship the forms of its
#: answer, each its templates and the verbs that fit them.
VOCABULARY = config("entity_vocabulary.json")

#: What each class is called, the same names the image captioning set uses.
CLASSES = config("vocabulary.json")["classes"]

#: How many of a plan's entities its conversation asks about; a plan with fewer is asked about all.
QUESTIONS = 10

#: What names the record in a question.
RECORD = "<record>"

#: How far the central band reaches from the plan's middle along each axis, as a share of the plan's size
#: along it. An entity is on a side only if all of it is past the band, so one crossing the middle is not.
CLEAR = 0.1

FEATURES = Features({"id": Value("string"), "image": ImageFeature(), "record": RECORD_FEATURE,
                     "conversations": List({"from": Value("string"), "value": Value("string")})})
SCHEMA = FEATURES.arrow_schema


def labels(record: dict) -> list:
    """'E0', 'E1', …, 'R0', 'R1', …: a label for every node of the record and then every relationship,
    each in its order."""
    return [f"E{at}" for at in range(len(record["nodes"]))] + [f"R{at}" for at in range(len(record["relationships"]))]


def question(rng: random.Random, label: str, first: bool) -> str:
    """A question about `label`; only the first of a conversation names the record."""
    text = rng.choice(VOCABULARY["prompts"]).format(label=label)
    return text if first else text.replace(RECORD, "").strip()


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
    layout', 'a horizontal partition wall in the floorplan' …: only the sides it is clearly on are named."""
    direction, row_at, column = place(node, low, high)
    position = " ".join(f"{rng.choice(VOCABULARY['rows'][row_at])} {rng.choice(VOCABULARY['columns'][column])}".split())
    template = VOCABULARY["answer"] if position else VOCABULARY["unplaced_answer"]
    return template.format(direction=rng.choice(VOCABULARY["directions"][direction]),
                           kind=rng.choice(CLASSES[node["class"]]["singular"]), position=position,
                           plan=rng.choice(VOCABULARY["plans"]))


def relationship_answer(rng: random.Random, relationship: dict) -> str:
    """'E3 is joined to E7', 'E7 and E3 are linked', 'E41 is set in E3', 'E3 contains E41' …: the two nodes
    it links. A form of the type is picked, then one of its templates and verbs; in a template `a` and `b`
    are the nodes in the record's order."""
    first, second = relationship["nodes"]
    form = rng.choice(VOCABULARY["relationships"][relationship["type"]])
    return rng.choice(form["templates"]).format(a=f"E{first}", b=f"E{second}", verb=rng.choice(form["verbs"]))


def conversation(record: dict) -> list:
    """Questions about some of the plan's entities, each answered by what it is."""
    rng = random.Random(record["seed"])
    named = labels(record)
    nodes = len(record["nodes"])
    low, high = bounds(record)
    turns = []
    for at in rng.sample(range(len(named)), min(QUESTIONS, len(named))):
        asked = question(rng, named[at], first=not turns)
        said = answer(rng, record["nodes"][at], low, high) if at < nodes else relationship_answer(rng, record["relationships"][at - nodes])
        turns += [{"from": "human", "value": asked}, {"from": "gpt", "value": said}]
    return turns


def row(record: dict, image: bytes) -> dict:
    name = f"{record['seed']:07d}"
    return {"id": name, "image": {"bytes": image, "path": f"{name}.png"}, "record": stored_record(record),
            "conversations": conversation(record)}


def main() -> None:
    write_split(parse_args("The LLaVA entity captioning set of a split.", "entity-captioning"), SCHEMA, row)


if __name__ == "__main__":
    main()
