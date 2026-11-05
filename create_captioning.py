"""The image captioning set of a split, in the conversation format of LLaVA-Pretrain
(https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) and stored as a Hugging Face dataset: per
plan its id, its image (an `Image` feature, the PNG inside the row) and one conversation asking for a
summary of the image, answered by how many of each class its record holds, in words ('a single', 'a
few', 'many' …), in a shuffled order. The question is one of LLaVA-Pretrain's own, taken in turn by
seed; the caption's choices are drawn from the seed too.

The splits are read and written as `captioning_utils.py` does for every captioning set.

    uv run create_captioning.py --dataset-dir ./synthetic --split val --output-dir ./captioning
    uv run create_captioning.py --dataset-dir ./synthetic --split train --output-dir ./captioning
"""

import random
import re
from collections import Counter

import numpy as np
import shapely
from datasets import Features, Image as ImageFeature, List, Value
from shapely.geometry import LineString

from captioning_utils import RECORD_FEATURE, config, parse_args, stored_record, write_split

#: The questions and the words a caption is made of: what it calls the image (with a place for its
#: orientation), the words for how the plan lies (a list per orientation, see `orientation`), the verb
#: leading into what the image holds, the words for how many of a class there are (a list per tier, see
#: `tier`), and each class's name.
VOCABULARY = config("vocabulary.json")

#: The questions LLaVA-Pretrain opens its conversations with, each phrase with the image before and after it.
PROMPTS = VOCABULARY["prompts"]

#: Per class, the limits between the tiers past one, in rising order: a count up to the first limit is
#: tier 2, up to the second tier 3, and so on; past the last it is the top tier.
LIMITS = {"outside wall": (6, 10), "inside wall": (6, 10), "door": (3, 7), "outside door": (3, 7), "window": (4, 8),
          "room": (4, 8)}

#: How often a plan's doors are named by the rooms they lead into instead.
ROOMS_SHARE = 0.5
#: How often windows and doors are said to stand in the walls they do.
PLACEMENT_SHARE = 0.3
#: The classes a caption counts by the walls the drawing shows, its straight runs, not the record's pieces.
RUNS = ("outside wall", "inside wall")

# Tiers 0 and 1, then one more than there are limits: the vocabulary needs words for each.
for _kind, _limits in LIMITS.items():
    if len(VOCABULARY["quantities"]) != len(_limits) + 3:
        raise ValueError(f"{_kind}: {len(_limits)} limits make {len(_limits) + 3} tiers, "
                         f"but the vocabulary has words for {len(VOCABULARY['quantities'])}")

#: How many times longer one side of the outline may be than the other for it to be evidently square,
#: and must be at least for it to be evidently wide or tall; in between, a caption names no orientation.
SQUARE = 1.1
ELONGATED = 1.5

FEATURES = Features({"id": Value("string"), "image": ImageFeature(), "record": RECORD_FEATURE,
                     "conversations": List({"from": Value("string"), "value": Value("string")})})
SCHEMA = FEATURES.arrow_schema


def prompt(seed: int) -> str:
    """The question for a plan: the prompts in turn, by seed, so that consecutive plans get the next one."""
    return PROMPTS[seed % len(PROMPTS)]


def tier(kind: str, count: int) -> int:
    """The tier of `count` of a class, the index of its words in the vocabulary's quantities: 0 for
    none, 1 for one, then 2 plus how many of the class's limits the count is past."""
    if count <= 1:
        return count
    return 2 + sum(count > limit for limit in LIMITS[kind])


def straight_runs(record: dict, kind: str) -> int:
    """How many walls of a class a plan shows: its straight runs. The record ends a wall wherever another
    meets it, so that walls are joined where they share an end; a run joins the pieces of a class that
    carry on along the same line from where one ends, as the drawing shows them, one wall."""
    pieces = sorted((y0 == y1, y0 if y0 == y1 else x0, min(x0, x1) if y0 == y1 else min(y0, y1),
                     max(x0, x1) if y0 == y1 else max(y0, y1))
                    for node in record["nodes"] if node["class"] == kind for (x0, y0), (x1, y1) in [node["points"]])
    runs = 0
    for at, (horizontal, line, start, _) in enumerate(pieces):
        # A piece starts a run unless the one before it on the same line ends where it starts.
        before = pieces[at - 1] if at else None
        runs += not (before and before[0] == horizontal and before[1] == line and before[3] == start)
    return runs


def orientation(record: dict) -> int:
    """How the plan's outline lies, the index of its words in the vocabulary's orientations: 0 when it is
    not evident, 1 square, 2 wide, 3 tall. Measured on the box around the outside walls."""
    points = np.array([point for node in record["nodes"] if node["class"] == "outside wall" for point in node["points"]])
    width, height = np.ptp(points, axis=0)
    if max(width, height) <= SQUARE * min(width, height):
        return 1
    if width >= ELONGATED * height:
        return 2
    if height >= ELONGATED * width:
        return 3
    return 0


def entrance(record: dict) -> int | None:
    """Which side of the plan its entrance is on, the index of its words in the vocabulary's entrance
    sides: 0 top, 1 bottom, 2 left, 3 right; None unless the plan has exactly one entrance.

    A horizontal entrance faces the top or the bottom, a vertical one the left or the right; which of
    the two, by where the outside is: the footprint, the outside walls joined into polygons, lies on
    the other side of it. So an entrance in a facade set back into the plan faces the way it opens.
    """
    doors = [node["points"] for node in record["nodes"] if node["class"] == VOCABULARY["entrance"]["class"]]
    if len(doors) != 1:
        return None
    (x0, y0), (x1, y1) = doors[0]
    walls = [LineString(node["points"]) for node in record["nodes"] if node["class"] == "outside wall"]
    footprint = shapely.union_all(shapely.get_parts(shapely.polygonize(walls)))
    middle_x, middle_y = (x0 + x1) / 2, (y0 + y1) / 2
    # A pixel past the entrance towards the top, or the left: inside the footprint, it faces the other way.
    # The image's y runs downwards, so the top is where y is smaller.
    if y0 == y1:
        return 1 if shapely.contains_xy(footprint, middle_x, middle_y - 1) else 0
    return 3 if shapely.contains_xy(footprint, middle_x - 1, middle_y) else 2


def entrance_phrase(rng: random.Random, side: int) -> str:
    """'an entrance on the left wall', 'the entry door along the top side' …: a plan has one entrance, so
    it is named and placed, not counted."""
    words = VOCABULARY["entrance"]
    noun = rng.choice(VOCABULARY["classes"][words["class"]]["singular"])
    return (f"{rng.choice(words['articles'])} {noun} {rng.choice(words['prepositions'])} the {rng.choice(words['sides'][side])} "
            f"{rng.choice(words['nouns'])}")


def amount(rng: random.Random, kind: str, count: int, name: dict) -> str:
    """'no inside walls', 'an entrance', 'a single window', 'a few doors', 'many outside walls' …"""
    noun = rng.choice(name["singular"] if count == 1 else name["plural"])
    word = rng.choice(VOCABULARY["quantities"][tier(kind, count)])
    return f"{word} {noun}"


def placement(rng: random.Random, kind: str) -> str:
    """'in the exterior walls', 'on the partition walls' …: the walls a class's openings stand in."""
    words = VOCABULARY["placements"]
    walls = VOCABULARY["classes"][words["walls"][kind]]["plural"]
    return f"{rng.choice(words['prepositions'])} the {rng.choice(walls)}"


def rooms(rng: random.Random, doors: int) -> str:
    """'a single room', 'a few rooms', 'many rooms' …: every room but the first is reached through a door
    of its own, so a plan has one room more than doors."""
    if doors == 0:
        return VOCABULARY["rooms"]["single"]
    return amount(rng, "room", doors + 1, VOCABULARY["rooms"])


def caption(record: dict) -> str:
    """'the wide plan of a floor showing several windows, many outside walls, the entrance on the left
    side, a lot of inside walls and a few doors.'

    The doors are named either as doors or as the rooms they lead into, and always as 'a single room' when
    there are none; a plan of a single room leaves out its wall counts. The entrance is placed, not
    counted. Windows and doors are sometimes said to stand in their walls.
    Every word choice, these ones and the order of the classes are drawn at random, from the plan's seed,
    so that a plan always gets the same caption.
    """
    rng = random.Random(record["seed"])
    counts = Counter(node["class"] for node in record["nodes"])
    for kind in RUNS:
        counts[kind] = straight_runs(record, kind)
    doors = VOCABULARY["rooms"]["class"]
    single = counts[doors] == 0
    kinds = [kind for kind in VOCABULARY["classes"] if not (single and kind in VOCABULARY["rooms"]["single_leaves_out"])]
    side = entrance(record)
    items = []
    for kind in kinds:
        if kind == doors and (single or rng.random() < ROOMS_SHARE):
            items.append(rooms(rng, counts[kind]))
        elif kind == VOCABULARY["entrance"]["class"] and side is not None:
            items.append(entrance_phrase(rng, side))
        else:
            phrase = amount(rng, kind, counts[kind], VOCABULARY["classes"][kind])
            if kind in VOCABULARY["placements"]["walls"] and rng.random() < PLACEMENT_SHARE:
                phrase = f"{phrase} {placement(rng, kind)}"
            items.append(phrase)
    rng.shuffle(items)
    subject = rng.choice(VOCABULARY["subjects"]).format(
        orientation=rng.choice(VOCABULARY["orientations"][orientation(record)]))
    # An orientation left out leaves two spaces where it would have been.
    subject = " ".join(subject.split())
    return articles(f"{subject} {rng.choice(VOCABULARY['verbs'])} {', '.join(items[:-1])} and {items[-1]}.")


def articles(text: str) -> str:
    """'a' before a vowel made 'an', wherever the words put together set one there."""
    return re.sub(r"\ba (?=[aeiou])", "an ", text)


def row(record: dict, image: bytes) -> dict:
    name = f"{record['seed']:07d}"
    return {"id": name, "image": {"bytes": image, "path": f"{name}.png"}, "record": stored_record(record),
            "conversations": [{"from": "human", "value": prompt(record["seed"])},
                              {"from": "gpt", "value": caption(record)}]}


def main() -> None:
    write_split(parse_args("The LLaVA image captioning set of a split.", "captioning"), SCHEMA, row)


if __name__ == "__main__":
    main()
