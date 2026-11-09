"""The relationship captioning set of a split, in the conversation format of LLaVA and stored as a Hugging
Face dataset: per plan its id, its image (an `Image` feature, the PNG inside the row), its whole record
and one conversation of questions about some of its relationships, each answered by the two nodes it
links.

The relationships are labelled `R0`, `R1`, … and the nodes they link `E0`, `E1`, …, each in the record's
order. A conversation asks about `QUESTIONS` relationships, picked at random from the plan's seed, in the
order they are picked; the record is named once, by `<record>` in the first question. The questions and
the words of the answers are in `captioning_config/entity_vocabulary.json`, which the node captioning set
shares.

The splits are read and written as `captioning_utils.py` does for every captioning set.

    uv run create_relationship_captioning.py --dataset-dir ./synthetic --split val --output-dir ./relationship-captioning
    uv run create_relationship_captioning.py --dataset-dir ./synthetic --split train --output-dir ./relationship-captioning
"""

import random

from datasets import Features, Image as ImageFeature, List, Value

from captioning_utils import RECORD_FEATURE, config, entity_conversation, parse_args, write_split

#: The questions, each with a place for the label it asks about and with the record before or after it,
#: and per type of relationship the forms of its answer, each its templates and the verbs that fit them.
VOCABULARY = config("entity_vocabulary.json")

#: How many of a plan's relationships its conversation asks about; a plan with fewer is asked about all.
QUESTIONS = 10

FEATURES = Features({"id": Value("string"), "image": ImageFeature(), "record": RECORD_FEATURE,
                     "conversations": List({"from": Value("string"), "value": Value("string")})})
SCHEMA = FEATURES.arrow_schema


def labels(record: dict) -> list:
    """'R0', 'R1', …: a label for every relationship of the record, in its order."""
    return [f"R{at}" for at in range(len(record["relationships"]))]


def answer(rng: random.Random, relationship: dict) -> str:
    """'E3 is joined to E7', 'E7 and E3 are linked', 'E41 sits in E3', 'E3 contains E41' …: the two nodes it
    links. A form of the type is picked, then one of its templates and verbs; in a template `a` and `b` are
    the nodes in the record's order."""
    first, second = relationship["nodes"]
    form = rng.choice(VOCABULARY["relationships"][relationship["type"]])
    return rng.choice(form["templates"]).format(a=f"E{first}", b=f"E{second}", verb=rng.choice(form["verbs"]))


def conversation(record: dict) -> list:
    """Questions about some of the plan's relationships, each answered by the nodes it links."""
    rng = random.Random(record["seed"])
    return entity_conversation(rng, VOCABULARY["prompts"], labels(record), QUESTIONS,
                               lambda at: answer(rng, record["relationships"][at]))


def main() -> None:
    write_split(parse_args("The LLaVA relationship captioning set of a split.", "relationship-captioning"), SCHEMA,
                conversation)


if __name__ == "__main__":
    main()
