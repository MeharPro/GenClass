"""W8: null, "" and desc == label all render a choice item as the bare label (matches api.normalize_request)."""

from jev_local.schema import ChoiceQuestion
from jev_local.serialize import question_block


def test_w8_bare_label_renderings_are_identical():
    variants = [
        {"Positive": None, "Negative": None},
        {"Positive": "", "Negative": ""},
        {"Positive": "Positive", "Negative": " Negative "},
    ]
    blocks = [question_block("q", ChoiceQuestion(instructions="Sentiment?", criteria=c)) for c in variants]
    assert all(b.items == ("Positive", "Negative") for b in blocks)
    assert all(b.labels == ("Positive", "Negative") for b in blocks)


def test_w8_real_description_still_rendered():
    b = question_block("q", ChoiceQuestion(instructions="Sentiment?", criteria={"Positive": "good feeling", "Negative": None}))
    assert b.items == ("Positive: good feeling", "Negative")
