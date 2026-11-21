"""Unit tests for the span-level metric (python -m pytest tests -q)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pii_rl.metrics import aggregate, composite_reward, score_sample  # noqa: E402

SRC = "Dear John Smith, your passport 123456789 expires. Mail john@x.com today."
SPANS = [
    {"start": 5, "end": 9, "label": "GIVENNAME1"},
    {"start": 10, "end": 15, "label": "LASTNAME1"},
    {"start": 31, "end": 40, "label": "PASSPORT"},
    {"start": 55, "end": 65, "label": "EMAIL"},
]
TGT = "Dear [GIVENNAME1] [LASTNAME1], your passport [PASSPORT] expires. Mail [EMAIL] today."
W = {"w_protect": 2.0, "w_typed": 1.0, "w_preserve": 1.5, "w_invalid": 0.5, "w_halluc": 0.5}


def test_spans_are_correct():
    for s in SPANS:
        assert SRC[s["start"]:s["end"]]


def status(pred):
    return [st for _, st in score_sample(SRC, pred, SPANS, TGT)["spans"]]


def test_perfect():
    assert status(TGT) == ["MASKED_CORRECT"] * 4
    assert composite_reward(SRC, TGT, SPANS, W) == 4.5


def test_identity_leaks_everything():
    assert status(SRC) == ["LEAKED_FULL"] * 4


def test_right_tag_count_but_wrong_place_is_caught():
    # bag-of-tags F1 would be perfect here; the passport number is still leaked
    pred = "Dear [GIVENNAME1] [LASTNAME1], your passport 123456789 expires. Mail [EMAIL] [PASSPORT] today."
    st = status(pred)
    assert st[2] == "LEAKED_FULL"


def test_wrong_type_and_removed():
    pred = "Dear [LASTNAME1] [LASTNAME1], your passport expires. Mail [EMAIL] today."
    st = status(pred)
    assert st[0] == "MASKED_WRONG" and st[2] == "REMOVED"


def test_truncation_is_dropped():
    pred = "Dear [GIVENNAME1] [LASTNAME1], your passport"
    st = status(pred)
    assert st[2] == "DROPPED" and st[3] == "DROPPED"
    r = score_sample(SRC, pred, SPANS, TGT)
    assert r["non_pii_preserved"] < r["non_pii_total"]


def test_aggregate():
    sc = [{"r": score_sample(SRC, p, SPANS, TGT), "prediction": p, "target": TGT} for p in [TGT, SRC]]
    a = aggregate(sc)
    assert a["span_leak_rate"] == 0.5 and a["sample_leak_rate"] == 0.5


def test_adjacent_spans_with_repeated_separators():
    # regression: difflib mis-attributed placeholders between adjacent spans separated by ':'
    src = "Name: Madame: 01/10/1990: HGQ: +31-329 692 7430: Iwa: Mondolot: Brocco\n"
    vals = [("Madame", "TITLE"), ("01/10/1990", "BOD"), ("HGQ", "USERNAME"), ("+31-329 692 7430", "TEL"),
            ("Iwa", "GIVENNAME1"), ("Mondolot", "LASTNAME1"), ("Brocco", "LASTNAME2")]
    spans, pos = [], 0
    for v, l in vals:
        a = src.index(v, pos); spans.append({"start": a, "end": a + len(v), "label": l}); pos = a + len(v)
    tgt = "Name: [TITLE]: [BOD]: [USERNAME]: [TEL]: [GIVENNAME1]: [LASTNAME1]: [LASTNAME2]\n"
    assert [s for _, s in score_sample(src, tgt, spans, tgt)["spans"]] == ["MASKED_CORRECT"] * 7
    swapped = "Name: [TITLE]: [BOD]: [USERNAME]: [TEL]: [LASTNAME1]: [GIVENNAME1]: [LASTNAME2]\n"
    st = [s for _, s in score_sample(src, swapped, spans, tgt)["spans"]]
    assert st[4] == "MASKED_WRONG" and st[5] == "MASKED_WRONG" and st[6] == "MASKED_CORRECT"


def test_leak_with_appended_tag_is_leak_and_fp():
    pred = TGT.replace("[PASSPORT]", "123456789") + " [PASSPORT]"
    r = score_sample(SRC, pred, SPANS, TGT)
    assert r["spans"][2][1] == "LEAKED_FULL" and r["fp_tags"] >= 1
