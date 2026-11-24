"""Offline tests for `tagging/tag_posts.py twins` (LLM twins for training and testing).

No model is loaded: a fake generator returns canned replies, so these run anywhere the
biascompare package (for its gender lexicon) imports.
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("tag_posts", ROOT / "tagging" / "tag_posts.py")
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)

RES = tp.twin_resources()


def check(text, axis, **raw):
    return tp.check_twin(text, axis, raw, RES)


# ---------------------------------------------------------------- apply_swaps

def test_swaps_apply_in_one_pass_and_keep_case():
    text = "She said he told her that HE was right. Women's rights, she’d say."
    twin, missing = tp.apply_swaps(text, [["she", "he"], ["he", "she"], ["told her", "told him"], ["women", "men"]])
    assert twin == "He said she told him that SHE was right. Men's rights, he’d say."
    assert missing == []


def test_swaps_respect_word_boundaries_and_report_missing():
    twin, missing = tp.apply_swaps("the theme is his", [["he", "she"], ["his", "her"]])
    assert twin == "the theme is her"
    assert missing == ["he"]


def test_empty_replacement_closes_the_gap():
    twin, _ = tp.apply_swaps("Deaf people need jobs. My deaf cousin agrees, he is deaf.",
                             [["Deaf", ""], ["deaf cousin", "cousin"]])
    assert twin == "People need jobs. My cousin agrees, he is."


# ---------------------------------------------------------------- check_twin: gender

def test_gender_twin_passes():
    c = check("The new manager, a woman from sales, thinks her ideas are better.", "gender",
              usable=True, group="female", swaps=[["a woman", "a man"], ["her", "his"]], kept=[])
    assert c["twin_ok"], c["issues"]
    assert c["twin_text"] == "The new manager, a man from sales, thinks his ideas are better."
    assert c["group"] == "female" and c["n_changed"] == 2


def test_mixed_gender_post_swaps_everyone():
    c = check("My brother told his wife that he was done.", "gender", usable=True, group="mixed",
              swaps=[["brother", "sister"], ["his wife", "her husband"], ["he", "she"]])
    assert c["twin_ok"], c["issues"]
    assert c["twin_text"] == "My sister told her husband that she was done."


def test_missed_pronoun_is_flagged():
    c = check("She said her boss was wrong.", "gender", usable=True, group="female", swaps=[["She", "He"]])
    assert not c["twin_ok"]
    assert any(i.startswith("gender_words_left") and "her" in i for i in c["issues"])


def test_kept_word_is_not_flagged():
    raw = dict(usable=True, group="male", swaps=[["he", "she"]])
    assert check("Man, he was fast.", "gender", **raw, kept=["Man"])["twin_ok"]
    assert not check("Man, he was fast.", "gender", **raw)["twin_ok"]


def test_plural_nouns_count_as_gender_words():
    c = check("The boys said they love their sons.", "gender", usable=True, group="male",
              swaps=[["boys", "girls"], ["sons", "daughters"]])
    assert c["twin_ok"], c["issues"]


@pytest.mark.parametrize("swaps, issue", [
    ([["he", "they"]], "gender_neutral_replacement"),
    ([["he", "him"]], "same_gender"),
    ([["he", "she"], ["his car", "her truck"]], "swap_changes_other_words"),
    ([["he", "she"], ["moon", "sun"]], "swap_not_in_post"),
])
def test_bad_gender_swaps_are_flagged(swaps, issue):
    c = check("I think he drove his car.", "gender", usable=True, group="male", swaps=swaps)
    assert not c["twin_ok"]
    assert any(i.startswith(issue) for i in c["issues"]), c["issues"]


def test_group_mismatch_and_inferred_group():
    wrong = check("She left.", "gender", usable=True, group="male", swaps=[["She", "He"]])
    assert any(i.startswith("group_mismatch") for i in wrong["issues"])
    inferred = check("She left.", "gender", usable=True, swaps=[["She", "He"]])
    assert inferred["twin_ok"] and inferred["group"] == "female"


# ---------------------------------------------------------------- check_twin: disability

def test_disability_twin_passes_with_article():
    c = check("An autistic boy in my class won.", "disability", usable=True, group="disabled",
              swaps=[["An autistic boy", "A boy"]])
    assert c["twin_ok"], c["issues"]
    assert c["twin_text"] == "A boy in my class won." and c["group"] == "disabled"


def test_wrong_article_is_flagged():
    c = check("An autistic boy in my class won.", "disability", usable=True, swaps=[["autistic", ""]])
    assert c["twin_text"] == "An boy in my class won."
    assert any(i.startswith("article") for i in c["issues"])


def test_disability_left_and_added_words_are_flagged():
    text = "My deaf cousin turned a blind eye."
    left = check(text, "disability", usable=True, swaps=[["deaf cousin", "cousin"]])
    assert any(i.startswith("disability_terms_left") for i in left["issues"])
    assert check(text, "disability", usable=True, swaps=[["deaf cousin", "cousin"]], kept=["blind"])["twin_ok"]
    added = check(text, "disability", usable=True, swaps=[["deaf cousin", "cousin who hears"]], kept=["blind"])
    assert any(i.startswith("replacement_adds_words") for i in added["issues"])


# ---------------------------------------------------------------- check_twin: no twin

def test_no_twin_reasons():
    fig = check("He has a blind spot.", "disability", usable=False, reason="figurative")
    assert fig["usable"] is False and fig["reason"] == "figurative" and not fig["twin_ok"]
    assert check("x", "disability", usable="false", reason="Not about a person")["reason"] == "other"
    assert check("x", "gender", usable=False, reason="no person")["reason"] == "no_person"
    bad = tp.check_twin("x", "gender", None, RES)
    assert bad["usable"] is None and bad["issues"] == ["llm_parse_error"]


# ---------------------------------------------------------------- the twins command

POSTS = [
    {"id": "p1", "text": "She said her boss was wrong.", "label": 0, "split": "train", "source": "cc",
     "axis": "gender", "gender_terms": "she|her", "gender_group": "female", "disability_terms": ""},
    {"id": "p2", "text": "My deaf cousin keeps texting me.", "label": 0, "split": "train", "source": "cc",
     "axis": "disability", "gender_terms": "", "gender_group": "", "disability_terms": "deaf"},
    {"id": "p3", "text": "He is blind to the facts.", "label": 1, "split": "train", "source": "cc",
     "axis": "both", "gender_terms": "he", "gender_group": "male", "disability_terms": "blind"},
    {"id": "p4", "text": "The weather is nice today.", "label": 0, "split": "train", "source": "cc",
     "axis": "none", "gender_terms": "", "gender_group": "", "disability_terms": ""},
]
REPLIES = {
    ("p1", "gender"): {"usable": True, "group": "female", "swaps": [["She", "He"], ["her", "his"]], "kept": []},
    ("p2", "disability"): {"usable": True, "group": "disabled", "swaps": [["deaf cousin", "cousin"]], "kept": []},
    ("p3", "gender"): {"usable": True, "group": "male", "swaps": [["He", "She"]], "kept": []},
    ("p3", "disability"): {"usable": False, "reason": "figurative", "swaps": [], "kept": ["blind"]},
}


class FakeGenerator:
    def __init__(self):
        self.calls = 0

    def __call__(self, conversations):
        replies = []
        for messages in conversations:
            self.calls += 1
            axis = "gender" if "Axis: gender" in messages[0]["content"] else "disability"
            text = messages[-1]["content"].removeprefix("Post: ")
            pid = next(p["id"] for p in POSTS if p["text"] == text)
            replies.append("Sure.\n" + json.dumps(REPLIES[(pid, axis)]))
        return replies


def write_posts(path, posts):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(posts[0]))
        writer.writeheader()
        writer.writerows(posts)


def twin_args(tmp_path, **kw):
    base = dict(input=str(tmp_path / "train.csv"), out=str(tmp_path / "out" / "twins_train.csv"),
                axes=["gender", "disability"], ids=None, model="fake/model", quantization="auto", batch_size=2,
                max_new_tokens=64, chunk_size=2, fresh=False, limit=None)
    return argparse.Namespace(**{**base, **kw})


def read_rows(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_twins_writes_one_row_per_post_and_axis(tmp_path):
    write_posts(tmp_path / "train.csv", POSTS)
    records, summary = tp.twins(twin_args(tmp_path), generate=FakeGenerator())
    rows = {(r["id"], r["axis"]): r for r in read_rows(tmp_path / "out" / "twins_train.csv")}
    assert set(rows) == set(REPLIES)  # p4 mentions no axis: no job
    assert rows[("p1", "gender")]["twin_text"] == "He said his boss was wrong."
    assert rows[("p1", "gender")]["twin_ok"] == "True"
    assert rows[("p3", "disability")]["usable"] == "False" and rows[("p3", "disability")]["reason"] == "figurative"
    assert summary["disability"] == {**summary["disability"], "posts": 2, "twin_ok": 1, "no_twin": {"figurative": 1}}
    saved = json.loads((tmp_path / "out" / "twins_train_summary.json").read_text())
    assert saved["axes"]["gender"]["twin_ok"] == 2

    twins = tp.load_twins(tmp_path / "out" / "twins_train.csv")
    assert twins["p2"] == [("disability", "My cousin keeps texting me.", "disabled")]
    assert twins["p3"] == [("gender", "She is blind to the facts.", "male")]
    assert tp.load_twins(tmp_path / "out" / "twins_train.csv", axes=("disability",)).keys() == {"p2"}


def test_twins_resumes_and_survives_a_cut_off_line(tmp_path):
    write_posts(tmp_path / "train.csv", POSTS[:2])
    first = FakeGenerator()
    tp.twins(twin_args(tmp_path), generate=first)
    assert first.calls == 2
    raw = tmp_path / "out" / "twins_train_raw.jsonl"
    with open(raw, "a", encoding="utf-8") as handle:
        handle.write('{"id": "p9", "axis": "gen')  # the run was stopped mid-write

    write_posts(tmp_path / "train.csv", POSTS)
    second = FakeGenerator()
    records, _ = tp.twins(twin_args(tmp_path), generate=second)
    assert second.calls == 2  # only p3's two axes are new
    assert len(records) == 4
    assert len(tp._load_replies(raw)[0]) == 4

    with pytest.raises(SystemExit):
        tp.twins(twin_args(tmp_path, model="other/model"), generate=FakeGenerator())
    third = FakeGenerator()
    tp.twins(twin_args(tmp_path, fresh=True), generate=third)
    assert third.calls == 4 and list((tmp_path / "out").glob("*.bak.jsonl"))


def test_twins_ids_axes_and_limit(tmp_path):
    write_posts(tmp_path / "train.csv", POSTS)
    (tmp_path / "ids.txt").write_text("p3\np4\n")
    records, _ = tp.twins(twin_args(tmp_path, ids=str(tmp_path / "ids.txt"), axes=["disability"]),
                          generate=FakeGenerator())
    assert [(r["id"], r["axis"]) for r in records] == [("p3", "disability")]
    records, _ = tp.twins(twin_args(tmp_path, limit=1, out=str(tmp_path / "x.csv")), generate=FakeGenerator())
    assert [r["id"] for r in records] == ["p1"]


def test_review_sheet_and_agreement(tmp_path):
    write_posts(tmp_path / "train.csv", POSTS)
    tp.twins(twin_args(tmp_path), generate=FakeGenerator())
    sheet_path = tmp_path / "check.csv"
    tp.main(["twins-sheet", "--twins", str(tmp_path / "out" / "twins_train.csv"), "--out", str(sheet_path), "--n", "4"])
    sheet = read_rows(sheet_path)
    assert len(sheet) == 4 and {r["axis"] for r in sheet} == {"gender", "disability"}
    assert sum(r["llm_decision"] == "twin" for r in sheet) == 3

    for r in sheet:
        r["r1_decision_ok"], r["r2_decision_ok"] = "yes", "y"
        if r["llm_decision"] == "twin":
            r["r1_twin_ok"] = "yes"
            r["r2_twin_ok"] = "no" if r["id"] == "p1" else "yes"
    with open(tmp_path / "filled.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sheet[0]))
        writer.writeheader()
        writer.writerows(sheet)
    tp.main(["twins-agree", "--sheet", str(tmp_path / "filled.csv")])
    report = json.loads((tmp_path / "filled.agreement.json").read_text())
    assert report["all"]["decision_ok"]["n"] == 4 and report["all"]["decision_ok"]["raw_agreement"] == 1.0
    assert report["all"]["no_twin_decision_ok"]["n"] == 1
    twin_ok = report["all"]["twin_ok"]
    assert twin_ok["n"] == 3 and twin_ok["raw_agreement"] == pytest.approx(2 / 3)
    assert twin_ok["yes_both"] == pytest.approx(2 / 3)


def test_prompt_examples_pass_their_own_checks():
    """The few-shot answers shown to the LLM must be twins the checks accept."""
    for axis, examples in tp.TWIN_EXAMPLES.items():
        for text, answer in examples:
            c = tp.check_twin(text, axis, answer, RES)
            if answer["usable"]:
                assert c["twin_ok"], (text, c["issues"])
            else:
                assert c["reason"] == answer["reason"]
            messages = tp.build_twin_messages("A post.", axis)
            assert messages[-1] == {"role": "user", "content": "Post: A post."}
