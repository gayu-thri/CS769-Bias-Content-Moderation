"""LLM tagging: swaps and control synonyms for the bias test, twins for training.

The proposal's tagging step. An 8B LLM (default Qwen3-8B) tags each post, per
bias axis it mentions, with

- `swaps`: every co-referring word and modifier that marks the group, with its
  counterpart (she/her/woman -> he/his/man; "deaf cousin" -> "cousin"), and
- `synonyms`: content words of the post with neutral replacements, which the
  framework uses to build the matched-magnitude controls.

Unlike labels, these tags feed Δ directly, so every LLM tag is checked twice:
by deterministic rules here (phrases occur in the post, the swapped text has no
leftover group words, synonyms avoid swap / identity / toxic words), and then by
people and Gemini on the review sheet. Only rows that pass review become the
evaluation set.

Two commands:

    # 1. tag: LLM + automatic checks -> tagged_posts.jsonl + tagging_review.csv
    python tagging/tag_posts.py tag --input data/processed/verification_300.csv --out-dir out/

    # 2. finalize: the filled-in review sheet -> verified_posts.jsonl (+ agreement)
    python tagging/tag_posts.py finalize --review out/tagging_review_filled.csv \
        --tagged out/tagged_posts.jsonl --out out/verified_posts.jsonl

The output rows are in the biascompare v1 format (bias-compare-poc docs/DATA.md
§2.2-2.3). The dataset's toxicity label goes in `meta.dataset_label`, never in
`label`: the evaluated posts need no label, and ToxicCommons labels are
model-derived.

Twins (counterfactuals for training M4 and for the transfer test). The rule
twins in train_lm.py cover only 3% of real disability posts, and most of M4's
disability twins insert a word into a post that never mentioned disability.
`twins` asks the LLM for one twin per post and axis instead: every gendered
word about a person swapped (both genders, so mixed posts work too), or every
disability marker about a person removed. The LLM also says when there is no
twin: the word is not about a person, is figurative or an insult, or the post
only makes sense for that group. Rule checks run on every twin; `twin_ok`
marks the ones a training run can use. Replies are saved as they come, so a
stopped Colab run resumes where it left off.

    # 3. twins: every train (or test) post the lexicon tagged -> twins_train.csv
    python tagging/tag_posts.py twins --input data/processed/train.csv --out out/twins_train.csv
    python tagging/tag_posts.py twins --input data/processed/test.csv --out out/twins_test.csv

    # 4. a 100-row sheet for two reviewers, then their agreement (Cohen's kappa)
    python tagging/tag_posts.py twins-sheet --twins out/twins_test.csv --out out/twins_check.csv
    python tagging/tag_posts.py twins-agree --sheet out/twins_check_filled.csv

`load_twins(path)` returns {post id: [(axis, twin text, group), ...]} for a
training run (see its docstring).
"""

from __future__ import annotations

import argparse
import csv
import difflib
import importlib.resources
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-8B"
AXES = ("gender", "disability")
OTHER_GROUP = {"male": "female", "female": "male", "disabled": "non-disabled"}

# The data notebook's disability term list (Content_Moderation.ipynb), used here to
# check that a disability swap removed every marker.
DISABILITY_TERMS = (
    "blind", "deaf", "hard of hearing", "visually impaired", "hearing impaired",
    "wheelchair", "paralyzed", "paralysed", "paraplegic", "quadriplegic",
    "amputee", "crippled", "cripple", "handicapped", "cerebral palsy", "dwarf", "dwarfism",
    "autistic", "autism", "asperger", "aspergers", "down syndrome", "down's syndrome",
    "downs syndrome", "dyslexic", "dyslexia", "adhd", "intellectual disability",
    "learning disability", "retard", "retarded", "retards", "mongoloid", "special needs",
    "mentally ill", "mental illness", "schizophrenic", "schizophrenia", "bipolar",
    "psychotic", "ptsd", "ocd", "clinical depression", "anxiety disorder", "lunatic", "psycho",
    "epileptic", "epilepsy", "chronic illness", "disabled", "disability", "disabilities",
    "deformed",
)
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


# ---------------------------------------------------------------- resources

def _resource(name):
    return json.loads((importlib.resources.files("biascompare") / "resources" / name).read_text(encoding="utf-8"))


def gender_sets():
    """The framework's gender lexicon as two word sets. All `twins` needs, so it runs without stopwords.json."""
    pairs = [(m.lower(), f.lower()) for m, f in _resource("gender_lexicon.json")["pairs"]]
    return {"male": frozenset(m for m, _ in pairs), "female": frozenset(f for _, f in pairs)}


def load_resources():
    return {
        **gender_sets(),
        "protected": frozenset(w.lower() for w in _resource("protected_words.json")["words"]),
        "stopwords": frozenset(w.lower() for w in _resource("stopwords.json")["words"]),
        "disability_words": frozenset(w for t in DISABILITY_TERMS for w in t.split()),
    }


def words(text):
    """Lowercase words, a possessive 's stripped ("women's" -> "women"), as in the data notebook."""
    return [re.sub(r"'s$", "", w.lower()) for w in WORD_RE.findall(text)]


def disability_hits(text):
    lowered = " " + " ".join(words(text)) + " "
    return [t for t in DISABILITY_TERMS if f" {' '.join(words(t))} " in lowered]


# ---------------------------------------------------------------- prompt

SYSTEM = """You annotate posts for a counterfactual bias test of toxicity classifiers.
For one post and one bias axis you return the words to swap and the words a control may change.
Reply with one JSON object only, no other text."""

AXIS_RULES = {
    "gender": """Axis: gender. The post refers to a {group} person or people.
- "swaps": EVERY word or phrase in the post that marks that referent's gender (pronouns, nouns, kinship
  terms, titles, gendered first names and modifiers), each paired with its {other} counterpart, so that
  after swapping the post refers to a {other} referent and nothing else changes.
- Copy each phrase exactly as it appears in the post. One phrase replaces every occurrence of it.
- "her" can be possessive (his) or object (him). If both uses occur, tag longer phrases
  such as ["told her", "told him"] and ["her car", "his car"] instead of a bare "her".
- If the post refers to people of BOTH genders, or gender words are not about a person
  (e.g. "Man, that was close", "you guys"), set "usable": false and say why.""",
    "disability": """Axis: disability. The post refers to a disabled person or people.
- "swaps": EVERY word or phrase that marks the disability, paired with the same phrase without the
  disability, so that after swapping the post refers to a non-disabled referent and nothing else changes:
  "deaf cousin" -> "cousin", "wheelchair user" -> "person", "the blind" -> "the people",
  "autistic kids" -> "kids". Keep grammar intact.
- Copy each phrase exactly as it appears in the post. One phrase replaces every occurrence of it.
- If the disability word is not about a person ("blind spot", "deaf to reason", "lame excuse"), or the
  post only names a disease with no person, set "usable": false and say why.""",
}

SYNONYM_RULES = """- "synonyms": at least as many words as the swap changes plus 2, usually 5 to 8: single words of the post that are NOT part of any swap phrase, NOT identity words
  (gender, disability, race, religion...) and NOT insults or profanity, each with 1 to 3 single-word
  replacements that keep the meaning, sentiment, toxicity, part of speech and inflection
  ("said" -> ["stated"], "house" -> ["home"], "quickly" -> ["rapidly"]). Prefer nouns, verbs, adjectives, adverbs.
JSON keys: "usable" (true/false), "reason" (short, if not usable), "swaps" (list of [phrase, replacement]),
"synonyms" (object: word -> list of replacements)."""

EXAMPLES = {
    "gender": [
        ("The new manager, a woman from sales, thinks her ideas are always better than ours.", "female",
         {"usable": True, "swaps": [["a woman", "a man"], ["her", "his"]],
          "synonyms": {"new": ["recent"], "thinks": ["believes"], "ideas": ["plans"], "always": ["constantly"],
                       "better": ["superior"]}}),
        ("My brother told his wife and my sister that he was done.", "male",
         {"usable": False, "reason": "refers to people of both genders", "swaps": [], "synonyms": {}}),
    ],
    "disability": [
        ("My deaf cousin keeps texting me about the game tonight.", "disabled",
         {"usable": True, "swaps": [["deaf cousin", "cousin"]],
          "synonyms": {"keeps": ["continues"], "texting": ["messaging"], "game": ["match"], "tonight": ["later"]}}),
    ],
}


def build_messages(text, axis, group):
    rules = AXIS_RULES[axis].format(group=group, other=OTHER_GROUP[group]) + "\n" + SYNONYM_RULES
    messages = [{"role": "system", "content": SYSTEM + "\n\n" + rules}]
    for ex_text, ex_group, ex_out in EXAMPLES[axis]:
        messages.append({"role": "user", "content": f"Group: {ex_group}\nPost: {ex_text}"})
        messages.append({"role": "assistant", "content": json.dumps(ex_out)})
    messages.append({"role": "user", "content": f"Group: {group}\nPost: {text}"})
    return messages


def parse_reply(reply):
    """The first JSON object in the reply (after any <think> block), or None."""
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL)
    start = reply.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(reply)):
            depth += reply[i] == "{"
            depth -= reply[i] == "}"
            if depth == 0:
                try:
                    obj = json.loads(reply[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    break
        start = reply.find("{", start + 1)
    return None


# ---------------------------------------------------------------- checks

def check_tags(text, axis, group, raw, res):
    """Cleans one LLM answer and returns (swaps, synonyms, swapped_text, issues).
    Invalid pairs and synonyms are dropped; anything a reviewer must look at is an issue."""
    from biascompare.perturb.tagged import apply_tagged_swap

    issues = []
    if raw is None:
        return [], {}, text, ["llm_parse_error"]
    if raw.get("usable") is False:
        issues.append(f"llm_unusable: {raw.get('reason') or 'no reason'}")

    swaps = []
    for pair in raw.get("swaps") or []:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2 and all(isinstance(x, str) for x in pair)):
            issues.append(f"bad_swap_pair: {pair!r}")
            continue
        phrase, repl = pair[0].strip(), pair[1].strip()
        if not phrase or phrase.lower() == repl.lower():
            continue
        # Phrases match case-insensitively and the framework re-capitalises the replacement
        # where the matched word is capitalised, so a lowercase replacement is right everywhere
        # ("She" -> "he" still gives "He" at a sentence start, and "he" mid-sentence).
        repl = repl[:1].lower() + repl[1:] if not repl.isupper() else repl
        if phrase.lower() in {p.lower() for p, _ in swaps}:
            continue
        swaps.append([phrase, repl])

    perturbation, missing = apply_tagged_swap(text, [tuple(p) for p in swaps], kind=axis)
    if missing:
        issues.append(f"swap_not_in_post: {missing}")
        swaps = [p for p in swaps if p[0] not in missing]
        perturbation, _ = apply_tagged_swap(text, [tuple(p) for p in swaps], kind=axis)
    swapped = perturbation.text
    if not swaps and raw.get("usable") is not False:
        issues.append("no_swaps")

    if axis == "gender":
        own, other = res[group], res[OTHER_GROUP[group]]
        before_own = [w for w in words(text) if w in own and w not in other]
        if not before_own:
            issues.append(f"no_{group}_lexicon_words")
        introduced = {w for _, r in swaps for w in words(r)}  # "wife" -> "husband" adds a male word on purpose
        left = [w for w in words(swapped) if w in own and w not in other and w not in introduced]
        if left and swaps:
            issues.append(f"incomplete_swap: {group} words left {sorted(set(left))}")
        mixed = [w for w in words(text) if w in other and w not in own]
        if mixed:
            issues.append(f"mentions_{OTHER_GROUP[group]}_too: {sorted(set(mixed))}")
    else:
        if not disability_hits(text):
            issues.append("no_disability_terms")
        introduced = set(disability_hits(" ".join(r for _, r in swaps)))
        left = [t for t in disability_hits(swapped) if t not in introduced]
        if left and swaps:
            issues.append(f"incomplete_swap: disability terms left {left}")

    swap_words = {w for p in swaps for side in p for w in words(side)}
    blocked = (swap_words | res["male"] | res["female"] | res["protected"] | res["disability_words"]
               | res["stopwords"])  # function words: "that" -> "as" changes grammar, not just wording
    post_words = set(words(text))
    synonyms = {}
    raw_syn = raw.get("synonyms") if isinstance(raw.get("synonyms"), dict) else {}
    for key, repls in raw_syn.items():
        k = str(key).strip()
        if len(words(k)) != 1 or words(k)[0] != k.lower() or k.lower() not in post_words or k.lower() in blocked:
            continue
        repls = [repls] if isinstance(repls, str) else list(repls or [])
        good = []
        for r in repls:
            r = str(r).strip()
            if len(words(r)) == 1 and words(r)[0] == r.lower() and r.lower() != k.lower() and r.lower() not in blocked:
                good.append(r)
        if good:
            synonyms[k] = sorted(set(good), key=good.index)
    dropped = len(raw_syn) - len(synonyms)
    if dropped:
        issues.append(f"synonyms_dropped: {dropped}")
    swap_size = perturbation.n_changed  # word edits, the unit the framework matches controls on
    if len(synonyms) < max(1, swap_size):
        issues.append(f"few_synonyms: {len(synonyms)} for a swap of {swap_size} word edits")
    return swaps, synonyms, swapped, issues


# ---------------------------------------------------------------- LLM

class HFGenerator:
    """Greedy chat generation with a Hugging Face causal LM, batched with left padding."""

    def __init__(self, model_name, quantization="auto", batch_size=8, max_new_tokens=384, token=None, verbose=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name, token=token)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kwargs = {"token": token, "device_map": "auto"}
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if quantization == "auto":  # an 8B model in 16-bit needs ~17 GB of GPU memory
            mem = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
            quantization = "none" if mem >= 22 else "4bit"
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
        else:
            kwargs["dtype"] = torch.bfloat16 if bf16 else torch.float16
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        except TypeError:  # transformers < 4.56 calls it torch_dtype
            kwargs["torch_dtype"] = kwargs.pop("dtype", None)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self.batch_size, self.max_new_tokens = batch_size, max_new_tokens
        self.quantization, self.verbose = quantization, verbose

    def _render(self, messages):
        try:  # Qwen3 is a hybrid reasoning model: answer directly, no thinking block
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def __call__(self, conversations):
        import torch

        replies = []
        for i in range(0, len(conversations), self.batch_size):
            prompts = [self._render(m) for m in conversations[i:i + self.batch_size]]
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            with torch.inference_mode():
                out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            replies += self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            if self.verbose:
                print(f"  tagged {min(i + self.batch_size, len(conversations))}/{len(conversations)}", flush=True)
        return replies


# ---------------------------------------------------------------- tag

def tasks_for(row):
    """(axis, group) pairs to tag for one post, from the data notebook's lexicon pre-filter."""
    tasks = []
    if row.get("gender_group") in ("male", "female"):
        tasks.append(("gender", row["gender_group"]))
    if str(row.get("disability_terms") or "").strip():
        tasks.append(("disability", "disabled"))
    return tasks


def tag(args, generate=None):
    import pandas as pd

    res = load_resources()
    frame = pd.read_csv(args.input, keep_default_na=False)
    if args.limit:
        frame = frame.head(args.limit)
    rows = frame.to_dict("records")
    jobs = [(i, axis, group) for i, row in enumerate(rows) for axis, group in tasks_for(row)]
    skipped = [r["id"] for r in rows if not tasks_for(r)]
    print(f"{len(rows)} posts, {len(jobs)} (post, axis) jobs; {len(skipped)} posts mention no tagged axis")

    if generate is None:
        import os
        generate = HFGenerator(args.model, args.quantization, args.batch_size,
                               token=os.environ.get("HF_TOKEN") or None)
    replies = generate([build_messages(rows[i]["text"], axis, group) for i, axis, group in jobs])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tagged, review = {}, []
    with open(out_dir / "llm_raw.jsonl", "w", encoding="utf-8") as raw_file:
        for (i, axis, group), reply in zip(jobs, replies):
            row = rows[i]
            raw = parse_reply(reply)
            raw_file.write(json.dumps({"id": row["id"], "axis": axis, "group": group, "reply": reply}) + "\n")
            swaps, synonyms, swapped, issues = check_tags(row["text"], axis, group, raw, res)
            entry = tagged.setdefault(row["id"], {
                "id": str(row["id"]), "text": row["text"], "attributes": {}, "swaps": {}, "synonyms": {},
                "meta": {k: row[k] for k in ("source", "platform", "verify_axis", "annotators") if k in row}
                        | {"dataset_label": row.get("label"), "split": "test", "tagger": args.model, "tag_issues": {}},
            })
            entry["attributes"][axis] = group
            entry["swaps"][axis] = swaps
            for k, v in synonyms.items():
                entry["synonyms"][k] = sorted(set(entry["synonyms"].get(k, [])) | set(v), key=(entry["synonyms"].get(k, []) + v).index)
            entry["meta"]["tag_issues"][axis] = issues
            review.append({
                "id": row["id"], "annotators": row.get("annotators", ""), "axis": axis, "group": group,
                "auto_ok": not issues, "issues": "; ".join(issues), "text": row["text"], "swapped_text": swapped,
                "swaps": json.dumps(swaps), "synonyms": json.dumps(synonyms),
                "human_verdict": "", "human2_verdict": "", "gemini_verdict": "", "corrected_swaps": "", "corrected_synonyms": "", "notes": "",
            })

    # Synonyms are shared by every axis of a post, so a word swapped on one axis may not be a control on another.
    for entry in tagged.values():
        swap_words = {w for pairs in entry["swaps"].values() for p in pairs for side in p for w in words(side)}
        entry["synonyms"] = {k: v for k, v in entry["synonyms"].items() if k.lower() not in swap_words}

    with open(out_dir / "tagged_posts.jsonl", "w", encoding="utf-8") as handle:
        for entry in tagged.values():
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # Preliminary, UNVERIFIED subset: only the axes whose tags passed every automatic check
    # (a dropped synonym is fine). For a first look before the human / Gemini review.
    n_auto = 0
    with open(out_dir / "tagged_posts_autopass.jsonl", "w", encoding="utf-8") as handle:
        for entry in tagged.values():
            ok = [a for a, issues in entry["meta"]["tag_issues"].items()
                  if all(i.startswith("synonyms_dropped") for i in issues)]
            if ok:
                row = {**entry, "attributes": {a: entry["attributes"][a] for a in ok},
                       "swaps": {a: entry["swaps"][a] for a in ok}}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_auto += 1

    with open(out_dir / "tagging_review.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review[0]) if review else ["id"])
        writer.writeheader()
        writer.writerows(review)

    counts = Counter((r["axis"], r["auto_ok"]) for r in review)
    issue_counts = Counter(i.split(":")[0] for r in review for i in r["issues"].split("; ") if i)
    print(f"\nWrote {len(tagged)} posts to {out_dir / 'tagged_posts.jsonl'} and {len(review)} review rows.")
    for axis in AXES:
        print(f"  {axis:>10}: {counts[(axis, True)]} pass the automatic checks, {counts[(axis, False)]} need a look")
    print("  issues:", dict(issue_counts.most_common()))
    print(f"  {n_auto} posts with at least one auto-passing axis in tagged_posts_autopass.jsonl (unverified)")
    return tagged, review


# ---------------------------------------------------------------- finalize

ACCEPT, FIX, REJECT = "accept", "fix", "reject"


def cohen_kappa(a, b):
    labels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def pair_agreement(review, a, b):
    rows = review[(review[a] != "") & (review[b] != "")]
    x, y = list(rows[a]), list(rows[b])
    if not x:
        return {"n": 0, "raw": None, "kappa": None}
    return {"n": len(x), "raw": sum(p == q for p, q in zip(x, y)) / len(x), "kappa": cohen_kappa(x, y)}


def finalize(args):
    """Applies the review: per (post, axis), the human verdict wins over Gemini's.
    `accept` keeps the LLM tags, `fix` uses corrected_swaps / corrected_synonyms (JSON),
    `reject` (or no verdict) drops that axis. Posts with no axis left are dropped.

    Two people may review the same rows (`human_verdict`, `human2_verdict`). Their
    agreement is the ceiling for human-model agreement, so both are reported. A row where
    the two people disagree is left out until they settle it (one verdict in both columns)."""
    import pandas as pd

    res = load_resources()
    review = pd.read_csv(args.review, keep_default_na=False)
    tagged = {}
    with open(args.tagged, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                tagged[row["id"]] = row

    if "human2_verdict" not in review.columns:  # sheets made before the second reviewer column existed
        review["human2_verdict"] = ""
    for column in ("human_verdict", "human2_verdict", "gemini_verdict"):
        review[column] = review[column].astype(str).str.strip().str.lower()
    agreement = {f"{a}_vs_{b}": pair_agreement(review, f"{a}_verdict", f"{b}_verdict")
                 for a, b in (("human", "human2"), ("human", "gemini"), ("human2", "gemini"))}

    final, decisions = {}, Counter()
    for r in review.to_dict("records"):
        humans = [v for v in (r["human_verdict"], r["human2_verdict"]) if v]
        if len(set(humans)) > 1:
            decisions["humans_disagree"] += 1
            print(f"NOTE {r['id']} {r['axis']}: the two reviewers disagree ({humans}); left out until settled")
            continue
        verdict = humans[0] if humans else r["gemini_verdict"]
        decisions[verdict or "none"] += 1
        if verdict not in (ACCEPT, FIX):
            continue
        src = tagged[str(r["id"])]
        axis = r["axis"]
        swaps = json.loads(r["corrected_swaps"]) if verdict == FIX and r["corrected_swaps"] else json.loads(r["swaps"])
        synonyms = json.loads(r["corrected_synonyms"]) if verdict == FIX and r["corrected_synonyms"] else json.loads(r["synonyms"])
        swaps, synonyms, _, issues = check_tags(src["text"], axis, r["group"], {"swaps": swaps, "synonyms": synonyms}, res)
        hard = [i for i in issues if i.startswith(("swap_not_in_post", "no_swaps", "llm_parse_error"))]
        if hard:
            print(f"WARNING {r['id']} {axis}: reviewed tags still fail {hard}; dropped")
            continue
        out = final.setdefault(src["id"], {"id": src["id"], "text": src["text"], "attributes": {}, "swaps": {},
                                           "synonyms": {}, "meta": dict(src["meta"])})
        out["attributes"][axis] = r["group"]
        out["swaps"][axis] = swaps
        out["synonyms"].update(synonyms)
        out["meta"].setdefault("review", {})[axis] = {"verdict": verdict, "by": "human" if humans else "gemini"}

    for out in final.values():
        out["meta"].pop("tag_issues", None)
        swap_words = {w for pairs in out["swaps"].values() for p in pairs for side in p for w in words(side)}
        out["synonyms"] = {k: v for k, v in out["synonyms"].items() if k.lower() not in swap_words}
    with open(args.out, "w", encoding="utf-8") as handle:
        for out in final.values():
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")

    per_axis = Counter(a for out in final.values() for a in out["attributes"])
    groups = Counter((a, g) for out in final.values() for a, g in out["attributes"].items())
    print(f"Verdicts: {dict(decisions)}")
    print(f"Wrote {len(final)} verified posts to {args.out}: {dict(per_axis)}; groups {dict(groups)}")
    print("Agreement (inter-human is the ceiling for human-model):")
    for name, a in agreement.items():
        shown = "no rows reviewed by both" if not a["n"] else f"{a['n']} rows, raw {a['raw']:.2f}, Cohen's kappa {a['kappa']:.2f}"
        print(f"  {name:>17}: {shown}")
    with open(Path(args.out).with_suffix(".agreement.json"), "w") as handle:
        json.dump({"verdicts": dict(decisions), "posts": len(final), "per_axis": dict(per_axis), "agreement": agreement},
                  handle, indent=2)
    return final, agreement


# ---------------------------------------------------------------- twins (training and test counterfactuals)

TWIN_SYSTEM = """You write counterfactual twins of social media posts, for training and testing a toxicity classifier.
A twin changes the group of the people the post talks about, and nothing else.
Reply with one JSON object only, no other text."""

TWIN_RULES = {
    "gender": """Axis: gender. The twin swaps the gender of every person the post talks about.
- "swaps": EVERY word or phrase that marks the gender of a person in the post (pronouns, nouns, kinship terms,
  titles, first names, gendered modifiers), each paired with its counterpart of the other gender, so that every
  man becomes a woman and every woman a man. Surnames stay.
- Copy each phrase exactly as it appears in the post. One phrase replaces every occurrence of it, so tag longer
  phrases when a word has two uses: "her" can be possessive (his) or object (him), so tag ["told her", "told him"]
  and ["her car", "his car"] instead of a bare "her".
- "group": "male" or "female" if the post talks only about men or only about women, "mixed" if about both.
- "kept": gender words you leave unchanged because they are not about a person ("Man, that was close",
  "you guys" said to anyone, "mark my words").
- "usable": false, with a "reason", when the post has no twin:
  "no_person": no gender word in the post is about a person;
  "group_specific": the post only makes sense for one gender (pregnancy, breastfeeding, sex-specific anatomy).""",
    "disability": """Axis: disability. The twin removes the disability of every disabled person the post talks about.
- "swaps": EVERY word or phrase that says a person in the post is disabled, each paired with the same phrase
  without the disability, so that the person is not disabled and nothing else changes: "deaf cousin" -> "cousin",
  "wheelchair user" -> "person", "the blind" -> "the people", "people with autism" -> "people". Include an article
  that has to change ("an autistic boy" -> "a boy").
- Copy each phrase exactly as it appears in the post. One phrase replaces every occurrence of it.
- "group": "disabled".
- "kept": disability words you leave unchanged because they are not about a disabled person.
- "usable": false, with a "reason", when the post has no twin:
  "no_person": the post names a condition but no person who has it ("autism rates are rising");
  "figurative": the word is a figure of speech ("blind spot", "turn a blind eye", "deaf to reason");
  "insult": the word is an insult, not a description of a disabled person ("what a lunatic", "you retard");
  "group_specific": the rest of the post only makes sense for disabled people (sign language, ramps, braille).""",
}
TWIN_KEYS = """JSON keys: "usable" (true/false), "reason" (one of the codes above, only when not usable), "group",
"swaps" (list of [phrase, replacement]), "kept" (list of words)."""
TWIN_REASONS = {"gender": ("no_person", "group_specific"),
                "disability": ("no_person", "figurative", "insult", "group_specific")}
TWIN_GROUPS = {"gender": ("male", "female", "mixed"), "disability": ("disabled",)}

TWIN_EXAMPLES = {
    "gender": [
        ("The new manager, a woman from sales, thinks her ideas are always better than ours.",
         {"usable": True, "group": "female", "swaps": [["a woman", "a man"], ["her", "his"]], "kept": []}),
        ("My brother told his wife that he was done, man.",
         {"usable": True, "group": "mixed",
          "swaps": [["brother", "sister"], ["his wife", "her husband"], ["he", "she"]], "kept": ["man"]}),
        ("Man, that was a close game last night.",
         {"usable": False, "reason": "no_person", "group": "", "swaps": [], "kept": ["Man"]}),
        ("Women who breastfeed in public should be left alone.",
         {"usable": False, "reason": "group_specific", "group": "female", "swaps": [], "kept": []}),
    ],
    "disability": [
        ("My deaf cousin keeps texting me about the game tonight.",
         {"usable": True, "group": "disabled", "swaps": [["deaf cousin", "cousin"]], "kept": []}),
        ("An autistic boy in my class won the spelling bee.",
         {"usable": True, "group": "disabled", "swaps": [["An autistic boy", "A boy"]], "kept": []}),
        ("He has a huge blind spot on trade.",
         {"usable": False, "reason": "figurative", "group": "", "swaps": [], "kept": ["blind"]}),
        ("Only a lunatic would vote for this.",
         {"usable": False, "reason": "insult", "group": "", "swaps": [], "kept": ["lunatic"]}),
    ],
}

# More gendered person nouns for the twin checks only. The framework lexicon has 30 pairs and few plurals, so
# with it alone a correct "boys" -> "girls" swap would look like a changed non-gender word.
GENDER_EXTRA_PAIRS = (
    ("boy", "girl"), ("boys", "girls"), ("sons", "daughters"), ("fathers", "mothers"), ("husbands", "wives"),
    ("brothers", "sisters"), ("uncles", "aunts"), ("nephews", "nieces"), ("kings", "queens"),
    ("gentlemen", "ladies"), ("males", "females"), ("boyfriends", "girlfriends"), ("grandfathers", "grandmothers"),
    ("grandson", "granddaughter"), ("grandsons", "granddaughters"), ("grandpa", "grandma"), ("dad", "mom"),
    ("dads", "moms"), ("dad", "mum"), ("daddy", "mommy"), ("papa", "mama"), ("mr", "mrs"), ("mr", "miss"),
    ("sir", "madam"), ("lad", "lass"), ("dude", "gal"), ("dudes", "gals"), ("bro", "sis"), ("fiance", "fiancee"),
    ("widower", "widow"), ("groom", "bride"), ("prince", "princess"), ("duke", "duchess"),
    ("chairman", "chairwoman"), ("congressman", "congresswoman"), ("spokesman", "spokeswoman"),
    ("businessman", "businesswoman"), ("businessmen", "businesswomen"), ("policeman", "policewoman"),
    ("salesman", "saleswoman"),
)
# A gender swap must not neutralise: "he" -> "they" is not a twin.
GENDER_NEUTRAL = frozenset({"they", "them", "their", "theirs", "themselves", "person", "people", "someone",
                            "somebody", "partner", "partners", "spouse", "spouses", "sibling", "siblings", "parent",
                            "parents", "child", "children", "kid", "kids", "individual", "folks"})
# Words a disability removal may add: an article or a plain person noun ("wheelchair user" -> "person").
DISABILITY_FILLERS = frozenset({"a", "an", "the", "person", "people", "someone", "somebody", "individual",
                                "individuals", "folks", "one", "ones", "others"})
_DELETED, _DELETED_CAP = "\x00", "\x01"
_ARTICLE_RE = re.compile(r"\b(a|an)\s+([A-Za-z]+)", re.I)


def twin_resources():
    lex = gender_sets()
    return {"male": lex["male"] | {m for m, _ in GENDER_EXTRA_PAIRS},
            "female": lex["female"] | {f for _, f in GENDER_EXTRA_PAIRS}}


def build_twin_messages(text, axis):
    messages = [{"role": "system", "content": f"{TWIN_SYSTEM}\n\n{TWIN_RULES[axis]}\n{TWIN_KEYS}"}]
    for ex_text, ex_out in TWIN_EXAMPLES[axis]:
        messages.append({"role": "user", "content": f"Post: {ex_text}"})
        messages.append({"role": "assistant", "content": json.dumps(ex_out)})
    messages.append({"role": "user", "content": f"Post: {text}"})
    return messages


def _norm_phrase(s):
    return re.sub(r"\s+", " ", s.replace("’", "'")).strip().lower()


def _phrase_pattern(phrase):
    parts = [re.escape(w).replace("'", "['’]") for w in _norm_phrase(phrase).split(" ")]
    return r"(?<![\w'’])" + r"\s+".join(parts) + r"(?!\w)"


def _match_case(found, repl):
    if not repl:
        return _DELETED_CAP if found[:1].isupper() else _DELETED
    if len(found) > 1 and found.isupper():
        return repl.upper()
    if found[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl if repl.isupper() and len(repl) > 1 else repl[:1].lower() + repl[1:]


def _close_gaps(text):
    """Removes the marks an empty replacement left, with one space, and capitalises the next
    word when a capitalised sentence start was deleted ("Deaf people ..." -> "People ...")."""
    while True:
        m = re.search(f"[{_DELETED}{_DELETED_CAP}]", text)
        if not m:
            return text
        before, after = text[:m.start()], text[m.end():]
        if re.match(r"\s*([.,;:!?)\]]|$)", after):
            before = before.rstrip()
        else:
            after = after.lstrip()
            if m.group(0) == _DELETED_CAP and (not before.strip() or re.search(r"[.!?]\s*$", before)):
                after = after[:1].upper() + after[1:]
        text = before + after


def apply_swaps(text, swaps, spans=None):
    """Applies [phrase, replacement] pairs in one pass, so he -> she and she -> he don't undo
    each other; where phrases overlap the longest wins. Matches are case-insensitive, at word
    boundaries ("he" never matches inside "the" or "she"), and a possessive stays attached
    ("women's" -> "men's"). A replacement takes the case of the text it replaces.
    Returns (twin text, the phrases that do not occur in the text). A `spans` list, if given,
    receives the (start, end) of every replaced stretch of `text`."""
    missing, table = [], {}
    for phrase, repl in swaps:
        key = _norm_phrase(phrase)
        if not key:
            continue
        if re.search(_phrase_pattern(phrase), text, re.I):
            table.setdefault(key, repl)
        else:
            missing.append(phrase)
    if not table:
        return text, missing
    pattern = re.compile("|".join(_phrase_pattern(k) for k in sorted(table, key=len, reverse=True)), re.I)

    def replace(m):
        if spans is not None:
            spans.append((m.start(), m.end()))
        return _match_case(m.group(0), table[_norm_phrase(m.group(0))])

    return _close_gaps(pattern.sub(replace, text)), missing


def _base(word):
    return re.sub(r"'[a-z]+$", "", word.lower())  # "she's", "he'd", "women's" -> she, he, women


def _genders(text, res):
    found = set()
    for w in WORD_RE.findall(text):
        b = _base(w)
        found |= {"male"} if b in res["male"] else set()
        found |= {"female"} if b in res["female"] else set()
    return found


def _other_words(text, markers):
    """The lowercase words of a swap phrase that are not gender words. Capitalised words are left out:
    names ("Bill" -> "Jill") and sentence starts ("The man" -> "The woman") may differ."""
    return Counter(_base(w) for w in WORD_RE.findall(text) if not w[:1].isupper() and _base(w) not in markers)


def _bad_articles(text):
    return {f"{a.lower()} {w.lower()}" for a, w in _ARTICLE_RE.findall(text)
            if (a.lower() == "a") == (w[:1].lower() in "aeiou")}


def check_twin(text, axis, raw, res):
    """Cleans one LLM twin answer and checks it. Returns a dict with usable (None when the reply
    did not parse), reason (a TWIN_REASONS code or "other"), reason_text, group, swaps, kept,
    twin_text, n_changed (word edits), issues, and twin_ok: a usable twin with no issues, the
    only kind a training run or the transfer test should use."""
    out = {"usable": None, "reason": "", "reason_text": "", "group": "", "swaps": [], "kept": [],
           "twin_text": "", "n_changed": 0, "issues": [], "twin_ok": False}
    issues = out["issues"]
    if raw is None:
        issues.append("llm_parse_error")
        return out
    group = str(raw.get("group") or "").strip().lower()
    out["usable"] = not (raw.get("usable") is False or str(raw.get("usable")).strip().lower() == "false")
    if not out["usable"]:
        out["reason_text"] = str(raw.get("reason") or "").strip()
        code = re.sub(r"[\s-]+", "_", out["reason_text"].lower())
        out["reason"] = code if code in TWIN_REASONS[axis] else "other"
        out["group"] = group if group in TWIN_GROUPS[axis] else ""
        return out

    kept = raw.get("kept") or []
    kept = [kept] if isinstance(kept, str) else [k for k in kept if isinstance(k, str)]
    post_words = [_base(w) for w in WORD_RE.findall(text)]
    out["kept"] = [k.strip() for k in kept if k.strip()]
    kept_words = {_base(w) for k in out["kept"] for w in WORD_RE.findall(k)} & set(post_words)

    swaps, seen = [], set()
    for pair in raw.get("swaps") or []:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2 and all(isinstance(x, str) for x in pair)):
            issues.append(f"bad_swap_pair: {pair!r}")
            continue
        phrase, repl = pair[0].strip(), pair[1].strip()
        key = _norm_phrase(phrase)
        if not key or key == _norm_phrase(repl) or key in seen:
            continue
        seen.add(key)
        swaps.append([phrase, repl])
    spans = []
    twin, missing = apply_swaps(text, swaps, spans)
    if missing:
        issues.append(f"swap_not_in_post: {missing}")
        swaps = [p for p in swaps if p[0] not in missing]
    out["swaps"], out["twin_text"] = swaps, twin
    if not swaps:
        issues.append("no_swaps")
    elif twin == text:
        issues.append("no_change")

    twin_words = [_base(w) for w in WORD_RE.findall(twin)]
    opcodes = difflib.SequenceMatcher(None, post_words, twin_words, autojunk=False).get_opcodes()
    out["n_changed"] = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in opcodes if op != "equal")
    unchanged = [_base(m.group(0)) for m in WORD_RE.finditer(text) if not any(s <= m.start() < e for s, e in spans)]

    if axis == "gender":
        markers = res["male"] | res["female"]
        left = sorted({w for w in unchanged if w in markers and w not in kept_words})
        if left and swaps:
            issues.append(f"gender_words_left: {left}")
        swapped_genders = set()
        for phrase, repl in swaps:
            gp, gr = _genders(phrase, res), _genders(repl, res)
            swapped_genders |= gp
            if gp and not gr and {_base(w) for w in WORD_RE.findall(repl)} <= GENDER_NEUTRAL:
                issues.append(f"gender_neutral_replacement: {[phrase, repl]}")
            elif gp and gp == gr and len(gp) == 1:
                issues.append(f"same_gender: {[phrase, repl]}")
            if _other_words(phrase, markers) != _other_words(repl, markers):
                issues.append(f"swap_changes_other_words: {[phrase, repl]}")
        if group in ("male", "female") and swapped_genders - {group}:
            issues.append(f"group_mismatch: llm says {group}, swaps change {sorted(swapped_genders)}")
        if group not in TWIN_GROUPS["gender"]:
            group = "mixed" if len(swapped_genders) == 2 else next(iter(swapped_genders), "")
            if not group:
                issues.append("no_group")
    else:
        left = [t for t in disability_hits(twin) if not {_base(w) for w in t.split()} <= kept_words]
        if left:
            issues.append(f"disability_terms_left: {left}")
        for phrase, repl in swaps:
            if {_base(w) for w in WORD_RE.findall(repl)} - {_base(w) for w in WORD_RE.findall(phrase)} - DISABILITY_FILLERS:
                issues.append(f"replacement_adds_words: {[phrase, repl]}")
        group = "disabled"
    new_articles = _bad_articles(twin) - _bad_articles(text)
    if new_articles:
        issues.append(f"article: {sorted(new_articles)}")
    out["group"] = group
    out["twin_ok"] = not issues
    return out


def twin_tasks(row, axes=AXES):
    """The axes to write a twin for: every axis the data notebook's lexicon tagged (gender posts of
    both genders included: the twin swaps everyone)."""
    axis = str(row.get("axis") or "")
    tasks = []
    if "gender" in axes and (str(row.get("gender_terms") or "").strip() or axis in ("gender", "both")):
        tasks.append("gender")
    if "disability" in axes and (str(row.get("disability_terms") or "").strip() or axis in ("disability", "both")):
        tasks.append("disability")
    return tasks


def _read_ids(path):
    if str(path).endswith((".csv", ".csv.gz")):
        import pandas as pd
        return set(pd.read_csv(path, usecols=["id"], dtype=str, keep_default_na=False)["id"])
    with open(path, encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def _load_replies(raw_path):
    replies, models = {}, set()
    if raw_path.exists():
        with open(raw_path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:  # a line cut off when the run stopped
                    continue
                replies[(str(r["id"]), r["axis"])] = r["reply"]
                models.add(r.get("model"))
    return replies, models


def _truthy(value):
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def twin_summary(records, axes=AXES):
    out = {}
    for axis in axes:
        rows = [r for r in records if r["axis"] == axis]
        ok = [r for r in rows if r["twin_ok"]]
        edits = sorted(r["n_changed"] for r in ok)
        out[axis] = {
            "posts": len(rows),
            "parsed": sum(r["usable"] is not None for r in rows),
            "usable": sum(r["usable"] is True for r in rows),
            "twin_ok": len(ok),
            "twin_ok_share": round(len(ok) / len(rows), 4) if rows else None,
            "no_twin": dict(Counter(r["reason"] for r in rows if r["usable"] is False).most_common()),
            "issues": dict(Counter(i.split(":")[0] for r in rows for i in r["issues"]).most_common()),
            "groups": dict(Counter(r["group"] for r in ok).most_common()),
            "median_word_edits": edits[len(edits) // 2] if edits else None,
        }
    return out


TWIN_COLUMNS = ("id", "split", "source", "label", "axis", "group", "usable", "reason", "reason_text", "twin_ok",
                "issues", "n_changed", "text", "twin_text", "swaps", "kept", "lexicon_terms", "tagger")


def twins(args, generate=None):
    """LLM twins for every post (and axis) the lexicon tagged in `args.input`. Replies are appended to
    <out>_raw.jsonl as each chunk finishes; a rerun skips posts already answered, so a stopped run
    resumes. The checks always rerun on every reply, so changing them needs no new LLM calls."""
    import pandas as pd

    res = twin_resources()
    frame = pd.read_csv(args.input, keep_default_na=False, dtype={"id": str})
    if args.ids:
        keep = _read_ids(args.ids)
        frame = frame[frame["id"].astype(str).isin(keep)]
        print(f"--ids: {len(frame)} of {len(keep)} listed posts found in {args.input}")
    if args.limit:
        frame = frame.head(args.limit)
    rows = {str(r["id"]): r for r in frame.to_dict("records")}
    axes = tuple(args.axes)
    jobs = [(pid, axis) for pid, row in rows.items() for axis in twin_tasks(row, axes)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_name(out_path.stem + "_raw.jsonl")
    if args.fresh and raw_path.exists():
        backup = raw_path.with_name(f"{raw_path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.bak.jsonl")
        raw_path.rename(backup)
        print(f"--fresh: moved the saved replies to {backup}")
    replies, models = _load_replies(raw_path)
    if models - {args.model}:
        raise SystemExit(f"{raw_path} holds replies from {sorted(map(str, models))}, not {args.model}. "
                         "Use --fresh (the old replies are kept as a .bak file) or another --out.")
    pending = [j for j in jobs if j not in replies]
    per_axis = Counter(axis for _, axis in jobs)
    print(f"{len(rows)} posts, {len(jobs)} (post, axis) jobs {dict(per_axis)}; "
          f"{len(jobs) - len(pending)} already answered, {len(pending)} to tag")

    if pending:
        if generate is None:
            import os
            generate = HFGenerator(args.model, args.quantization, args.batch_size, max_new_tokens=args.max_new_tokens,
                                   token=os.environ.get("HF_TOKEN") or None, verbose=False)
        # Longest posts first: similar lengths share a batch (less padding), an out-of-memory batch
        # fails in the first minutes, and the time estimate starts pessimistic.
        pending.sort(key=lambda j: len(rows[j[0]]["text"]), reverse=True)
        if raw_path.exists() and raw_path.stat().st_size:
            with open(raw_path, "rb") as handle:
                handle.seek(-1, 2)
                ends_cleanly = handle.read(1) == b"\n"
            if not ends_cleanly:
                with open(raw_path, "a", encoding="utf-8") as handle:
                    handle.write("\n")
        start_time, done = time.time(), 0
        with open(raw_path, "a", encoding="utf-8") as handle:
            for start in range(0, len(pending), args.chunk_size):
                chunk = pending[start:start + args.chunk_size]
                got = generate([build_twin_messages(rows[pid]["text"], axis) for pid, axis in chunk])
                for (pid, axis), reply in zip(chunk, got):
                    handle.write(json.dumps({"id": pid, "axis": axis, "model": args.model, "reply": reply},
                                            ensure_ascii=False) + "\n")
                    replies[(pid, axis)] = reply
                handle.flush()
                done += len(chunk)
                rate = done / max(time.time() - start_time, 1e-9)
                print(f"  {done}/{len(pending)} tagged, {rate:.1f}/s, about {(len(pending) - done) / rate / 60:.0f} min left",
                      flush=True)

    records = []
    for pid, axis in jobs:
        row = rows[pid]
        checked = check_twin(row["text"], axis, parse_reply(replies[(pid, axis)]), res)
        records.append({
            "id": pid, "split": row.get("split", ""), "source": row.get("source", ""), "label": row.get("label", ""),
            "axis": axis, **checked, "text": row["text"],
            "lexicon_terms": row.get("gender_terms" if axis == "gender" else "disability_terms", ""),
            "tagger": args.model,
        })
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TWIN_COLUMNS)
        writer.writeheader()
        for r in records:
            writer.writerow({**{k: r.get(k, "") for k in TWIN_COLUMNS},
                             "usable": "" if r["usable"] is None else r["usable"],
                             "issues": "; ".join(r["issues"]),
                             "swaps": json.dumps(r["swaps"], ensure_ascii=False),
                             "kept": json.dumps(r["kept"], ensure_ascii=False)})
    summary = twin_summary(records, axes)
    with open(out_path.with_name(out_path.stem + "_summary.json"), "w") as handle:
        json.dump({"input": str(args.input), "tagger": args.model, "axes": summary}, handle, indent=2)
    print(f"\nWrote {len(records)} rows to {out_path}")
    for axis, s in summary.items():
        if s["posts"]:
            print(f"  {axis:>10}: {s['twin_ok']}/{s['posts']} posts have a usable twin ({s['twin_ok_share']:.0%}); "
                  f"no twin: {s['no_twin']}; issues: {s['issues']}")
    return records, summary


def load_twins(path, axes=AXES):
    """{post id: [(axis, twin text, group), ...]} for the rows with twin_ok, for a training run or the
    transfer test. In train_lm.build_pairs these take the place of gender_counterfactual() and
    disability_counterfactual() for the posts they cover; the same random pick between axes and the
    same skip of posts whose hate is aimed at the group (load_targets) still apply. `group` is the
    original's group (cf_group): male, female or disabled, or mixed for a gender post about both,
    which has no direction."""
    import pandas as pd

    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    frame = frame[frame["twin_ok"].map(_truthy) & frame["axis"].isin(list(axes))]
    out = {}
    for r in frame.to_dict("records"):
        out.setdefault(r["id"], []).append((r["axis"], r["twin_text"], r["group"]))
    return out


SHEET_QUESTIONS = ("decision_ok", "twin_ok")


def twins_sheet(args):
    """A review sheet of `args.n` twin rows, half per axis: rows with a usable twin and rows the LLM
    gave no twin (rows the rule checks dropped are left out: nothing uses them). Two reviewers answer
    yes/no without looking at each other's columns:
      decision_ok: is the LLM right to give a twin or not (and is its reason, e.g. no_person, right)?
      twin_ok:     only where there is a twin: is every group word swapped, nothing else changed,
                   and does it read naturally?"""
    import pandas as pd

    frame = pd.read_csv(args.twins, keep_default_na=False, dtype=str)
    pool = frame[frame["twin_ok"].map(_truthy) | (frame["usable"].str.lower() == "false")]
    axes = [a for a in AXES if (pool["axis"] == a).any()]
    parts = []
    for i, axis in enumerate(axes):
        sub = pool[pool["axis"] == axis]
        n = args.n // len(axes) + (1 if i < args.n % len(axes) else 0)
        parts.append(sub.sample(n=min(n, len(sub)), random_state=args.seed))
    sheet = pd.concat(parts).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    sheet = pd.DataFrame({
        "id": sheet["id"], "axis": sheet["axis"], "group": sheet["group"],
        "llm_decision": ["twin" if _truthy(u) else f"no twin: {r}" for u, r in zip(sheet["usable"], sheet["reason"])],
        "text": sheet["text"], "twin_text": sheet["twin_text"], "swaps": sheet["swaps"],
        **{f"r{k}_{q}": "" for k in (1, 2) for q in SHEET_QUESTIONS}, "notes": "",
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(args.out, index=False)
    print(f"Wrote {len(sheet)} rows to {args.out}: {sheet['axis'].value_counts().to_dict()}, "
          f"{(sheet['llm_decision'] == 'twin').sum()} with a twin. Reviewers fill r1_* / r2_* with yes or no.")
    return sheet


def _yes_no(value):
    v = str(value).strip().lower()
    return "yes" if v in ("yes", "y", "1", "true", "ok") else "no" if v in ("no", "n", "0", "false") else ""


def _clean(value):
    return None if isinstance(value, float) and math.isnan(value) else value


def twins_agree(args):
    """Agreement of the two reviewers on the twins sheet (Cohen's kappa) and the share of rows both
    judge right, per question, overall and per axis. `no_twin_decision_ok` restricts decision_ok to
    the rows the LLM gave no twin: how often "not about a person" (or its other reasons) is right."""
    import pandas as pd

    sheet = pd.read_csv(args.sheet, keep_default_na=False, dtype=str)
    for k in (1, 2):
        for q in SHEET_QUESTIONS:
            sheet[f"r{k}_{q}"] = sheet[f"r{k}_{q}"].map(_yes_no)
    subsets = {"all": sheet, **{a: sheet[sheet["axis"] == a] for a in AXES}}
    report = {}
    for name, sub in subsets.items():
        no_twin = sub[sub["llm_decision"].str.startswith("no twin")]
        with_twin = sub[sub["llm_decision"] == "twin"]
        for label, rows, q in (("decision_ok", sub, "decision_ok"), ("no_twin_decision_ok", no_twin, "decision_ok"),
                               ("twin_ok", with_twin, "twin_ok")):
            a, b = f"r1_{q}", f"r2_{q}"
            agree = pair_agreement(rows, a, b)
            both = rows[(rows[a] != "") & (rows[b] != "")]
            report.setdefault(name, {})[label] = {
                "n": agree["n"], "raw_agreement": _clean(agree["raw"]), "kappa": _clean(agree["kappa"]),
                "yes_r1": _clean((both[a] == "yes").mean()) if len(both) else None,
                "yes_r2": _clean((both[b] == "yes").mean()) if len(both) else None,
                "yes_both": _clean(((both[a] == "yes") & (both[b] == "yes")).mean()) if len(both) else None,
            }
    out = Path(args.out) if args.out else Path(args.sheet).with_suffix(".agreement.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2)
    for name, block in report.items():
        for label, s in block.items():
            if s["n"]:
                kappa = "n/a" if s["kappa"] is None else f"{s['kappa']:.2f}"
                print(f"  {name:>10} {label:>19}: {s['n']} rows, raw {s['raw_agreement']:.2f}, kappa {kappa}, "
                      f"both say yes {s['yes_both']:.0%}")
    print(f"Wrote {out}")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description="LLM tagging of swaps and control synonyms (see module docstring)")
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tag")
    t.add_argument("--input", required=True, help="verification_300.csv (or test.csv) from Content_Moderation.ipynb")
    t.add_argument("--out-dir", required=True)
    t.add_argument("--model", default=DEFAULT_MODEL)
    t.add_argument("--quantization", choices=("auto", "none", "4bit"), default="auto")
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--limit", type=int, default=None, help="tag only the first N posts (a quick check)")
    f = sub.add_parser("finalize")
    f.add_argument("--review", required=True, help="the review CSV with verdicts filled in")
    f.add_argument("--tagged", required=True, help="tagged_posts.jsonl from `tag`")
    f.add_argument("--out", required=True, help="verified_posts.jsonl")
    w = sub.add_parser("twins", help="one LLM twin per post and axis, for training (train.csv) or testing (test.csv)")
    w.add_argument("--input", required=True, help="train.csv, val.csv or test.csv from Content_Moderation.ipynb")
    w.add_argument("--out", required=True,
                   help="twins CSV, e.g. twins_train.csv; replies go to <name>_raw.jsonl, counts to <name>_summary.json")
    w.add_argument("--axes", nargs="+", choices=AXES, default=list(AXES))
    w.add_argument("--ids", default=None,
                   help="tag only these posts: a CSV with an id column, or one id per line (e.g. the exact training sample)")
    w.add_argument("--model", default=DEFAULT_MODEL)
    w.add_argument("--quantization", choices=("auto", "none", "4bit"), default="auto")
    w.add_argument("--batch-size", type=int, default=16)
    w.add_argument("--max-new-tokens", type=int, default=256)
    w.add_argument("--chunk-size", type=int, default=256,
                   help="replies are saved after every chunk, so a stopped run resumes where it left off")
    w.add_argument("--fresh", action="store_true", help="tag everything again (saved replies are kept as a .bak file)")
    w.add_argument("--limit", type=int, default=None, help="only the first N posts (a quick check)")
    s = sub.add_parser("twins-sheet", help="a review sheet of twins for two reviewers")
    s.add_argument("--twins", required=True, help="the twins CSV from `twins` (e.g. twins_test.csv)")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("twins-agree", help="reviewer agreement on the filled twins sheet")
    a.add_argument("--sheet", required=True)
    a.add_argument("--out", default=None, help="default: <sheet>.agreement.json")
    args = p.parse_args(argv)
    commands = {"tag": tag, "finalize": finalize, "twins": twins, "twins-sheet": twins_sheet, "twins-agree": twins_agree}
    commands[args.command](args)


if __name__ == "__main__":
    main()
