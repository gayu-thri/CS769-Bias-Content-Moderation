"""LLM tagging of held-out posts: targeted swaps and control synonyms.

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
"""

from __future__ import annotations

import argparse
import csv
import importlib.resources
import json
import re
import sys
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


def load_resources():
    pairs = [(m.lower(), f.lower()) for m, f in _resource("gender_lexicon.json")["pairs"]]
    return {
        "male": frozenset(m for m, _ in pairs),
        "female": frozenset(f for _, f in pairs),
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

    def __init__(self, model_name, quantization="auto", batch_size=8, max_new_tokens=384, token=None):
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
        self.quantization = quantization

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
    args = p.parse_args(argv)
    (tag if args.command == "tag" else finalize)(args)


if __name__ == "__main__":
    main()
