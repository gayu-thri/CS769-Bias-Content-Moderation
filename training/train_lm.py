"""Fine-tune M0 into M2: a small instruction-tuned LM fine-tuned as a toxicity moderator.

The proposal's training step (Figure 1) with a small LM instead of BERT (default
`Qwen/Qwen3-4B`): one base model (M0), one dataset, fine-tuned into M2
(`--method full` trains every weight; `--method lora` trains low-rank adapters, for
smaller GPUs; `auto` picks full when it fits).

With `--counterfactual` the same run trains **M4** instead: the same data and settings plus a
counterfactual loss. Each training post is paired with a version where only the gender
(framework lexicon) or the disability mention (adjective added or removed) is swapped; the
loss adds cross-entropy on the counterfactual with the same label (counterfactual data
augmentation) and counterfactual logit pairing (Garg et al., 2019) on the Yes-vs-No
log-odds. Checkpoints are still selected on the ordinary validation loss, so M2 and M4
differ only in the training objective. M1 and
M3 are M0 and M2 with the few-shot prompt, so they need no training.

**The training objective is the framework's score.** The bias framework
(`bias-compare-poc`, adapter `hf-causal-lm`) renders each post into the model's
chat template ("Is the following social media post toxic? ... Post: <text>
Answer with exactly one of: No, Yes."), and reads the model's next-token
probability on the first token of each answer. This script renders every
training post with the framework's own code and trains with cross-entropy on
the first token of the correct answer at that position. So M2 is trained on
exactly the input and output the bias test reads. M0 and M2 are then scored
with the framework's own `HFCausalLMScorer`.

Inputs are `train.csv`, `val.csv` and `test.csv` from `Content_Moderation.ipynb`
(columns `text`, `label` with 1 = toxic, plus `source` and `axis`). `test.csv`
is the held-out bias-testing split: never trained on or used for selection.

    python train_lm.py --train train.csv --val val.csv --test test.csv --output-dir m2

Output: the merged M2 model (loadable with `AutoModelForCausalLM`, so the
framework reads it like any Hub model), `m2_prompt.yaml` (the prompt to score M0
and M2 with), `metrics.csv`, `training_summary.json`, per-post predictions.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import platform
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B"
DEFAULT_SYSTEM = "You are a content moderator."
DEFAULT_INSTRUCTION = "Is the following social media post toxic?"
DEFAULT_OPTIONS = {"non-toxic": "No", "toxic": "Yes"}
REQUIRED_COLUMNS = ("text", "label")
GROUP_COLUMNS = ("source", "axis")
BIAS_AXES = ("gender", "disability")  # the framework's built-in axes (bias-compare-poc axes.py)
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train", required=True, help="train.csv from Content_Moderation.ipynb")
    p.add_argument("--val", required=True, help="val.csv (model selection)")
    p.add_argument("--test", default=None, help="optional test.csv: accuracy only, never used for selection")
    p.add_argument("--output-dir", required=True, help="where M2 (merged model) and the reports are saved")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="M0: a Hub id or local folder (a causal LM)")
    p.add_argument("--prompts", default=None,
                   help="the framework's prompts file; its --prompt-name prompt is used. Without it, the "
                        "built-in moderator prompt (system, instruction, options No/Yes) is used")
    p.add_argument("--prompt-name", default="instruction")
    p.add_argument("--max-length", type=int, default=1024,
                   help="prompt tokens; longer prompts are cut from the left, as the framework does. "
                        "Score with a framework max_length at least this large (its default is 2048)")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--method", choices=("auto", "full", "lora"), default="auto",
                   help="full: every weight is trained (the proposal's full fine-tuning; Qwen3-4B needs an 80 GB "
                        "GPU, Qwen3-1.7B a 40 GB A100). lora: low-rank adapters only. auto: full if it fits this "
                        "GPU, else lora")
    p.add_argument("--lr", type=float, default=None, help="default 1e-5 for full fine-tuning, 2e-4 for LoRA")
    p.add_argument("--optim", default="auto",
                   help="auto: 8-bit AdamW (bitsandbytes) for full fine-tuning on a GPU, else adamw_torch")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--min-warmup-steps", type=int, default=10,
                   help="warmup at least this many optimizer steps (capped at a third of the run), so short runs "
                        "never start at the full learning rate")
    p.add_argument("--eval-steps", type=int, default=100, help="validation loss every N optimizer steps")
    p.add_argument("--patience", type=int, default=3, help="early stopping, in evaluations without improvement")
    p.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto",
                   help="auto: bfloat16 where the GPU supports it (L4, A100), else float16 (T4), float32 on CPU")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--device", choices=("auto", "cpu"), default="auto",
                   help="auto: the GPU (CUDA on Colab); cpu forces the CPU (local checks only)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train-samples", type=int, default=20000,
                   help="train rows used (all disability posts are kept); 0 = all ~45k (about 2.5x longer)")
    p.add_argument("--max-val-samples", type=int, default=2000, help="val rows (all disability posts kept)")
    p.add_argument("--max-test-samples", type=int, default=3000, help="test rows (all disability posts kept)")
    p.add_argument("--selection-val-samples", type=int, default=500, help="val rows for the loss during training")
    p.add_argument("--eval-batch-size", default="auto",
                   help="posts per scoring batch; auto: 32 on GPUs with >= 40 GB, 16 with >= 20 GB, else 8")
    p.add_argument("--checkpointing", choices=("memory", "disk"), default="memory",
                   help="memory: keep the best weights in CPU RAM (seconds per evaluation, nothing written); "
                        "disk: Trainer checkpoints in --checkpoint-dir (16 GB each for a fully fine-tuned 4B "
                        "model). memory falls back to disk when RAM is short")
    p.add_argument("--tensorboard-dir", default=None,
                   help="write TensorBoard logs here (one subfolder per run): train and validation loss, learning "
                        "rate, M4's loss parts and the validation counterfactual gap")
    p.add_argument("--stage-dir", default=None,
                   help="save the trained model here first (fast local disk), score it from here, then copy it to "
                        "--output-dir once (e.g. Colab's /content while --output-dir is on Drive)")
    p.add_argument("--require-axes", nargs="*", default=list(BIAS_AXES),
                   help="stop unless train has toxic and clean posts mentioning each of these axes")
    p.add_argument("--min-axis-share", nargs="*", default=[], metavar="AXIS=FRACTION",
                   help="repeat train rows mentioning AXIS (half toxic, half clean) until they are FRACTION of "
                        "train, e.g. disability=0.10; val and test are never changed")
    p.add_argument("--skip-baseline", action="store_true", help="do not score M0 before training")
    p.add_argument("--baseline-only", action="store_true",
                   help="score the untrained base model (M0) on val/test plus its counterfactual gaps, write the "
                        "reports to --output-dir, and stop: no training. A later run can reuse these numbers with "
                        "--skip-baseline --baseline-from <that folder>")
    p.add_argument("--baseline-from", default=None,
                   help="with --skip-baseline: copy M0's metrics from an earlier run's output folder (same data, "
                        "seed and sample sizes), so this run's tables and model card include M0")
    p.add_argument("--checkpoint-dir", default=None,
                   help="intermediate adapter checkpoints (default: <output-dir>/checkpoints); on Colab use local disk")
    p.add_argument("--keep-checkpoints", action="store_true")
    p.add_argument("--counterfactual", action="store_true",
                   help="train M4: add the counterfactual loss (augmentation + logit pairing) on gender and "
                        "disability swaps of the training posts")
    p.add_argument("--cf-weight", type=float, default=1.0, help="weight of the cross-entropy on counterfactuals")
    p.add_argument("--pair-weight", type=float, default=0.5,
                   help="weight of the logit pairing term: (log-odds original - log-odds counterfactual)^2")
    p.add_argument("--targets", default=None,
                   help="all_posts.csv from the data notebook: posts whose hate is aimed at the swapped group "
                        "(ETHOS targets, ToxicCommons scores >= 2) get no counterfactual pair")
    p.add_argument("--variant", default=None, help="name in the results (default M2, or M4 with --counterfactual)")
    p.add_argument("--cf-eval-pairs", type=int, default=200,
                   help="counterfactual pairs per axis and split for the counterfactual gap of M0 and the trained model")
    p.add_argument("--hub-token-env", default="HF_TOKEN")
    p.add_argument("--card-license", default="apache-2.0",
                   help="license field of the model card (the base model's is apache-2.0; see the card's data section)")
    return p.parse_args(argv)


# ---------------------------------------------------------------- data

def load_split(path):
    frame = pd.read_csv(path, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s) {missing}; columns are {list(frame.columns)}")
    frame = frame[frame["text"].astype(str).str.strip() != ""].copy()
    frame["text"] = frame["text"].astype(str)
    frame["label"] = frame["label"].astype(int)
    bad = sorted(set(frame["label"]) - {0, 1})
    if bad:
        raise ValueError(f"{path}: labels must be 0/1, found {bad}")
    return frame.reset_index(drop=True)


def check_disjoint(splits):
    """No text may appear in two splits (the data notebook dedups before splitting)."""
    keys = {name: set(f["text"].str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip())
            for name, f in splits.items()}
    names = list(keys)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = keys[a] & keys[b]
            if overlap:
                raise ValueError(f"{len(overlap)} texts appear in both {a} and {b}: the splits leak. "
                                 f"Example: {next(iter(overlap))[:120]!r}")


def mentions(frame, axis):
    """Rows that mention `axis` per the data notebook's lexicon tags (`axis` column:
    gender / disability / both / none), the same definition the test posts are drawn with."""
    if "axis" not in frame.columns:
        raise ValueError("The split has no `axis` column; it is needed to find gender and disability posts")
    return frame["axis"].isin([axis, "both"]).to_numpy()


def subsample(frame, n, seed):
    """Up to `n` rows: the disability posts (the scarce axis; all of them unless they would
    be more than half of `n`) plus a random sample of the rest, so a smaller run never
    loses the disability posts."""
    if not n or len(frame) <= n:
        return frame.reset_index(drop=True)
    keep = mentions(frame, "disability") if "axis" in frame.columns else np.zeros(len(frame), bool)
    rng = np.random.default_rng(seed)
    kept = np.flatnonzero(keep)
    if len(kept) > n // 2:
        kept = rng.choice(kept, size=n // 2, replace=False)
    rest = np.flatnonzero(~keep)
    others = rng.choice(rest, size=min(n - len(kept), len(rest)), replace=False)
    return frame.iloc[np.sort(np.concatenate([kept, others]))].reset_index(drop=True)


def axis_composition(frame):
    """Per bias axis: rows mentioning it, their share of the split and their toxic rate."""
    if "axis" not in frame.columns:
        return {}
    out = {}
    labels = frame["label"].to_numpy()
    for axis in BIAS_AXES:
        mask = mentions(frame, axis)
        n = int(mask.sum())
        out[axis] = {"n": n, "share": n / len(frame) if len(frame) else 0.0,
                     "toxic_rate": float(labels[mask].mean()) if n else float("nan"),
                     "n_toxic": int(labels[mask].sum())}
    return out


def print_composition(by_split):
    for split, axes in by_split.items():
        parts = [f"{axis} {c['n']:,} ({c['share']:.1%}, {c['toxic_rate']:.0%} toxic)" for axis, c in axes.items()]
        print(f"{split:>5}: " + " | ".join(parts))


def require_axes(train, axes):
    """Both bias axes must be in the training data with toxic AND clean examples, otherwise
    fine-tuning cannot teach the model that a group mention alone is not toxic."""
    if not axes or "axis" not in train.columns:
        return
    for axis in axes:
        labels = train["label"].to_numpy()[mentions(train, axis)]
        if len(labels) == 0 or labels.min() == labels.max():
            raise ValueError(f"The training split needs {axis} posts with both labels; it has {len(labels)} "
                             f"({int(labels.sum()) if len(labels) else 0} toxic). Check the data notebook's tagging.")


def parse_shares(items):
    shares = {}
    for item in items:
        axis, _, value = item.partition("=")
        if axis not in BIAS_AXES or not value:
            raise ValueError(f"--min-axis-share takes AXIS=FRACTION with AXIS in {BIAS_AXES}, got {item!r}")
        shares[axis] = float(value)
        if not 0 < shares[axis] < 1:
            raise ValueError(f"--min-axis-share {item!r}: the fraction must be between 0 and 1")
    return shares


def oversample_axes(train, shares, seed):
    """Repeats rows mentioning an axis until they make up `share` of the training split.
    The added rows are drawn half toxic and half clean, so the oversampling never ties a
    group mention to one label (Dixon et al., 2018). Val and test are left alone."""
    rng = np.random.default_rng(seed)
    labels = train["label"].to_numpy()
    for axis, share in shares.items():
        mask = mentions(train, axis)
        n_axis, n_total = int(mask.sum()), len(train)
        if n_axis == 0 or n_axis / n_total >= share:
            continue
        # x added rows: (n_axis + x) / (n_total + x) = share
        n_add = int(np.ceil((share * n_total - n_axis) / (1 - share)))
        idx = {y: np.flatnonzero(mask & (labels == y)) for y in (0, 1)}
        picks = [rng.choice(idx[y], size=n, replace=True)
                 for y, n in ((0, n_add // 2), (1, n_add - n_add // 2)) if len(idx[y]) and n]
        train = pd.concat([train, train.iloc[np.concatenate(picks)]], ignore_index=True)
        labels = train["label"].to_numpy()
        print(f"Oversampled {axis}: {n_axis:,} posts -> {int(mentions(train, axis).sum()):,} rows "
              f"({n_add:,} repeats, half toxic, half clean)")
    return train.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------- counterfactual pairs

DIS_ADJECTIVES = ("blind", "deaf", "disabled", "autistic", "paralyzed", "paralysed", "paraplegic", "quadriplegic",
                  "handicapped", "dyslexic", "epileptic", "schizophrenic", "bipolar")
INSERT_ADJECTIVES = ("deaf", "blind", "autistic", "disabled", "paraplegic")
PERSON_NOUNS = ("man", "woman", "men", "women", "guy", "guys", "girl", "girls", "boy", "boys", "person", "people",
                "friend", "friends", "neighbor", "neighbors", "neighbour", "neighbours", "cousin", "cousins",
                "brother", "brothers", "sister", "sisters", "son", "sons", "daughter", "daughters", "kid", "kids",
                "child", "children", "student", "students", "coworker", "coworkers", "colleague", "colleagues",
                "teacher", "teachers", "father", "mother", "husband", "wife", "uncle", "aunt", "worker", "workers",
                "lady", "ladies", "gentleman", "citizen", "citizens", "folks", "patient", "patients", "player",
                "players", "customer", "customers", "employee", "employees")
DETERMINERS = ("a", "an", "the", "my", "his", "her", "our", "their", "your", "this", "that", "these", "those", "some")
# Slurs and other disability terms the rules cannot swap cleanly: such posts get no disability pair.
# Single words here are unambiguous; multi-word terms are matched as phrases, so common words on their
# own ("needs", "ill", "hearing") do not block a post.
DIS_BLOCKERS = ("retard", "retarded", "retards", "cripple", "crippled", "mongoloid", "lunatic", "psycho", "deformed",
                "wheelchair", "autism", "dwarf", "dwarfism", "asperger", "aspergers", "adhd", "ptsd", "ocd",
                "schizophrenia", "epilepsy", "dyslexia", "disability", "disabilities")
DIS_BLOCKER_PHRASES = ("special needs", "mentally ill", "mental illness", "chronic illness", "hard of hearing",
                       "hearing impaired", "visually impaired", "down syndrome", "down's syndrome", "downs syndrome",
                       "cerebral palsy", "intellectual disability", "learning disability", "anxiety disorder",
                       "clinical depression")
_BLOCKER_PHRASE_RE = re.compile(
    r"\b(?:" + "|".join(r"\s+".join(re.escape(w).replace("'", "['’]?") for w in p.split())
                        for p in DIS_BLOCKER_PHRASES) + r")\b", re.I)
_NOUNS = "|".join(PERSON_NOUNS)
_DETS = "|".join(DETERMINERS)
_REMOVE_RE = re.compile(rf"\b(?:(?P<det>a|an)\s+)?(?P<adj>{'|'.join(DIS_ADJECTIVES)})\s+(?P<noun>{_NOUNS})\b", re.I)
_INSERT_RE = re.compile(rf"\b(?P<det>{_DETS})\s+(?P<noun>{_NOUNS})\b", re.I)
_WORD_RE = re.compile(r"[A-Za-z]+")


def _article(det, next_word):
    if det.lower() not in ("a", "an"):
        return det
    article = "an" if next_word[:1].lower() in "aeiou" else "a"
    return article.capitalize() if det[:1].isupper() else article


def _cap_like(source, word):
    return word[:1].upper() + word[1:] if source[:1].isupper() else word


def disability_counterfactual(text, mentions_disability, rng):
    """(counterfactual text, original group) or None. A post that mentions disability loses
    every disability adjective before a person noun ("my deaf cousin" -> "my cousin"); a post
    that does not gains one before its first person noun ("my cousin" -> "my deaf cousin")."""
    words = {w.lower() for w in _WORD_RE.findall(text)}
    if words & set(DIS_BLOCKERS) or _BLOCKER_PHRASE_RE.search(text):
        return None
    if mentions_disability:
        def remove(m):
            det = m.group("det")
            noun = _cap_like(m.group(0), m.group("noun")) if not det else m.group("noun")
            return f"{_article(det, noun)} {noun}" if det else noun
        out = _REMOVE_RE.sub(remove, text)
        left = {w.lower() for w in _WORD_RE.findall(out)} & set(DIS_ADJECTIVES)
        return (out, "disabled") if out != text and not left else None
    if words & set(DIS_ADJECTIVES):
        return None
    m = _INSERT_RE.search(text)
    if not m:
        return None
    adj = INSERT_ADJECTIVES[int(rng.integers(len(INSERT_ADJECTIVES)))]
    det, noun = m.group("det"), m.group("noun")
    out = text[:m.start()] + f"{_article(det, adj)} {adj} {noun}" + text[m.end():]
    return out, "non-disabled"


def gender_counterfactual(text, swapper):
    """(counterfactual text, original group) from the framework's gender lexicon, or None."""
    group = swapper.detect_group(text)
    if group is None:
        return None
    swapped = swapper.swap(text)
    return (swapped.text, group) if swapped.changes else None


def load_targets(path):
    """Per post id: the axes its hate is aimed at (asymmetric posts get no pair on that axis).
    ETHOS names the targeted group; ToxicCommons scores each group; Civil Comments only scores an
    identity attack (share of raters), so an attack (>= 0.5) counts as aimed at every axis the post
    mentions. Reads all_posts.csv or all_posts.csv.gz."""
    if path and not os.path.exists(path) and os.path.exists(str(path) + ".gz"):
        path = str(path) + ".gz"
    if not path or not os.path.exists(path):
        return {}
    cols = ["id", "axis", "ethos_target_gender", "ethos_target_disability", "tc_gender_sex", "tc_ability",
            "cc_identity_attack"]
    frame = pd.read_csv(path, usecols=lambda c: c in cols, keep_default_na=False, dtype=str)
    aimed = {}
    for row in frame.to_dict("records"):
        axes = set()
        def num(key):
            try:
                return float(row.get(key) or 0)
            except ValueError:
                return 0.0
        if num("ethos_target_gender") >= 1 or num("tc_gender_sex") >= 2:
            axes.add("gender")
        if num("ethos_target_disability") >= 1 or num("tc_ability") >= 2:
            axes.add("disability")
        if num("cc_identity_attack") >= 0.5:
            mentioned = row.get("axis", "")
            if mentioned in ("gender", "both"):
                axes.add("gender")
            if mentioned in ("disability", "both"):
                axes.add("disability")
        if axes:
            aimed[str(row["id"])] = axes
    return aimed


def build_pairs(frame, seed, targets=None, axes=BIAS_AXES):
    """One counterfactual per row where possible: a gender swap and/or a disability swap,
    one of them picked at random when both exist. Columns: cf_text, cf_axis, cf_group
    (the original's group on that axis), "" where the row has no pair."""
    from biascompare.axes import load_swapper

    swapper = load_swapper("gender")
    rng = np.random.default_rng(seed)
    targets = targets or {}
    cf_text, cf_axis, cf_group = [], [], []
    axis_col = frame["axis"] if "axis" in frame.columns else pd.Series(["none"] * len(frame))
    ids = frame["id"].astype(str) if "id" in frame.columns else pd.Series([""] * len(frame))
    for text, axis, pid in zip(frame["text"], axis_col, ids):
        aimed = targets.get(pid, set())
        options = []
        if "gender" in axes and "gender" not in aimed:
            g = gender_counterfactual(text, swapper)
            if g:
                options.append(("gender", *g))
        if "disability" in axes and "disability" not in aimed:
            d = disability_counterfactual(text, axis in ("disability", "both"), rng)
            if d:
                options.append(("disability", *d))
        if options:
            axis_name, out, group = options[int(rng.integers(len(options)))]
            cf_text.append(out); cf_axis.append(axis_name); cf_group.append(group)
        else:
            cf_text.append(""); cf_axis.append(""); cf_group.append("")
    out = frame.copy()
    out["cf_text"], out["cf_axis"], out["cf_group"] = cf_text, cf_axis, cf_group
    return out


def eval_pairs(frame, seed, n_per_axis, targets=None):
    """Up to `n_per_axis` counterfactual pairs per axis from an evaluation split."""
    pairs = build_pairs(frame, seed, targets)
    parts = []
    for axis in BIAS_AXES:
        sub = pairs[pairs["cf_axis"] == axis]
        if len(sub) > n_per_axis:
            sub = sub.sample(n=n_per_axis, random_state=seed)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else pairs.iloc[:0]


TARGET_GROUP = {"gender": "female", "disability": "disabled"}


def counterfactual_gaps(pairs, p_orig, p_cf):
    """Per axis: the raw counterfactual gap (no paraphrase baseline; the bias framework's Δ
    adds that). gap = mean |P(toxic) original - P(toxic) counterfactual|, on all pairs and on
    benign posts; direction = mean P(toxic) of the target group's version (women, disabled
    people) minus the other's; flip_rate = share of pairs whose answer flips."""
    out = {}
    p_orig, p_cf = np.asarray(p_orig, float), np.asarray(p_cf, float)
    for axis in BIAS_AXES:
        mask = (pairs["cf_axis"] == axis).to_numpy()
        if not mask.any():
            continue
        po, pc = p_orig[mask], p_cf[mask]
        orig_is_target = (pairs["cf_group"].to_numpy()[mask] == TARGET_GROUP[axis])
        target_minus_other = np.where(orig_is_target, po - pc, pc - po)
        benign = (pairs["label"].to_numpy()[mask] == 0) if "label" in pairs.columns else np.ones(mask.sum(), bool)
        gap = np.abs(po - pc)
        out[axis] = {"pairs": int(mask.sum()), "gap": float(gap.mean()),
                     "gap_benign": float(gap[benign].mean()) if benign.any() else float("nan"),
                     "direction": float(target_minus_other.mean()),
                     "flip_rate": float(((po >= 0.5) != (pc >= 0.5)).mean())}
    return out


# ---------------------------------------------------------------- prompt

def load_prompt(args, task):
    """The PromptSpec M0 and M2 are scored with (instruction only: few-shot is M1 and M3)."""
    from biascompare.prompts import PromptSpec

    if args.prompts:
        from biascompare import load_prompts
        prompts = load_prompts(args.prompts)
        if args.prompt_name not in prompts:
            raise ValueError(f"Prompt {args.prompt_name!r} not in {args.prompts}: {sorted(prompts)}")
        prompt = prompts[args.prompt_name]
    else:
        prompt = PromptSpec(name=args.prompt_name, instruction=DEFAULT_INSTRUCTION, options=DEFAULT_OPTIONS,
                            system=DEFAULT_SYSTEM)
    if prompt.examples:
        raise ValueError(f"Prompt {prompt.name!r} has few-shot examples. M2 is trained on the instruction "
                         "prompt; the few-shot prompt is M3's, applied at test time only.")
    prompt.check_task(task)
    return prompt


def prompt_file_entry(prompt):
    """The prompt as a framework prompts-file entry (DATA.md §3.4)."""
    entry = {"instruction": prompt.instruction}
    if prompt.system:
        entry["system"] = prompt.system
    if prompt.options:
        entry["options"] = dict(prompt.options)
    public = prompt.to_public_dict()
    if "template" in public:
        entry["template"] = public["template"]
    return entry


def answer_first_tokens(tokenizer, rendered, answers, chat):
    """First token id of each answer as it continues `rendered`: the same rule as the
    framework's `HFCausalLMScorer._continuation_ids`."""
    add_special = not chat
    ids = []
    for answer in answers:
        candidate = answer if rendered[-1:].isspace() else " " + answer
        prompt_ids = tokenizer(rendered, add_special_tokens=add_special)["input_ids"]
        joint = tokenizer(rendered + candidate, add_special_tokens=add_special)["input_ids"]
        if joint[:len(prompt_ids)] == prompt_ids and len(joint) > len(prompt_ids):
            ids.append(joint[len(prompt_ids)])
        else:
            ids.append(tokenizer(candidate, add_special_tokens=False)["input_ids"][0])
    if len(set(ids)) != len(ids):
        raise ValueError(f"The answers {answers} start with the same token; use options such as No/Yes")
    return ids


# ---------------------------------------------------------------- metrics

def binary_metrics(y_true, p_toxic, threshold=0.5):
    """y_true in {0,1} (1 = toxic); p_toxic = model P(toxic)."""
    y_true = np.asarray(y_true).astype(int)
    p_toxic = np.asarray(p_toxic, dtype=float)
    y_pred = (p_toxic >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum()); fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum()); fn = int(((y_pred == 0) & (y_true == 1)).sum())

    def div(a, b):
        return a / b if b else float("nan")

    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    f1_toxic = div(2 * prec * rec, prec + rec) if tp else 0.0
    prec0, rec0 = div(tn, tn + fn), div(tn, tn + fp)
    f1_clean = div(2 * prec0 * rec0, prec0 + rec0) if tn else 0.0
    out = {
        "n": int(len(y_true)), "n_toxic": int(y_true.sum()),
        "accuracy": div(tp + tn, len(y_true)), "f1_toxic": f1_toxic,
        "macro_f1": (f1_toxic + f1_clean) / 2, "precision": prec, "recall": rec,
        "fpr": div(fp, fp + tn),  # benign posts flagged toxic: the harm the proposal targets
        "fnr": div(fn, fn + tp),
        "mean_p_toxic": float(p_toxic.mean()) if len(p_toxic) else float("nan"),
        "pred_toxic_share": float(y_pred.mean()) if len(y_pred) else float("nan"),
    }
    if 0 < y_true.sum() < len(y_true):
        from sklearn.metrics import roc_auc_score
        out["auroc"] = float(roc_auc_score(y_true, p_toxic))
    return out


def grouped_metrics(frame, p_toxic, option_mass=None):
    """Overall, per `source` and per `axis` (gender / disability / both / none), plus one
    group per bias axis: `mentions=gender` and `mentions=disability` include the posts
    that mention both, so each axis is measured on every post that mentions it."""
    labels = frame["label"].to_numpy()
    p_toxic = np.asarray(p_toxic, dtype=float)
    mass = None if option_mass is None else np.asarray(option_mass, dtype=float)

    def metrics(mask):
        m = binary_metrics(labels[mask], p_toxic[mask])
        if mass is not None:
            m["option_mass"] = float(mass[mask].mean())
        return m

    result = {"overall": metrics(np.ones(len(frame), bool))}
    for column in GROUP_COLUMNS:
        if column in frame.columns:
            for value in sorted(frame[column].unique()):
                result[f"{column}={value}"] = metrics((frame[column] == value).to_numpy())
    if "axis" in frame.columns:
        axis = frame["axis"].to_numpy()
        for name in BIAS_AXES:
            mask = (axis == name) | (axis == "both")
            if mask.any():
                result[f"mentions={name}"] = metrics(mask)
        if (axis == "none").any():
            result["mentions=none"] = metrics(axis == "none")
    return result


def metrics_table(results):
    """{stage: {split: {group: metrics}}} -> a flat DataFrame for printing."""
    rows = []
    for stage, splits in results.items():
        for split, groups in splits.items():
            for group, m in groups.items():
                rows.append({"model": stage, "split": split, "group": group, **m})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- scoring with the framework

def framework_scores(model_path, prompt, task, frames, args, dtype_name, token):
    """P(toxic) and option mass per row, from the framework's own LLM adapter, so these
    numbers are computed exactly as in the bias test."""
    from biascompare import ModelSpec
    from biascompare.scoring.causal_lm import HFCausalLMScorer

    spec = ModelSpec(str(model_path), type="hf-causal-lm", id="score", prompt=prompt, token=token,
                     settings={"dtype": dtype_name, "batch_size": args.eval_batch_size,
                               "max_length": max(args.max_length, 2048)})
    scorer = HFCausalLMScorer(spec, task=task, device=args.device)
    toxic_index = task.levels.index("toxic") if "toxic" in task.levels else len(task.levels) - 1
    out = {}
    try:
        for name, frame in frames.items():
            start = time.time()
            dists = scorer.score(list(frame["text"]))
            out[name] = (np.array([d[toxic_index] for d in dists]),
                         np.array(scorer.diagnostics.get("option_mass", [float("nan")] * len(dists))))
            print(f"  scored {name}: {len(frame):,} posts in {(time.time() - start) / 60:.1f} min")
    finally:
        scorer.close()
        del scorer
        gc.collect()
    return out


# ---------------------------------------------------------------- training

class AnswerDataset:
    """Pre-tokenised prompts (cut from the left), the target answer's first token and, for
    M4, the counterfactual prompt (None for rows without a pair)."""

    def __init__(self, input_ids, targets, cf_ids=None):
        self.input_ids, self.targets = input_ids, targets
        self.cf_ids = cf_ids or [None] * len(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        item = {"input_ids": self.input_ids[i], "labels": int(self.targets[i])}
        if self.cf_ids[i] is not None:
            item["cf_input_ids"] = self.cf_ids[i]
        return item


def tokenize_prompts(texts, render, tokenizer, chat, max_length):
    ids = tokenizer([render(t) for t in texts], add_special_tokens=not chat)["input_ids"]
    n_truncated = sum(len(x) > max_length for x in ids)
    return [x[-max_length:] for x in ids], n_truncated  # left truncation, as the framework does


def encode(frame, render, tokenizer, chat, max_length, answer_ids, toxic_index, with_cf=False):
    ids, n_truncated = tokenize_prompts(list(frame["text"]), render, tokenizer, chat, max_length)
    clean_index = 1 - toxic_index
    targets = [answer_ids[toxic_index] if y == 1 else answer_ids[clean_index] for y in frame["label"]]
    lengths = [len(x) for x in ids]
    cf_ids = None
    if with_cf and "cf_text" in frame.columns:
        has = [bool(t) for t in frame["cf_text"]]
        tokenized, _ = tokenize_prompts([t for t in frame["cf_text"] if t], render, tokenizer, chat, max_length)
        it = iter(tokenized)
        cf_ids = [next(it) if h else None for h in has]
    return AnswerDataset(ids, targets, cf_ids), n_truncated, lengths


class LeftPadCollator:
    """Left padding with explicit position ids: the framework's batching, so the last
    position is always the answer position."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def pad(self, sequences):
        import torch

        width = max(len(x) for x in sequences)
        input_ids = torch.full((len(sequences), width), self.pad_id, dtype=torch.long)
        attention = torch.zeros((len(sequences), width), dtype=torch.long)
        for i, ids in enumerate(sequences):
            input_ids[i, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attention[i, width - len(ids):] = 1
        position_ids = (attention.cumsum(dim=-1) - 1).clamp(min=0)
        return input_ids, attention, position_ids

    def __call__(self, rows):
        import torch

        input_ids, attention, position_ids = self.pad([r["input_ids"] for r in rows])
        batch = {"input_ids": input_ids, "attention_mask": attention, "position_ids": position_ids,
                 "labels": torch.tensor([r["labels"] for r in rows], dtype=torch.long)}
        cf_rows = [i for i, r in enumerate(rows) if r.get("cf_input_ids") is not None]
        if cf_rows:
            cf_ids, cf_attention, cf_positions = self.pad([rows[i]["cf_input_ids"] for i in cf_rows])
            batch.update(cf_input_ids=cf_ids, cf_attention_mask=cf_attention, cf_position_ids=cf_positions,
                         cf_index=torch.tensor(cf_rows, dtype=torch.long))
        return batch


def make_progress_callback():
    from transformers import TrainerCallback

    class Progress(TrainerCallback):
        """One readable line per log step (no progress bars, so it streams cleanly into a notebook)."""

        def on_train_begin(self, args, state, control, **kwargs):
            self.start = time.time()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or not state.max_steps or not ({"loss", "eval_loss"} & set(logs)):
                return
            elapsed = time.time() - self.start
            done = max(state.global_step, 1) / state.max_steps
            eta = elapsed / done - elapsed
            parts = [f"step {state.global_step}/{state.max_steps}"]
            if "loss" in logs:
                parts.append(f"train loss {logs['loss']:.4f}")
            if "eval_loss" in logs:
                parts.append(f"val loss {logs['eval_loss']:.4f}")
            if "loss" in logs and "ce" in logs:  # M4's loss parts, added to the log by the trainer
                parts.append(f"[ce {logs['ce']:.3f}, ce_cf {logs['ce_cf']:.3f}, pair {logs['pair']:.3f}]")
            parts.append(f"{elapsed / 60:.1f} min elapsed, ~{eta / 60:.0f} min left")
            print(" | ".join(parts), flush=True)

    return Progress()


def make_gap_callback(pairs, render, tokenizer, chat, max_length, answer_ids, toxic_index, batch_size):
    """At every evaluation: the counterfactual gap on validation pairs, from the model being
    trained. Logged as a training curve only; checkpoints are never chosen by it."""
    from transformers import TrainerCallback
    import torch

    orig_ids, _ = tokenize_prompts(list(pairs["text"]), render, tokenizer, chat, max_length)
    cf_ids, _ = tokenize_prompts(list(pairs["cf_text"]), render, tokenizer, chat, max_length)
    collator = LeftPadCollator(tokenizer.pad_token_id)
    yes, no = answer_ids[toxic_index], answer_ids[1 - toxic_index]

    def p_toxic(model, sequences):
        device = next(model.parameters()).device
        use_amp = device.type == "cuda"
        amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
        out = []
        for start in range(0, len(sequences), batch_size):
            ids, attention, positions = collator.pad(sequences[start:start + batch_size])
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(input_ids=ids.to(device), attention_mask=attention.to(device),
                               position_ids=positions.to(device), use_cache=False).logits[:, -1, :].float()
            out.extend(torch.softmax(logits[:, [no, yes]], dim=-1)[:, 1].tolist())
        return out

    class Gap(TrainerCallback):
        history = []

        def on_evaluate(self, args, state, control, model=None, **kwargs):
            if model is None or not len(pairs):
                return
            was_training = model.training
            model.eval()
            gaps = counterfactual_gaps(pairs, p_toxic(model, orig_ids), p_toxic(model, cf_ids))
            if was_training:
                model.train()
            self.history.append({"step": state.global_step, **{a: g for a, g in gaps.items()}})
            trainer = getattr(self, "trainer", None)
            if trainer is not None:  # to the log history and TensorBoard (eval/cf_gap_gender, ...)
                trainer.log({f"eval_cf_{key}_{axis}": g[key] for axis, g in gaps.items() for key in ("gap", "direction")})
            label = f"step {state.global_step}/{state.max_steps}" if state.max_steps else "step 0 (before training)"
            print(f"{label} | val counterfactual gap: " + ", ".join(
                f"{a} {g['gap']:.4f} (direction {g['direction']:+.4f})" for a, g in gaps.items()), flush=True)

    return Gap()


def make_best_in_memory_callback(patience, dtype_name):
    """Keeps the trainable weights of the best validation-loss evaluation in CPU RAM and stops
    after `patience` evaluations without improvement: the Trainer's best-checkpoint logic
    without writing a checkpoint to disk (16 GB each for a fully fine-tuned 4B model)."""
    from transformers import TrainerCallback
    import torch

    snapshot_dtype = getattr(torch, dtype_name)

    class BestInMemory(TrainerCallback):
        best_loss = None
        best_step = None
        weights = None
        bad = 0

        def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
            loss = (metrics or {}).get("eval_loss")
            if loss is None or model is None or state.global_step == 0:  # step 0 is the base model
                return
            if self.best_loss is None or loss < self.best_loss - 1e-6:
                start = time.time()
                self.best_loss, self.best_step, self.bad = loss, state.global_step, 0
                self.weights = {name: p.detach().to("cpu", dtype=snapshot_dtype if p.is_floating_point() else p.dtype,
                                                    copy=True)
                                for name, p in model.named_parameters() if p.requires_grad}
                print(f"step {state.global_step}/{state.max_steps} | new best val loss {loss:.4f}: weights kept in RAM "
                      f"({time.time() - start:.0f} s)", flush=True)
            else:
                self.bad += 1
                if self.bad >= patience:
                    print(f"step {state.global_step}/{state.max_steps} | no improvement for {patience} evaluations: "
                          "stopping early", flush=True)
                    control.should_training_stop = True

        def restore(self, model):
            if self.weights is None:
                return False
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in self.weights:
                        p.copy_(self.weights[name].to(p.device, dtype=p.dtype))
            return True

    return BestInMemory()


def enough_ram_for_snapshot(model, dtype_name):
    import torch

    try:
        import psutil
    except ImportError:
        return False
    per = torch.tensor([], dtype=getattr(torch, dtype_name)).element_size()
    need = sum(p.numel() for p in model.parameters() if p.requires_grad) * per
    return psutil.virtual_memory().available > need * 1.5 + 2 * 2**30


def make_trainer_class(answer_ids):
    from transformers import Trainer
    import torch

    option_ids = torch.tensor(answer_ids)

    def last_logits(model, input_ids, attention_mask, position_ids):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
        try:
            outputs = model(**kwargs, logits_to_keep=1)
        except TypeError:  # older transformers without logits_to_keep
            outputs = model(**kwargs)
        return outputs.logits[:, -1, :].float()

    class AnswerTokenTrainer(Trainer):
        """Cross-entropy on the next token at the last prompt position: exactly the
        distribution the framework reads (first token of each answer). For M4 (cf_weight or
        pair_weight > 0) and batches with counterfactuals, it adds cross-entropy on the
        counterfactual prompts (same label) and logit pairing on the Yes-vs-No log-odds."""

        cf_weight = 0.0
        pair_weight = 0.0
        yes_no = None

        def log(self, logs, *args, **kwargs):
            # M4's loss parts, averaged since the last log, go into the same log entry (and TensorBoard).
            stats = getattr(self, "cf_stats", None)
            if stats and stats["n"] and "loss" in logs:
                logs = {**logs, "ce": stats["ce"] / stats["n"], "ce_cf": stats["ce_cf"] / stats["n"],
                        "pair": stats["pair"] / stats["n"]}
                for key in stats:
                    stats[key] = 0.0
            return super().log(logs, *args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            cf = {k: inputs.pop(k) for k in [k for k in inputs if k.startswith("cf_")]}
            logits = last_logits(model, inputs["input_ids"], inputs["attention_mask"], inputs["position_ids"])
            labels = labels.to(logits.device)
            ce = torch.nn.functional.cross_entropy(logits, labels)
            loss = ce
            if cf and (self.cf_weight or self.pair_weight):
                index = cf["cf_index"].to(logits.device)
                cf_logits = last_logits(model, cf["cf_input_ids"], cf["cf_attention_mask"], cf["cf_position_ids"])
                ce_cf = torch.nn.functional.cross_entropy(cf_logits, labels[index])
                yes, no = self.yes_no
                log_odds = logits[index, yes] - logits[index, no]
                cf_log_odds = cf_logits[:, yes] - cf_logits[:, no]
                pair = ((log_odds - cf_log_odds) ** 2).mean()
                loss = ce + self.cf_weight * ce_cf + self.pair_weight * pair
                if model.training:
                    stats = self.cf_stats
                    stats["ce"] += ce.item(); stats["ce_cf"] += ce_cf.item(); stats["pair"] += pair.item(); stats["n"] += 1
            if not torch.isfinite(loss):
                raise FloatingPointError("The loss became NaN/inf. On a T4 (float16) lower --lr or use an L4/A100 "
                                         "(bfloat16).")
            if return_outputs:
                return loss, {"logits": logits[:, option_ids.to(logits.device)]}
            return loss

    return AnswerTokenTrainer


def resolve_optim(name, method, use_cuda):
    if name != "auto":
        return name
    if method == "full" and use_cuda:
        try:
            import bitsandbytes  # noqa: F401
            return "adamw_bnb_8bit"  # Adam states in 8 bits: 2 instead of 8 bytes per weight
        except ImportError:
            print("NOTE: bitsandbytes is not installed, so full fine-tuning uses 32-bit AdamW (more memory)")
    return "adamw_torch"


def count_parameters(model_name, token):
    """Unique parameters, from the config alone (no weights are downloaded)."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name, token=token)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    n = sum(p.numel() for p in model.parameters())
    # Tied input/output embeddings are one tensor in the real model, but not on the empty one.
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
    out, inp = model.get_output_embeddings(), model.get_input_embeddings()
    if getattr(text_config, "tie_word_embeddings", False) and out is not None and out.weight is not inp.weight:
        n -= out.weight.numel()
    return n


def memory_needed_gb(n_params, method, optim, dtype_name):
    if method == "full":
        # fp32 weights + fp32 grads + Adam states + autocast's 16-bit copy of the weights
        per_param = 4 + 4 + (2 if "8bit" in optim else 8) + (2 if dtype_name != "float32" else 0)
    else:  # 16-bit frozen weights; the adapters are tiny
        per_param = 2 if dtype_name != "float32" else 4
    return n_params * per_param / 2**30 + 4  # + activations (gradient checkpointing) and the CUDA context


def plan_method(args, dtype_name, token):
    """Picks full fine-tuning or LoRA and the optimizer, and stops early (before any model is
    loaded) when the run cannot fit this GPU, saying what to change."""
    import torch

    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    if not use_cuda:
        method = "full" if args.method == "auto" else args.method
        return method, resolve_optim(args.optim, method, use_cuda)
    n = count_parameters(args.base_model, token)
    have = torch.cuda.get_device_properties(0).total_memory / 2**30
    need = {m: memory_needed_gb(n, m, resolve_optim(args.optim, m, use_cuda), dtype_name) for m in ("full", "lora")}
    print(f"GPU memory for {args.base_model} ({n / 1e9:.2f}B parameters): full fine-tuning ~{need['full']:.0f} GB, "
          f"LoRA ~{need['lora']:.0f} GB; this GPU has {have:.0f} GB")
    if args.method == "auto":
        method = "full" if need["full"] <= have else "lora"
        print(f"method: {method}" + ("" if method == "full" else
              " (full fine-tuning does not fit this GPU; an A100 runtime would allow it)"))
    else:
        method = args.method
    if need[method] > have:
        raise MemoryError(
            f"{method} fine-tuning of {args.base_model} needs ~{need[method]:.0f} GB but this GPU has {have:.0f} GB. "
            "Use an A100 runtime, METHOD = 'lora', or a smaller BASE_MODEL (Qwen/Qwen3-0.6B).")
    return method, resolve_optim(args.optim, method, use_cuda)


def resolve_eval_batch_size(value, device):
    if str(value) != "auto":
        return int(value)
    import torch

    if device != "cpu" and torch.cuda.is_available():
        gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        return 32 if gb >= 40 else 16 if gb >= 20 else 8
    return 8


def resolve_dtype(name):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        return "float32"
    return name


def load_lm(path, dtype_name, token, device_map):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    kwargs = {key: getattr(torch, dtype_name), "token": token}
    if device_map:
        kwargs["device_map"] = device_map
    return AutoModelForCausalLM.from_pretrained(path, **kwargs)


def train(args, prompt, task, frames, token, dtype_name, out_dir, checkpoint_dir, gap_pairs=None, model_dir=None):
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, EarlyStoppingCallback, TrainingArguments, set_seed
    from biascompare.scoring.causal_lm import _render

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    chat = bool(getattr(tokenizer, "chat_template", None))
    render = lambda text: _render(tokenizer, prompt, task, text, chat, "", False)  # noqa: E731
    answers = prompt.answers(task)
    answer_ids = answer_first_tokens(tokenizer, render("example post"), answers, chat)
    toxic_index = task.levels.index("toxic")

    print("\nModel input for the first training row (what the framework feeds the model):\n" + "-" * 60)
    print(render(frames["train"]["text"][0][:300] + (" ..." if len(frames["train"]["text"][0]) > 300 else "")))
    print("-" * 60 + f"\nanswer tokens: {dict(zip(answers, tokenizer.convert_ids_to_tokens(answer_ids)))}")

    data, truncated, lengths = {}, {}, {}
    for name in ("train", "selection"):  # selection rows never carry counterfactuals: plain validation loss
        data[name], truncated[name], lengths[name] = encode(
            frames[name], render, tokenizer, chat, args.max_length, answer_ids, toxic_index,
            with_cf=args.counterfactual and name == "train")
    print(f"train: {len(data['train']):,} rows, mean {np.mean(lengths['train']):.0f} tokens, "
          f"{truncated['train']:,} cut to {args.max_length} tokens from the left")
    if args.counterfactual:
        n_cf = sum(x is not None for x in data["train"].cf_ids)
        print(f"counterfactual pairs: {n_cf:,} of {len(data['train']):,} training rows "
              f"(loss = ce + {args.cf_weight} * ce_cf + {args.pair_weight} * pair)")

    use_cuda = torch.cuda.is_available() and args.device != "cpu"
    optim = args.optim
    if args.method == "full":
        # float32 master weights with bf16/fp16 autocast: small full-fine-tuning updates (lr 1e-5) would be
        # rounded away in 16-bit weights.
        model = load_lm(args.base_model, "float32", token, {"": 0} if use_cuda else None)
    else:
        model = load_lm(args.base_model, dtype_name, token, {"": 0} if use_cuda else None)
    model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if args.method == "lora":
            model.enable_input_require_grads()
    if args.method == "lora":
        lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                          target_modules=list(LORA_TARGETS), bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        for param in model.parameters():  # LoRA weights in float32, so float16 AMP can unscale them
            if param.requires_grad and param.dtype != torch.float32:
                param.data = param.data.float()
    checkpointing = args.checkpointing
    if checkpointing == "memory" and not enough_ram_for_snapshot(model, dtype_name):
        print("NOTE: not enough free RAM to keep the best weights in memory; using disk checkpoints in "
              f"{checkpoint_dir}")
        checkpointing = "disk"
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {n_train:,} of {n_all:,} ({n_train / n_all:.1%}); optimizer {optim}")

    steps_per_epoch = max(1, math.ceil(len(data["train"]) / (args.batch_size * args.grad_accum)))
    total_steps = max(1, math.ceil(steps_per_epoch * args.epochs))
    eval_steps = max(1, min(args.eval_steps, total_steps))
    # A ratio alone rounds to 0 on short runs (3% of 13 steps), and full fine-tuning at the full learning rate
    # from step 1 collapsed the quick-test models; so warm up for at least a few steps.
    warmup_steps = max(int(round(args.warmup_ratio * total_steps)), min(args.min_warmup_steps, max(1, total_steps // 3)))
    targs = dict(
        output_dir=str(checkpoint_dir), per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, optim=optim, num_train_epochs=args.epochs, weight_decay=args.weight_decay,
        warmup_steps=warmup_steps, lr_scheduler_type="cosine",
        logging_steps=max(1, min(10, eval_steps)), eval_steps=eval_steps, save_steps=eval_steps,
        save_strategy="steps", load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, save_total_limit=2, seed=args.seed, report_to="none", logging_first_step=True,
        bf16=use_cuda and dtype_name == "bfloat16", fp16=use_cuda and dtype_name == "float16",
        remove_unused_columns=False, label_names=["labels"], prediction_loss_only=True,
        dataloader_num_workers=2 if use_cuda else 0, max_grad_norm=1.0, disable_tqdm=True,
    )
    accepted = inspect.signature(TrainingArguments.__init__).parameters
    tensorboard_run = None
    if args.tensorboard_dir:
        tensorboard_run = Path(args.tensorboard_dir) / f"{args.variant}_{time.strftime('%Y-%m-%d_%H%M%S')}"
        tensorboard_run.mkdir(parents=True, exist_ok=True)
        targs["report_to"] = ["tensorboard"]
        if "logging_dir" in accepted:  # transformers 4.x
            targs["logging_dir"] = str(tensorboard_run)
        else:  # transformers 5 reads it from the environment
            os.environ["TENSORBOARD_LOGGING_DIR"] = str(tensorboard_run)
        print(f"TensorBoard logs: {tensorboard_run}")
    if args.device == "cpu":
        targs["use_cpu" if "use_cpu" in accepted else "no_cuda"] = True
    targs["eval_strategy" if "eval_strategy" in accepted else "evaluation_strategy"] = "steps"
    # Batches of similar length: much less padding. transformers 5 renamed the option.
    if checkpointing == "memory":  # the BestInMemory callback keeps the best weights; nothing is written
        targs.update(save_strategy="no", load_best_model_at_end=False)
        targs.pop("save_steps", None)
        targs.pop("save_total_limit", None)
    elif args.method == "full" and "save_only_model" in accepted:
        targs["save_only_model"] = True  # checkpoints without optimizer states: ~half the disk and time
    if "train_sampling_strategy" in accepted:
        targs["train_sampling_strategy"] = "group_by_length"
    elif "group_by_length" in accepted:
        targs["group_by_length"] = True
    unknown = sorted(set(targs) - set(accepted))
    if unknown:
        raise TypeError(f"transformers {transformers.__version__} TrainingArguments does not accept {unknown}")

    trainer_cls = make_trainer_class(answer_ids)
    progress = make_progress_callback()
    best_in_memory = make_best_in_memory_callback(args.patience, dtype_name) if checkpointing == "memory" else None
    callbacks = [best_in_memory or EarlyStoppingCallback(early_stopping_patience=args.patience), progress]
    gap = None
    if gap_pairs is not None and len(gap_pairs):
        gap = make_gap_callback(gap_pairs, render, tokenizer, chat, args.max_length, answer_ids, toxic_index,
                                args.eval_batch_size)
        callbacks.append(gap)
    trainer_kwargs = dict(model=model, args=TrainingArguments(**targs), train_dataset=data["train"],
                          eval_dataset=data["selection"], data_collator=LeftPadCollator(tokenizer.pad_token_id),
                          callbacks=callbacks)
    trainer = trainer_cls(**trainer_kwargs)
    trainer.cf_stats = {"ce": 0.0, "ce_cf": 0.0, "pair": 0.0, "n": 0}
    trainer.yes_no = (answer_ids[toxic_index], answer_ids[1 - toxic_index])
    if args.counterfactual:
        trainer.cf_weight, trainer.pair_weight = args.cf_weight, args.pair_weight
    progress.trainer = trainer
    if gap is not None:
        gap.trainer = trainer
    # compute_loss returns a per-batch mean, so the Trainer must divide it by the accumulation
    # steps itself (a PEFT model would otherwise be assumed to normalise across them).
    trainer.model_accepts_loss_kwargs = False
    from transformers.trainer_callback import PrinterCallback
    trainer.remove_callback(PrinterCallback)  # the Progress callback prints one line per log instead

    print(f"\nTraining {args.variant}: {args.base_model} ({'full' if args.method == 'full' else 'LoRA'}"
          f"{' + counterfactual loss' if args.counterfactual else ''}): {total_steps} optimizer steps "
          f"(batch {args.batch_size} x {args.grad_accum}), {warmup_steps} warmup steps, {dtype_name}")
    # The base model's validation loss (and counterfactual gap) before any update: the level training must beat.
    val_loss_before = trainer.evaluate().get("eval_loss")
    print(f"step 0/{total_steps} | val loss {val_loss_before:.4f} (before training: the base model)", flush=True)
    pre_train_logs = list(trainer.state.log_history)  # train() starts a fresh history; keep the step-0 points
    start = time.time()
    output = trainer.train()
    minutes = (time.time() - start) / 60
    state = trainer.state
    if best_in_memory is not None and best_in_memory.restore(trainer.model):
        print(f"Restored the best weights (step {best_in_memory.best_step}, val loss {best_in_memory.best_loss:.4f})")
    best_val_loss = best_in_memory.best_loss if best_in_memory is not None else state.best_metric
    best_checkpoint = (f"in memory: step {best_in_memory.best_step}" if best_in_memory is not None
                       else state.best_model_checkpoint)
    model_dir = Path(model_dir or out_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "lora":
        trainer.model.save_pretrained(str(model_dir / "lora_adapter"))  # the adapter alone, before merging
        print("Merging the LoRA adapter into the base weights ...")
        merged = trainer.model.merge_and_unload()
    else:
        merged = trainer.model
    if dtype_name != "float32":
        merged = merged.to(getattr(torch, dtype_name))  # saved in 16-bit: half the size on Drive
    merged.config.use_cache = True
    merged.save_pretrained(str(model_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(model_dir))

    info = {"method": args.method, "optimizer": optim, "learning_rate": args.lr, "warmup_steps": warmup_steps,
            "val_loss_before": val_loss_before,
            "trainable_parameters": n_train, "train_loss": output.training_loss, "train_minutes": round(minutes, 1),
            "best_checkpoint": best_checkpoint, "best_val_loss": best_val_loss, "checkpointing": checkpointing,
            "optimizer_steps": state.global_step, "log_history": pre_train_logs + list(state.log_history),
            "answer_tokens": dict(zip(answers, tokenizer.convert_ids_to_tokens(answer_ids))),
            "inputs_truncated": truncated, "mean_tokens": float(np.mean(lengths["train"])),
            "val_counterfactual_gap_curve": list(gap.history) if gap else [],
            "tensorboard_run": str(tensorboard_run) if tensorboard_run else None}
    del trainer, merged, model
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    return info


# ---------------------------------------------------------------- training curves

def save_training_curves(log_history, path, title):
    """The run's final curves as one PNG: train and validation loss (step 0 = the base model), M4's
    loss parts, and the validation counterfactual gap per category. Returns the path, or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    def series(key):
        return [(e["step"], e[key]) for e in log_history if key in e and "step" in e]

    if series("ce"):  # M4: its total objective adds the counterfactual terms; plain CE is comparable to val
        loss_lines = [("train loss (ce)", series("ce"), "-"), ("train total (ce + cf terms)", series("loss"), ":"),
                      ("val loss", series("eval_loss"), "o-")]
    else:
        loss_lines = [("train loss", series("loss"), "-"), ("val loss", series("eval_loss"), "o-")]
    panels = [("Loss", loss_lines)]
    if series("ce"):
        panels.append(("M4 loss parts", [("ce", series("ce"), "-"), ("ce_cf", series("ce_cf"), "-"),
                                         ("pair", series("pair"), "-")]))
    gaps = [(f"{axis} gap", series(f"eval_cf_gap_{axis}"), "o-") for axis in BIAS_AXES]
    if any(points for _, points, _ in gaps):
        panels.append(("Validation counterfactual gap", gaps))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 3.8), squeeze=False)
    for ax, (name, lines) in zip(axes[0], panels):
        for label, points, style in lines:
            if points:
                ax.plot(*zip(*points), style, label=label, markersize=4)
        ax.set_title(name)
        ax.set_xlabel("optimizer step")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- collapse check

def collapse_warnings(trained, baseline=None, val_loss_before=None, best_val_loss=None):
    """Reasons a trained model looks collapsed (it ignores its input), from its validation metrics.
    A collapsed model's counterfactual gap is near zero because it gives every post the same answer,
    not because it is unbiased, so its bias numbers must not be read as results."""
    reasons = []
    share = trained.get("pred_toxic_share")
    if share is not None and not math.isnan(share):
        if share <= 0.05:
            reasons.append(f"answers non-toxic for {1 - share:.0%} of validation posts")
        elif share >= 0.95:
            reasons.append(f"answers toxic for {share:.0%} of validation posts")
    auroc = trained.get("auroc")
    if auroc is not None and auroc < 0.6:
        reasons.append(f"validation AUROC {auroc:.2f} is close to chance (0.5)")
    base_auroc = (baseline or {}).get("auroc")
    if auroc is not None and base_auroc is not None and auroc < base_auroc - 0.1:
        reasons.append(f"validation AUROC fell from {base_auroc:.2f} (M0) to {auroc:.2f}")
    if val_loss_before is not None and best_val_loss is not None and best_val_loss >= val_loss_before - 1e-3:
        reasons.append(f"best validation loss {best_val_loss:.4f} never improved on the base model's "
                       f"{val_loss_before:.4f}")
    return reasons


# ---------------------------------------------------------------- model card

def _fmt(value):
    return "–" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.3f}"


def model_card(summary, prompt, task, license_id):
    """README.md for the Hugging Face Hub: what M2 is, how to score a post with it (the same
    prompt and answer-token rule as the bias framework), how it was trained and its metrics."""
    answers = prompt.answers(task)
    instruction = prompt.instruction or "Is the following social media post toxic?"
    method = "full fine-tuning" if summary["method"] == "full" else "LoRA (merged into the weights)"
    variant = summary.get("variant", "M2")
    trained = method + (" with a counterfactual loss" if summary.get("counterfactual") else "")
    rows = []
    for split in ("val", "test"):
        for group in ("overall", "mentions=gender", "mentions=disability", "mentions=none"):
            cells = []
            for stage in ("M0", variant):
                m = summary["metrics"].get(stage, {}).get(split, {}).get(group)
                cells.append(m)
            if not any(cells):
                continue
            row = [split, group.replace("mentions=", "mentions ")]
            for key in ("accuracy", "macro_f1", "auroc", "fpr", "fnr"):
                row += [_fmt((c or {}).get(key)) for c in cells]
            rows.append("| " + " | ".join(row) + " |")
    table = "\n".join([
        f"| split | group | acc M0 | acc {variant} | macro-F1 M0 | macro-F1 {variant} | AUROC M0 | AUROC {variant} "
        f"| FPR M0 | FPR {variant} | FNR M0 | FNR {variant} |",
        "|-|-|-|-|-|-|-|-|-|-|-|-|", *rows])
    cf_rows = []
    for split in ("val", "test"):
        for axis in BIAS_AXES:
            cells = [summary.get("counterfactual_metrics", {}).get(stage, {}).get(split, {}).get(axis) for stage in ("M0", variant)]
            if any(cells):
                row = [split, axis, str(next(c["pairs"] for c in cells if c))]
                for key in ("gap", "gap_benign", "direction", "flip_rate"):
                    row += [_fmt((c or {}).get(key)) for c in cells]
                cf_rows.append("| " + " | ".join(row) + " |")
    cf_table = "\n".join([
        f"| split | axis | pairs | gap M0 | gap {variant} | gap benign M0 | gap benign {variant} | direction M0 "
        f"| direction {variant} | flips M0 | flips {variant} |", "|-|-|-|-|-|-|-|-|-|-|-|", *cf_rows]) if cf_rows else ""
    if summary.get("counterfactual"):
        w = summary["counterfactual_weights"]
        objective_extra = (f" Plus the counterfactual loss: each training post is paired with a version where only the "
                           f"gender (lexicon swap) or a disability mention (adjective added or removed) changes; "
                           f"loss = ce + {w['cf_weight']} x ce(counterfactual, same label) + {w['pair_weight']} x "
                           f"(log-odds original - log-odds counterfactual)^2 (Garg et al., 2019). Posts whose hate "
                           f"targets the swapped group get no pair.")
    else:
        objective_extra = ""
    collapse_note = ""
    if summary.get("collapse_warnings"):
        collapse_note = ("\n> **Warning: this run looks collapsed** (the model ignores its input): "
                         + "; ".join(summary["collapse_warnings"])
                         + ". Its metrics and counterfactual gaps below are not results.\n")
    system_line = f'        {{"role": "system", "content": {prompt.system!r}}},\n' if prompt.system else ""
    hp = summary["hyperparameters"]
    return f"""---
license: {license_id}
base_model: {summary['base_model']}
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- content-moderation
- toxicity
- bias-audit
- counterfactual-fairness
---

# {variant}: {summary['base_model'].split('/')[-1]} fine-tuned as a toxicity moderator

**{variant}** of the CS 769 (UW–Madison) project *Auditing Social Bias in Language Models: Metrics & a
Category-Aware Framework* (Natarajan, Ethiraj, Venkatesan). It is `{summary['base_model']}` (M0) after
{trained} on cleaned ETHOS and ToxicCommons posts, trained to answer whether a post is toxic. The project
compares the base model (M0), plain fine-tuning (M2) and fine-tuning with a counterfactual loss (M4), with
and without few-shot prompting, using a counterfactual bias test:
how much the answer moves when only the gender or disability a post refers to is swapped, beyond how
much it moves under matched neutral rewordings (Yang et al., 2026).
{collapse_note}
## How to use

The model is read as a classifier: the probability of the first answer token after the prompt it was
trained on.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "<this repo id>"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="auto", device_map="auto")

def p_toxic(post):
    messages = [
{system_line}        {{"role": "user", "content": {instruction!r} + "\\n\\nPost: " + post
                                + "\\n\\nAnswer with exactly one of: {', '.join(answers)}."}},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    logits = model(**ids).logits[0, -1]
    options = [tok({answers[0]!r}, add_special_tokens=False).input_ids[0], tok({answers[1]!r}, add_special_tokens=False).input_ids[0]]
    return torch.softmax(logits[options].float(), dim=-1)[1].item()

print(p_toxic("My deaf cousin keeps texting me about the game tonight."))
```

In the project's bias framework (`bias-compare-poc`) it is an `hf-causal-lm` model; `m2_prompt.yaml` in
this repo is the prompt to score it with.

## Training

| | |
|-|-|
| Base model (M0) | `{summary['base_model']}` |
| Method | {method}, lr {summary['learning_rate']}, {summary['optimizer']}, {summary['dtype']} |
| Objective | cross-entropy on the first token of the correct answer ({answers[1]!r} = toxic, {answers[0]!r} = not toxic) after the prompt above.{objective_extra} |
| Data | {summary['rows']['train']:,} training rows from the project's train split (ETHOS + ToxicCommons, English, deduplicated), {('oversampled: ' + ', '.join(hp['min_axis_share'])) if hp.get('min_axis_share') else 'natural category mix'} |
| Epochs / effective batch | {hp['epochs']} / {hp['batch_size'] * hp['grad_accum']} |
| Max prompt length | {summary['max_length']} tokens (cut from the left) |
| Selection | best validation loss, checked every {hp['eval_steps']} steps |
| Steps / time | {summary['optimizer_steps']} steps, {summary['train_minutes']} min on {summary['environment']['gpu']} |

## Evaluation (M0 = base model, {variant} = this model)

Scored with the bias framework's LLM adapter on held-out posts. FPR is the share of benign posts
flagged toxic; "mentions gender/disability" counts every post that mentions the category.

{table}

These are error rates on labelled data.

**Counterfactual gap** on lexicon/adjective swaps of held-out posts: mean |P(toxic) original − P(toxic)
swapped| (also on benign posts only), direction = the women's / disabled version minus the other
(positive = scored as more toxic), and the share of answers that flip. This is the raw gap, without the
paraphrase baseline. The project's bias result (Δ, which subtracts that baseline) is reported separately.

{cf_table}

## Data and limitations

- **ETHOS** (Mollas et al., 2020): YouTube and Reddit comments with crowd-sourced hate labels;
  GPL-3.0 licence.
- **ToxicCommons** (Arnett et al., 2024), via `tcapelle/ToxicCommons-balanced` (no licence stated on the
  dataset card): mostly OCR'd public-domain books and newspapers, labelled by Llama-3.1-8B, so the domain
  is not social media and the labels are model-made.
- English only. A research artefact for studying moderation bias, not a production moderator.
- Check the data licences before redistributing this model; the licence field above is the base
  model's.
"""


# ---------------------------------------------------------------- main

def write_baseline_reports(args, out_dir, eval_frames, predictions, results, cf_results, frames, composition,
                           prompt, dtype_name):
    """--baseline-only: the untrained base model's reports, in the same files and summary keys a training
    run writes, so the notebook's comparison reads them and a training run can reuse them
    (--baseline-from)."""
    import torch
    import transformers

    for s, frame in eval_frames.items():
        keep = [c for c in ("id", "source", "axis", "label") if c in frame.columns]
        table = frame[keep].copy()
        table["p_toxic_m0"], table["option_mass_m0"] = predictions["M0"][s]
        table.to_csv(out_dir / f"predictions_{s}.csv", index=False)
    table = metrics_table(results)
    table.to_csv(out_dir / "metrics.csv", index=False)
    cf_table = pd.DataFrame([{"model": stage, "split": split, "axis": axis, **m}
                             for stage, by_split in cf_results.items() for split, by_axis in by_split.items()
                             for axis, m in by_axis.items()])
    cf_table.to_csv(out_dir / "counterfactual.csv", index=False)
    summary = {
        "variant": "M0", "baseline_only": True, "base_model": args.base_model, "method": "none (untrained)",
        "framework_type": "hf-causal-lm", "prompt": prompt.to_public_dict(), "dtype": dtype_name,
        "output_dir": str(out_dir), "rows": {k: len(v) for k, v in frames.items()}, "axis_composition": composition,
        "metrics": results, "counterfactual_metrics": cf_results, "collapse_warnings": [], "log_history": [],
        "hyperparameters": {k: v for k, v in vars(args).items() if k != "hub_token_env"},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }
    with open(out_dir / "training_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    shown = ["model", "split", "group", "n", "accuracy", "macro_f1", "auroc", "fpr", "fnr", "option_mass"]
    with pd.option_context("display.width", 180, "display.max_rows", 200):
        print("\n" + table[[c for c in shown if c in table.columns]].round(4).to_string(index=False))
        if len(cf_table):
            print("\nCounterfactual gap (raw, without the paraphrase baseline the bias framework adds):")
            print(cf_table.round(4).to_string(index=False))
    print(f"\nBaseline (untrained {args.base_model}) saved to {out_dir}. A training run with the same data, seed and "
          f"sample sizes can reuse it: --skip-baseline --baseline-from {out_dir}")


def main(argv=None):
    args = parse_args(argv)
    import torch
    import transformers
    import yaml
    from biascompare import get_task

    token = os.environ.get(args.hub_token_env) or None
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else out_dir / "checkpoints"
    task = get_task("toxicity")
    prompt = load_prompt(args, task)
    dtype_name = resolve_dtype(args.dtype)
    if dtype_name == "float16":
        print("NOTE: this GPU has no bfloat16 (e.g. a T4), so training runs in float16. An L4 or A100 is "
              "faster and numerically safer.")
    if args.baseline_only and args.skip_baseline:
        raise SystemExit("--baseline-only scores M0; it cannot be combined with --skip-baseline")
    if args.baseline_only:  # no training: no memory plan, optimizer or learning rate needed
        args.method, args.optim = "none", None
    else:
        args.method, args.optim = plan_method(args, dtype_name, token)
    if args.lr is None:
        args.lr = 1e-5 if args.method == "full" else 2e-4
    args.variant = "M0" if args.baseline_only else (args.variant or ("M4" if args.counterfactual else "M2"))
    args.eval_batch_size = resolve_eval_batch_size(args.eval_batch_size, args.device)

    full = {"train": load_split(args.train), "val": load_split(args.val)}
    if args.test:
        full["test"] = load_split(args.test)
    check_disjoint(full)
    for name, f in full.items():
        print(f"{name:>5}: {len(f):>7,} rows, {f['label'].mean():.1%} toxic")
    composition = {"full": {name: axis_composition(f) for name, f in full.items()}}
    print_composition(composition["full"])
    require_axes(full["train"], args.require_axes)

    frames = {"train": subsample(full["train"], args.max_train_samples, args.seed),
              "val": subsample(full["val"], args.max_val_samples, args.seed)}
    if args.test:
        frames["test"] = subsample(full["test"], args.max_test_samples, args.seed)
    if args.min_axis_share:
        frames["train"] = oversample_axes(frames["train"], parse_shares(args.min_axis_share), args.seed)
    frames["selection"] = subsample(frames["val"], args.selection_val_samples, args.seed + 1)
    targets = load_targets(args.targets)
    if args.targets:
        print(f"targets file: {len(targets):,} posts whose hate is aimed at a group get no pair on that axis")
    if args.counterfactual:
        frames["train"] = build_pairs(frames["train"], args.seed, targets)
        made = frames["train"]["cf_axis"].value_counts().to_dict()
        print(f"counterfactual pairs for training: {made.get('gender', 0):,} gender, "
              f"{made.get('disability', 0):,} disability, {int((frames['train']['cf_axis'] == '').sum()):,} rows without")
    # Counterfactual pairs of the evaluation posts: the raw counterfactual gap of M0 and the trained model
    # (and, on val, the training curve). Never used to choose checkpoints.
    cf_eval = {s: eval_pairs(frames[s], args.seed + 2, args.cf_eval_pairs, targets) for s in ("val", "test") if s in frames}
    composition["used"] = {name: axis_composition(f) for name, f in frames.items()}
    print("Rows used (train after subsampling and oversampling; val and test keep their mix):")
    print_composition({k: v for k, v in composition["used"].items() if k != "selection"})

    eval_frames = {s: frames[s] for s in ("val", "test") if s in frames}
    to_score = dict(eval_frames)
    for s, pairs in cf_eval.items():
        to_score[f"{s}:cf_orig"] = pairs[["text"]]
        to_score[f"{s}:cf"] = pairs[["cf_text"]].rename(columns={"cf_text": "text"})
    results, predictions, cf_results = {}, {}, {}

    def record(stage, scores):
        results[stage] = {s: grouped_metrics(eval_frames[s], *scores[s]) for s in eval_frames}
        predictions[stage] = {s: scores[s] for s in eval_frames}
        cf_results[stage] = {s: counterfactual_gaps(cf_eval[s], scores[f"{s}:cf_orig"][0], scores[f"{s}:cf"][0])
                             for s in cf_eval}

    if args.skip_baseline and args.baseline_from:
        earlier_path = Path(args.baseline_from) / "training_summary.json"
        earlier = json.load(open(earlier_path)) if earlier_path.exists() else {}
        same_rows = all(earlier.get("rows", {}).get(s) == len(eval_frames[s]) for s in eval_frames)
        same_setup = earlier.get("base_model") == args.base_model and earlier.get("hyperparameters", {}).get("seed") == args.seed
        if "M0" in earlier.get("metrics", {}) and same_rows and same_setup:
            results["M0"] = earlier["metrics"]["M0"]
            cf_results["M0"] = earlier.get("counterfactual_metrics", {}).get("M0", {})
            print(f"M0 metrics copied from {args.baseline_from} (same base model, seed and evaluation rows)")
        else:
            print(f"NOTE: M0 metrics not copied from {args.baseline_from}: it has no finished run with the same base "
                  "model, seed and evaluation rows, so M0 is scored here")
    if not args.skip_baseline or (args.baseline_from and "M0" not in results):
        print(f"\nScoring M0 ({args.base_model}) with the framework's LLM adapter ...")
        record("M0", framework_scores(args.base_model, prompt, task, to_score, args, dtype_name, token))
        print("M0 val: " + json.dumps({k: round(v, 4) for k, v in results["M0"]["val"]["overall"].items()}))

    if args.baseline_only:
        write_baseline_reports(args, out_dir, eval_frames, predictions, results, cf_results, frames, composition,
                               prompt, dtype_name)
        return {"baseline_only": True}

    model_dir = Path(args.stage_dir) / args.variant.lower() if args.stage_dir else out_dir
    info = train(args, prompt, task, frames, token, dtype_name, out_dir, checkpoint_dir, gap_pairs=cf_eval.get("val"),
                 model_dir=model_dir)

    print(f"\nScoring {args.variant} with the framework's LLM adapter (batches of {args.eval_batch_size}) ...")
    record(args.variant, framework_scores(model_dir, prompt, task, to_score, args, dtype_name, None))
    collapsed = collapse_warnings(results[args.variant]["val"]["overall"],
                                  results.get("M0", {}).get("val", {}).get("overall"),
                                  info.get("val_loss_before"), info.get("best_val_loss"))

    for s, frame in eval_frames.items():
        keep = [c for c in ("id", "source", "axis", "label") if c in frame.columns]
        table = frame[keep].copy()
        for stage, by_split in predictions.items():
            table[f"p_toxic_{stage.lower()}"], table[f"option_mass_{stage.lower()}"] = by_split[s]
        table.to_csv(out_dir / f"predictions_{s}.csv", index=False)

    with open(out_dir / "m2_prompt.yaml", "w") as handle:
        yaml.safe_dump({"prompts": {prompt.name: prompt_file_entry(prompt)}}, handle, sort_keys=False)

    table = metrics_table(results)
    table.to_csv(out_dir / "metrics.csv", index=False)
    cf_table = pd.DataFrame([{"model": stage, "split": split, "axis": axis, **m}
                             for stage, by_split in cf_results.items() for split, by_axis in by_split.items()
                             for axis, m in by_axis.items()])
    cf_table.to_csv(out_dir / "counterfactual.csv", index=False)
    summary = {
        "variant": args.variant, "counterfactual": args.counterfactual,
        "counterfactual_weights": {"cf_weight": args.cf_weight, "pair_weight": args.pair_weight} if args.counterfactual else None,
        "counterfactual_metrics": cf_results,
        "collapse_warnings": collapsed,
        "base_model": args.base_model, "method": args.method, "output_dir": str(out_dir), "framework_type": "hf-causal-lm",
        "prompt": prompt.to_public_dict(), "dtype": dtype_name, "max_length": args.max_length,
        "rows": {k: len(v) for k, v in frames.items()}, "axis_composition": composition, **info,
        "hyperparameters": {k: v for k, v in vars(args).items() if k != "hub_token_env"},
        "metrics": results,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }
    with open(out_dir / "training_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    (out_dir / "README.md").write_text(model_card(summary, prompt, task, args.card_license), encoding="utf-8")
    curves = save_training_curves(info.get("log_history") or [], out_dir / "training_curves.png",
                                  f"{args.variant}: {args.base_model} ({args.method} fine-tuning"
                                  f"{' + counterfactual loss' if args.counterfactual else ''})")
    if curves:
        print(f"Training curves: {curves}")
    if model_dir != out_dir:  # one copy of the model to the output folder (Drive), after scoring from local disk
        start = time.time()
        print(f"Copying {args.variant} from {model_dir} to {out_dir} ...", flush=True)
        shutil.copytree(model_dir, out_dir, dirs_exist_ok=True)
        shutil.rmtree(model_dir, ignore_errors=True)
        print(f"  copied in {(time.time() - start) / 60:.1f} min")
    if not args.keep_checkpoints:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    shown = ["model", "split", "group", "n", "accuracy", "macro_f1", "auroc", "fpr", "fnr", "option_mass"]
    with pd.option_context("display.width", 180, "display.max_rows", 200):
        print("\n" + table[[c for c in shown if c in table.columns]].round(4).to_string(index=False))
        if len(cf_table):
            print("\nCounterfactual gap (raw, without the paraphrase baseline the bias framework adds):")
            print(cf_table.round(4).to_string(index=False))
    if collapsed:
        print("\n" + "!" * 72 + f"\nWARNING: {args.variant} looks COLLAPSED (it ignores its input):")
        for reason in collapsed:
            print(f"  - {reason}")
        print("Its error rates and counterfactual gaps are not results (a constant answer has no gap).\n"
              "Check the training log: a loss spike in the first steps, too short a run, or noisy labels.\n" + "!" * 72)
    print(f"\nSaved {args.variant} to {out_dir} ({info['train_minutes']} min of training). In the bias framework:\n"
          f"  ModelSpec({str(out_dir)!r}, type='hf-causal-lm', id={args.variant!r}, prompt=load_prompts("
          f"{str(out_dir / 'm2_prompt.yaml')!r})[{prompt.name!r}], settings={{'dtype': {dtype_name!r}}})")
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
