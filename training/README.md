# Training (M0 → M2, M4)

Run order on Colab:

| Step | Notebook | Output |
|-|-|-|
| 1. Clean and split | `Content_Moderation.ipynb` | `train.csv`, `val.csv`, `test.csv`, `verification_300.csv` |
| 2. Train M2 and M4 | `training/Train_M2_M4.ipynb` (80 GB GPU for full fine-tuning) | `m2/`, `m4/` next to the notebook (model, model card, prompt, metrics, counterfactual gaps) and `comparison/` (M0 vs M2 vs M4 tables and figure); optional upload to the Hugging Face Hub |
| 3. Tag the 300 posts | `tagging/Tag_Posts.ipynb` (GPU) | `tagged_posts.jsonl`, `tagging_review.csv` |
| 4. Review, then `finalize` | Google Sheets, then the last cell of `Tag_Posts.ipynb` | `verified_posts.jsonl`, agreement report |
| 5. Bias test M0–M4 | `bias-compare-poc/notebooks/bias_study.ipynb` (models as `hf-causal-lm`) | `study.json`, plots |

Steps 2 and 3 are independent and can run at the same time.

For step 2, put `Train_M2_M4.ipynb`, `train_lm.py`, the three CSVs and (optionally) `all_posts.csv` in one
Drive folder. The notebook finds that folder and writes `m2/`, `m4/` and `comparison/` into it.

## Design decisions (and why)

**Three models: base, fine-tuned, fine-tuned with a counterfactual loss.**

| Model | Training | Answers |
|-|-|-|
| M0 | none | the baseline |
| M2 | ordinary fine-tuning | the proposal's question: what does ordinary fine-tuning do to bias? |
| M4 | the same fine-tuning plus a counterfactual loss | does training on counterfactuals reduce bias beyond that, and at what accuracy cost? |

M2 and M4 share posts, prompt, hyperparameters and checkpoint selection (ordinary validation loss), so their
difference is the loss alone. M2 stays a plain control: if it were trained to be swap-invariant, M2 vs M0
would no longer measure ordinary fine-tuning. M1 and M3 (few-shot) need no training.

**M4's counterfactual loss.**

- **Pairs:** each training post gets a counterfactual where only the group changes. Gender uses the
  framework's lexicon swap. Disability adds or removes a disability adjective before a person noun
  ("my cousin" ↔ "my deaf cousin").
- **Posts without a pair:** posts with slurs or disability words the rules can't swap cleanly, and posts
  whose hate is aimed at the swapped group (ETHOS targets, Civil Comments `identity_attack ≥ 0.5`, from
  `all_posts.csv`). Those are the asymmetric cases where a different score is legitimate (Garg et al.,
  2019).
- **Loss:** `ce + 1.0 × ce(counterfactual, same label) + 0.5 × (log-odds(Yes) original − log-odds(Yes)
  counterfactual)²`. That's counterfactual augmentation plus counterfactual logit pairing.
- **What it doesn't optimise:** the pairing term works on log-odds, not on the JSD that Δ measures, and it
  uses lexicon/rule swaps, not the human-verified multi-word test swaps. So the final Δ is not optimised
  directly, and it also tests whether the effect generalises.
- **During training:** the validation counterfactual gap of both models is logged every 100 steps, as a
  curve. It's never used to select checkpoints.
- **Cost:** about 1.7× M2's compute.

**M0 = `Qwen/Qwen3-4B`, a small instruction-tuned LM, not BERT.** In feedback on our proposal, Lucy pointed
out that BERT is dated and suggested small, laptop-sized LMs (Gemma, Phi, small Qwen). Why Qwen3-4B:

- It's text-only, has a standard architecture, uses the Apache 2.0 licence, and isn't gated.
- 4B is the size the Qwen team ships as a moderator: Qwen3Guard-Gen-4B (Qwen Team, 2025) is fine-tuned
  from Qwen3-4B. Llama Guard 3 (1B/8B) and ShieldGemma (2B/9B) put industry moderators in the same range.
- It's the largest Qwen3 that can be *fully* fine-tuned on Colab Pro's 80 GB GPU (~53 GB).
- It's large enough for few-shot prompting (M1, M3) to work. 0.6B scored an obvious insult at only
  P(toxic) = 0.30 zero-shot.
- It isn't the Qwen3-8B used as the tagger and in the 8B LLM comparison.

The newer Qwen3.5 and Gemma 4 small models are multimodal: they load through different model classes than
the framework's text-LLM adapter. `Qwen/Qwen3-1.7B` is the faster fallback.

**The training objective is the framework's score.** The bias framework's `hf-causal-lm` adapter puts
each post in the chat template ("Is the following social media post toxic? ... Answer with exactly one
of: No, Yes.") and reads the next-token probability of each answer's first token. `train_lm.py` renders
every training post with the framework's own `_render` function and minimises cross-entropy on the
correct answer token at exactly that position. M0 and M2 are then scored with the framework's own
`HFCausalLMScorer`, so the training report and the bias test use identical code. The prompt is saved as
`m2/m2_prompt.yaml` for the bias study.

**Full fine-tuning when the GPU allows it, otherwise LoRA.** The proposal says *fully* fine-tune, and
`METHOD = "auto"` does that when it fits:

- **Full:** float32 master weights with bf16/fp16 compute, 8-bit AdamW, lr 1e-5. About 53 GB for
  Qwen3-4B (80 GB GPU) and 23 GB for Qwen3-1.7B (40 GB A100).
- **LoRA:** rank 16 on every attention and MLP projection, lr 2e-4, about 12 GB for 4B. The adapter is
  merged back into the weights, so M2 loads like any model.

The script estimates memory from the model config before loading anything and stops with what to
change if the run cannot fit. The method used is recorded in `training_summary.json`.

**Fast checkpointing and scoring.** These options cut each run by roughly 15–30 minutes on Colab.
- **Best weights in RAM** (`--checkpointing memory`, the default). When validation loss improves, the
  trainable weights are copied to CPU RAM: about 8 GB in seconds, with no 16 GB checkpoint file written.
  The run stops early after `--patience` evaluations without improvement, and the best weights are
  restored at the end. If RAM is short it falls back to Trainer checkpoints on disk
  (`--checkpointing disk`). Tested: both modes reach the same best validation loss, and restored weights
  equal the saved ones.
- **Local staging** (`--stage-dir`). The finished model is saved to Colab's local disk, scored from there,
  then copied to Drive once, instead of being read back from Drive.
- **GPU-sized scoring batches** (`--eval-batch-size auto`): 32 posts on GPUs with 40 GB or more.

**Loss curves, live and final.**
- **Live:** `--tensorboard-dir` logs to TensorBoard (one subfolder per run, e.g. `M2_2026-10-26_1530`). In
  the notebook, run section 3b before training: a TensorBoard panel opens and refreshes every 15 s.
  - `train/`: `loss`, `learning_rate`, `grad_norm`, and for M4 `ce`, `ce_cf` and `pair`.
  - `eval/`: `loss` from step 0 (the base model), plus `cf_gap_*` and `cf_direction_*` per category.

  The logs stay in `tensorboard/` on Drive.
- **Final:** each run saves `training_curves.png` in its model folder (loss, M4's loss parts, validation
  counterfactual gap), and the notebook saves `comparison/training_curves_M2_M4.png`. For M4, the plain
  cross-entropy part is plotted as its train loss, so it's comparable to M2 and to validation; its total
  objective, which adds the counterfactual terms, is dotted.

**Training data size.** 20,000 of the ~44,200 training posts for one epoch (all disability posts kept).
Posts average about 125 tokens with the prompt, so a run takes well under an hour.
`MAX_TRAIN_SAMPLES = 0` uses everything.

**Both bias categories are in the training data.** The framework tests gender and disability. About 36%
of training posts mention gender (56% toxic) and 8% (3,555) mention disability (48% toxic). The script
prints this mix and stops if train lacks toxic or clean posts for either category. Disability posts are
repeated (half toxic, half clean) to 10% of the training rows. Val and test keep their natural mix.

**Uploadable to Hugging Face.** `m2/` is a complete model repo: weights, tokenizer and a model card
(`README.md`). The card covers what M2 is, a usage example, the training settings, M0 vs M2 metrics,
data licences and limitations. The card's example reproduces the framework's score exactly (checked:
0.344284 vs 0.344285). The notebook's upload cell creates the repo **private** with a write token from
the `HF_TOKEN` Colab secret, and refuses to upload the quick-test model. The card's licence field is the
base model's Apache-2.0, but ETHOS is GPL-3.0 (Civil Comments is CC0), so the team
should settle the licence before making the repo public.

**No leakage.** The data notebook deduplicates before splitting, and the script checks again that no
text appears in two splits. `test.csv` is never trained on or used for selection.

**Group error rates are not the bias metric.** `metrics.csv` reports accuracy, macro-F1, AUROC, FPR,
FNR and option mass for M0 and M2, overall and per category. That's a sanity check. The study's
metric is still the counterfactual Δ from the framework. The results cell prints a one-sentence takeaway
under the figure, for the caption (Lucy's tip for the final report).

**Tagging: the LLM proposes, rules and people check.** Tags enter Δ directly. Every LLM tag passes
deterministic checks:

- each phrase occurs in the post;
- no word of the original group is left after the swap;
- there's one referent;
- synonyms avoid swap, identity, toxic and function words;
- there are enough synonyms for the swap size.

Then people and Gemini review the tags. On the 100 doubly reviewed posts two teammates review
independently (`human_verdict`, `human2_verdict`), and `finalize` reports **inter-human** agreement next
to **human–Gemini** agreement (Cohen's κ). That follows Lucy's point that inter-human agreement is the
ceiling for human-model agreement.

## Known limitations

- Training data is Civil Comments + ETHOS (see the [data README](../data/processed/README.md)). The
  first quick run below used the earlier ToxicCommons-based data.
- The disability counterfactual rules are simple (adjective added or removed before a person noun), so
  M4's disability pairs are mostly insertions into posts without disability words. Gender pairs come
  from the same lexicon the framework falls back on.
- M4 sees each paired post twice (original and counterfactual), so it gets more forward passes than M2.
  The comparison is "same data, extra counterfactual signal", not "same compute".
- `train_lm.py` and the notebook were tested end to end on a tiny random Qwen3 with the real Qwen3
  tokenizer, on real posts: full fine-tuning, LoRA, M4's counterfactual loss (and its gradients), the
  quick test, scoring through the framework, and the Hub upload (mocked).

## Quick run on Colab (25 Oct 2026, A100 80 GB)

Full log: [`results/quicktest_2026-10-25.txt`](results/quicktest_2026-10-25.txt). The run used Qwen3-4B,
full fine-tuning, 400 training posts and 13 optimizer steps per model.

**Works:**
- the memory check (~49 GB needed, full fine-tuning chosen);
- 8-bit AdamW with bf16;
- M0 answers in the requested format (option mass 1.0);
- scoring through the framework, counterfactual pairs, and the M0/M2/M4 comparison;
- M4 reuses M0's numbers.

The `lm_head.weight` "missing key" warning is harmless: Qwen3-4B ties its input and output embeddings.

**Not usable as results: both models collapsed.**

| test | M0 | M2 | M4 |
|-|-|-|-|
| AUROC | 0.71 | 0.53 | 0.46 |
| FPR / FNR | 0.68 / 0.08 | 0.66 / 0.28 | 0.00 / 1.00 (always "No") |

- The training loss averaged 5.7 (M2) and 4.8 (M4) over the first 5 steps, against a starting level
  around 1.
- 3% warmup of 13 steps rounds to zero, so full fine-tuning started at the full learning rate.
- M4's pairing term then fell to ~0 only because the model gave every post the same answer, so a
  collapsed model's low counterfactual gap means nothing.
- The full run (~650 steps, ~19 warmup steps) is less exposed.

**Fixed in `train_lm.py` after this run:**
- **Minimum warmup:** at least 10 steps, capped at a third of the run (`--min-warmup-steps`). The quick
  test now warms up for 4 of its 13 steps.
- **Validation before training:** the log starts with
  `step 0/N | val loss … (before training: the base model)` and the base model's counterfactual gap.
  This is the level training must beat.
- **Collapse check:** a model is flagged as collapsed if:
  - more than 95% of its validation answers are one class;
  - its validation AUROC is below 0.6, or more than 0.1 below M0's;
  - or its best validation loss never beats the step-0 level.

  The flag is printed in a warning box, stored as `collapse_warnings` in `training_summary.json`, and
  put at the top of the model card. The notebook leaves collapsed models out of the takeaway sentence.
- **Disability pair rules:** they block phrases ("special needs", "mentally ill", "hard of hearing",
  "Down's syndrome", …), so common words like "needs", "ill" and "hearing" no longer stop a post from
  getting a pair.

**Also fixed since:** the training data (now Civil Comments + ETHOS, see the
[data README](../data/processed/README.md)). Re-run the quick test on it before the full run.
