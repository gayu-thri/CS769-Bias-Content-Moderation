# Auditing Social Bias in Content Moderation

CS 769 (UW–Madison, Fall 2026) project by **Shreyas Natarajan, Gayathri Ethiraj and Dharini Venkatesan**.

**Question.** How do fine-tuning and prompt engineering, alone and combined, change bias in a content
moderation model? Bias is measured as counterfactual sensitivity beyond paraphrase-induced variation.

**Idea in one example.** A moderator should give these the same toxicity score:

| | Post |
|-|-|
| Original | My **deaf cousin** keeps texting me about the game. |
| Group swap | My **cousin** keeps texting me about the game. |
| Rewording (baseline) | My deaf cousin keeps **messaging** me about the game. |

A score change after the swap counts as bias only if it's larger than what a neutral rewording of the
same size does (Yang et al., 2026). Per post, that's

```
Δ = JSD(original, swap) − mean JSD(original, 5 rewordings)
```

A model is biased only if Δ is consistently positive (one-sided paired t-test). Models are compared
pairwise (Wilcoxon, Holm-corrected). The swaps change **every** word that marks the group ("she told her
sister" → "he told his brother"), and they cover **gender** and **disability**.

## Models

All five variants share one base model, a small instruction-tuned LM, **`Qwen/Qwen3-4B`**. It replaced
BERT after instructor feedback. Each variant changes one thing:

| Model | What it is | Trained here |
|-|-|-|
| **M0** | Qwen3-4B as is | no |
| **M1** | M0 + few-shot prompt | no |
| **M2** | M0 + ordinary fine-tuning | `training/` |
| **M3** | M2 + few-shot prompt | no |
| **M4** | M0 + fine-tuning **with a counterfactual loss** (augmentation + logit pairing, Garg et al. 2019) | `training/` |

M2 vs M0 is the effect of fine-tuning, M1/M3 the effect of prompting, and M4 vs M2 the effect of the
counterfactual loss.

## Repository layout

| Path | What it is |
|-|-|
| [`Content_Moderation.ipynb`](Content_Moderation.ipynb) | Data pipeline: loads ETHOS + ToxicCommons, cleans, deduplicates, labels, tags gender/disability mentions, splits 70/10/20, samples the 300 verification posts |
| [`data/processed/`](data/processed) | The pipeline's output: `train/val/test.csv`, `verification_300.csv`, `all_posts.csv.gz` (see its [README](data/processed/README.md)) |
| [`training/`](training) | `train_lm.py` + `Train_M2_M4.ipynb`: trains M2 and M4, compares M0/M2/M4, writes Hugging Face model cards (see its [README](training/README.md)) |
| [`tagging/`](tagging) | `tag_posts.py` + `Tag_Posts.ipynb`: an 8B LLM tags the swaps and control synonyms of the 300 verification posts; rule checks; review sheet for two humans + Gemini; agreement report |
| [`docs/`](docs) | Proposal and midterm slides |
| `src/`, `poc.ipynb`, `DESIGN.md`, `tests/` | **Legacy**: the original 10-post proof of concept of the bias framework. The current framework (M0–M4, gender + disability, LLM scoring) lives in [`shreyasnat2804/bias-compare-poc`](https://github.com/shreyasnat2804/bias-compare-poc), branch `claude/testing-framework-llm-bert-njfxy2` |

## How to run (Google Colab)

| Step | Run | Output |
|-|-|-|
| 1. Build the data | `Content_Moderation.ipynb` (CPU, High-RAM; `SAVE_TO_DRIVE = True`) | the splits, or use `data/processed/` from this repo |
| 2. Train M2 and M4 | `training/Train_M2_M4.ipynb` on the **80 GB GPU**, with `train_lm.py` and the CSVs in the same Drive folder | `m2/`, `m4/`, `comparison/` |
| 3. Tag 300 test posts | `tagging/Tag_Posts.ipynb` (GPU) | `tagged_posts.jsonl`, `tagging_review.csv` |
| 4. Review the tags | Google Sheets (two humans + Gemini), then `finalize` in `Tag_Posts.ipynb` | `verified_posts.jsonl`, agreement |
| 5. Bias test M0–M4 | `bias-compare-poc/notebooks/bias_study.ipynb` (models as `hf-causal-lm`) | Δ, direction, pairwise comparisons |

Steps 2 and 3 are independent. `all_posts.csv.gz` is gzipped to stay under GitHub's 100 MB file limit.
pandas reads it directly, or run `gunzip` to get `all_posts.csv` for the training notebook.

## Data

| | |
|-|-|
| Sources | ETHOS (Mollas et al. 2020; 998 YouTube/Reddit comments, human labels) and ToxicCommons (Arnett et al. 2024; via `tcapelle/ToxicCommons-balanced`, labels by Llama-3.1-8B) |
| After cleaning | 65,117 posts: train 45,581 / val 6,511 / test 13,025 (stratified, deduplicated across sources) |
| Categories | gender in 77% of posts, disability in 2.7%, each about 50% toxic |
| Verification set | 300 test posts (150 gender + 150 disability) for verified multi-word swaps |

## Status (25 Oct 2026)

**Done**
- Data pipeline run and its outputs committed.
- Training (M2, M4) and tagging pipelines built and tested end to end on a tiny stand-in model.
- A quick run on Colab's A100 80 GB finished, with Qwen3-4B fully fine-tuned on 400 posts: setup,
  memory check, scoring, counterfactual gaps and the comparison all work. Log:
  [`training/results/quicktest_2026-10-25.txt`](training/results/quicktest_2026-10-25.txt).

**Found in the quick run and the data review: fix before the full run**
1. **Training labels are noisy.**
   - 68% of ToxicCommons' toxic labels in train (15,132 of 22,128) are on posts ToxicCommons itself
     rates "clean". They're toxic only because of the pipeline's rule that one category scoring ≥ 2 makes
     a post toxic.
   - Samples are newspaper ads, church history and war reports. Even its "mild" and "toxic" tiers are
     mostly historical text that describes violence.
   - Options: drop the rule (`TC_AXIS_TOXIC_MIN = None`), or train on human-labelled social media
     (Civil Comments, as the proposal planned) plus ETHOS.
2. **12.6% of ToxicCommons training posts aren't English** (German, Dutch, Polish and Latin OCR), and
   65% of those are labelled toxic. Needs a stricter language filter.
3. **The quick-run models collapsed.** M2 fell to near-chance AUROC, and M4 answered "No" to everything.
   With only 13 steps, 3% warmup rounds to zero, so full fine-tuning hit the model at full learning rate.
   The full run has warmup, but `train_lm.py` should still enforce a minimum warmup, validate before
   training, and flag collapsed models.
4. **Minor:**
   - Possessive gender words ("women's") aren't tagged as gender (148 posts).
   - M4's disability-pair rules block common words like "needs" and "ill" instead of the phrases
     "special needs" and "mentally ill".

**Next:** fix 1–4, run M2 and M4 in full (about 25 Colab compute units), review the 300 tagged posts,
run the bias test on M0–M4, and write the final report.

## References

- Yang, Levy, Goldberg & Wallace (2026). *Compared to What? Baselines and Metrics for Counterfactual
  Prompting.* arXiv:2605.01048.
- Garg et al. (2019). *Counterfactual Fairness in Text Classification through Robustness.* AIES.
- Dixon et al. (2018); Hutchinson et al. (2020); Venkit et al. (2023): identity-term and disability bias
  in toxicity models.
- Mollas et al. (2020), ETHOS. Arnett et al. (2024), Toxicity of the Commons.
- Qwen Team (2025), Qwen3 / Qwen3Guard.
