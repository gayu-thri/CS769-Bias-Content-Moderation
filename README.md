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
| [`Content_Moderation.ipynb`](Content_Moderation.ipynb) | Data pipeline: loads ETHOS + Civil Comments (ToxicCommons optional), cleans, deduplicates, labels, tags gender/disability mentions, splits 70/10/20, samples the 300 verification posts |
| [`data/processed/`](data/processed) | The pipeline's output: `train/val/test.csv`, `verification_300.csv`, `all_posts.csv` (see its [README](data/processed/README.md)) |
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

Steps 2 and 3 are independent.

## Data

| | |
|-|-|
| Sources | **Civil Comments** (Borkan et al. 2019; `google/civil_comments`, 1.8M news-site comments, toxicity = share of crowd raters, CC0) and **ETHOS** (Mollas et al. 2020; 998 YouTube/Reddit comments, human labels, GPL-3.0). ToxicCommons is optional and off (see below) |
| Labels | toxic if at least half the raters said toxic (Civil Comments `toxicity ≥ 0.5`); ETHOS `isHate ≥ 0.5` |
| Sampling | 60k Civil Comments (half toxic) plus every toxic disability-mentioning comment and as many clean ones |
| After cleaning | 65,150 posts: train 45,605 / val 6,515 / test 13,030 (stratified, deduplicated across sources), 49.6% toxic |
| Categories | gender in 37% of posts (56% toxic), disability in 10.4% (48% toxic) |
| Length | median 38 words; ~127 tokens with the prompt |
| Verification set | 300 test posts (150 gender + 150 disability) for verified multi-word swaps |

Details, columns and the data review: [`data/processed/README.md`](data/processed/README.md).

## Status (25 Oct 2026)

**Done**
- Data pipeline rebuilt on Civil Comments + ETHOS; 26 offline tests pass.
- Training (M2, M4) and tagging pipelines built and tested end to end on a tiny stand-in model, on
  the new data too.
- A quick run on Colab's A100 80 GB (on the old data) finished, with Qwen3-4B fully fine-tuned on 400
  posts. Setup, memory check, scoring, counterfactual gaps and the comparison all work. Log:
  [`training/results/quicktest_2026-10-25.txt`](training/results/quicktest_2026-10-25.txt).

**Fixed after the quick run and the data review**
1. **Noisy training labels.** The first build trained on ToxicCommons: OCR'd books and newspapers labelled
   by an LLM. 68% of its toxic labels in train were on posts it rated "clean" (ads, church history, war
   news), toxic only through a "one category ≥ 2" rule. Training now uses human-labelled Civil Comments
   + ETHOS. ToxicCommons can be switched back on (`USE_TOXICCOMMONS = True`) with that rule off.
2. **Non-English text.** 12.6% of ToxicCommons posts were German, Dutch, Polish or Latin OCR. Civil
   Comments and ETHOS are English. ToxicCommons, if used, now needs ≥ 15% common English words.
3. **Possessives.** "women's", "mother's" and similar are now tagged as gender (0 of 297 such posts
   missed, from 148 missed before).
4. **Collapsed quick-run models.** Zero warmup on a 13-step run. `train_lm.py` now enforces a minimum
   warmup, validates before training, and flags collapsed models
   ([`training/README.md`](training/README.md)).
5. **M4's disability pairs** block phrases ("special needs", "mentally ill"), not the common words
   "needs" and "ill".

**Next:** re-run the quick test on the new data (it should not collapse), then train M2 and M4 in full.
After that, tag and review the 300 test posts, run the bias test on M0–M4, and write the final report.

## References

- Yang, Levy, Goldberg & Wallace (2026). *Compared to What? Baselines and Metrics for Counterfactual
  Prompting.* arXiv:2605.01048.
- Garg et al. (2019). *Counterfactual Fairness in Text Classification through Robustness.* AIES.
- Dixon et al. (2018); Hutchinson et al. (2020); Venkit et al. (2023): identity-term and disability bias
  in toxicity models.
- Mollas et al. (2020), ETHOS. Arnett et al. (2024), Toxicity of the Commons.
- Qwen Team (2025), Qwen3 / Qwen3Guard.
