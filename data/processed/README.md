# Processed data

Output of [`Content_Moderation.ipynb`](../../Content_Moderation.ipynb) (seed 0). Rebuilding it with that
notebook gives the same files.

| File | Rows | Use |
|-|-|-|
| `train.csv` | 45,581 | fine-tuning M2 / M4 |
| `val.csv` | 6,511 | checkpoint selection |
| `test.csv` | 13,025 | held out: never trained on; source of the bias-test posts |
| `verification_300.csv` | 300 | test posts for LLM tagging and human/Gemini review (150 gender + 150 disability) |
| `all_posts.csv.gz` | 65,117 | every cleaned post with its split, dataset target labels and ToxicCommons scores. Gzipped because the CSV (177 MB) exceeds GitHub's 100 MB limit. `pd.read_csv` reads it directly; `gunzip` it for the training notebook, which uses it to skip counterfactual pairs on posts whose hate targets a group |
| `biascompare_gender_test.jsonl` | 9,508 | gender test posts in the old proof-of-concept format |

## Columns (`train/val/test.csv`)

| Column | Meaning |
|-|-|
| `id` | `ethos-NNNN` or `tc-NNNNNNN` |
| `source`, `platform` | ETHOS (YouTube/Reddit) or ToxicCommons (public-domain text) |
| `text` | cleaned post (URLs removed, `@user` for handles) |
| `label` | 1 = toxic, 0 = not toxic |
| `toxicity_tier` | ToxicCommons' tier: clean / mild / toxic |
| `axis` | lexicon tags: `gender`, `disability`, `both` or `none` |
| `gender_group`, `gender_terms`, `disability_terms` | the matched terms |

## Known issues (found 25 Oct 2026; fix before the full training run)

- **Label noise in ToxicCommons.**
  - 15,132 of the 22,128 toxic ToxicCommons labels in train are on posts ToxicCommons rates "clean".
    They're toxic only because one category scored ≥ 2 (`TC_AXIS_TOXIC_MIN = 2` in the notebook).
  - Random samples are ads, book reviews, church history and war news. Even the "mild" and "toxic"
    tiers are mostly historical text describing violence.
  - Example: `tc-0560925`, a patent-medicine advert labelled toxic for "gender 2".
- **Non-English text.** 5,638 ToxicCommons train posts (12.6%) are German, Dutch, Polish or Latin
  OCR; 65% are labelled toxic. The `looks_english` filter is too loose.
- **Possessives.** 148 posts with "women's", "mother's" and similar aren't tagged as gender, because the
  lexicon matches whole tokens.

Labels are used only for training and for accuracy reports. The bias metric Δ compares each model with
itself and uses no labels.
