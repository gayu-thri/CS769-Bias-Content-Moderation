# Processed data

Output of [`Content_Moderation.ipynb`](../../Content_Moderation.ipynb) (seed 0, Civil Comments + ETHOS).
Rebuilding it with that notebook gives the same files.

| File | Rows | Use |
|-|-|-|
| `train.csv` | 45,605 | fine-tuning M2 / M4 |
| `val.csv` | 6,515 | checkpoint selection |
| `test.csv` | 13,030 | held out: never trained on; source of the bias-test posts |
| `verification_300.csv` | 300 | test posts for LLM tagging and human/Gemini review (150 gender + 150 disability) |
| `all_posts.csv` | 65,150 | every cleaned post with its split and source scores (ETHOS targets, Civil Comments `cc_toxicity`, `cc_identity_attack`, `cc_insult`, `cc_threat`). The training notebook uses it to skip counterfactual pairs on posts that attack a group |
| `biascompare_gender_test.jsonl` | 4,623 | gender test posts in the old proof-of-concept format |

## Sources and labels

| Source | Rows kept | Label |
|-|-|-|
| Civil Comments (`google/civil_comments`, Borkan et al. 2019; CC0) | 65,791 sampled from 1.8M: 60k half toxic / half clean, plus every toxic disability-mentioning comment and as many clean ones | toxic if `toxicity ≥ 0.5` (at least half the crowd raters said toxic) |
| ETHOS (Mollas et al. 2020; GPL-3.0) | 998 | hate if `isHate ≥ 0.5` (5 annotators) |

After cleaning and cross-source deduplication: 65,150 posts, 49.6% toxic. Each split has the same mix
(stratified by source × label × category).

| Category (lexicon tags) | Share of posts | Toxic |
|-|-|-|
| gender (incl. possessives like "women's") | 37.0% | 56% |
| disability | 10.4% | 48% |

Posts are short: median 38 words, about 127 tokens with the training prompt.

## Columns (`train/val/test.csv`)

| Column | Meaning |
|-|-|
| `id` | `cc-NNNNNNN` (row in Civil Comments' train split) or `ethos-NNNN` |
| `source`, `platform` | `civil_comments` (news comments) or `ethos` (YouTube/Reddit) |
| `text` | cleaned post (URLs removed, `@user` for handles) |
| `label` | 1 = toxic, 0 = not toxic |
| `toxicity_tier` | `toxic` / `clean` |
| `axis` | lexicon tags: `gender`, `disability`, `both` or `none` |
| `gender_group`, `gender_terms`, `disability_terms` | the matched terms |

## History: why not ToxicCommons (25 Oct 2026 review)

The first build used ToxicCommons (OCR'd public-domain books and newspapers, scored by Llama-3.1-8B).
A review of its training split found:

- **Wrong labels.** 15,132 of its 22,128 toxic labels were on posts it rates "clean", made toxic by a
  "one category ≥ 2" rule. Random samples were ads, book reviews, church history and war news; even its
  "mild"/"toxic" tiers were mostly history describing violence.
- **Non-English text.** 12.6% of its posts (German, Dutch, Polish, Latin OCR), 65% of them labelled toxic.

It's still in the notebook, off by default (`USE_TOXICCOMMONS = False`). If switched on, the "≥ 2" rule
is off (`TC_AXIS_TOXIC_MIN = None`) and posts need ≥ 15% common English words.

## Known limitations

- **Figurative disability tags.** Some disability tags are figurative or insults ("too blind to
  notice", "a mentally ill despot", "lunatic"). That's fine for training. In the bias test, the tagger
  and reviewers mark such posts unusable, because the word isn't about a disabled person.
- **Labels are used only for training and accuracy reports.** The bias metric Δ compares each model
  with itself and uses no labels.
