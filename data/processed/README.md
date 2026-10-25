# Processed data

Output of [`Content_Moderation.ipynb`](../../Content_Moderation.ipynb) (seed 0, Civil Comments + ETHOS).
Rebuilding it with that notebook gives the same files.

| File | Rows | Use |
|-|-|-|
| `train.csv` | 44,234 | fine-tuning M2 / M4 |
| `val.csv` | 6,321 | checkpoint selection |
| `test.csv` | 12,640 | held out: never trained on; source of the bias-test posts |
| `verification_300.csv` | 300 | test posts for LLM tagging and human/Gemini review (150 gender: 77 male / 73 female, no mixed-gender posts; 150 disability) |
| `all_posts.csv` | 63,195 | every cleaned post with its split and source scores (ETHOS targets, Civil Comments `cc_toxicity`, `cc_identity_attack`, `cc_insult`, `cc_threat`). The training notebook uses it to skip counterfactual pairs on posts that attack a group |
| `biascompare_gender_test.jsonl` | 4,313 | gender test posts in the old proof-of-concept format |

## Sources and labels

| Source | Rows kept | Label |
|-|-|-|
| Civil Comments (`google/civil_comments`, Borkan et al. 2019; CC0) | 64,118 sampled from 1.8M: 60k half toxic / half clean, plus every toxic disability-mentioning comment and as many clean ones | toxic if `toxicity > 0.5` (a strict majority of crowd raters); clean if `< 0.3`; the 159,651 comments in between (rater ties and near-ties) are dropped |
| ETHOS (Mollas et al. 2020; GPL-3.0) | 998 | hate if `isHate ≥ 0.5` (5 annotators) |

**Cleaning**, in order:

1. Unescape HTML entities and remove HTML tags.
2. NFKC-normalize Unicode.
3. Remove URLs; replace handles with `@user`.
4. Drop posts that are under 4 words (1,610), over 2,000 characters, mostly non-letters (37) or
   comment spam (18; earnings / work-from-home pitches, "fancy" Unicode lettering).
5. Remove exact duplicates (218) and near-duplicates (37; same first 100 normalized characters), across
   all sources before splitting, keeping the toxic copy.

Result: 63,195 posts, 49.5% toxic. Each split has the same mix (stratified by source × label ×
category), and no post or near-duplicate appears in two splits.

| Category (lexicon tags) | Share of posts | Toxic |
|-|-|-|
| gender (incl. possessives like "women's") | 35.7% | 56% |
| disability | 8.0% | 48% |

Posts are short: median 35 words, about 125 tokens with the training prompt.

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
