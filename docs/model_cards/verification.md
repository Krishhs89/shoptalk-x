# Model Card: Visual Verification Head

## Intended use
Given a photo of a received item and an order's `item_id`, decides
**match / mismatch / suspect** (design doc §3.3b) — detects wrong-item
shipments or potential counterfeits. `suspect` verdicts are explicitly
**not** an automatic accusation; they're routed to human review (design doc
§8 "human-review path for verification 'suspect' verdicts").

## Architecture
Siamese comparison: both images (received-item photo, catalog photo) are
CLIP-embedded (`open_clip` ViT-B-32), then combined into a pairwise feature
vector `[emb_a, emb_b, |emb_a - emb_b|, emb_a * emb_b]` (concat of both
embeddings, their absolute difference, and elementwise product — standard
featurization for pairwise metric-learning classifiers, giving the head both
similarity and directional cues). A small MLP
(`shoptalk.verification.model.VerificationMLP`: 2048→128→64→1, ReLU,
dropout 0.2) outputs a match probability via sigmoid.

## Training data
Positive pairs: ABO's multi-view images (`main_image_id` +
`other_image_id`) of the *same* product — two genuinely different photos of
one item. Negative pairs: **hard negatives** — the nearest CLIP neighbor
from a *different* product in the *same category* (visually similar,
genuinely confusable), mined from the Day-3 image index; falls back to a
random same-category product if the ANN lookup is empty.
(`shoptalk.verification.build_pairs`)

## Eval results (500-product local deployment: 378 pairs)
| Metric | Value | (earlier 100-product smoke test) |
|---|---|---|
| Validation ROC-AUC | 0.964 | 0.926 |
| Threshold (Youden's J) | 0.304 | 0.508 |
| FAR (false accept rate) | 0.067 | 0.048 |
| FRR (false reject rate) | 0.080 | 0.200 |

Training converged cleanly (val AUC 0.899 → 0.964 over 15 epochs, loss
monotonically decreasing to 0.015) — see `results/day5_verification_eval.md`
for the run against your data. Scaling the training set from 100 to 500
products meaningfully improved calibration (FRR nearly halved) — the trend
the smoke-test card originally predicted. Live-tested end-to-end via the
deployed API: a genuine mismatch photo scored confidence 0.009 (threshold
0.304), a confident correct rejection. Expect continued improvement toward
the full ~10k catalog's much larger positive/hard-negative pool (more
multi-view products, richer category coverage for hard-negative
mining).

## Limitations / failure modes
- **`suspect` band is a fixed ±0.15 margin** around the trained threshold
  (`SUSPECT_MARGIN` in `verify.py`), not learned or calibrated per-category
  — a category with systematically noisier CLIP embeddings will route more
  cases to `suspect` than one with cleaner separation. Revisit with
  per-category calibration if the human-review queue becomes imbalanced.
- **Single catalog image compared**: `verify_photo` compares against the
  ordered item's `main_image_id` only, not an ensemble across all its
  catalog views — a genuine match photographed at an unusual angle is more
  likely to land in the `suspect` band. Averaging embeddings across all
  available catalog views is a natural improvement.
- **No adversarial robustness testing**: not evaluated against deliberately
  deceptive photos (e.g. a counterfeit photographed to closely mimic the
  catalog image). The FAR/FRR numbers above reflect *accidental*
  wrong-item confusion (same-category hard negatives), not adversarial
  fraud attempts.
- **Small-scale training caveat**: 183 pairs is a smoke test, not a
  production-scale training run — re-run `build_pairs.py` +
  `train_verification.py` against the full catalog before treating this
  model's calibration as final.
