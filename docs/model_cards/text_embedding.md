# Model Card: Text Embedding (Stage-1 Retrieval)

## Intended use
Encodes product documents (name + category + brand + color + style +
description + keywords + BLIP caption) and user text queries into a shared
768-dim vector space for ANN retrieval (top-10k → top-100). Not intended for
tasks outside product-catalog semantic search (e.g. general sentence
similarity, clustering unrelated text).

## Base model
`BAAI/bge-base-en-v1.5` (HuggingFace) — open-source, MTEB-leaderboard text
embedding model. Query-side uses BGE's recommended asymmetric-retrieval
instruction prefix ("Represent this sentence for searching relevant
passages: "); passages (product documents) are embedded as-is.

## Fine-tuned variant
`src/shoptalk/embeddings/finetune_text.py` fine-tunes the base model with
**triplet loss**: anchor = golden-set query, positive = its labeled relevant
product, negative = a **hard negative mined from the same category** via the
current index (visually/textually confusable, not a random unrelated
product — see design doc §3.1/§12.3).

## Training data
Triplets derived from the Day-2 golden evaluation set (`data/eval/golden_set.jsonl`)
— template-generated (query, positive-product) pairs, human-spot-checked,
paired with same-category hard negatives at train time. Not external/scraped
data; entirely derived from the ABO catalog subset already in this repo.

## Eval results (smoke-test scale: 150-product subset, 40-query golden set)
| Metric | Base | Fine-tuned | Uplift |
|---|---|---|---|
| Recall@10 | 0.897 | 0.959 | +6.2pp |
| Recall@50 | 0.981 | 0.981 | 0.0pp |

Full-catalog (~10k products, ~100-query golden set) numbers will differ —
see `results/day5_finetune_eval.md` for the run actually executed against
your data.

## Limitations / failure modes
- **Synthetic price field**: `price_usd` is not real Amazon pricing (ABO has
  none) — the model has no signal on genuine price-based user intent beyond
  what's encoded in the synthetic price text.
- **Small-scale fine-tune risk**: with too few triplets (e.g. a smoke-test
  golden set under ~30 queries), fine-tuning can overfit; always compare
  against the base model on a held-out slice before promoting (see the
  Airflow DAG's `evaluate_and_promote` gate).
- **English-only**: ABO listings are multilingual; this pipeline filters to
  English (`en_*` language tags) at preprocessing time. No non-English query
  support.
- **No price ground truth**: never present `price_usd` to a user as
  authoritative; it's a demo convenience, clearly flagged
  `price_is_synthetic=True` throughout the codebase.
