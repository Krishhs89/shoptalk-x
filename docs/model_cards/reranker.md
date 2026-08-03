# Model Card: Cross-Encoder Reranker (Stage-2 Retrieval)

## Intended use
Re-scores the stage-1 ANN's top-100 (or top-50 for image queries) candidates
by jointly attending to (query, product-document) pairs, producing a more
accurate top-K ranking than the bi-encoder alone. Never run over the full
catalog — too slow; only ever sees pre-filtered candidates.

## Model
`cross-encoder/ms-marco-MiniLM-L-6-v2` (HuggingFace, off-the-shelf, not
fine-tuned in this project). Trained on MS MARCO passage ranking; transfers
reasonably to product search since both are short-query, short-document
relevance-ranking tasks.

## Training data
None (used as pre-trained) — an explicit scope decision to spend the
project's fine-tuning effort on the embedding model and verification head
instead, where the ABO-specific signal (hard negatives, multi-view images)
adds the most value. Fine-tuning this reranker on the golden set is a
natural next step if reranking quality becomes the bottleneck.

## Eval results (smoke-test scale: 300-product subset, 30-query golden set)
| Metric | Stage-1 only | +Rerank | Uplift |
|---|---|---|---|
| Recall@10 | 0.864 | 0.954 | +10.5pp |
| MRR | 0.846 | 0.942 | +11.3pp |
| NDCG@10 | 0.810 | 0.929 | +14.6pp |
| Recall@100 | 0.992 | 0.992 | 0.0pp (expected — rerank reorders, doesn't expand, the candidate pool) |

See `results/day2_retrieval_eval.md` for the run against your actual data.

## Limitations / failure modes
- **Photo-query pseudo-text**: for image search, there's no natural text
  query to pair against candidates, so the query photo is BLIP-captioned to
  manufacture one (`shoptalk.retrieval.image_search`). Reranking quality for
  photo search is therefore bounded by caption quality — a vague or
  inaccurate caption (cluttered/ambiguous photo) degrades reranking.
- **Latency**: batched cross-encoder scoring over 100 candidates is the
  single most expensive retrieval-side stage (design doc §4.2 target
  <300ms P95); reducing `retrieval.stage1_k` in `configs/config.yaml`
  trades recall for latency if needed.
- **Not fine-tuned on ABO**: scores reflect MS MARCO's notion of relevance,
  not ABO-specific product semantics; if the golden-set eval shows reranking
  hurting a specific category, that's the signal to fine-tune it.
