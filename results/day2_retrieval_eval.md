# Day 2 — Two-stage retrieval evaluation

Golden set: 100 queries (/Users/krishnakumar/Documents/Krishna/Interview Kickstart ML SwitchUp/Project/Capstone Project 4,Project 5 and Project 6 Live Class with Lakshaya - Lakshaya - August 02, 2026/Shoptalk/data/eval/golden_set.jsonl)
Stage-1 embedding model: `data/models/bge-finetuned`

| metric       |   stage1_baseline |   two_stage_reranked |      uplift |   uplift_pct |
|:-------------|------------------:|---------------------:|------------:|-------------:|
| precision@10 |          0.21     |             0.221    |  0.011      |          5.2 |
| recall@10    |          0.828833 |             0.846917 |  0.0180833  |          2.2 |
| recall@50    |          0.916833 |             0.934667 |  0.0178333  |          1.9 |
| recall@100   |          0.943417 |             0.943417 |  0          |          0   |
| mrr          |          0.81138  |             0.808169 | -0.00321107 |         -0.4 |
| ndcg@10      |          0.772715 |             0.77655  |  0.0038341  |          0.5 |

> **Reading the uplift_pct column:** the reranker's uplift over stage-1 shrinks whenever stage-1 itself gets better -- e.g. after the fine-tuned embedding model is wired into serving (see Day 5), stage-1 alone already recovers most of the relevant results, so the reranker has less room left to add on top. A smaller uplift_pct here is a sign the *upstream* model improved, not that reranking stopped working -- compare against `git log` on this file to see the pre-fine-tune baseline uplift for context.
> **Precision@10 vs Recall@10:** Precision@10 is naturally much lower -- most golden-set queries have only 1-2 labeled positives, capping Precision@10 at 0.1-0.2 even for perfect retrieval; a few queries with up to 8 positives pull the average above that floor.
