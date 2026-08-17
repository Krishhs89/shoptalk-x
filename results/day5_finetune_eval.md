# Day 5 — Embedding fine-tune evaluation

Golden set: 100 queries

| metric       |   base_model |   finetuned_model |    uplift |
|:-------------|-------------:|------------------:|----------:|
| recall@10    |      0.71525 |          0.832167 | 0.116917  |
| recall@50    |      0.91    |          0.920167 | 0.0101667 |
| precision@10 |      0.171   |          0.212    | 0.041     |
| precision@50 |      0.0534  |          0.0552   | 0.0018    |

> Precision@k is much lower than Recall@k here by construction, not because retrieval is weaker than Recall@k suggests -- most golden-set queries have only 1-2 labeled positives, which caps Precision@10 at 0.1-0.2 even for perfect retrieval; a handful of queries with up to 8 positives pull the average up from there.
