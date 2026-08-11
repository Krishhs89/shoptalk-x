# Day 2 — Two-stage retrieval evaluation

Golden set: 100 queries (/home/ubuntu/shoptalk-x/data/eval/golden_set.jsonl)

| metric     |   stage1_baseline |   two_stage_reranked |    uplift |   uplift_pct |
|:-----------|------------------:|---------------------:|----------:|-------------:|
| recall@10  |          0.68825  |             0.789083 | 0.100833  |         14.7 |
| recall@50  |          0.8585   |             0.871833 | 0.0133333 |          1.6 |
| recall@100 |          0.878083 |             0.878083 | 0         |          0   |
| mrr        |          0.609717 |             0.761613 | 0.151897  |         24.9 |
| ndcg@10    |          0.590443 |             0.723177 | 0.132733  |         22.5 |
