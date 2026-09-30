# Results

Duplicate question detection on the Quora Question Pairs dataset — pretrained vs fine-tuned SBERT, LSH for fast retrieval, Union-Find for clustering.

## Baseline (pretrained `all-MiniLM-L6-v2`)

Cosine similarity threshold swept on the test split (60,644 pairs):

| Threshold | F1 |
|---|---|
| 0.70 | 0.730 |
| **0.75** | **0.734** |
| 0.80 | 0.720 |

Best: **F1 = 0.734** at threshold 0.75 (Precision 0.634, Recall 0.871).

## LSH (from scratch, random hyperplane)

Tested against exact brute-force search at 60,644 questions.

- Config used: 32 tables, 12 bits
- **6.5x faster** than vectorized exact search
- Checks only **1.35%** of the index per query
- Keeps **~95%** of exact search's retrieval accuracy (pair-hit rate)

Fewer tables/bits is faster but loses too much accuracy; more checks too much of the index to be worth it. 32/12 was the best trade-off tested.

## Clustering (Union-Find)

Pairs found by LSH are merged into clusters instead of scored one-by-one.

- Best result: **F1 = 0.765** at threshold 0.80 — better than the plain pairwise baseline (0.734)
- At a lower threshold (0.75), clustering chains together too aggressively — one cluster grew to 18,000+ questions. Raising the threshold to 0.80 fixes this.

## Fine-tuning (SBERT on Quora pairs)

Fine-tuned `all-MiniLM-L6-v2` on the train split, evaluated on the same held-out test split as the baseline.

| Model | Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Pretrained | 0.75 | 0.767 | 0.634 | 0.871 | 0.734 |
| **Fine-tuned** | **0.60** | **0.839** | **0.745** | 0.855 | **0.797** |

**F1 improved by 8.5%.** Most of the gain is in precision (+17.5%) — the pretrained model was flagging too many topically-similar-but-different questions as duplicates; fine-tuning fixed that while barely touching recall.

Note: the fine-tuned model's optimal threshold (0.60) is quite different from the pretrained model's (0.75) — reusing the old threshold would have been wrong.

## API

Built with FastAPI + SQLite + the fine-tuned model. On each new question:
1. Embed it
2. Query the LSH index for candidates
3. Union any match above threshold into a cluster
4. Propagate the cluster ID to every affected question in the DB

Tested end-to-end with the real model — new questions get matched, clustered, and retrievable via `/clusters` and `/questions/{id}/duplicates`, and clustering survives a full restart (rebuilt from the database).

## Known limitations

- Clustering can only merge, never split — one bad match can pull unrelated questions into the same group.
- The API's similarity threshold (0.80) was tuned on the pretrained model's embeddings, not the fine-tuned ones — a proper re-tune is a good next step.
- `pair_hit`/F1 numbers only count labeled pairs; true duplicates that were never labeled as a pair aren't reflected in these scores.
