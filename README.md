# Duplicate Question Detector

Detects and clusters duplicate questions (Quora-style) using SBERT embeddings, a from-scratch LSH index for fast similarity search, and Union-Find for clustering. Served as a FastAPI backend.

Baseline (pretrained SBERT) gets **F1 = 0.734**. Fine-tuning on the Quora Question Pairs dataset improves that to **F1 = 0.797**. Full numbers in [`RESULTS.md`](./RESULTS.md).

## How it works

1. A new question comes in → embedded with SBERT (fine-tuned on Quora pairs).
2. The embedding is queried against an LSH index (custom implementation, not FAISS) to find candidate near-duplicates without scanning the whole database.
3. Matches above a similarity threshold are merged using Union-Find, so duplicates are grouped into clusters — not just scored pair-by-pair.
4. Everything is persisted to SQLite; the LSH index and Union-Find structure are rebuilt from the database on startup.

## Repo layout

```
api/            FastAPI app — routes, request/response schemas, in-memory index state
src/
  search/       Brute-force search + the LSH index (both from scratch)
  clustering/   Union-Find
  embeddings/   Wraps SentenceTransformer for request-time embedding
  db/           SQLAlchemy models + session setup
tests/          pytest — unit tests + full API integration tests
scripts/        Benchmarking (LSH vs. brute force) and clustering sweep scripts
notebooks/      Colab notebooks: embedding generation, fine-tuning SBERT
data/           Dataset splits, embeddings, the fine-tuned model (gitignored)
results/        Benchmark/sweep output CSVs (gitignored, regenerate via scripts/)
RESULTS.md      Full write-up of numbers for every phase
```

## Why Colab + local

Embedding generation and fine-tuning need a GPU, so those steps ran on Colab — see `notebooks/`. Everything else (LSH, Union-Find, the API, tests) runs locally and only needs a CPU.

## Setup

```bash
git clone <repo-url>
cd duplicate-question-detector
python -m venv venv
venv\Scripts\activate        # or source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
```

You'll also need:
- The Quora Question Pairs dataset (`data/raw/train.csv`) — not included, download from Kaggle.
- The fine-tuned model folder at `data/finetuned_sbert/` — produced by `notebooks/02_finetune_sbert.ipynb`, or point `src/embeddings/embedder.py` at `"all-MiniLM-L6-v2"` to use the pretrained model instead.

## Running the API

```bash
uvicorn api.main:app --reload
```

Then open `http://127.0.0.1:8000/docs` for the interactive API.

| Endpoint | What it does |
|---|---|
| `POST /questions` | Submit a new question — embeds it, finds/clusters duplicates, returns matches |
| `GET /questions/{id}/duplicates` | All other questions in the same cluster, ranked by similarity |
| `GET /clusters` | All clusters, largest first |
| `GET /clusters/{id}` | Full question list for one cluster |

## Tests

```bash
pytest tests/ -v
```

All tests run against synthetic/fake data (no GPU or real model download needed) except the manual end-to-end check against the real fine-tuned model, done separately.

## Reproducing the benchmarks

```bash
python -m scripts.benchmark_brute_vs_lsh   # LSH vs. exact search, speed + accuracy
python -m scripts.cluster_questions        # Union-Find clustering sweep
```

## Tech stack

Python, PyTorch, Sentence-Transformers, FastAPI, SQLAlchemy, NumPy, pytest. LSH and Union-Find are implemented from scratch (no FAISS, no off-the-shelf DSU library).

## Known limitations

- Union-Find can only merge clusters, never split them — one bad match can pull unrelated questions together permanently.
- The API's clustering threshold was tuned on pretrained-model embeddings; re-tuning against the fine-tuned model's embeddings is a natural next step.
- No frontend yet.

See [`RESULTS.md`](./RESULTS.md) for full numbers and discussion.
