"""
Benchmark: brute-force vs LSH on your real SBERT embeddings.



What it measures, for several index sizes (so you can see how time SCALES):
  1. Brute force, pure Python  (your BruteForceIndex)   - slow on purpose, so
     it only runs on NUM_SLOW_QUERIES queries. Compare per-query averages.
  2. Brute force, NumPy vectorized (one matrix-vector product) - the honest
     "fast exact" reference. Also used as ground truth for LSH recall.
  3. LSH (your LSHIndex) for each (num_tables, hash_size) config.

Metrics:
  - build_s        : time to insert everything into the index
  - avg_query_ms   : average time per query
  - recall_at_k    : fraction of the exact top-K that the method also returns
  - pair_hit_rate  : for queries whose pair is a labeled duplicate, how often
                     the paired question1 is in the returned top-K
  - avg_candidates : (LSH only) how many vectors LSH actually compared
"""

import os
import time

import numpy as np
import pandas as pd

from src.search.brute_force import BruteForceIndex
from src.search.lsh_index import LSHIndex

Q1_PATH = "data/embeddings/q1_embeddings_baseline.npy"
Q2_PATH = "data/embeddings/q2_embeddings_baseline.npy"
CSV_PATH = "data/embeddings/test_split_with_similarity.csv"
OUT_PATH = "results/benchmark_brute_vs_lsh.csv"

INDEX_SIZES = [1000, 5000, 20000, None]   # None = use all rows
NUM_QUERIES = 3000          # queries for NumPy brute force and LSH
NUM_SLOW_QUERIES = 30      # queries for pure-Python brute force (it is slow)
TOP_K = 5
LSH_CONFIGS = [(8, 12), (16, 10), (20, 8), (20, 10), (24, 10), (32, 12)]   # (num_tables, hash_size)
SEED = 42


def exact_topk(norm_matrix, query, k):
    """Exact top-k ids via one vectorized matrix-vector product."""
    q = query / np.linalg.norm(query)
    sims = norm_matrix @ q
    idx = np.argpartition(-sims, k)[:k]
    return idx[np.argsort(-sims[idx])].tolist()


def recall_at_k(returned_ids, exact_ids):
    return len(set(returned_ids) & set(exact_ids)) / len(exact_ids)




def run(index_sizes=INDEX_SIZES, num_queries=NUM_QUERIES, num_slow=NUM_SLOW_QUERIES,
        top_k=TOP_K, lsh_configs=LSH_CONFIGS):
    df = pd.read_csv(CSV_PATH)
    q1 = np.load(Q1_PATH).astype(np.float64)
    q2 = np.load(Q2_PATH).astype(np.float64)
    assert len(df) == len(q1) == len(q2), "CSV and embeddings are misaligned"

    is_dup = df["is_duplicate"].values
    dim = q1.shape[1]
    rng = np.random.default_rng(SEED)   # this is used for sampling query rows and generating synthetic data
    results = []

    for size in index_sizes:
        n = len(df) if size is None else min(size, len(df))
        print(f"\n=== Index size: {n} ===")

        index_vecs = q1[:n]
        norm_matrix = index_vecs / np.linalg.norm(index_vecs, axis=1, keepdims=True)

        # Query = question2 of randomly chosen rows inside the index range
        nq = min(num_queries, n)
        query_rows = rng.choice(n, size=nq, replace=False)

        # ---- 2. Exact NumPy brute force (also ground truth) ----
        t0 = time.perf_counter()
        exact_results = [exact_topk(norm_matrix, q2[r], top_k) for r in query_rows]
        exact_time = time.perf_counter() - t0

        def pair_hit(rows, returned_lists):
            hits = [r in ids for r, ids in zip(rows, returned_lists) if is_dup[r] == 1]
            return float(np.mean(hits)) if hits else float("nan")

        results.append(dict(
            index_size=n, method="brute force (NumPy)", build_s=0.0,
            avg_query_ms=exact_time / nq * 1000, recall_at_k=1.0,
            pair_hit_rate=pair_hit(query_rows, exact_results), avg_candidates=n))

        # ---- 1. Pure-Python brute force (slow, fewer queries) ----
        ns = min(num_slow, nq)
        bf = BruteForceIndex()
        t0 = time.perf_counter()
        bf.insert_many(list(range(n)), index_vecs)
        bf_build = time.perf_counter() - t0

        slow_rows = query_rows[:ns]
        t0 = time.perf_counter()
        slow_results = [[i for i, _ in bf.query(q2[r], top_k=top_k)] for r in slow_rows]
        slow_time = time.perf_counter() - t0
        del bf

        results.append(dict(
            index_size=n, method="brute force (pure Python)", build_s=bf_build,
            avg_query_ms=slow_time / ns * 1000,
            recall_at_k=float(np.mean([recall_at_k(a, b) for a, b in zip(slow_results, exact_results[:ns])])),
            pair_hit_rate=pair_hit(slow_rows, slow_results), avg_candidates=n))

        # ---- 3. LSH configs ----
        for num_tables, hash_size in lsh_configs:
            lsh = LSHIndex(dim=dim, num_tables=num_tables, hash_size=hash_size, seed=SEED)
            t0 = time.perf_counter()
            for i in range(n):
                lsh.insert(i, index_vecs[i])
            build = time.perf_counter() - t0

            t0 = time.perf_counter()
            lsh_results = [[i for i, _ in lsh.query(q2[r], top_k=top_k)] for r in query_rows]
            lsh_time = time.perf_counter() - t0

            # Candidate counting is done outside the timed region
            cand_counts = [len(lsh._candidates(q2[r])) for r in query_rows[:100]]

            results.append(dict(
                index_size=n, method=f"LSH (tables={num_tables}, bits={hash_size})",
                build_s=build, avg_query_ms=lsh_time / nq * 1000,
                recall_at_k=float(np.mean([recall_at_k(a, b) for a, b in zip(lsh_results, exact_results)])),
                pair_hit_rate=pair_hit(query_rows, lsh_results),
                avg_candidates=float(np.mean(cand_counts))))
            del lsh

    out = pd.DataFrame(results)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("\n", out.to_string(index=False))

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved to {OUT_PATH}")
    return out


if __name__ == "__main__":
    run()