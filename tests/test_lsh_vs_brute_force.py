"""
Tests: does LSH agree with brute force?


Part 1: synthetic agreement tests (fast, deterministic, no data files needed)
Part 2: real-data sample (auto-skipped if your embedding files aren't present)

These tests check CORRECTNESS and RECALL. Speed at scale is measured
separately by scripts/benchmark_brute_vs_lsh.py, because timing on a few
hundred items says almost nothing about behaviour at tens of thousands.
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.search.brute_force import BruteForceIndex
from src.search.lsh_index import LSHIndex


# ---------------------------------------------------------------------------
# Part 1: synthetic agreement tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clustered_data():
    """200 clusters x 10 members each. Members of a cluster are close to each other."""
    rng = np.random.default_rng(0)
    dim, n_clusters, per_cluster = 64, 200, 10
    centers = rng.normal(size=(n_clusters, dim))
    vectors, ids = [], []
    for c in range(n_clusters):
        for m in range(per_cluster):
            vectors.append(centers[c] + rng.normal(scale=0.3, size=dim))
            ids.append(f"c{c}_m{m}")
    return dict(dim=dim, centers=centers, vectors=np.array(vectors), ids=ids, rng=rng)


@pytest.fixture(scope="module")
def indexes(clustered_data):
    d = clustered_data
    bf = BruteForceIndex()
    bf.insert_many(d["ids"], d["vectors"])
    lsh = LSHIndex(dim=d["dim"], num_tables=20, hash_size=8)
    lsh.insert_many(d["ids"], d["vectors"])
    return bf, lsh


def make_queries(d, n=50):
    """New points near existing clusters (like a new question paraphrasing an old one)."""
    rng = np.random.default_rng(99)
    picks = rng.choice(len(d["centers"]), size=n, replace=False)
    return [d["centers"][c] + rng.normal(scale=0.3, size=d["dim"]) for c in picks]


def test_lsh_scores_match_brute_force_scores(clustered_data, indexes):
    """Every (id, similarity) LSH returns must carry the TRUE cosine similarity."""
    bf, lsh = indexes
    all_sims = None
    for q in make_queries(clustered_data, n=10):
        all_sims = dict(bf.query(q, top_k=len(bf.vectors)))
        for item_id, sim in lsh.query(q, top_k=10):
            assert sim == pytest.approx(all_sims[item_id], abs=1e-5)


def test_lsh_results_sorted_and_within_top_k(clustered_data, indexes):
    _, lsh = indexes
    for q in make_queries(clustered_data, n=10):
        results = lsh.query(q, top_k=5)
        assert len(results) <= 5
        sims = [s for _, s in results]
        assert sims == sorted(sims, reverse=True)


def test_lsh_top1_matches_brute_force_top1(clustered_data, indexes):
    """For queries with close neighbours, LSH should almost always find the same best match."""
    bf, lsh = indexes
    queries = make_queries(clustered_data, n=50)
    agree = 0
    for q in queries:
        bf_top = bf.query(q, top_k=1)[0][0]
        lsh_res = lsh.query(q, top_k=1)
        if lsh_res and lsh_res[0][0] == bf_top:
            agree += 1
    rate = agree / len(queries)
    print(f"\nTop-1 agreement with brute force: {rate:.0%}")
    assert rate >= 0.9


def test_lsh_recall_at_5_vs_brute_force(clustered_data, indexes):
    bf, lsh = indexes
    recalls = []
    for q in make_queries(clustered_data, n=50):
        exact = {i for i, _ in bf.query(q, top_k=5)}
        approx = {i for i, _ in lsh.query(q, top_k=5)}
        recalls.append(len(exact & approx) / 5)
    mean_recall = float(np.mean(recalls))
    print(f"\nMean recall@5 vs brute force: {mean_recall:.2%}")
    assert mean_recall >= 0.9


def test_lsh_checks_far_fewer_candidates_than_brute_force():
    """The point of LSH: on unrelated data it should only look at a small slice of the index."""
    rng = np.random.default_rng(5)
    n, dim = 2000, 64
    vectors = rng.normal(size=(n, dim))
    lsh = LSHIndex(dim=dim, num_tables=20, hash_size=8)
    lsh.insert_many(list(range(n)), vectors)

    counts = [len(lsh.get_candidates(rng.normal(size=dim))) for _ in range(50)]
    avg = float(np.mean(counts))
    print(f"\nAverage candidates checked: {avg:.0f} of {n} ({avg / n:.1%})")
    assert avg < 0.25 * n


def test_lsh_never_returns_ids_that_are_not_in_the_index(clustered_data, indexes):
    _, lsh = indexes
    valid = set(clustered_data["ids"])
    for q in make_queries(clustered_data, n=10):
        assert all(i in valid for i, _ in lsh.query(q, top_k=10))


# ---------------------------------------------------------------------------
# Part 2: real-data sample (your Quora test split + SBERT embeddings)
# ---------------------------------------------------------------------------

Q1_PATH = "data/embeddings/q1_embeddings_baseline.npy"
Q2_PATH = "data/embeddings/q2_embeddings_baseline.npy"
CSV_PATH = "data/embeddings/test_split_with_similarity.csv"
real_data_available = all(os.path.exists(p) for p in (Q1_PATH, Q2_PATH, CSV_PATH))

SAMPLE_SIZE = 300
CONFIGS = [(8, 12), (16, 10), (20, 8)]   # (num_tables, hash_size)


@pytest.mark.skipif(not real_data_available, reason="Real embedding files not found in repo")
def test_real_data_lsh_vs_brute_force():
    df = pd.read_csv(CSV_PATH).iloc[:SAMPLE_SIZE].reset_index(drop=True)
    emb1 = np.load(Q1_PATH)[:SAMPLE_SIZE]
    emb2 = np.load(Q2_PATH)[:SAMPLE_SIZE]
    assert len(df) == len(emb1) == len(emb2)

    ids = list(range(SAMPLE_SIZE))
    dim = emb1.shape[1]
    dup_rows = df.index[df["is_duplicate"] == 1].tolist()

    bf = BruteForceIndex()
    bf.insert_many(ids, emb1)
    bf_top = {i: [x for x, _ in bf.query(emb2[i], top_k=5)] for i in dup_rows}
    bf_hit = float(np.mean([i in bf_top[i] for i in dup_rows]))

    print(f"\n{len(dup_rows)} duplicate pairs | brute-force pair hit rate: {bf_hit:.1%}")
    print(f"{'config':<22}{'pair_hit':>10}{'recall@5':>10}{'avg_cands':>11}")

    last_hit = None
    for tables, bits in CONFIGS:
        lsh = LSHIndex(dim=dim, num_tables=tables, hash_size=bits)
        lsh.insert_many(ids, emb1)

        hits, recalls, cands = [], [], []
        for n_checked, i in enumerate(dup_rows):
            res = lsh.query(emb2[i], top_k=5)
            res_ids = [x for x, _ in res]
            hits.append(i in res_ids)
            recalls.append(len(set(res_ids) & set(bf_top[i])) / 5)
            cands.append(len(lsh.get_candidates(emb2[i])))

            # Scores LSH returns must equal true similarities (spot-check 20 queries)
            if n_checked < 20:
                truth = dict(bf.query(emb2[i], top_k=SAMPLE_SIZE))
                for item_id, sim in res:
                    assert sim == pytest.approx(truth[item_id], abs=1e-4)

        last_hit = float(np.mean(hits))
        print(f"tables={tables:<3} bits={bits:<3}      "
              f"{last_hit:>9.1%}{np.mean(recalls):>10.1%}{np.mean(cands):>11.1f}")

    # Loose bound on the most recall-friendly config - tune after seeing your numbers.
    assert last_hit >= 0.8 * bf_hit