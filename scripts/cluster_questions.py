"""
Phase 5 driver: cluster questions with LSH + Union-Find, and measure how good
the clusters are at several similarity thresholds.

Run from the project root:
    python -m scripts.cluster_questions
    python -m scripts.cluster_questions --limit-pairs 5000        # quick trial
    python -m scripts.cluster_questions --thresholds 0.8 0.85 0.9 --inspect 0.85

Pipeline:
  1. Collect every UNIQUE question (by qid) from question1 and question2.
  2. Index all of them in an LSHIndex.
  3. Query each question once (top-k neighbours, keeping similarity >= the
     lowest threshold). Every hit is a candidate duplicate pair.
  4. For each threshold: union the pairs at or above it, giving clusters.
  5. Evaluate against the labeled pairs in the CSV.

How the evaluation works (pairwise, on labeled pairs only):
  - A labeled duplicate pair whose two questions land in the same cluster = TP
  - A labeled NON-duplicate pair whose questions land in the same cluster = FP
    (a bad merge, usually caused by chaining through transitivity)
  - A labeled duplicate pair in different clusters = FN
  The same numbers are also computed for "direct threshold, no clustering"
  (the Phase 2 method) so you can see what LSH + transitivity change.

Caveat: only pairs that have labels can be scored. Questions that are true
duplicates but were never labeled as a pair are invisible to this metric.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.clustering.union_find import UnionFind
from src.search.lsh_index import LSHIndex

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional
    def tqdm(iterable, **kwargs):
        return iterable


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--q1", default="data/embeddings/q1_embeddings_baseline.npy")
    p.add_argument("--q2", default="data/embeddings/q2_embeddings_baseline.npy")
    p.add_argument("--csv", default="data/embeddings/test_split_with_similarity.csv")
    p.add_argument("--limit-pairs", type=int, default=None, help="use only the first N labeled pairs")
    p.add_argument("--tables", type=int, default=32)
    p.add_argument("--bits", type=int, default=12)
    p.add_argument("--top-k", type=int, default=10, help="neighbours fetched per question")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.75, 0.80, 0.85, 0.90])
    p.add_argument("--inspect", type=float, default=None,
                   help="threshold to print example clusters for (default: the middle threshold)")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def short(text, n=90):
    text = str(text)
    return text if len(text) <= n else text[: n - 3] + "..."


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # ---------------- Load data ----------------
    df = pd.read_csv(args.csv)
    needed = {"qid1", "qid2", "question1", "question2", "is_duplicate"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing columns: {sorted(missing)}")

    if args.limit_pairs:
        df = df.iloc[: args.limit_pairs]
    q1 = np.load(args.q1)[: len(df)].astype(np.float32)
    q2 = np.load(args.q2)[: len(df)].astype(np.float32)
    assert len(df) == len(q1) == len(q2), "CSV and embeddings are misaligned"

    # ---------------- Unique questions ----------------
    qids_all = np.concatenate([df["qid1"].values, df["qid2"].values])
    texts_all = np.concatenate([df["question1"].values, df["question2"].values])
    emb_all = np.vstack([q1, q2])
    _, first = np.unique(qids_all, return_index=True)
    qids = [int(x) for x in qids_all[first]]
    texts = dict(zip(qids, texts_all[first]))
    emb = emb_all[first]
    print(f"{len(df):,} labeled pairs -> {len(qids):,} unique questions")

    # ---------------- Thresholds ----------------
    thresholds = sorted(set(args.thresholds))
    inspect_t = args.inspect if args.inspect is not None else thresholds[len(thresholds) // 2]
    if inspect_t not in thresholds:
        thresholds = sorted(thresholds + [inspect_t])
    min_thr = thresholds[0]

    # ---------------- LSH candidate pairs ----------------
    t0 = time.perf_counter()
    lsh = LSHIndex(dim=emb.shape[1], num_tables=args.tables, hash_size=args.bits)
    lsh.insert_many(qids, emb)
    print(f"Built LSH index (tables={args.tables}, bits={args.bits}) in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    pairs = []
    for i, qid in enumerate(tqdm(qids, desc="Querying")):
        for other, sim in lsh.query(emb[i], top_k=args.top_k + 1, min_similarity=min_thr):
            if other != qid:
                pairs.append((qid, other, sim))
    print(f"Found {len(pairs):,} candidate pairs (sim >= {min_thr}) in {time.perf_counter() - t0:.1f}s")

    # ---------------- Labeled pairs for evaluation ----------------
    a = df["qid1"].values.astype(int)
    b = df["qid2"].values.astype(int)
    y = df["is_duplicate"].values.astype(bool)
    n1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
    n2 = q2 / np.linalg.norm(q2, axis=1, keepdims=True)
    direct_sims = np.sum(n1 * n2, axis=1)   # cosine similarity per labeled pair

    # ---------------- Sweep thresholds ----------------
    results, inspect_uf = [], None
    for t in thresholds:
        uf = UnionFind(qids)
        merges = uf.union_pairs([p for p in pairs if p[2] >= t])

        same = np.array([uf.connected(int(x), int(z)) for x, z in zip(a, b)])
        c_p, c_r, c_f = prf((same & y).sum(), (same & ~y).sum(), (~same & y).sum())

        pred = direct_sims >= t
        d_p, d_r, d_f = prf((pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum())

        sizes = sorted((len(m) for m in uf.clusters().values()), reverse=True)
        results.append(dict(
            threshold=t,
            merges=merges,
            clusters=len(sizes),
            multi_item_clusters=sum(s >= 2 for s in sizes),
            singleton_pct=100 * sum(s == 1 for s in sizes) / len(sizes),
            largest_cluster=sizes[0],
            top5_sizes=str(sizes[:5]),
            bad_merges_FP=int((same & ~y).sum()),
            cluster_precision=c_p, cluster_recall=c_r, cluster_f1=c_f,
            direct_precision=d_p, direct_recall=d_r, direct_f1=d_f,
        ))
        if t == inspect_t:
            inspect_uf = uf

    out = pd.DataFrame(results)
    with pd.option_context("display.width", 250, "display.max_columns", None,
                           "display.float_format", "{:.3f}".format):
        print("\nCluster structure:")
        print(out[["threshold", "merges", "clusters", "multi_item_clusters",
                   "singleton_pct", "largest_cluster", "top5_sizes"]].to_string(index=False))
        print("\nQuality on labeled pairs (cluster = LSH + Union-Find; direct = threshold only):")
        print(out[["threshold", "bad_merges_FP", "cluster_precision", "cluster_recall", "cluster_f1",
                   "direct_precision", "direct_recall", "direct_f1"]].to_string(index=False))

    os.makedirs(args.out_dir, exist_ok=True)
    sweep_path = os.path.join(args.out_dir, "cluster_sweep.csv")
    out.to_csv(sweep_path, index=False)

    # ---------------- Inspect clusters by eye ----------------
    clusters = inspect_uf.clusters(min_size=2)
    by_size = sorted(clusters.values(), key=len, reverse=True)
    print(f"\n=== Example clusters at threshold {inspect_t} ({len(by_size):,} clusters with duplicates) ===")

    print("\nLargest clusters (large ones are where chaining shows up):")
    for members in by_size[:3]:
        print(f"\n  size {len(members)}")
        for m in members[:6]:
            print(f"    - {short(texts[m])}")
        if len(members) > 6:
            print(f"    ... and {len(members) - 6} more")

    small = [m for m in by_size if 2 <= len(m) <= 5]
    print("\nRandom small clusters:")
    for idx in rng.choice(len(small), size=min(3, len(small)), replace=False) if small else []:
        print(f"\n  size {len(small[idx])}")
        for m in small[idx]:
            print(f"    - {short(texts[m])}")

    rows = [dict(cluster_id=root, cluster_size=len(members), qid=m, question=texts[m])
            for root, members in clusters.items() for m in members]
    assign_path = os.path.join(args.out_dir, f"clusters_t{inspect_t:.2f}.csv")
    pd.DataFrame(rows).sort_values(["cluster_size", "cluster_id"], ascending=[False, True]) \
        .to_csv(assign_path, index=False)
    print(f"\nSaved sweep table to {sweep_path}")
    print(f"Saved clusters with duplicates (threshold {inspect_t}) to {assign_path}")


if __name__ == "__main__":
    main()