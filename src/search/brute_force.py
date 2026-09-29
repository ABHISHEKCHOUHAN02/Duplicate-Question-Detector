"""
Phase 3: Brute-force duplicate search.

Purpose: establish a CORRECT baseline before optimizing with LSH (Phase 4).
This compares a query embedding against EVERY stored embedding (O(n) per
query, O(n^2) if you check all pairs) — slow at scale, but simple and
guaranteed correct. LSH's job later is to approximate this fast, and you'll
measure LSH's recall against what brute-force finds here.
"""

import numpy as np

# this is a simple brute-force index that stores embeddings in memory and
# compares them using cosine similarity.
class BruteForceIndex:
    def __init__(self):
        # Maps item_id -> embedding vector
        self.vectors: dict[str, np.ndarray] = {}    # it is a dictionary that maps item IDs to their corresponding embedding vectors.

    def insert(self, item_id, vector: np.ndarray):
        """Store an item's embedding."""
        self.vectors[item_id] = np.asarray(vector, dtype=np.float64)

    def insert_many(self, ids: list, vectors: np.ndarray):
        """Bulk-insert. `vectors` shape: (n, dim), aligned with `ids`."""
        for item_id, vec in zip(ids, vectors):
            self.insert(item_id, vec)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def query(self, vector: np.ndarray, top_k: int = 5, min_similarity: float = 0.0):
        """
        Compare `vector` against every stored item (true brute force).
        Returns top_k matches as [(item_id, similarity), ...], sorted
        descending by similarity.
        """
        vector = np.asarray(vector, dtype=np.float64)

        scored = []
        for item_id, stored_vec in self.vectors.items():
            sim = self._cosine_sim(vector, stored_vec)
            if sim >= min_similarity:
                scored.append((item_id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def find_all_duplicate_pairs(self, threshold: float = 0.75):
        """
        Check ALL pairs among stored items (O(n^2)) and return every pair
        whose similarity >= threshold. Only use this on small sets (a few
        hundred to low thousands of items) — it does not scale, which is
        exactly the problem LSH (Phase 4) solves.

        Returns: [(id_a, id_b, similarity), ...]
        """
        ids = list(self.vectors.keys())
        pairs = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                sim = self._cosine_sim(self.vectors[id_a], self.vectors[id_b])
                if sim >= threshold:
                    pairs.append((id_a, id_b, sim))
        return pairs


# ---------------------------------------------------------------------------
# Quick correctness check using small synthetic data.
# Swap this for real SBERT embeddings + your test split before trusting it.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dim = 32

    index = BruteForceIndex()

    base = rng.normal(size=dim)
    near_duplicates = {f"dup_{i}": base + rng.normal(scale=0.05, size=dim) for i in range(3)}
    unrelated = {f"unrelated_{i}": rng.normal(size=dim) for i in range(10)}

    for item_id, vec in {**near_duplicates, **unrelated}.items():
        index.insert(item_id, vec)

    query_vec = base + rng.normal(scale=0.05, size=dim)
    results = index.query(query_vec, top_k=5)

    print("Top matches (should mostly be dup_* items):")
    for item_id, sim in results:
        print(f"  {item_id}: {sim:.4f}")

    print("\nAll duplicate pairs found (threshold=0.9):")
    for id_a, id_b, sim in index.find_all_duplicate_pairs(threshold=0.9):
        print(f"  {id_a} <-> {id_b}: {sim:.4f}")