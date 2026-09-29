"""
Phase 4: Locality-Sensitive Hashing (LSH) index for cosine similarity.

Method: random hyperplane hashing (a.k.a. SimHash).
  - Each hash table uses `hash_size` random hyperplanes. A vector's hash is
    the sequence of sides (dot product > 0 or not) it falls on, packed into
    one integer.
  - Vectors with a small angle between them (high cosine similarity) land on
    the same side of most hyperplanes, so they tend to share a bucket.
  - With `num_tables` independent tables, two vectors become CANDIDATES if
    they share a bucket in ANY table.
  - Candidates are then re-ranked by their true cosine similarity. LSH only
    shortlists; it never decides "duplicate" by itself.

Tuning (per-table collision probability = (1 - angle/pi) ** hash_size):
  - more num_tables  -> higher recall, more memory, more candidates
  - more hash_size   -> fewer false candidates, but lower recall
  Use scripts/benchmark_brute_vs_lsh.py to pick values from real numbers.

Design notes:
  - Vectors are stored L2-normalized in one NumPy matrix, so cosine
    similarity becomes a dot product and candidate re-ranking is one
    vectorized matrix-vector product instead of a Python loop.
  - Hashing of all tables is a single matrix multiply.
  - Stored as float32 to halve memory (60K x 384 is ~92 MB).
"""

import pickle
from collections import defaultdict

import numpy as np


class LSHIndex:
    # this function initializes the LSHIndex class with the specified parameters. It sets up the necessary data structures and generates random hyperplanes for hashing the vectors. The hyperplanes are used to create hash keys for the vectors, which are then stored in hash tables for efficient retrieval of similar vectors.
    def __init__(self, dim: int, num_tables: int = 16, hash_size: int = 10, seed: int = 42):
        """
        dim:        embedding dimensionality (384 for all-MiniLM-L6-v2)
        num_tables: number of independent hash tables (bands)
        hash_size:  hyperplanes (bits) per table, 1..62
        seed:       RNG seed for the hyperplanes (fixed = reproducible index)
        """
        
        if not 1 <= hash_size <= 62:
            raise ValueError("hash_size must be between 1 and 62")
        if num_tables < 1:
            raise ValueError("num_tables must be >= 1")

        self.dim = dim
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.seed = seed

        rng = np.random.default_rng(seed) # this line creates a random number generator (RNG) using the specified seed. The RNG is used to generate random hyperplanes for hashing the vectors. By using a fixed seed, the same set of hyperplanes will be generated each time the index is created, ensuring reproducibility of the results.
        
        # All hyperplanes for all tables stacked: (num_tables * hash_size, dim), this line generates a set of random hyperplanes for all hash tables. The shape of the array is (num_tables * hash_size, dim), where each row represents a hyperplane in the embedding space.
        self._planes = rng.normal(size=(num_tables * hash_size, dim)).astype(np.float32)
        
        # Weights used to pack a row of bits into a single integer key. this line creates an array of powers of 2, which is used to convert a binary representation of the hash (a sequence of bits) into a single integer key. Each bit corresponds to a hyperplane, and the powers of 2 are used to compute the integer value of the binary representation.
        self._powers = 1 << np.arange(hash_size, dtype=np.int64)

        # tables[t]: {bucket_key (int) -> [internal positions]}
        self._tables = [defaultdict(list) for _ in range(num_tables)]

        # Storage: original ids, id -> position map, normalized vectors
        self._ids: list = []
        self._id_to_pos: dict = {}
        self._matrix = np.empty((1024, dim), dtype=np.float32)
        self._size = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    
    # this function computes the cosine similarity between two vectors a and b.
    def _normalize(self, vectors) -> np.ndarray:
        """L2-normalize one vector (dim,) or a batch (n, dim). Returns float32."""
        v = np.asarray(vectors, dtype=np.float32)
        if v.shape[-1] != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {v.shape[-1]}")
        # norms are computed along the last axis (the embedding dimension) and kept as a separate dimension for broadcasting. The vectors are then divided by their norms to produce unit vectors. If any vector has a norm of zero, a ValueError is raised to prevent indexing or querying with a zero vector.
        norms = np.linalg.norm(v, axis=-1, keepdims=True) 
        if np.any(norms == 0):
            raise ValueError("Cannot index or query a zero vector")
        return v / norms


    
    # this function computes the hash keys for a set of normalized vectors. It takes in an array of normalized vectors and returns an array of integer bucket keys for each vector, which can be used to look up candidates in the hash tables.
    def _hash_keys(self, normalized: np.ndarray) -> np.ndarray:
        """(n, dim) normalized vectors -> (n, num_tables) integer bucket keys."""
        bits = (normalized @ self._planes.T) > 0
        bits = bits.reshape(len(normalized), self.num_tables, self.hash_size)
        return bits.astype(np.int64) @ self._powers



    # this function ensures that the internal storage matrix has enough capacity to accommodate additional vectors. If the current capacity is insufficient, it grows the matrix by creating a new larger matrix and copying the existing vectors into it. The new capacity is determined by doubling the current size, ensuring it meets the required size, and having a minimum capacity of 1024.
    def _ensure_capacity(self, extra: int):
        needed = self._size + extra
        if needed <= self._matrix.shape[0]:
            return
        new_cap = max(self._matrix.shape[0] * 2, needed, 1024)
        grown = np.empty((new_cap, self.dim), dtype=np.float32)
        grown[: self._size] = self._matrix[: self._size]
        self._matrix = grown

    
    
    # this function retrieves the positions of candidate vectors that share a bucket with the given normalized query vector in any of the hash tables. It computes the hash keys for the query vector, looks up the corresponding buckets in each table, and collects the positions of all candidates into a set. This set of positions can then be used to retrieve the actual vectors for further similarity computation.
    def _candidate_positions(self, normalized_query: np.ndarray) -> set:
        keys = self._hash_keys(normalized_query[None, :])[0].tolist()
        candidates = set()
        for t, key in enumerate(keys):
            bucket = self._tables[t].get(key)
            if bucket:
                candidates.update(bucket)
        return candidates

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    
    # this function inserts multiple vectors into the LSH index. It takes a list of item IDs and a corresponding array of vectors, normalizes the vectors, computes their hash keys, and stores them in the appropriate hash tables. It also updates the internal storage matrix and maintains mappings from item IDs to their positions in the matrix. The function checks for duplicate IDs and ensures that the input dimensions are valid before proceeding with the insertion.
    def insert_many(self, ids, vectors):
        """Bulk insert. `vectors` shape (n, dim), aligned with `ids`."""
        ids = list(ids)
        vectors = np.asarray(vectors)
        if vectors.ndim != 2 or len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length; vectors must be 2-D")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate ids within the batch")
        clashes = [i for i in ids if i in self._id_to_pos]
        if clashes:
            raise ValueError(f"ids already in the index: {clashes[:5]}")

        normalized = self._normalize(vectors)
        keys = self._hash_keys(normalized)  # (n, num_tables)

        self._ensure_capacity(len(ids))
        start = self._size
        self._matrix[start : start + len(ids)] = normalized

        for offset, item_id in enumerate(ids):
            pos = start + offset
            self._ids.append(item_id)
            self._id_to_pos[item_id] = pos
            for t in range(self.num_tables):
                self._tables[t][int(keys[offset, t])].append(pos)
        self._size += len(ids)

    def insert(self, item_id, vector):
        """Insert a single item."""
        self.insert_many([item_id], np.asarray(vector)[None, :])

    def get_candidates(self, vector) -> set:
        """Ids that share a bucket with `vector` in ANY table (before re-ranking)."""
        qn = self._normalize(vector)
        return {self._ids[p] for p in self._candidate_positions(qn)}

    # Kept so the benchmark script (which calls _candidates) keeps working.
    _candidates = get_candidates

    
    
    
    # this function queries the LSH index with a given vector and retrieves the top_k most similar items based on cosine similarity. It first normalizes the query vector, finds candidate positions that share a bucket with the query in any of the hash tables, and then computes the cosine similarities between the query vector and the candidate vectors. The results are filtered based on a minimum similarity threshold and sorted to return the top_k matches as a list of (item_id, similarity) pairs.
    def query(self, vector, top_k: int = 5, min_similarity: float = 0.0):
        """
        Return up to top_k (id, cosine_similarity) pairs, best first.
        Only LSH candidates are scored - this is NOT a full scan.
        """
        qn = self._normalize(vector)
        positions = self._candidate_positions(qn)
        if not positions:
            return []

        pos = np.fromiter(positions, dtype=np.int64, count=len(positions))
        sims = self._matrix[pos] @ qn

        keep = sims >= min_similarity
        pos, sims = pos[keep], sims[keep]
        order = np.argsort(-sims)[:top_k]
        return [(self._ids[pos[i]], float(sims[i])) for i in order]

    # this function computes the cosine similarity between two vectors a and b. It calculates the dot product of the two vectors and divides it by the product of their L2 norms. If either vector has a norm of zero, it returns a similarity of 0.0 to avoid division by zero. The result is a float representing the cosine similarity between the two vectors.
    def stats(self) -> dict:
        """Bucket statistics - useful for diagnosing bad parameter choices."""
        sizes = [len(b) for table in self._tables for b in table.values()]
        return {
            "num_items": self._size,
            "num_tables": self.num_tables,
            "hash_size": self.hash_size,
            "num_buckets": len(sizes),
            "avg_bucket_size": float(np.mean(sizes)) if sizes else 0.0,
            "max_bucket_size": max(sizes) if sizes else 0,
        }

    def __len__(self):
        return self._size

    def __contains__(self, item_id):
        return item_id in self._id_to_pos

    
    # Persistence (needed in Phase 6 so the API doesn't rebuild on startup)
  
    def save(self, path: str):
        state = {
            "dim": self.dim,
            "num_tables": self.num_tables,
            "hash_size": self.hash_size,
            "seed": self.seed,
            "planes": self._planes,
            "ids": self._ids,
            "matrix": self._matrix[: self._size].copy(),
            "tables": [dict(t) for t in self._tables],
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "LSHIndex":
        """Load an index saved with save(). Only load files you created yourself
        (pickle can execute arbitrary code from untrusted files)."""
        with open(path, "rb") as f:
            s = pickle.load(f)
        obj = cls(s["dim"], s["num_tables"], s["hash_size"], s["seed"])
        obj._planes = s["planes"]
        obj._ids = s["ids"]
        obj._id_to_pos = {item_id: p for p, item_id in enumerate(s["ids"])}
        obj._matrix = s["matrix"]
        obj._size = len(s["ids"])
        obj._tables = [defaultdict(list, t) for t in s["tables"]]
        return obj


