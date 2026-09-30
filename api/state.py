"""
In-memory application state: the LSH index and Union-Find structure.

The database is the source of truth. These two structures are caches,
rebuilt from the database once at startup (see rebuild_from_db). They are
NOT persisted separately — for a project at this scale, rebuilding from the
DB on restart is simpler than keeping a pickle file in sync with it. (For a
much larger DB, you'd switch to LSHIndex.save()/load() and only rebuild
Union-Find, to avoid re-hashing everything on every restart.)

Threading note: FastAPI can run multiple requests concurrently. This state
is mutated in place with no locking, which is fine for a single-process
dev/demo server but would need a lock (or a move to shared external storage)
under real concurrent write load.
"""

from collections import defaultdict

import numpy as np
from sqlalchemy.orm import Session

from src.clustering.union_find import UnionFind
from src.db.models import Question
from src.embeddings.embedder import EMBEDDING_DIM, embed_text
from src.search.lsh_index import LSHIndex

# From Phase 4 / Phase 5 tuning (see RESULTS.md).
NUM_TABLES = 32
HASH_SIZE = 12
SIMILARITY_THRESHOLD = 0.80
TOP_K = 10


class AppState:
    def __init__(self, dim: int = EMBEDDING_DIM, threshold: float = SIMILARITY_THRESHOLD):
        self.dim = dim
        self.threshold = threshold
        self.embed_fn = embed_text   # swappable in tests, so they don't need the real model
        self.lsh = LSHIndex(dim=dim, num_tables=NUM_TABLES, hash_size=HASH_SIZE)
        self.uf = UnionFind()

    def rebuild_from_db(self, db: Session) -> None:
        """Rebuild the LSH index and Union-Find structure from every stored question."""
        rows = db.query(Question).all()

        self.lsh = LSHIndex(dim=self.dim, num_tables=NUM_TABLES, hash_size=HASH_SIZE)
        self.uf = UnionFind()
        if not rows:
            return

        ids = [r.id for r in rows]
        vectors = np.stack([np.frombuffer(r.embedding, dtype=np.float32) for r in rows])
        self.lsh.insert_many(ids, vectors)
        for item_id in ids:
            self.uf.add(item_id)

        groups = defaultdict(list)
        for r in rows:
            if r.cluster_id is not None:
                groups[r.cluster_id].append(r.id)
        for members in groups.values():
            for other in members[1:]:
                self.uf.union(members[0], other)


def add_question(state: AppState, db: Session, text: str):
    """
    Insert one new question end to end:
      embed -> find candidates (pre-insert) -> insert into LSH -> union
      duplicates in Union-Find -> propagate the resulting cluster_id to
      every affected row in the database.

    Returns (Question row, list[(candidate_id, similarity)], is_new_cluster).
    """
    vector = state.embed_fn(text)
    if vector.shape != (state.dim,):
        raise ValueError(f"Embedding has shape {vector.shape}, expected ({state.dim},)")

    # 1. Find candidates BEFORE inserting, so the question can't match itself.
    candidates = state.lsh.query(vector, top_k=TOP_K, min_similarity=state.threshold)

    # 2. Persist the new row to get its id.
    row = Question(text=text, embedding=vector.tobytes(), cluster_id=None)
    db.add(row)
    db.commit()
    db.refresh(row)

    # 3. Add it to the in-memory structures.
    state.lsh.insert(row.id, vector)
    state.uf.add(row.id)

    if not candidates:
        return row, [], False

    # Did any matched candidate already belong to a cluster? Tells the caller
    # whether this request joined an existing cluster or formed a brand-new one.
    candidate_ids = [c for c, _ in candidates]
    candidate_rows = db.query(Question).filter(Question.id.in_(candidate_ids)).all()
    had_existing_cluster = any(r.cluster_id is not None for r in candidate_rows)

    # 4. Union with every match at/above the threshold.
    for candidate_id, _ in candidates:
        state.uf.union(row.id, candidate_id)

    # 5. Propagate the (possibly new) root as cluster_id to every group member.
    root = state.uf.find(row.id)
    member_ids = state.uf.members(row.id)
    db.query(Question).filter(Question.id.in_(member_ids)).update(
        {"cluster_id": root}, synchronize_session=False
    )
    db.commit()

    return row, candidates, not had_existing_cluster