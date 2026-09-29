"""
Tests for src/clustering/union_find.py

Run from the project root:
    pytest tests/test_union_find.py -v
"""

import random
from collections import defaultdict, deque

import numpy as np
import pytest

from src.clustering.union_find import UnionFind
from src.search.brute_force import BruteForceIndex
from src.search.lsh_index import LSHIndex


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

def test_new_items_start_as_separate_clusters():
    uf = UnionFind(["a", "b", "c"])
    assert len(uf) == 3
    assert uf.num_clusters == 3
    assert not uf.connected("a", "b")


def test_add_is_idempotent():
    uf = UnionFind()
    assert uf.add("a") is True
    assert uf.add("a") is False
    assert len(uf) == 1


def test_union_returns_true_only_when_it_merges():
    uf = UnionFind(["a", "b"])
    assert uf.union("a", "b") is True
    assert uf.union("a", "b") is False
    assert uf.union("b", "a") is False
    assert uf.num_clusters == 1


def test_union_is_transitive():
    uf = UnionFind(range(4))
    uf.union(0, 1)
    uf.union(1, 2)
    assert uf.connected(0, 2)          # never paired directly
    assert not uf.connected(0, 3)


def test_find_is_stable_within_a_cluster():
    uf = UnionFind(range(6))
    for a, b in [(0, 1), (1, 2), (3, 4), (2, 4)]:
        uf.union(a, b)
    roots = {uf.find(i) for i in [0, 1, 2, 3, 4]}
    assert len(roots) == 1
    assert uf.find(5) == 5


def test_group_size_and_members():
    uf = UnionFind(["a", "b", "c", "d"])
    uf.union("a", "b")
    uf.union("b", "c")
    assert uf.group_size("a") == 3
    assert sorted(uf.members("c")) == ["a", "b", "c"]
    assert uf.members("d") == ["d"]


def test_members_returns_a_copy():
    uf = UnionFind(["a", "b"])
    uf.union("a", "b")
    m = uf.members("a")
    m.append("junk")
    assert sorted(uf.members("a")) == ["a", "b"]


def test_clusters_min_size_filters_singletons():
    uf = UnionFind(range(5))
    uf.union(0, 1)
    uf.union(2, 3)
    all_groups = uf.clusters()
    dup_groups = uf.clusters(min_size=2)
    assert len(all_groups) == 3
    assert len(dup_groups) == 2
    assert {frozenset(m) for m in dup_groups.values()} == {frozenset({0, 1}), frozenset({2, 3})}


def test_unknown_item_raises_keyerror():
    uf = UnionFind(["a"])
    with pytest.raises(KeyError):
        uf.find("nope")
    with pytest.raises(KeyError):
        uf.union("a", "nope")


def test_contains_and_len():
    uf = UnionFind(["a", "b"])
    assert "a" in uf and "z" not in uf
    assert len(uf) == 2


# ---------------------------------------------------------------------------
# Optimizations actually happen
# ---------------------------------------------------------------------------

def test_path_compression_flattens_chains():
    # Uses the private _parent map on purpose: this checks the optimization,
    # not just the answers.
    uf = UnionFind(range(8))
    for i in range(7):
        uf.union(i, i + 1)
    root = uf.find(0)
    for i in range(8):
        uf.find(i)
    assert all(uf._parent[i] == root for i in range(8))


def test_union_by_size_keeps_larger_groups_root():
    uf = UnionFind(range(6))
    uf.union(0, 1)
    uf.union(1, 2)                      # group {0,1,2}
    big_root = uf.find(0)
    uf.union(3, 0)                      # single item joins the big group
    assert uf.find(3) == big_root


def test_deep_chain_does_not_hit_recursion_limit():
    n = 50_000
    uf = UnionFind(range(n))
    for i in range(n - 1):
        uf.union(i, i + 1)
    assert uf.num_clusters == 1
    assert uf.group_size(0) == n


# ---------------------------------------------------------------------------
# union_pairs and from_clusters
# ---------------------------------------------------------------------------

def test_union_pairs_ignores_extra_tuple_entries_and_counts_merges():
    uf = UnionFind(range(5))
    merges = uf.union_pairs([(0, 1, 0.93), (1, 2, 0.88), (0, 2, 0.90)])
    assert merges == 2                  # third pair was already connected
    assert uf.connected(0, 2)


def test_union_pairs_add_missing():
    uf = UnionFind()
    assert uf.union_pairs([(1, 2), (2, 3)], add_missing=True) == 2
    assert uf.connected(1, 3)


def test_union_pairs_without_add_missing_raises():
    with pytest.raises(KeyError):
        UnionFind().union_pairs([(1, 2)])


def test_from_clusters_round_trip():
    uf = UnionFind(range(10))
    for a, b in [(0, 1), (1, 2), (5, 6), (8, 9)]:
        uf.union(a, b)
    rebuilt = UnionFind.from_clusters(uf.clusters().values())
    assert {frozenset(m) for m in rebuilt.clusters().values()} == \
           {frozenset(m) for m in uf.clusters().values()}
    assert rebuilt.num_clusters == uf.num_clusters


# ---------------------------------------------------------------------------
# Correctness against an independent implementation
# ---------------------------------------------------------------------------

def bfs_components(n, edges):
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen, comps = set(), []
    for s in range(n):
        if s in seen:
            continue
        comp, queue = {s}, deque([s])
        seen.add(s)
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    comp.add(v)
                    queue.append(v)
        comps.append(frozenset(comp))
    return set(comps)


def test_matches_bfs_connected_components_on_random_graphs():
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(1, 60)
        edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(0, 80))]
        uf = UnionFind(range(n))
        for a, b in edges:
            uf.union(a, b)
        got = {frozenset(m) for m in uf.clusters().values()}
        assert got == bfs_components(n, edges)
        assert uf.num_clusters == len(got)


# ---------------------------------------------------------------------------
# Integration with the search classes
# ---------------------------------------------------------------------------

def make_grouped_vectors(n_groups=5, per_group=4, dim=48, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    ids, vectors, truth = [], [], {}
    for g in range(n_groups):
        center = rng.normal(size=dim)
        for m in range(per_group):
            item_id = f"g{g}_m{m}"
            ids.append(item_id)
            vectors.append(center + rng.normal(scale=noise, size=dim))
            truth[item_id] = g
    return ids, np.array(vectors), truth


def test_brute_force_pairs_produce_the_planted_groups():
    ids, vectors, truth = make_grouped_vectors()
    bf = BruteForceIndex()
    bf.insert_many(ids, vectors)

    uf = UnionFind(ids)
    uf.union_pairs(bf.find_all_duplicate_pairs(threshold=0.9))

    found = {frozenset(m) for m in uf.clusters().values()}
    expected = defaultdict(set)
    for item_id, g in truth.items():
        expected[g].add(item_id)
    assert found == {frozenset(s) for s in expected.values()}


def test_lsh_neighbours_produce_the_planted_groups():
    ids, vectors, truth = make_grouped_vectors()
    lsh = LSHIndex(dim=vectors.shape[1], num_tables=20, hash_size=8)
    lsh.insert_many(ids, vectors)

    uf = UnionFind(ids)
    for item_id, vec in zip(ids, vectors):
        for other, sim in lsh.query(vec, top_k=6, min_similarity=0.9):
            if other != item_id:
                uf.union(item_id, other)

    # Every planted group must be fully connected and never merged with another.
    for a in ids:
        for b in ids:
            assert uf.connected(a, b) == (truth[a] == truth[b])