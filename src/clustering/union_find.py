"""
Phase 5: Union-Find (Disjoint Set Union) for duplicate-question clustering.

Problem: LSH + similarity gives you PAIRS ("A ~ B", "B ~ C"). You want
CLUSTERS, where A, B and C all end up in one group because duplicates are
transitive. Union-Find maintains those groups incrementally:

    union(a, b)      -> merge the groups containing a and b
    find(a)          -> the representative ("root") of a's group
    connected(a, b)  -> True if a and b are in the same group

Optimizations (together they make each operation nearly O(1), formally
O(alpha(n)), where alpha is the inverse Ackermann function):
  - Path compression: find() re-points every node it visits directly at the
    root, so later lookups are shorter.
  - Union by size: the smaller group is attached under the larger one, which
    keeps the trees shallow.

Extras beyond the textbook version (needed for the API in Phase 6):
  - Works with arbitrary hashable ids (question ids, strings, ints...).
  - Keeps a member list per group, merged small-into-large, so
    "give me all duplicates of question X" is O(group size) instead of a
    scan over every question.

Limitations:
  - Groups can only merge, never split. If one bad pair (a false positive
    from the similarity threshold) joins two groups, the merge cannot be
    undone. Removing an item is also unsupported: mark it deleted in your
    database, or rebuild the structure from the remaining pairs.
  - Transitivity can chain: A~B and B~C merge A and C even if A and C are
    not similar. Use a fairly strict threshold when feeding pairs in.
"""

from typing import Hashable, Iterable


class UnionFind:
    def __init__(self, items: Iterable[Hashable] = ()):
        self._parent: dict = {}    # item -> parent item (a root points to itself)
        self._members: dict = {}   # root -> list of every item in its group
        for item in items:
            self.add(item)

    # ------------------------------------------------------------------
    # Building the structure
    # ------------------------------------------------------------------
    def add(self, item: Hashable) -> bool:
        """Add `item` as its own group. Returns False if it already exists."""
        if item in self._parent:
            return False
        self._parent[item] = item
        self._members[item] = [item]
        return True

    def find(self, item: Hashable) -> Hashable:
        """Return the root (group representative) of `item`, compressing the path."""
        if item not in self._parent:
            raise KeyError(f"Unknown item: {item!r} (call add() first)")

        # Pass 1: walk up to the root.
        root = item
        while self._parent[root] != root:
            root = self._parent[root]

        # Pass 2: point every node on the path straight at the root.
        while item != root:
            next_item = self._parent[item]
            self._parent[item] = root
            item = next_item
        return root

    def union(self, a: Hashable, b: Hashable) -> bool:
        """
        Merge the groups of `a` and `b`.
        Returns True if two different groups were merged, False if they
        were already in the same group.
        """
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False

        # Union by size: the smaller group goes under the larger one.
        if len(self._members[root_a]) < len(self._members[root_b]):
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        self._members[root_a].extend(self._members.pop(root_b))
        return True

    def union_pairs(self, pairs: Iterable, add_missing: bool = False) -> int:
        """
        Union many pairs. Each pair is a tuple whose first two entries are
        the ids; extra entries (e.g. a similarity score) are ignored, so the
        output of BruteForceIndex.find_all_duplicate_pairs() works directly.
        Returns the number of merges that actually happened.
        """
        merges = 0
        for pair in pairs:
            a, b = pair[0], pair[1]
            if add_missing:
                self.add(a)
                self.add(b)
            if self.union(a, b):
                merges += 1
        return merges

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------
    def connected(self, a: Hashable, b: Hashable) -> bool:
        """True if `a` and `b` are in the same group."""
        return self.find(a) == self.find(b)

    def group_size(self, item: Hashable) -> int:
        """Number of items in `item`'s group (including itself)."""
        return len(self._members[self.find(item)])

    def members(self, item: Hashable) -> list:
        """All items in `item`'s group (including itself). O(group size)."""
        return list(self._members[self.find(item)])

    def clusters(self, min_size: int = 1) -> dict:
        """
        Return {root: [members]} for every group with at least `min_size`
        items. Use min_size=2 to list only groups that actually contain
        duplicates. Roots are only stable until the next union().
        """
        return {
            root: list(members)
            for root, members in self._members.items()
            if len(members) >= min_size
        }

    @property
    def num_clusters(self) -> int:
        return len(self._members)

    def __len__(self) -> int:
        """Total number of items (not groups)."""
        return len(self._parent)

    def __contains__(self, item: Hashable) -> bool:
        return item in self._parent

    def __repr__(self) -> str:
        return f"UnionFind(items={len(self)}, clusters={self.num_clusters})"

    # ------------------------------------------------------------------
    # Rebuilding from stored clusters (e.g. loaded from your database)
    # ------------------------------------------------------------------
    @classmethod
    def from_clusters(cls, clusters: Iterable[Iterable[Hashable]]) -> "UnionFind":
        """
        Rebuild from an iterable of groups, e.g. `uf.clusters().values()`,
        or rows grouped by cluster_id when loading from a database.
        """
        uf = cls()
        for group in clusters:
            group = list(group)
            for item in group:
                uf.add(item)
            for item in group[1:]:
                uf.union(group[0], item)
        return uf


# ---------------------------------------------------------------------------
# Demo (run: python -m src.clustering.union_find)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uf = UnionFind(range(1, 9))          # 8 questions, ids 1..8
    print(uf)

    # Pairs found by LSH + similarity threshold:
    pairs = [(1, 2, 0.93), (2, 3, 0.88), (5, 6, 0.91), (7, 8, 0.85), (6, 7, 0.82)]
    merges = uf.union_pairs(pairs)
    print(f"{merges} merges ->", uf)

    print("1 and 3 connected (never paired directly)?", uf.connected(1, 3))
    print("1 and 5 connected?", uf.connected(1, 5))
    print("Duplicates of question 8:", uf.members(8))
    print("Groups with duplicates:", list(uf.clusters(min_size=2).values()))