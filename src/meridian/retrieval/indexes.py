"""Index structures: Okapi BM25 (inverted index) and a flat cosine vector index.

Both support incremental `add` and `remove` so documents can be re-ingested
(upserted) without rebuilding, and both accept a metadata filter that is applied
*before* scoring rather than after truncation — so filtered queries still get a
full top-k.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable

import numpy as np

from meridian.retrieval.embeddings import tokenize

Filter = Callable[[str], bool]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._doc_len: dict[str, int] = {}
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._doc_len)

    def add(self, key: str, text: str) -> None:
        if key in self._doc_len:
            self.remove(key)
        tf = Counter(tokenize(text))
        for term, n in tf.items():
            self._postings[term][key] = n
        length = sum(tf.values())
        self._doc_len[key] = length
        self._total_len += length

    def remove(self, key: str) -> None:
        length = self._doc_len.pop(key, None)
        if length is None:
            return
        self._total_len -= length
        for term in [t for t, p in self._postings.items() if key in p]:
            del self._postings[term][key]
            if not self._postings[term]:
                del self._postings[term]

    def search(self, query: str, k: int, allow: Filter | None = None) -> list[tuple[str, float]]:
        n = len(self._doc_len)
        if n == 0:
            return []
        avgdl = self._total_len / n
        scores: dict[str, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for key, tf in postings.items():
                if allow and not allow(key):
                    continue
                dl = self._doc_len[key]
                scores[key] += idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                )
        return sorted(scores.items(), key=lambda kv: -kv[1])[:k]


class VectorIndex:
    """Exact (flat) cosine search over L2-normalised vectors.

    Rows live in a contiguous float32 matrix; deletes are tombstoned and compacted
    lazily. Exact search is the right call up to ~1M vectors on one node — beyond
    that swap in an ANN backend behind the same interface (see ADR-0002).
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._matrix = np.zeros((0, dim), dtype=np.float32)
        self._keys: list[str] = []
        self._row: dict[str, int] = {}
        self._dead: set[int] = set()

    def __len__(self) -> int:
        return len(self._row)

    def add(self, keys: list[str], vectors: np.ndarray) -> None:
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vectors.shape[1]}")
        for k in keys:
            self.remove(k)
        start = len(self._keys)
        self._matrix = np.vstack([self._matrix, vectors.astype(np.float32)])
        for i, k in enumerate(keys):
            self._keys.append(k)
            self._row[k] = start + i
        if len(self._dead) > 0.3 * len(self._keys):
            self._compact()

    def remove(self, key: str) -> None:
        row = self._row.pop(key, None)
        if row is not None:
            self._dead.add(row)

    def _compact(self) -> None:
        live = [i for i in range(len(self._keys)) if i not in self._dead]
        self._matrix = self._matrix[live]
        self._keys = [self._keys[i] for i in live]
        self._row = {k: i for i, k in enumerate(self._keys)}
        self._dead.clear()

    def vector(self, key: str) -> np.ndarray:
        return self._matrix[self._row[key]]

    def search(self, query: np.ndarray, k: int,
               allow: Filter | None = None) -> list[tuple[str, float]]:
        if not self._row:
            return []
        sims = self._matrix @ query.astype(np.float32)
        if self._dead:
            sims[list(self._dead)] = -np.inf
        if allow:
            mask = np.array([allow(key) for key in self._keys])
            sims[~mask] = -np.inf
        k = min(k, len(self._row))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(self._keys[i], float(sims[i])) for i in top if np.isfinite(sims[i])]
