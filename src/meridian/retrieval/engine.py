"""Retrieval sub-engine: ingest → chunk → index (lexical + dense) → hybrid search.

Search pipeline:
    1. Dense top-N and BM25 top-N candidates (metadata filter applied pre-scoring).
    2. Reciprocal Rank Fusion merges the two rankings — robust to their very
       different score scales, no tuning of weights required.
    3. Maximal Marginal Relevance re-ranks the fused list to trade relevance
       against redundancy, so the context window isn't spent on near-duplicates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from meridian.core.config import RetrievalConfig
from meridian.core.tracing import Tracer
from meridian.retrieval.chunking import Chunk, chunk_document
from meridian.retrieval.embeddings import Embedder, HashingEmbedder
from meridian.retrieval.indexes import BM25Index, VectorIndex


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.id,
            "doc_id": self.chunk.doc_id,
            "section": self.chunk.section,
            "text": self.chunk.text,
            "score": round(self.score, 6),
            "dense_rank": self.dense_rank,
            "lexical_rank": self.lexical_rank,
            "metadata": self.chunk.metadata,
        }


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """score(d) = Σ_r 1 / (k + rank_r(d)),  rank is 1-based (Cormack et al., 2009)."""
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            fused[key] += 1.0 / (k + rank)
    return dict(fused)


def mmr(query: np.ndarray, candidates: list[str], vectors: dict[str, np.ndarray],
        relevance: dict[str, float], k: int, lam: float = 0.7) -> list[str]:
    """Greedy MMR: argmax λ·rel(d) − (1−λ)·max_{s∈S} sim(d, s).

    `relevance` is min-max normalised so it lives on the same [0,1] scale as cosine.
    """
    if not candidates:
        return []
    rel = np.array([relevance[c] for c in candidates], dtype=np.float32)
    lo, hi = float(np.min(rel)), float(np.max(rel))
    span = hi - lo
    rel = (rel - lo) / span if span > 0 else np.ones_like(rel)
    mat = np.stack([vectors[c] for c in candidates])
    selected: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < k:
        if selected:
            redundancy = (mat[remaining] @ mat[selected].T).max(axis=1)
        else:
            redundancy = np.zeros(len(remaining), dtype=np.float32)
        scores = lam * rel[remaining] - (1 - lam) * redundancy
        best = remaining[int(np.argmax(scores))]
        selected.append(best)
        remaining.remove(best)
    return [candidates[i] for i in selected]


class RetrievalEngine:
    def __init__(self, config: RetrievalConfig | None = None, embedder: Embedder | None = None,
                 tracer: Tracer | None = None) -> None:
        self.config = config or RetrievalConfig()
        self.embedder = embedder or HashingEmbedder(self.config.embedding_dim)
        self.tracer = tracer or Tracer()
        self.bm25 = BM25Index()
        self.vectors = VectorIndex(self.embedder.dim)
        self.chunks: dict[str, Chunk] = {}
        self._doc_chunks: dict[str, list[str]] = defaultdict(list)

    # ------------------------------------------------------------------ ingest
    def ingest(self, doc_id: str, text: str, metadata: dict[str, Any] | None = None) -> int:
        """Upsert a document. Re-ingesting the same doc_id replaces its chunks atomically
        from the caller's point of view (old chunks are removed first)."""
        self.delete(doc_id)
        chunks = chunk_document(text, doc_id, chunk_size=self.config.chunk_size,
                                overlap=self.config.chunk_overlap, metadata=metadata)
        if not chunks:
            return 0
        vecs = self.embedder.embed([c.indexed_text for c in chunks])
        self.vectors.add([c.id for c in chunks], vecs)
        for c in chunks:
            self.bm25.add(c.id, c.indexed_text)
            self.chunks[c.id] = c
            self._doc_chunks[doc_id].append(c.id)
        return len(chunks)

    def delete(self, doc_id: str) -> int:
        ids = self._doc_chunks.pop(doc_id, [])
        for cid in ids:
            self.bm25.remove(cid)
            self.vectors.remove(cid)
            self.chunks.pop(cid, None)
        return len(ids)

    @property
    def stats(self) -> dict[str, int]:
        return {"documents": len(self._doc_chunks), "chunks": len(self.chunks)}

    # ------------------------------------------------------------------ search
    async def search(self, query: str, k: int | None = None,
                     filters: dict[str, Any] | None = None) -> list[Hit]:
        k = k or self.config.top_k
        n = max(self.config.candidate_k, k)
        allow = self._make_filter(filters) if filters else None

        async with self.tracer.span("retrieval.search", k=k) as span:
            q = self.embedder.embed([query])[0]
            dense = self.vectors.search(q, n, allow)
            lexical = self.bm25.search(query, n, allow)
            dense_rank = {key: i for i, (key, _) in enumerate(dense, start=1)}
            lex_rank = {key: i for i, (key, _) in enumerate(lexical, start=1)}

            fused = reciprocal_rank_fusion([[key for key, _ in dense],
                                            [key for key, _ in lexical]], k=self.config.rrf_k)
            ordered = sorted(fused, key=lambda key: -fused[key])
            vecs = {key: self.vectors.vector(key) for key in ordered}
            final = mmr(q, ordered, vecs, fused, k, self.config.mmr_lambda)
            span.set(dense=len(dense), lexical=len(lexical), returned=len(final))

        return [Hit(self.chunks[key], fused[key], dense_rank.get(key), lex_rank.get(key))
                for key in final]

    def _make_filter(self, filters: dict[str, Any]):
        def allow(chunk_id: str) -> bool:
            meta = self.chunks[chunk_id].metadata
            for field, want in filters.items():
                have = self.chunks[chunk_id].doc_id if field == "doc_id" else meta.get(field)
                if isinstance(want, list | tuple | set):
                    if have not in want:
                        return False
                elif have != want:
                    return False
            return True
        return allow
