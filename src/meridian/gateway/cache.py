"""Two-tier semantic response cache.

Tier 1: exact match on a canonical hash of (model, messages, temperature, tools).
Tier 2: approximate match — cosine similarity of the *question* embedding against
        cached entries in the same scope, above `threshold`.

Scoping matters more than the threshold. Embedding the whole prompt is a classic
false-hit bug: in RAG the retrieved context dominates the text, so two different
questions over the same passages look ~identical. Instead the semantic key is only
the final user turn (or `metadata["semantic_key"]`), and everything else — tenant,
model, prior turns, plus an optional `metadata["semantic_scope"]` such as the
retrieved chunk ids — must match exactly.

Only deterministic requests (temperature == 0, no tools) are cached, since a
semantic hit on a sampled or tool-using request would be wrong. Entries carry a
TTL and the store is LRU-bounded.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from meridian.core.types import CompletionRequest, CompletionResponse
from meridian.retrieval.embeddings import Embedder, HashingEmbedder


@dataclass
class _Entry:
    key: str
    scope: str
    vector: np.ndarray
    response: CompletionResponse
    expires_at: float


class SemanticCache:
    def __init__(
        self,
        embedder: Embedder | None = None,
        threshold: float = 0.95,
        ttl_s: float = 3600,
        capacity: int = 10_000,
        clock=time.monotonic,
    ) -> None:
        self.embedder = embedder or HashingEmbedder(256)
        self.threshold = threshold
        self.ttl_s = ttl_s
        self.capacity = capacity
        self._clock = clock
        self._entries: OrderedDict[str, _Entry] = OrderedDict()

    @staticmethod
    def cacheable(req: CompletionRequest) -> bool:
        return req.cacheable and req.temperature == 0 and not req.tools

    @staticmethod
    def _scope(req: CompletionRequest, model: str) -> str:
        prior = [m.model_dump(exclude_none=True) for m in req.messages[:-1]]
        extra = req.metadata.get("semantic_scope", "")
        digest = hashlib.sha256(json.dumps([prior, extra], sort_keys=True).encode()).hexdigest()
        return f"{req.tenant}:{model}:{digest[:24]}"

    @staticmethod
    def _semantic_text(req: CompletionRequest) -> str:
        if key := req.metadata.get("semantic_key"):
            return str(key)
        return req.messages[-1].content if req.messages else ""

    @staticmethod
    def exact_key(req: CompletionRequest, model: str) -> str:
        canon = json.dumps(
            {
                "m": model,
                "t": req.tenant,
                "msgs": [m.model_dump(exclude_none=True) for m in req.messages],
                "max": req.max_tokens,
            },
            sort_keys=True,
        )
        return hashlib.sha256(canon.encode()).hexdigest()

    def _evict_expired(self) -> None:
        now = self._clock()
        for k in [k for k, e in self._entries.items() if e.expires_at <= now]:
            del self._entries[k]

    def get(self, req: CompletionRequest, model: str) -> tuple[CompletionResponse, str] | None:
        """Returns (response, 'exact'|'semantic') or None."""
        if not self.cacheable(req):
            return None
        self._evict_expired()
        key = self.exact_key(req, model)
        if (hit := self._entries.get(key)) is not None:
            self._entries.move_to_end(key)
            return hit.response, "exact"

        scope = self._scope(req, model)
        candidates = [e for e in self._entries.values() if e.scope == scope]
        if not candidates:
            return None
        q = self.embedder.embed([self._semantic_text(req)])[0]
        matrix = np.stack([e.vector for e in candidates])
        sims = matrix @ q
        best = int(np.argmax(sims))
        if float(sims[best]) >= self.threshold:
            entry = candidates[best]
            self._entries.move_to_end(entry.key)
            return entry.response, "semantic"
        return None

    def put(self, req: CompletionRequest, model: str, resp: CompletionResponse) -> None:
        if not self.cacheable(req):
            return
        key = self.exact_key(req, model)
        vec = self.embedder.embed([self._semantic_text(req)])[0]
        self._entries[key] = _Entry(
            key, self._scope(req, model), vec, resp, self._clock() + self.ttl_s
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)
