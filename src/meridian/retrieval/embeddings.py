"""Embedding backends.

`HashingEmbedder` is the default: a signed feature-hashing embedder over word
unigrams, bigrams and character trigrams. It needs no model download, is
deterministic across processes, and is surprisingly strong for lexical-semantic
matching — good enough to make the whole engine run (and be tested) offline.
Swap in `ProviderEmbedder` for a hosted dense model in production.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Awaitable, Callable, Sequence
from itertools import pairwise
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or that the this to "
    "was were what when where which who why will with you your do does did can".split()
)


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    return [t for t in toks if t not in STOPWORDS] if drop_stopwords else toks


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def _h(feature: str) -> int:
    return int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = tokenize(text)
        feats: list[tuple[str, float]] = [(f"w:{w}", 1.0) for w in words]
        feats += [(f"b:{a}_{b}", 0.7) for a, b in pairwise(words)]
        for w in words:
            padded = f"#{w}#"
            feats += [(f"c:{padded[i:i + 3]}", 0.35) for i in range(len(padded) - 2)]
        return feats

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text):
                h = _h(feat)
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[row, h % self.dim] += sign * weight
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class ProviderEmbedder:
    """Adapter for a hosted embedding endpoint (async fn: list[str] -> list[list[float]])."""

    def __init__(self, fn: Callable[[list[str]], Awaitable[list[list[float]]]], dim: int) -> None:
        self._fn = fn
        self.dim = dim

    async def aembed(self, texts: Sequence[str]) -> np.ndarray:
        vecs = np.asarray(await self._fn(list(texts)), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - sync shim
        raise NotImplementedError("use aembed() for remote embedders")
