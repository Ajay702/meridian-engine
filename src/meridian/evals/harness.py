"""Evaluation sub-engine: retrieval and answer-quality metrics + a regression harness.

Dataset format (JSONL), one case per line:
    {"question": "...", "relevant_docs": ["doc-a"], "answer_contains": ["30 days"]}

Metrics
    recall@k, MRR, nDCG@k   – retrieval quality against labelled relevant docs
    groundedness            – share of answer sentences supported by the context
    citation_validity       – answer cites only sources that exist
    answer_match            – required facts present in the answer
    latency p50/p95, cost   – the operational side of quality

`EvalReport.passes(thresholds)` turns a run into a CI gate, so a prompt or
retrieval change that regresses quality fails the build.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from meridian.guardrails.engine import validate_citations
from meridian.retrieval.embeddings import tokenize

if TYPE_CHECKING:
    from meridian.engine import Engine


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 1.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    seen: set[str] = set()
    dcg = 0.0
    for i, d in enumerate(retrieved[:k], start=1):
        if d in relevant and d not in seen:
            dcg += 1 / math.log2(i + 1)
            seen.add(d)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 1.0


_SENT = re.compile(r"(?<=[.!?])\s+")
_CITE = re.compile(r"\[\d+\]")


def groundedness(answer: str, contexts: list[str], min_overlap: float = 0.6) -> float:
    """Fraction of answer sentences whose content words are ≥`min_overlap` covered by
    some context passage. A lexical proxy for faithfulness — cheap enough to run on
    every request; use an LLM judge offline for the nuanced cases."""
    sents = [s for s in _SENT.split(_CITE.sub("", answer)) if tokenize(s)]
    if not sents:
        return 0.0
    ctx_terms = [set(tokenize(c)) for c in contexts]
    supported = 0
    for s in sents:
        terms = set(tokenize(s))
        best = max((len(terms & c) / len(terms) for c in ctx_terms), default=0.0)
        supported += best >= min_overlap
    return supported / len(sents)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))
    return s[idx]


@dataclass
class CaseResult:
    question: str
    answer: str
    retrieved_docs: list[str]
    recall: float
    mrr: float
    ndcg: float
    groundedness: float
    citation_valid: bool
    answer_match: float
    latency_ms: float
    cost_usd: float


@dataclass
class EvalReport:
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, float]:
        if not self.cases:
            return {}

        def mean(attr: str) -> float:
            return round(statistics.fmean(float(getattr(c, attr)) for c in self.cases), 4)

        lat = [c.latency_ms for c in self.cases]
        return {
            "n": len(self.cases),
            "recall@k": mean("recall"),
            "mrr": mean("mrr"),
            "ndcg@k": mean("ndcg"),
            "groundedness": mean("groundedness"),
            "citation_validity": mean("citation_valid"),
            "answer_match": mean("answer_match"),
            "latency_p50_ms": round(percentile(lat, 50), 2),
            "latency_p95_ms": round(percentile(lat, 95), 2),
            "total_cost_usd": round(sum(c.cost_usd for c in self.cases), 6),
        }

    def passes(self, thresholds: dict[str, float]) -> tuple[bool, list[str]]:
        s = self.summary
        failures = [f"{k}={s.get(k, 0):.3f} < {v}" for k, v in thresholds.items()
                    if s.get(k, 0.0) < v]
        return not failures, failures

    def to_json(self) -> str:
        return json.dumps({"summary": self.summary, "cases": [asdict(c) for c in self.cases]},
                          indent=2)


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


async def run_eval(engine: Engine, dataset: list[dict[str, Any]], *, k: int = 5,
                   concurrency: int = 8) -> EvalReport:
    sem = asyncio.Semaphore(concurrency)

    async def one(case: dict[str, Any]) -> CaseResult:
        async with sem:
            res = await engine.ask(case["question"], k=k)
        docs: list[str] = []
        for h in res.sources:
            if h["doc_id"] not in docs:
                docs.append(h["doc_id"])
        relevant = set(case.get("relevant_docs", []))
        must = [m.lower() for m in case.get("answer_contains", [])]
        hits = sum(m in res.answer.lower() for m in must)
        return CaseResult(
            question=case["question"],
            answer=res.answer,
            retrieved_docs=docs,
            recall=recall_at_k(docs, relevant, k),
            mrr=mrr(docs, relevant),
            ndcg=ndcg_at_k(docs, relevant, k),
            groundedness=groundedness(res.answer, [h["text"] for h in res.sources]),
            citation_valid=not [p for p in validate_citations(res.answer, len(res.sources))
                                if p.startswith("citation_out_of_range")],
            answer_match=hits / len(must) if must else 1.0,
            latency_ms=res.latency_ms,
            cost_usd=res.cost_usd,
        )

    return EvalReport(list(await asyncio.gather(*(one(c) for c in dataset))))
