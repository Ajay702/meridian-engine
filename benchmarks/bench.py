"""Reproducible benchmarks. Run: python benchmarks/bench.py

1. Chaos failover — primary provider fails 30% of calls (+ 40ms latency) while a
   backup is healthy. Compares availability with failover on vs. off.
2. Retrieval latency — hybrid search (dense + BM25 + RRF + MMR) over a synthetic
   corpus of N chunks; reports p50/p95 per query.
3. Semantic cache — hit rate and latency on a paraphrase-heavy workload.
"""

from __future__ import annotations

import asyncio
import random
import statistics
import time

from meridian.core.config import GatewayConfig, ModelConfig
from meridian.core.errors import MeridianError
from meridian.core.types import CompletionRequest, Message
from meridian.evals.harness import percentile
from meridian.gateway.gateway import ModelGateway
from meridian.gateway.providers.mock import MockProvider
from meridian.retrieval.engine import RetrievalEngine

MODELS = [
    ModelConfig(name="primary-model", provider="primary", input_per_mtok=3, output_per_mtok=15,
                quality=0.9),
    ModelConfig(name="backup-model", provider="backup", input_per_mtok=0.8, output_per_mtok=4,
                quality=0.7),
]


def gateway() -> ModelGateway:
    g = ModelGateway(MODELS, GatewayConfig(cache_enabled=False, max_retries=1,
                                           breaker_failure_threshold=5, breaker_reset_s=0.5))
    g.register_provider(MockProvider("primary", failure_rate=0.3, latency_ms=40, seed=7),
                        rpm=10**6)
    g.register_provider(MockProvider("backup", latency_ms=60, seed=8), rpm=10**6)
    return g


async def run_chaos(fallback: bool, n: int, concurrency: int) -> tuple[int, list[float], float]:
    g = gateway()
    sem = asyncio.Semaphore(concurrency)
    lat: list[float] = []

    async def one(i: int) -> bool:
        req = CompletionRequest(messages=[Message(role="user", content=f"q{i}")],
                                model="primary-model", metadata={"allow_fallback": fallback})
        async with sem:
            t0 = time.perf_counter()
            try:
                await g.complete(req)
            except MeridianError:
                return False
            lat.append((time.perf_counter() - t0) * 1000)
            return True

    t0 = time.perf_counter()
    ok = sum(await asyncio.gather(*(one(i) for i in range(n))))
    return ok, lat, time.perf_counter() - t0


async def chaos(n: int = 1000, concurrency: int = 50) -> None:
    print(f"\n== chaos failover: {n} requests, primary failure_rate=0.30 ==")
    for fallback in (False, True):
        ok, lat, wall = await run_chaos(fallback, n, concurrency)
        label = "failover ON " if fallback else "failover OFF"
        print(f"  {label}: success {ok / n:6.1%}  p50 {percentile(lat, 50):6.1f}ms  "
              f"p95 {percentile(lat, 95):6.1f}ms  throughput {n / wall:6.0f} req/s")


WORDS = ("invoice refund latency cluster shard replica token budget tenant quota region "
         "failover cache vector index policy audit encryption key rotation backup restore "
         "deploy rollout canary alert incident pager schema migration queue worker").split()


async def retrieval(n_docs: int = 2000) -> None:
    rng = random.Random(0)
    r = RetrievalEngine()
    t0 = time.perf_counter()
    for d in range(n_docs):
        paras = [" ".join(rng.choices(WORDS, k=60)) + "." for _ in range(3)]
        r.ingest(f"doc-{d}", f"# Doc {d}\n\n" + "\n\n".join(paras))
    ingest_s = time.perf_counter() - t0
    lat = []
    for _ in range(200):
        q = " ".join(rng.choices(WORDS, k=5))
        t = time.perf_counter()
        await r.search(q, k=5)
        lat.append((time.perf_counter() - t) * 1000)
    chunks = r.stats["chunks"]
    print(f"\n== hybrid retrieval: {chunks} chunks ==")
    print(f"  ingest {chunks / ingest_s:,.0f} chunks/s   "
          f"search p50 {statistics.median(lat):.2f}ms  p95 {percentile(lat, 95):.2f}ms")


async def cache(n: int = 500) -> None:
    g = ModelGateway(MODELS[1:], GatewayConfig(cache_threshold=0.9))
    p = MockProvider("backup", latency_ms=30)
    g.register_provider(p, rpm=10**6)
    rng = random.Random(1)
    base = ["How do I rotate my API key", "What is the refund window",
            "How do I add a teammate", "Where are audit logs stored"]
    variants = [lambda s: s, lambda s: s.lower(), lambda s: s + "?", lambda s: s.lower() + "?"]
    lat = []
    for _ in range(n):
        q = rng.choice(variants)(rng.choice(base))
        t = time.perf_counter()
        await g.complete(CompletionRequest(messages=[Message(role="user", content=q)]))
        lat.append((time.perf_counter() - t) * 1000)
    hits = n - p.calls
    print(f"\n== semantic cache: {n} paraphrased requests over 4 intents ==")
    print(f"  hit rate {hits / n:.1%}  provider calls {p.calls}  "
          f"mean latency {statistics.fmean(lat):.2f}ms (uncached ≈30ms)")


async def main() -> None:
    await chaos()
    await retrieval()
    await cache()


if __name__ == "__main__":
    asyncio.run(main())
