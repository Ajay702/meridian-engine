"""End-to-end HTTP load test against a running `meridian serve` (offline mock provider).

    meridian --config meridian.example.toml serve --port 8080 &
    python benchmarks/http_load.py --url http://127.0.0.1:8080 --requests 3000 --concurrency 32

Ingests examples/corpus over HTTP, then fires the labelled questions in
examples/eval.jsonl at POST /v1/ask and checks every answer, not just the status code.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


async def main(url, n, conc, label):
    lines = (ROOT / "examples/eval.jsonl").read_text().splitlines()
    qs = [json.loads(line) for line in lines if line.strip()]
    async with httpx.AsyncClient(base_url=url, timeout=30,
                                 limits=httpx.Limits(max_connections=conc)) as c:
        for p in sorted((ROOT / "examples/corpus").glob("*.md")):
            doc = {"id": p.stem, "text": p.read_text()}
            (await c.post("/v1/documents", json=doc)).raise_for_status()
        lat, ok, correct = [], 0, 0
        sem = asyncio.Semaphore(conc)

        async def one(i):
            nonlocal ok, correct
            q = qs[i % len(qs)]
            async with sem:
                t = time.perf_counter()
                r = await c.post("/v1/ask", json={"question": q["question"]})
                lat.append((time.perf_counter() - t) * 1e3)
            if r.status_code == 200:
                ok += 1
                ans = r.json().get("answer", "")
                correct += all(s.lower() in ans.lower() for s in q["answer_contains"])

        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(n)))
        wall = time.perf_counter() - t0
    lat.sort()
    print(f"{label:<14} {n:>6} req  conc {conc:>3}  {n / wall:>7,.0f} req/s  "
          f"p50 {lat[n // 2]:6.1f} ms  p95 {lat[int(n * .95)]:6.1f} ms  "
          f"p99 {lat[int(n * .99)]:6.1f} ms  "
          f"2xx {ok / n:.1%}  correct {correct / n:.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--requests", type=int, default=3000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--label", default="ask")
    a = ap.parse_args()
    asyncio.run(main(a.url, a.requests, a.concurrency, a.label))
