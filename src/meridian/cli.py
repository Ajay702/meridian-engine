"""Command-line entry point.

    meridian ask  "How long do refunds take?" --corpus examples/corpus
    meridian eval examples/eval.jsonl --corpus examples/corpus --gate recall@k=0.8
    meridian serve --port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from meridian.core.config import EngineConfig
from meridian.engine import Engine
from meridian.evals.harness import load_dataset, run_eval


async def _ask(args: argparse.Namespace) -> int:
    engine = Engine(EngineConfig.load(args.config))
    stats = await engine.ingest_path(args.corpus)
    res = await engine.ask(args.question, k=args.k)
    print(f"\n{res.answer}\n")
    for i, s in enumerate(res.sources, start=1):
        sec = f" › {s['section']}" if s["section"] else ""
        print(f"  [{i}] {s['doc_id']}{sec}  (score {s['score']:.4f})")
    print(f"\n  model={res.model} latency={res.latency_ms}ms cost=${res.cost_usd:.6f} "
          f"chunks_indexed={sum(stats.values())} trace={res.trace_id[:12]}")
    if args.trace:
        spans = engine.tracer.get(res.trace_id)
        depth = {None: -1}
        for span in spans:
            depth[span["span_id"]] = depth.get(span["parent_id"], -1) + 1
            pad = "  " * depth[span["span_id"]]
            print(f"    {span['duration_ms']:>8.2f}ms  {pad}{span['name']}")
    await engine.aclose()
    return 0


async def _eval(args: argparse.Namespace) -> int:
    engine = Engine(EngineConfig.load(args.config))
    await engine.ingest_path(args.corpus)
    report = await run_eval(engine, load_dataset(args.dataset), k=args.k)
    print(json.dumps(report.summary, indent=2))
    if args.out:
        await asyncio.to_thread(Path(args.out).write_text, report.to_json())
    gates = dict(g.split("=") for g in args.gate)
    ok, failures = report.passes({k: float(v) for k, v in gates.items()})
    await engine.aclose()
    if not ok:
        print("EVAL GATE FAILED: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meridian")
    parser.add_argument("--config", default=None, help="path to meridian.toml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="ask a question over a local corpus")
    a.add_argument("question")
    a.add_argument("--corpus", default="examples/corpus")
    a.add_argument("-k", type=int, default=4)
    a.add_argument("--trace", action="store_true", help="print the span timeline")

    e = sub.add_parser("eval", help="run the eval harness (CI quality gate)")
    e.add_argument("dataset")
    e.add_argument("--corpus", default="examples/corpus")
    e.add_argument("-k", type=int, default=5)
    e.add_argument("--gate", action="append", default=[], help="metric=min, repeatable")
    e.add_argument("--out", default=None, help="write full JSON report here")

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)

    args = parser.parse_args(argv)
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("meridian.api.server:app", host=args.host, port=args.port)
        return 0
    return asyncio.run(_ask(args) if args.cmd == "ask" else _eval(args))


if __name__ == "__main__":
    raise SystemExit(main())
