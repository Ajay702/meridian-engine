# Architecture

## Goals

1. **Every LLM call goes through one place.** Routing, cost, rate limits and failover live there, not in each feature.
2. **Failures degrade, they don't cascade.** A slow provider, a dead session store or a flaky tool should cost quality, not availability.
3. **Every request can be observed.** A single trace id explains why an answer came out the way it did: which chunks, which model, what it cost, what was redacted.
4. **Quality is measured in CI.** A prompt or retrieval change that hurts answers fails the build.

## Layers

```
┌──────────────────────────────────────────────────────────────┐
│ API layer         FastAPI · tenancy header · error → status   │
├──────────────────────────────────────────────────────────────┤
│ Main engine       composes sub-engines into DAG pipelines     │
│                   owns Tracer · Metrics · EventBus            │
├───────────┬───────────┬────────────┬────────────┬────────────┤
│ Gateway   │ Retrieval │ Guardrails │ Agents     │ Evals      │
│ (sub-eng) │ (sub-eng) │ (sub-eng)  │ (sub-eng)  │ (sub-eng)  │
├───────────┴───────────┴────────────┴────────────┴────────────┤
│ Core              types · errors · config · pipeline · tracing│
└──────────────────────────────────────────────────────────────┘
```

Dependencies only point downward. Agents depend on the gateway *interface* (they need to call a model). Everything else is a sibling that only the main engine wires together.

## Main engine: the DAG scheduler

`Pipeline` ([core/pipeline.py](../src/meridian/core/pipeline.py)) is a small workflow runner, roughly a single-process Airflow or Temporal:

- **Eager scheduling.** Each node is an `asyncio.Task` that waits on its dependencies' `Event`s. Independent branches really do run at the same time, and a slow branch never holds up an unrelated one. (Level-by-level topological execution would.)
- **Failure semantics.**
  - A *required* node that fails raises `PipelineError` and cancels every in-flight sibling, so no orphan tasks keep spending tokens.
  - An *optional* node that fails is recorded in `ctx.errors`. Its *hard* dependents are skipped.
  - A *soft* dependency is waited on, but its failure doesn't skip the dependent. That is how `generate` uses `memory`.
- **Per-node policy.** Timeout (`asyncio.wait_for`), retries with jittered backoff, and a span named `<pipeline>.<node>`.

## Model gateway

The steps for each request, in order:

1. **Budget check.** A tenant over its USD budget is rejected before any work is done (402).
2. **Rank.** Filter models whose provider circuit is open or whose context window is too small, then sort by policy:
   - `quality`: highest quality first, cost as the tie-breaker
   - `cost`: cheapest estimated cost (prompt tokens + expected output)
   - `latency`: lowest EWMA latency (α=0.2) for that model
   - a pinned `model` goes first, and the rest stay as fallbacks unless `allow_fallback=false`.
3. **Cache lookup** for the top-ranked model (exact, then semantic).
4. **Walk the chain.** For each candidate: token-bucket acquire → circuit breaker → retry (retryable errors only, full jitter, never retrying an open circuit) → provider.
   - Retryable or circuit-open errors fail over to the next candidate and publish `gateway.failover`.
   - Fatal errors (4xx, bad request) are raised immediately. Failing over a malformed request would just repeat the error on every provider and pay for each attempt.
5. **Account.** Price the response with the model's real token usage, add it to the tenant's spend, emit metrics, cache the result, publish `gateway.completion`.

### Circuit breaker

```
          failures ≥ threshold
 CLOSED ─────────────────────────► OPEN
   ▲                                 │ reset_timeout elapsed
   │ probe succeeds                  ▼
   └──────────────────────────── HALF_OPEN ── probe fails ──► OPEN
```

Only `half_open_max_calls` probes get through in the half-open state, so a recovering provider isn't flooded the moment it comes back.

## Retrieval

```
document ─► recursive chunker (keeps heading path) ─┬─► HashingEmbedder ─► VectorIndex
                                                    └─► tokenizer ───────► BM25Index

query ─► dense top-N ┐
       ─► BM25 top-N ┴─► RRF fuse ─► MMR re-rank ─► top-k Hits
```

- **Why hybrid?** Dense retrieval misses exact identifiers (error codes, SKUs, "99.95"). BM25 misses paraphrases. RRF merges them by rank, so you don't need to tune score scales.
- **Why MMR?** Overlapping chunks are often near-duplicates. Without MMR, top-k can spend the whole context window on one passage.
- **Upserts.** Re-ingesting a `doc_id` removes its old chunks from both indexes first. Vector deletes are tombstoned and compacted lazily once 30% of rows are dead.
- **Filters run before scoring**, so a filtered query still returns a full top-k instead of whatever survived truncation.

## Guardrails

The input stage runs checks in order and stops at the first `BLOCK`: length → injection → PII redaction. Blocking comes before redaction so an attack payload is never stored in a redacted form. The output stage redacts PII again, because the model can repeat PII from retrieved context, and validates `[n]` citations against the number of passages actually supplied.

The injection detector is a weighted-signal heuristic with an exposed score. It is meant as a cheap first filter. A classifier model can handle the uncertain middle band using the same `Check` protocol.

## Agent runtime

Tools are plain functions. The JSON schema the model sees and the pydantic validator that checks its arguments are both built from the same type hints, so they can't drift apart. Each loop step:

1. call the gateway with the tool specs
2. no tool calls → done
3. check every call against the loop detector (same name and arguments more than N times → stop)
4. run all calls concurrently, and return errors (unknown tool, bad arguments, exception, timeout) as observations
5. stop on step, time or cost limits

The returned `AgentResult` has a `stop_reason` and the full step transcript for replay.

## Scaling out

| Concern | Single node (today) | Multi-replica |
|---|---|---|
| Semantic cache | in-process LRU | Redis (vector search) or a pgvector table |
| Session memory | in-process deque | Redis lists with TTL |
| Rate limits | in-process token bucket | Redis sliding window per provider key |
| Circuit breaker | per process | per process is fine, since each replica learns on its own |
| Vector index | exact numpy | HNSW / pgvector / a managed vector DB behind `VectorIndex` |
| Tracing | in-memory ring | OTLP exporter → Tempo/Jaeger |
| Ingestion | inline | queue + worker so the API never blocks on embedding |

Every row in the right-hand column replaces one class behind an existing interface. No call sites change, and that is the main reason for the sub-engine boundaries.
