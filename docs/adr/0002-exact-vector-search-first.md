# ADR-0002: Exact vector search first, ANN behind the same interface later

**Status:** accepted

## Context
Approximate nearest-neighbour indexes (HNSW, IVF) add build time, memory overhead, recall/latency tuning knobs and non-deterministic results.

## Decision
`VectorIndex` does exact cosine search with one matrix-vector product over a contiguous float32 matrix. Deletes are tombstoned and compacted lazily.

## Consequences
- 100% recall and deterministic results, which keeps tests and evals stable.
- Measured p50 is about 5 ms for a full hybrid query over ~7k chunks (see benchmarks). Exact search stays practical up to around 10⁶ vectors per node.
- Past that, an HNSW or pgvector backend implements the same `add` / `remove` / `search(query, k, allow)` interface. The pre-scoring `allow` filter has to be kept (as filtered HNSW) or filtered queries will lose recall.
