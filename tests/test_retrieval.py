import numpy as np

from meridian.retrieval.chunking import chunk_document
from meridian.retrieval.embeddings import HashingEmbedder
from meridian.retrieval.engine import RetrievalEngine, mmr, reciprocal_rank_fusion
from meridian.retrieval.indexes import BM25Index, VectorIndex


def test_chunker_respects_size_overlap_and_sections():
    text = "# Guide\n\n## Install\n" + ("Run the installer now. " * 40) + "\n\n## Usage\nCall it."
    chunks = chunk_document(text, "g", chunk_size=200, overlap=40)
    assert len(chunks) > 3
    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[1].section == "Guide > Install"
    assert {c.section for c in chunks} <= {"Guide", "Guide > Install", "Guide > Usage"}
    tail = chunk_document("# A\n\n## B\nshort.\n\n## C\n" + "word " * 60, "d", chunk_size=120,
                          overlap=10)
    assert tail[-1].section == "A > C"
    assert len({c.id for c in chunks}) == len(chunks)


def test_embedder_deterministic_and_normalised():
    e = HashingEmbedder(128)
    a, b = e.embed(["refund policy", "refund policy"])
    assert np.allclose(a, b) and np.isclose(np.linalg.norm(a), 1)
    q, near, far = e.embed(["card refund time", "how long do card refunds take", "AES keys"])
    assert q @ near > q @ far


def test_bm25_ranks_and_removes():
    idx = BM25Index()
    idx.add("a", "refund card business days")
    idx.add("b", "encryption keys rotated")
    assert idx.search("card refund", 5)[0][0] == "a"
    idx.remove("a")
    assert idx.search("card refund", 5) == []
    assert len(idx) == 1


def test_vector_index_upsert_and_filter():
    e = HashingEmbedder(64)
    vi = VectorIndex(64)
    vi.add(["x", "y"], e.embed(["alpha beta", "gamma delta"]))
    vi.add(["x"], e.embed(["gamma delta epsilon"]))  # upsert
    assert len(vi) == 2
    res = vi.search(e.embed(["gamma delta"])[0], 2, allow=lambda k: k == "x")
    assert [k for k, _ in res] == ["x"]


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert set(sorted(fused, key=fused.get, reverse=True)[:2]) == {"a", "b"}
    assert fused["d"] < fused["a"]


def test_mmr_prefers_diversity():
    v = {"a": np.array([1, 0], np.float32), "a2": np.array([1, 0], np.float32),
         "b": np.array([0, 1], np.float32)}
    picked = mmr(np.array([1, 0], np.float32), ["a", "a2", "b"], v,
                 {"a": 1.0, "a2": 0.99, "b": 0.5}, k=2, lam=0.5)
    assert picked == ["a", "b"]


async def test_hybrid_search_filters_and_reingest():
    r = RetrievalEngine()
    r.ingest("refunds", "Card refunds take 5 to 10 business days.", {"team": "billing"})
    r.ingest("security", "Keys are rotated every 90 days.", {"team": "secops"})
    hits = await r.search("how long for a card refund", k=2)
    assert hits[0].chunk.doc_id == "refunds"
    hits = await r.search("card refund", k=2, filters={"team": "secops"})
    assert all(h.chunk.doc_id == "security" for h in hits)
    r.ingest("refunds", "Refunds are instant now.")
    assert r.stats == {"documents": 2, "chunks": 2}
    assert "instant" in (await r.search("refund", k=1))[0].chunk.text
