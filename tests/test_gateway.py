import pytest

from meridian.core.errors import BudgetExceededError, NoHealthyProviderError
from meridian.core.types import CompletionRequest, Message, RoutingPolicy
from meridian.gateway.providers.mock import MockProvider
from meridian.gateway.resilience import BreakerState

from .conftest import two_model_gateway


def req(text="hello", **kw):
    return CompletionRequest(messages=[Message(role="user", content=text)], **kw)


async def test_policy_ranking():
    g = two_model_gateway(MockProvider("a"), MockProvider("b"))
    assert [m.name for m in g.rank(req(policy=RoutingPolicy.QUALITY))] == ["big", "small"]
    assert [m.name for m in g.rank(req(policy=RoutingPolicy.COST))] == ["small", "big"]
    g.latency.observe("big", 50)
    g.latency.observe("small", 900)
    assert g.rank(req(policy=RoutingPolicy.LATENCY))[0].name == "big"


async def test_failover_and_circuit_opening():
    primary = MockProvider("a", failure_rate=1.0)
    backup = MockProvider("b")
    g = two_model_gateway(primary, backup, max_retries=0, breaker_failure_threshold=2,
                          cache_enabled=False)
    r1 = await g.complete(req("one"))
    assert r1.model == "small" and r1.provider == "b"
    await g.complete(req("two"))
    assert g.breaker("a").state is BreakerState.OPEN
    calls_before = primary.calls
    r3 = await g.complete(req("three"))
    assert r3.provider == "b" and primary.calls == calls_before  # short-circuited


async def test_all_fail_raises():
    g = two_model_gateway(MockProvider("a", failure_rate=1.0), MockProvider("b", failure_rate=1.0),
                          max_retries=0, cache_enabled=False)
    with pytest.raises(NoHealthyProviderError):
        await g.complete(req())


async def test_pinned_model_without_fallback():
    g = two_model_gateway(MockProvider("a", failure_rate=1.0), MockProvider("b"), max_retries=0,
                          cache_enabled=False)
    with pytest.raises(NoHealthyProviderError):
        await g.complete(req(model="big", metadata={"allow_fallback": False}))
    assert (await g.complete(req(model="big"))).model == "small"


async def test_semantic_cache_hits_and_skips_sampled_requests():
    a = MockProvider("a")
    g = two_model_gateway(a, MockProvider("b"), cache_threshold=0.9)
    await g.complete(req("What is the refund window for monthly plans?"))
    hit = await g.complete(req("What is the refund window for monthly plans?"))
    assert hit.cached and hit.cost_usd == 0
    near = await g.complete(req("what is the refund window for monthly plans"))
    assert near.cached
    assert g.metrics.cache.value(result="semantic") == 1
    sampled = await g.complete(req("What is the refund window for monthly plans?",
                                   temperature=0.7))
    assert not sampled.cached
    assert a.calls == 2


async def test_cache_is_tenant_isolated():
    a = MockProvider("a")
    g = two_model_gateway(a, MockProvider("b"))
    await g.complete(req("secret question", tenant="acme"))
    r = await g.complete(req("secret question", tenant="globex"))
    assert not r.cached


async def test_budget_enforced_and_cost_accounted():
    g = two_model_gateway(MockProvider("a"), MockProvider("b"), tenant_budget_usd=0.0000001,
                          cache_enabled=False)
    r = await g.complete(req("x" * 400, tenant="t1"))
    assert r.cost_usd > 0 and g.spend("t1") == pytest.approx(r.cost_usd)
    with pytest.raises(BudgetExceededError):
        await g.complete(req("again", tenant="t1"))
    await g.complete(req("other tenant unaffected", tenant="t2"))


async def test_semantic_cache_scoped_by_context_not_fooled_by_it():
    """Regression: identical long context + different questions must not collide."""
    a = MockProvider("a")
    g = two_model_gateway(a, MockProvider("b"), cache_threshold=0.9)
    ctx = "Context: " + "refund policy text " * 200

    def rag(q, scope="c1"):
        return CompletionRequest(
            messages=[Message(role="user", content=f"{ctx}\nQuestion: {q}")],
            metadata={"semantic_key": q, "semantic_scope": scope})

    await g.complete(rag("How long do card refunds take?"))
    assert not (await g.complete(rag("Are overage charges refundable?"))).cached
    assert (await g.complete(rag("how long do card refunds take"))).cached
    assert not (await g.complete(rag("how long do card refunds take", scope="c2"))).cached
