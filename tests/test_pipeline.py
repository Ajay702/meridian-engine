import asyncio

import pytest

from meridian.core.errors import PipelineError
from meridian.core.pipeline import Pipeline


async def test_independent_branches_run_concurrently():
    p = Pipeline("t")
    order: list[str] = []

    async def slow(ctx):
        await asyncio.sleep(0.05)
        order.append("slow")
        return 1

    async def fast(ctx):
        order.append("fast")
        return 2

    async def join(ctx):
        return ctx["slow"] + ctx["fast"]

    p.add("slow", slow).add("fast", fast).add("join", join, deps=["slow", "fast"])
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    ctx = await p.run({})
    assert ctx["join"] == 3
    assert order == ["fast", "slow"]
    assert loop.time() - t0 < 0.1


async def test_optional_failure_degrades_and_skips_dependents():
    p = Pipeline("t")

    async def boom(ctx):
        raise RuntimeError("store down")

    async def ok(ctx):
        return "ok"

    p.add("mem", boom, optional=True).add("needs_mem", ok, deps=["mem"], optional=True)
    p.add("main", ok)
    ctx = await p.run({})
    assert ctx["main"] == "ok"
    assert "mem" in ctx.errors and "skipped" in ctx.errors["needs_mem"]


async def test_required_failure_raises_pipeline_error():
    p = Pipeline("t")

    async def boom(ctx):
        raise ValueError("bad")

    p.add("a", boom)
    with pytest.raises(PipelineError) as ei:
        await p.run({})
    assert ei.value.node == "a"


async def test_retry_then_succeed():
    calls = {"n": 0}

    async def flaky(ctx):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return "done"

    ctx = await Pipeline("t").add("x", flaky, retries=2).run({})
    assert ctx["x"] == "done" and calls["n"] == 3


async def test_timeout_enforced():
    async def hang(ctx):
        await asyncio.sleep(5)

    with pytest.raises(PipelineError):
        await Pipeline("t").add("h", hang, timeout_s=0.02).run({})


def test_cycle_and_unknown_dep_detection():
    async def f(ctx):
        return None

    p = Pipeline("t").add("a", f, deps=["b"]).add("b", f, deps=["a"])
    with pytest.raises(ValueError, match="cycle"):
        p.topological_order()
    with pytest.raises(ValueError, match="unknown"):
        Pipeline("t").add("a", f, deps=["zzz"]).topological_order()


async def test_spans_form_a_tree():
    async def f(ctx):
        return 1

    p = Pipeline("t").add("a", f).add("b", f, deps=["a"])
    ctx = await p.run({})
    spans = p.tracer.get(ctx.inputs["trace_id"])
    root = next(s for s in spans if s["parent_id"] is None)
    assert {s["name"] for s in spans if s["parent_id"] == root["span_id"]} == {"t.a", "t.b"}


async def test_soft_dependency_failure_does_not_skip():
    async def boom(ctx):
        raise RuntimeError("down")

    async def use(ctx):
        return ctx.get("mem", "fallback")

    p = Pipeline("t").add("mem", boom, optional=True).add("gen", use, soft_deps=["mem"])
    ctx = await p.run({})
    assert ctx["gen"] == "fallback"
