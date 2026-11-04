"""Async DAG executor — the scheduling heart of the main engine.

A `Pipeline` is a set of named nodes with declared dependencies. Execution is
*eager*: each node starts the moment all of its dependencies finish (not level by
level), so a slow branch never stalls an independent one. Every node gets its own
timeout, retry policy and tracing span, and can be marked optional so a failure
degrades the result instead of aborting the run.

    p = Pipeline("rag")
    p.add("guard_in", guard_fn)
    p.add("retrieve", retrieve_fn, deps=["guard_in"])
    p.add("rewrite", rewrite_fn, deps=["guard_in"])          # runs concurrently
    p.add("generate", generate_fn, deps=["retrieve", "rewrite"])
    result = await p.run({"question": q})
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from meridian.core.errors import PipelineError
from meridian.core.tracing import Tracer

NodeFn = Callable[["RunContext"], Awaitable[Any]]


@dataclass
class Node:
    name: str
    fn: NodeFn
    deps: tuple[str, ...] = ()
    soft_deps: tuple[str, ...] = ()
    """Waited on, but their failure does not skip this node (read them with ctx.get)."""
    timeout_s: float | None = 30.0
    retries: int = 0
    optional: bool = False


@dataclass
class RunContext:
    """Shared state for one pipeline run. Nodes read inputs and upstream outputs here."""

    inputs: dict[str, Any]
    outputs: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, node: str) -> Any:
        return self.outputs[node]

    def get(self, node: str, default: Any = None) -> Any:
        return self.outputs.get(node, default)


class Pipeline:
    def __init__(self, name: str, tracer: Tracer | None = None) -> None:
        self.name = name
        self.tracer = tracer or Tracer()
        self._nodes: dict[str, Node] = {}

    def add(
        self,
        name: str,
        fn: NodeFn,
        *,
        deps: list[str] | tuple[str, ...] = (),
        soft_deps: list[str] | tuple[str, ...] = (),
        timeout_s: float | None = 30.0,
        retries: int = 0,
        optional: bool = False,
    ) -> Pipeline:
        if name in self._nodes:
            raise ValueError(f"duplicate node '{name}'")
        self._nodes[name] = Node(name, fn, tuple(deps), tuple(soft_deps), timeout_s, retries,
                                 optional)
        return self

    # ------------------------------------------------------------------ validation
    def topological_order(self) -> list[str]:
        """Kahn's algorithm; raises on unknown deps or cycles."""
        for n in self._nodes.values():
            for d in (*n.deps, *n.soft_deps):
                if d not in self._nodes:
                    raise ValueError(f"node '{n.name}' depends on unknown node '{d}'")
        indegree = {name: len(n.deps) + len(n.soft_deps) for name, n in self._nodes.items()}
        children: dict[str, list[str]] = {name: [] for name in self._nodes}
        for n in self._nodes.values():
            for d in (*n.deps, *n.soft_deps):
                children[d].append(n.name)
        ready = sorted(k for k, v in indegree.items() if v == 0)
        order: list[str] = []
        while ready:
            cur = ready.pop(0)
            order.append(cur)
            for c in children[cur]:
                indegree[c] -= 1
                if indegree[c] == 0:
                    ready.append(c)
        if len(order) != len(self._nodes):
            cyclic = sorted(set(self._nodes) - set(order))
            raise ValueError(f"cycle detected among nodes: {cyclic}")
        return order

    # ------------------------------------------------------------------ execution
    async def run(self, inputs: dict[str, Any]) -> RunContext:
        self.topological_order()
        ctx = RunContext(inputs=dict(inputs))
        done: dict[str, asyncio.Event] = {n: asyncio.Event() for n in self._nodes}
        skipped: set[str] = set()

        async def run_node(node: Node) -> None:
            try:
                for d in (*node.deps, *node.soft_deps):
                    await done[d].wait()
                failed_deps = [d for d in node.deps if d in skipped]
                if failed_deps:
                    skipped.add(node.name)
                    ctx.errors[node.name] = f"skipped: upstream {failed_deps} failed"
                    return
                async with self.tracer.span(f"{self.name}.{node.name}") as span:
                    try:
                        ctx.outputs[node.name] = await self._execute(node, ctx)
                    except Exception as exc:
                        span.set(optional=node.optional)
                        if not node.optional:
                            raise PipelineError(node.name, exc) from exc
                        skipped.add(node.name)
                        ctx.errors[node.name] = repr(exc)
            finally:
                done[node.name].set()

        async with self.tracer.span(f"pipeline.{self.name}", nodes=len(self._nodes)) as root:
            ctx.inputs["trace_id"] = root.trace_id
            tasks = [asyncio.create_task(run_node(n)) for n in self._nodes.values()]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            root.set(degraded=sorted(ctx.errors))
        return ctx

    async def _execute(self, node: Node, ctx: RunContext) -> Any:
        attempt = 0
        while True:
            try:
                coro = node.fn(ctx)
                if node.timeout_s is None:
                    return await coro
                return await asyncio.wait_for(coro, timeout=node.timeout_s)
            except Exception:
                if attempt >= node.retries:
                    raise
                attempt += 1
                await asyncio.sleep(min(2.0, 0.05 * 2**attempt) * random.random())
