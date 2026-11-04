"""Agent runtime: a bounded tool-use loop on top of the Model Gateway.

    model ─► tool_calls? ──yes──► execute all calls concurrently ─► append results ─┐
      ▲                   no                                                       │
      │                   └──► final answer                                        │
      └────────────────────────────────────────────────────────────────────────────┘

Hard limits on steps, wall-clock time, cumulative cost and repeated identical
calls (loop detection) make the loop safe to expose to untrusted prompts. Every
step is a span and lands in the returned transcript for replay/debugging.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from meridian.agents.tools import ToolRegistry
from meridian.core.tracing import Tracer
from meridian.core.types import CompletionRequest, Message, RoutingPolicy, ToolCall
from meridian.gateway.gateway import ModelGateway


@dataclass
class AgentLimits:
    max_steps: int = 8
    max_seconds: float = 120.0
    max_cost_usd: float = 0.50
    max_identical_calls: int = 2


@dataclass
class StepRecord:
    step: int
    model: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    content: str = ""


@dataclass
class AgentResult:
    output: str
    stop_reason: str  # completed | max_steps | timeout | budget | loop_detected
    steps: list[StepRecord]
    cost_usd: float
    trace_id: str | None


class Agent:
    def __init__(self, gateway: ModelGateway, tools: ToolRegistry, *, system: str = "",
                 model: str | None = None, policy: RoutingPolicy = RoutingPolicy.QUALITY,
                 limits: AgentLimits | None = None, tracer: Tracer | None = None) -> None:
        self.gateway = gateway
        self.tools = tools
        self.system = system
        self.model = model
        self.policy = policy
        self.limits = limits or AgentLimits()
        self.tracer = tracer or gateway.tracer

    async def run(self, task: str, *, tenant: str = "default") -> AgentResult:
        messages: list[Message] = []
        if self.system:
            messages.append(Message(role="system", content=self.system))
        messages.append(Message(role="user", content=task))
        steps: list[StepRecord] = []
        seen: dict[str, int] = {}
        cost = 0.0
        started = time.monotonic()

        async with self.tracer.span("agent.run", tools=len(self.tools.specs)) as root:
            for step in range(1, self.limits.max_steps + 1):
                if time.monotonic() - started > self.limits.max_seconds:
                    return self._done("", "timeout", steps, cost, root.trace_id)
                if cost >= self.limits.max_cost_usd:
                    return self._done("", "budget", steps, cost, root.trace_id)

                async with self.tracer.span("agent.step", step=step):
                    resp = await self.gateway.complete(CompletionRequest(
                        messages=messages, model=self.model, policy=self.policy,
                        tools=self.tools.specs, tenant=tenant, cacheable=False))
                cost += resp.cost_usd
                record = StepRecord(step, resp.model, content=resp.content)
                steps.append(record)

                if not resp.tool_calls:
                    return self._done(resp.content, "completed", steps, cost, root.trace_id)

                messages.append(Message(role="assistant", content=resp.content,
                                        tool_calls=resp.tool_calls))
                for tc in resp.tool_calls:
                    sig = f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True)}"
                    seen[sig] = seen.get(sig, 0) + 1
                    if seen[sig] > self.limits.max_identical_calls:
                        return self._done(resp.content, "loop_detected", steps, cost,
                                          root.trace_id)

                results = await asyncio.gather(*(self._exec(tc) for tc in resp.tool_calls))
                for tc, (out, is_err) in zip(resp.tool_calls, results, strict=True):
                    record.tool_calls.append({"id": tc.id, "name": tc.name,
                                              "arguments": tc.arguments})
                    record.observations.append({"id": tc.id, "output": out, "error": is_err})
                    messages.append(Message(role="tool", content=out, tool_call_id=tc.id,
                                            name=tc.name))

            return self._done(steps[-1].content if steps else "", "max_steps", steps, cost,
                              root.trace_id)

    async def _exec(self, tc: ToolCall) -> tuple[str, bool]:
        async with self.tracer.span("agent.tool", tool=tc.name) as span:
            out, is_err = await self.tools.execute(tc.name, tc.arguments)
            span.set(error=is_err)
            return out, is_err

    @staticmethod
    def _done(output: str, reason: str, steps: list[StepRecord], cost: float,
              trace_id: str | None) -> AgentResult:
        return AgentResult(output, reason, steps, round(cost, 6), trace_id)
