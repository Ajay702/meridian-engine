"""Wire-level types shared by every sub-engine.

These are deliberately provider-agnostic: each provider adapter translates to/from
its vendor schema at the edge, so routing, caching, guardrails and tracing never
care which vendor served a request.
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    """Set on assistant messages that requested tool invocations."""


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class RoutingPolicy(StrEnum):
    COST = "cost"
    LATENCY = "latency"
    QUALITY = "quality"


class CompletionRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    """Pin a model. When None the router picks one according to `policy`."""
    policy: RoutingPolicy = RoutingPolicy.QUALITY
    max_tokens: int = 1024
    temperature: float = 0.0
    tools: list[ToolSpec] = Field(default_factory=list)
    tenant: str = "default"
    cacheable: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    def prompt_text(self) -> str:
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl_{uuid.uuid4().hex[:16]}")
    model: str
    provider: str
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    created_at: float = Field(default_factory=time.time)
    stop_reason: str = "end_turn"
