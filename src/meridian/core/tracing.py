"""Minimal, dependency-free distributed-tracing primitives.

Spans propagate through `contextvars`, so nested `async with tracer.span(...)`
blocks — even across `asyncio.gather` — form a correct parent/child tree.
The shape mirrors OpenTelemetry (trace_id / span_id / parent_id / attributes) so
an OTLP exporter can be dropped in without touching call sites.
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

_current_span: ContextVar[Span | None] = ContextVar("meridian_current_span", default=None)


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None
    start: float = field(default_factory=time.perf_counter)
    end: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        end = self.end if self.end is not None else time.perf_counter()
        return (end - self.start) * 1000

    def set(self, **attrs: Any) -> None:
        self.attributes.update(attrs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
        }


class Tracer:
    """Collects finished spans in a bounded in-memory store keyed by trace id."""

    def __init__(self, max_traces: int = 1000) -> None:
        self._traces: OrderedDict[str, list[Span]] = OrderedDict()
        self._max = max_traces

    @asynccontextmanager
    async def span(self, name: str, **attributes: Any) -> AsyncIterator[Span]:
        parent = _current_span.get()
        trace_id = parent.trace_id if parent else uuid.uuid4().hex
        s = Span(
            name=name,
            trace_id=trace_id,
            parent_id=parent.span_id if parent else None,
            attributes=dict(attributes),
        )
        token = _current_span.set(s)
        try:
            yield s
        except BaseException as exc:
            s.status = "error"
            s.error = repr(exc)
            raise
        finally:
            s.end = time.perf_counter()
            _current_span.reset(token)
            self._record(s)

    def _record(self, span: Span) -> None:
        bucket = self._traces.setdefault(span.trace_id, [])
        bucket.append(span)
        self._traces.move_to_end(span.trace_id)
        while len(self._traces) > self._max:
            self._traces.popitem(last=False)

    def get(self, trace_id: str) -> list[dict[str, Any]]:
        spans = self._traces.get(trace_id, [])
        return [s.to_dict() for s in sorted(spans, key=lambda s: s.start)]

    @staticmethod
    def current() -> Span | None:
        return _current_span.get()
