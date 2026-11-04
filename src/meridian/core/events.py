"""In-process async event bus.

Sub-engines publish domain events (`gateway.completion`, `guardrail.blocked`,
`retrieval.ingested`, ...) and anything can subscribe — audit logs, usage
billing, webhooks — without the publisher knowing about it. Handlers run
concurrently and a failing handler never breaks the publisher.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("meridian.events")

Handler = Callable[["Event"], Awaitable[None]]


@dataclass(frozen=True)
class Event:
    topic: str
    payload: dict[str, Any]
    ts: float = field(default_factory=time.time)


class EventBus:
    def __init__(self) -> None:
        self._subs: list[tuple[str, Handler]] = []

    def subscribe(self, pattern: str, handler: Handler) -> Callable[[], None]:
        """Subscribe with a glob pattern (e.g. ``gateway.*``). Returns an unsubscribe fn."""
        entry = (pattern, handler)
        self._subs.append(entry)
        return lambda: self._subs.remove(entry)

    async def publish(self, topic: str, **payload: Any) -> None:
        event = Event(topic, payload)
        handlers = [h for p, h in self._subs if fnmatch.fnmatchcase(topic, p)]
        if not handlers:
            return
        results = await asyncio.gather(*(h(event) for h in handlers), return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                log.warning("event handler failed for %s: %r", topic, r)
