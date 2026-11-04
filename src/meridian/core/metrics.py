"""Prometheus-compatible metrics registry with zero dependencies.

Exposes counters and fixed-bucket histograms and renders the text exposition
format served at `GET /metrics`.
"""

from __future__ import annotations

import bisect
import threading
from collections import defaultdict

LabelKey = tuple[tuple[str, str], ...]

DEFAULT_BUCKETS_MS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


def _key(labels: dict[str, str] | None) -> LabelKey:
    return tuple(sorted((labels or {}).items()))


def _fmt(labels: LabelKey, extra: dict[str, str] | None = None) -> str:
    items = list(labels) + sorted((extra or {}).items())
    if not items:
        return ""
    return "{" + ",".join(f'{k}="{v}"' for k, v in items) + "}"


class Counter:
    def __init__(self, name: str, help_: str) -> None:
        self.name, self.help = name, help_
        self._values: dict[LabelKey, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        with self._lock:
            self._values[_key(labels)] += amount

    def value(self, **labels: str) -> float:
        return self._values.get(_key(labels), 0.0)

    def render(self) -> list[str]:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        out += [f"{self.name}{_fmt(k)} {v}" for k, v in sorted(self._values.items())]
        return out


class Histogram:
    def __init__(self, name: str, help_: str, buckets: tuple[float, ...] = DEFAULT_BUCKETS_MS):
        self.name, self.help, self.buckets = name, help_, tuple(sorted(buckets))
        self._counts: dict[LabelKey, list[int]] = {}
        self._sum: dict[LabelKey, float] = defaultdict(float)
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: str) -> None:
        k = _key(labels)
        with self._lock:
            counts = self._counts.setdefault(k, [0] * (len(self.buckets) + 1))
            counts[bisect.bisect_left(self.buckets, value)] += 1
            self._sum[k] += value

    def count(self, **labels: str) -> int:
        return sum(self._counts.get(_key(labels), []))

    def render(self) -> list[str]:
        out = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for k, counts in sorted(self._counts.items()):
            cumulative = 0
            for bound, c in zip(self.buckets, counts, strict=False):
                cumulative += c
                out.append(f"{self.name}_bucket{_fmt(k, {'le': str(bound)})} {cumulative}")
            cumulative += counts[-1]
            out.append(f"{self.name}_bucket{_fmt(k, {'le': '+Inf'})} {cumulative}")
            out.append(f"{self.name}_sum{_fmt(k)} {self._sum[k]}")
            out.append(f"{self.name}_count{_fmt(k)} {cumulative}")
        return out


class Metrics:
    """The engine-wide registry. One instance per Engine."""

    def __init__(self) -> None:
        self.requests = Counter("meridian_requests_total", "Gateway requests by provider/outcome")
        self.tokens = Counter("meridian_tokens_total", "Tokens processed by provider/direction")
        self.cost = Counter("meridian_cost_usd_total", "Spend in USD by tenant")
        self.cache = Counter("meridian_cache_total", "Semantic cache lookups by result")
        self.guardrail = Counter("meridian_guardrail_total", "Guardrail verdicts by stage/action")
        self.latency = Histogram("meridian_provider_latency_ms", "Provider latency in ms")
        self.pipeline = Histogram("meridian_pipeline_latency_ms", "End-to-end pipeline latency")

    def render(self) -> str:
        lines: list[str] = []
        for m in (self.requests, self.tokens, self.cost, self.cache, self.guardrail,
                  self.latency, self.pipeline):
            lines += m.render()
        return "\n".join(lines) + "\n"
