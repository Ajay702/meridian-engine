"""Resilience primitives: circuit breaker, token-bucket rate limiter, retry with jitter.

All three are clock-injectable so their state machines are tested deterministically
instead of with sleeps.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import TypeVar

from meridian.core.errors import CircuitOpenError, MeridianError

T = TypeVar("T")
Clock = Callable[[], float]


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Classic three-state breaker.

    CLOSED    -> counts consecutive failures; trips to OPEN at `failure_threshold`.
    OPEN      -> rejects immediately until `reset_timeout_s` elapses.
    HALF_OPEN -> admits up to `half_open_max_calls` probes; one success closes the
                 circuit, one failure re-opens it.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout_s: float = 30.0,
        half_open_max_calls: int = 1,
        clock: Clock = time.monotonic,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self.half_open_max_calls = half_open_max_calls
        self._clock = clock
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_inflight = 0

    @property
    def state(self) -> BreakerState:
        if (
            self._state is BreakerState.OPEN
            and self._clock() - self._opened_at >= self.reset_timeout_s
        ):
            self._state = BreakerState.HALF_OPEN
            self._half_open_inflight = 0
        return self._state

    def allow(self) -> bool:
        state = self.state
        if state is BreakerState.CLOSED:
            return True
        if state is BreakerState.HALF_OPEN and self._half_open_inflight < self.half_open_max_calls:
            self._half_open_inflight += 1
            return True
        return False

    def retry_after(self) -> float:
        return max(0.0, self.reset_timeout_s - (self._clock() - self._opened_at))

    def record_success(self) -> None:
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._half_open_inflight = 0

    def record_failure(self) -> None:
        if self.state is BreakerState.HALF_OPEN:
            self._trip()
            return
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = self._clock()
        self._failures = 0

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        if not self.allow():
            raise CircuitOpenError(self.name, self.retry_after())
        try:
            result = await fn()
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result


class TokenBucket:
    """Async token bucket. `rate` tokens/sec refill, `capacity` burst."""

    def __init__(self, rate: float, capacity: float, clock: Clock = time.monotonic) -> None:
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._clock = clock
        self._last = clock()
        self._lock = asyncio.Lock()

    @classmethod
    def per_minute(cls, rpm: int) -> TokenBucket:
        return cls(rate=rpm / 60.0, capacity=max(1, rpm // 6))

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    async def acquire(self, tokens: float = 1.0, max_wait_s: float = 10.0) -> None:
        deadline = self._clock() + max_wait_s
        async with self._lock:
            while not self.try_acquire(tokens):
                wait = (tokens - self._tokens) / self.rate
                if self._clock() + wait > deadline:
                    from meridian.core.errors import RateLimitedError

                    raise RateLimitedError(f"rate limit wait {wait:.2f}s exceeds budget")
                await asyncio.sleep(wait)


def backoff_delay(attempt: int, base_s: float = 0.1, cap_s: float = 5.0) -> float:
    """AWS-style 'full jitter': uniform(0, min(cap, base * 2^attempt))."""
    return random.uniform(0, min(cap_s, base_s * (2**attempt)))


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int = 2,
    base_s: float = 0.1,
    is_retryable: Callable[[BaseException], bool] = lambda e: getattr(e, "retryable", False),
) -> T:
    attempt = 0
    while True:
        try:
            return await fn()
        except MeridianError as exc:
            if attempt >= retries or not is_retryable(exc):
                raise
            await asyncio.sleep(backoff_delay(attempt, base_s))
            attempt += 1
