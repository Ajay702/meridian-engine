import pytest

from meridian.core.errors import CircuitOpenError, RetryableProviderError
from meridian.gateway.resilience import BreakerState, CircuitBreaker, TokenBucket, retry_async


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_breaker_trips_half_opens_and_recovers():
    clock = FakeClock()
    b = CircuitBreaker("p", failure_threshold=3, reset_timeout_s=10, clock=clock)
    for _ in range(3):
        assert b.allow()
        b.record_failure()
    assert b.state is BreakerState.OPEN and not b.allow()
    clock.t = 10
    assert b.state is BreakerState.HALF_OPEN
    assert b.allow() and not b.allow()  # only one probe admitted
    b.record_success()
    assert b.state is BreakerState.CLOSED


def test_half_open_failure_reopens():
    clock = FakeClock()
    b = CircuitBreaker("p", failure_threshold=1, reset_timeout_s=5, clock=clock)
    b.record_failure()
    clock.t = 5
    assert b.allow()
    b.record_failure()
    assert b.state is BreakerState.OPEN
    assert b.retry_after() == pytest.approx(5)


async def test_breaker_call_rejects_when_open():
    b = CircuitBreaker("p", failure_threshold=1)
    b.record_failure()

    async def fn():
        return 1

    with pytest.raises(CircuitOpenError):
        await b.call(fn)


def test_token_bucket_refill():
    clock = FakeClock()
    tb = TokenBucket(rate=2, capacity=2, clock=clock)
    assert tb.try_acquire() and tb.try_acquire() and not tb.try_acquire()
    clock.t = 0.5
    assert tb.try_acquire() and not tb.try_acquire()


async def test_retry_only_retryable():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RetryableProviderError("p", "503")
        return "ok"

    assert await retry_async(flaky, retries=3, base_s=0) == "ok"
    calls["n"] = 0

    async def fatal():
        calls["n"] += 1
        from meridian.core.errors import ProviderError
        raise ProviderError("p", "400 bad request")

    with pytest.raises(Exception, match="400"):
        await retry_async(fatal, retries=3, base_s=0)
    assert calls["n"] == 1
