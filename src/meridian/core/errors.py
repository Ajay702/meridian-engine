"""Error taxonomy.

Errors are split by *what the caller should do*, not by where they came from:
retryable errors feed the retry/circuit-breaker machinery, everything else is
surfaced immediately.
"""

from __future__ import annotations


class MeridianError(Exception):
    """Base class for all engine errors."""

    retryable: bool = False


class ProviderError(MeridianError):
    def __init__(self, provider: str, message: str, *, status: int | None = None) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.status = status


class RetryableProviderError(ProviderError):
    """Timeouts, 429s, 5xx — safe to retry or fail over."""

    retryable = True


class CircuitOpenError(MeridianError):
    def __init__(self, name: str, retry_after_s: float) -> None:
        super().__init__(f"circuit '{name}' is open; retry after {retry_after_s:.1f}s")
        self.name = name
        self.retry_after_s = retry_after_s


class RateLimitedError(MeridianError):
    retryable = True


class NoHealthyProviderError(MeridianError):
    """Every candidate in the fallback chain failed or was short-circuited."""


class BudgetExceededError(MeridianError):
    pass


class GuardrailViolation(MeridianError):
    def __init__(self, stage: str, reasons: list[str]) -> None:
        super().__init__(f"guardrail blocked {stage}: {', '.join(reasons)}")
        self.stage = stage
        self.reasons = reasons


class PipelineError(MeridianError):
    def __init__(self, node: str, cause: BaseException) -> None:
        super().__init__(f"pipeline node '{node}' failed: {cause!r}")
        self.node = node
        self.cause = cause
