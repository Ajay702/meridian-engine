"""Model Gateway sub-engine.

The single choke point every LLM call goes through. Per request it:

1. Ranks candidate models by the request's routing policy (cost / latency / quality),
   filtering out models whose provider circuit is open or whose context window is
   too small for the prompt.
2. Enforces the tenant's spend budget.
3. Serves from the semantic cache when possible.
4. Calls the provider under a token-bucket rate limiter, retry-with-jitter and a
   per-provider circuit breaker — failing over down the ranked chain on
   retryable errors.
5. Prices the response, updates latency EWMAs, emits metrics, spans and events.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

from meridian.core.config import GatewayConfig, ModelConfig
from meridian.core.errors import (
    BudgetExceededError,
    CircuitOpenError,
    MeridianError,
    NoHealthyProviderError,
    RateLimitedError,
)
from meridian.core.events import EventBus
from meridian.core.metrics import Metrics
from meridian.core.tracing import Tracer
from meridian.core.types import CompletionRequest, CompletionResponse, RoutingPolicy
from meridian.gateway.cache import SemanticCache
from meridian.gateway.providers.base import Provider, estimate_tokens
from meridian.gateway.resilience import BreakerState, CircuitBreaker, TokenBucket, retry_async


@dataclass
class _ProviderSlot:
    provider: Provider
    breaker: CircuitBreaker
    limiter: TokenBucket


class LatencyTracker:
    """Exponentially-weighted moving average of latency per model."""

    def __init__(self, alpha: float = 0.2, prior_ms: float = 800.0) -> None:
        self.alpha, self.prior = alpha, prior_ms
        self._ewma: dict[str, float] = {}

    def observe(self, model: str, ms: float) -> None:
        prev = self._ewma.get(model)
        self._ewma[model] = ms if prev is None else self.alpha * ms + (1 - self.alpha) * prev

    def get(self, model: str) -> float:
        return self._ewma.get(model, self.prior)


class ModelGateway:
    def __init__(
        self,
        models: list[ModelConfig],
        config: GatewayConfig | None = None,
        *,
        metrics: Metrics | None = None,
        tracer: Tracer | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self.config = config or GatewayConfig()
        self.models = {m.name: m for m in models}
        self.metrics = metrics or Metrics()
        self.tracer = tracer or Tracer()
        self.bus = bus or EventBus()
        self.latency = LatencyTracker()
        self.cache = SemanticCache(
            threshold=self.config.cache_threshold,
            ttl_s=self.config.cache_ttl_s,
            capacity=self.config.cache_capacity,
        )
        self._slots: dict[str, _ProviderSlot] = {}
        self._spend: dict[str, float] = defaultdict(float)

    # ------------------------------------------------------------------ registry
    def register_provider(self, provider: Provider, *, rpm: int = 600) -> None:
        self._slots[provider.name] = _ProviderSlot(
            provider=provider,
            breaker=CircuitBreaker(
                provider.name,
                failure_threshold=self.config.breaker_failure_threshold,
                reset_timeout_s=self.config.breaker_reset_s,
            ),
            limiter=TokenBucket.per_minute(rpm),
        )

    def breaker(self, provider: str) -> CircuitBreaker:
        return self._slots[provider].breaker

    def spend(self, tenant: str) -> float:
        return self._spend[tenant]

    # ------------------------------------------------------------------ routing
    def estimate_cost(self, model: ModelConfig, in_tokens: int, out_tokens: int) -> float:
        return (in_tokens * model.input_per_mtok + out_tokens * model.output_per_mtok) / 1e6

    def rank(self, req: CompletionRequest) -> list[ModelConfig]:
        """Ordered fallback chain for this request."""
        in_tokens = estimate_tokens(req.prompt_text())
        expected_out = min(req.max_tokens, 512)
        eligible = [
            m for m in self.models.values()
            if m.provider in self._slots
            and m.context_window >= in_tokens + req.max_tokens
            and self._slots[m.provider].breaker.state is not BreakerState.OPEN
        ]

        def cost(m: ModelConfig) -> float:
            return self.estimate_cost(m, in_tokens, expected_out)

        if req.policy is RoutingPolicy.COST:
            eligible.sort(key=lambda m: (cost(m), -m.quality))
        elif req.policy is RoutingPolicy.LATENCY:
            eligible.sort(key=lambda m: (self.latency.get(m.name), cost(m)))
        else:
            eligible.sort(key=lambda m: (-m.quality, cost(m)))

        if req.model:
            pinned = [m for m in eligible if m.name == req.model]
            if not pinned and req.model not in self.models:
                raise MeridianError(f"unknown model '{req.model}'")
            rest = [m for m in eligible if m.name != req.model]
            allow_fallback = req.metadata.get("allow_fallback", True)
            eligible = pinned + (rest if allow_fallback else [])
        return eligible

    # ------------------------------------------------------------------ execution
    async def complete(self, req: CompletionRequest) -> CompletionResponse:
        budget = self.config.tenant_budget_usd
        if budget is not None and self._spend[req.tenant] >= budget:
            raise BudgetExceededError(
                f"tenant '{req.tenant}' spent ${self._spend[req.tenant]:.4f} of ${budget:.2f}"
            )

        async with self.tracer.span("gateway.complete", tenant=req.tenant,
                                    policy=req.policy.value) as span:
            chain = self.rank(req)
            if not chain:
                raise NoHealthyProviderError("no eligible model (all circuits open?)")
            span.set(chain=[m.name for m in chain])

            if self.config.cache_enabled:
                for m in chain[:1]:
                    if (hit := self.cache.get(req, m.name)) is not None:
                        resp, kind = hit
                        self.metrics.cache.inc(result=kind)
                        span.set(cache=kind, model=m.name)
                        return resp.model_copy(update={"cached": True, "cost_usd": 0.0,
                                                       "latency_ms": 0.0})
                self.metrics.cache.inc(result="miss")

            errors: list[str] = []
            for model in chain:
                slot = self._slots[model.provider]
                try:
                    resp = await self._call(slot, req, model)
                except (CircuitOpenError, RateLimitedError, MeridianError) as exc:
                    if isinstance(exc, MeridianError) and not (
                        exc.retryable or isinstance(exc, CircuitOpenError)
                    ):
                        self.metrics.requests.inc(provider=model.provider, outcome="error")
                        raise
                    self.metrics.requests.inc(provider=model.provider, outcome="failover")
                    errors.append(f"{model.name}: {exc}")
                    await self.bus.publish("gateway.failover", model=model.name, error=str(exc))
                    continue

                self._account(req, model, resp)
                span.set(model=model.name, provider=model.provider, cost_usd=resp.cost_usd,
                         attempts=len(errors) + 1)
                if self.config.cache_enabled:
                    self.cache.put(req, chain[0].name, resp)
                await self.bus.publish("gateway.completion", tenant=req.tenant,
                                       model=model.name, cost_usd=resp.cost_usd,
                                       tokens=resp.usage.total)
                return resp

            raise NoHealthyProviderError("all candidates failed: " + " | ".join(errors))

    async def _call(self, slot: _ProviderSlot, req: CompletionRequest,
                    model: ModelConfig) -> CompletionResponse:
        await slot.limiter.acquire(max_wait_s=5.0)

        async def attempt() -> CompletionResponse:
            return await slot.breaker.call(lambda: slot.provider.complete(req, model.name))

        t0 = time.perf_counter()
        async with self.tracer.span("provider.call", provider=slot.provider.name,
                                    model=model.name):
            resp = await retry_async(
                attempt,
                retries=self.config.max_retries,
                # never hammer a provider whose breaker just opened
                is_retryable=lambda e: getattr(e, "retryable", False)
                and not isinstance(e, CircuitOpenError),
            )
        elapsed = (time.perf_counter() - t0) * 1000
        resp.latency_ms = resp.latency_ms or elapsed
        self.latency.observe(model.name, resp.latency_ms)
        self.metrics.latency.observe(resp.latency_ms, provider=slot.provider.name)
        return resp

    def _account(self, req: CompletionRequest, model: ModelConfig,
                 resp: CompletionResponse) -> None:
        resp.cost_usd = self.estimate_cost(model, resp.usage.input_tokens,
                                           resp.usage.output_tokens)
        self._spend[req.tenant] += resp.cost_usd
        self.metrics.requests.inc(provider=model.provider, outcome="ok")
        self.metrics.tokens.inc(resp.usage.input_tokens, provider=model.provider, dir="in")
        self.metrics.tokens.inc(resp.usage.output_tokens, provider=model.provider, dir="out")
        self.metrics.cost.inc(resp.cost_usd, tenant=req.tenant)

    async def aclose(self) -> None:
        for slot in self._slots.values():
            await slot.provider.aclose()
