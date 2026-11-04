"""Provider adapter contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from meridian.core.types import CompletionRequest, CompletionResponse


def estimate_tokens(text: str) -> int:
    """~4 chars/token heuristic; adapters overwrite with real usage when the API returns it."""
    return max(1, len(text) // 4)


@runtime_checkable
class Provider(Protocol):
    name: str

    async def complete(self, req: CompletionRequest, model: str) -> CompletionResponse:
        """Execute one completion against `model`. Must raise RetryableProviderError for
        timeouts / 429 / 5xx so the router can retry or fail over."""
        ...

    async def aclose(self) -> None: ...
