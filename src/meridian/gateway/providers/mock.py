"""Offline providers.

`MockProvider` is a deterministic, *extractive* stand-in for an LLM: given a
prompt containing numbered context passages it answers with the sentences that
best overlap the question and cites them. That makes the full RAG pipeline —
retrieval quality, citations, groundedness evals — observable without an API key.
It also exposes `failure_rate` and `latency_ms` for chaos-testing the router.

`ScriptedProvider` replays a fixed list of responses (used to test the agent loop).
"""

from __future__ import annotations

import asyncio
import random
import re

from meridian.core.errors import RetryableProviderError
from meridian.core.types import CompletionRequest, CompletionResponse, ToolCall, Usage
from meridian.gateway.providers.base import estimate_tokens
from meridian.retrieval.embeddings import tokenize

_PASSAGE = re.compile(r"^\[(\d+)\]\s*(.+?)(?=^\[\d+\]|\Z)", re.S | re.M)
_SENT = re.compile(r"(?<=[.!?])\s+")


class MockProvider:
    def __init__(
        self,
        name: str = "local",
        *,
        failure_rate: float = 0.0,
        latency_ms: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.name = name
        self.failure_rate = failure_rate
        self.latency_ms = latency_ms
        self._rng = random.Random(seed)
        self.calls = 0

    async def complete(self, req: CompletionRequest, model: str) -> CompletionResponse:
        self.calls += 1
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)
        if self.failure_rate and self._rng.random() < self.failure_rate:
            raise RetryableProviderError(self.name, "injected failure", status=503)

        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        content = self._answer(last_user)
        return CompletionResponse(
            model=model,
            provider=self.name,
            content=content,
            usage=Usage(
                input_tokens=estimate_tokens(req.prompt_text()),
                output_tokens=estimate_tokens(content),
            ),
        )

    @staticmethod
    def _answer(prompt: str) -> str:
        if "Context:" not in prompt or "Question:" not in prompt:
            return f"(mock) {prompt[:200]}"
        context, _, question = prompt.partition("Question:")
        q_terms = set(tokenize(question))
        scored: list[tuple[float, int, str]] = []
        for m in _PASSAGE.finditer(context.split("Context:", 1)[1]):
            raw = re.sub(r"^#{1,6}\s.*$", "", m.group(2), flags=re.M)  # markdown headings
            idx, body = int(m.group(1)), " ".join(raw.split())
            body = re.sub(r"^\(.*?\)\s*", "", body)  # drop "(doc › section)" header
            for sent in _SENT.split(body):
                terms = set(tokenize(sent))
                if not terms:
                    continue
                overlap = len(q_terms & terms) / (len(q_terms) ** 0.5 * len(terms) ** 0.5 or 1)
                scored.append((overlap, idx, sent.strip()))
        scored.sort(key=lambda x: -x[0])
        picked = [s for s in scored[:2] if s[0] > 0]
        if not picked:
            return "I don't have enough information in the provided context to answer that."
        return " ".join(f"{sent} [{idx}]" for _, idx, sent in picked)

    async def aclose(self) -> None:
        return None


class ScriptedProvider:
    """Returns pre-baked responses in order; each item is text or a list of ToolCalls."""

    def __init__(self, script: list[str | list[ToolCall]], name: str = "scripted") -> None:
        self.name = name
        self._script = list(script)
        self.requests: list[CompletionRequest] = []

    async def complete(self, req: CompletionRequest, model: str) -> CompletionResponse:
        self.requests.append(req)
        if not self._script:
            raise RuntimeError("ScriptedProvider exhausted")
        step = self._script.pop(0)
        if isinstance(step, str):
            return CompletionResponse(model=model, provider=self.name, content=step,
                                      usage=Usage(input_tokens=10, output_tokens=10))
        return CompletionResponse(model=model, provider=self.name, content="", tool_calls=step,
                                  stop_reason="tool_use",
                                  usage=Usage(input_tokens=10, output_tokens=5))

    async def aclose(self) -> None:
        return None
