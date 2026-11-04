"""HTTP provider adapters: Anthropic Messages API and any OpenAI-compatible endpoint
(OpenAI, Azure OpenAI, vLLM, Ollama, LM Studio, Together, Groq, ...).

Each adapter owns one pooled `httpx.AsyncClient`, translates the neutral request
schema at the edge, and maps transport/HTTP failures onto the engine's error
taxonomy so the router can make retry / failover decisions uniformly.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

from meridian.core.errors import ProviderError, RetryableProviderError
from meridian.core.types import CompletionRequest, CompletionResponse, Message, ToolCall, Usage

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class _HTTPProvider:
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None,
        timeout_s: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            transport=transport,
        )

    async def _post(self, path: str, body: dict[str, Any], headers: dict[str, str]) -> Any:
        try:
            r = await self._client.post(path, json=body, headers=headers)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise RetryableProviderError(self.name, f"transport error: {exc!r}") from exc
        if r.status_code in RETRYABLE_STATUS:
            raise RetryableProviderError(self.name, r.text[:300], status=r.status_code)
        if r.status_code >= 400:
            raise ProviderError(self.name, r.text[:300], status=r.status_code)
        return r.json()

    async def aclose(self) -> None:
        await self._client.aclose()


class AnthropicProvider(_HTTPProvider):
    API_VERSION = "2023-06-01"

    def __init__(self, name: str = "anthropic", *, base_url: str = "https://api.anthropic.com",
                 api_key: str | None = None, timeout_s: float = 60.0,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        super().__init__(name, base_url, api_key or os.environ.get("ANTHROPIC_API_KEY"),
                         timeout_s, transport)

    @staticmethod
    def _convert(messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        system = "\n\n".join(m.content for m in messages if m.role == "system") or None
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id,
                         "content": m.content}
                # consecutive tool results must share one user turn
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m.role == "assistant" and m.tool_calls:
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                blocks += [{"type": "tool_use", "id": tc.id, "name": tc.name,
                            "input": tc.arguments} for tc in m.tool_calls]
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": m.role, "content": m.content})
        return system, out

    async def complete(self, req: CompletionRequest, model: str) -> CompletionResponse:
        system, messages = self._convert(req.messages)
        body: dict[str, Any] = {"model": model, "max_tokens": req.max_tokens,
                                "temperature": req.temperature, "messages": messages}
        if system:
            body["system"] = system
        if req.tools:
            body["tools"] = [{"name": t.name, "description": t.description,
                              "input_schema": t.parameters} for t in req.tools]
        t0 = time.perf_counter()
        data = await self._post("/v1/messages", body, {
            "x-api-key": self._api_key or "", "anthropic-version": self.API_VERSION})
        text = "".join(b.get("text", "") for b in data["content"] if b["type"] == "text")
        calls = [ToolCall(id=b["id"], name=b["name"], arguments=b.get("input") or {})
                 for b in data["content"] if b["type"] == "tool_use"]
        u = data.get("usage", {})
        return CompletionResponse(
            id=data.get("id", ""), model=data.get("model", model), provider=self.name,
            content=text, tool_calls=calls, stop_reason=data.get("stop_reason") or "end_turn",
            usage=Usage(input_tokens=u.get("input_tokens", 0),
                        output_tokens=u.get("output_tokens", 0)),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class OpenAICompatibleProvider(_HTTPProvider):
    def __init__(self, name: str, *, base_url: str, api_key: str | None = None,
                 timeout_s: float = 60.0,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        super().__init__(name, base_url, api_key, timeout_s, transport)

    @staticmethod
    def _convert(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            d: dict[str, Any] = {"role": m.role, "content": m.content}
            if m.role == "tool":
                d["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                d["tool_calls"] = [{"id": tc.id, "type": "function", "function": {
                    "name": tc.name, "arguments": json.dumps(tc.arguments)}}
                    for tc in m.tool_calls]
            out.append(d)
        return out

    async def complete(self, req: CompletionRequest, model: str) -> CompletionResponse:
        body: dict[str, Any] = {"model": model, "max_tokens": req.max_tokens,
                                "temperature": req.temperature,
                                "messages": self._convert(req.messages)}
        if req.tools:
            body["tools"] = [{"type": "function", "function": {
                "name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in req.tools]
        headers = {"authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        t0 = time.perf_counter()
        data = await self._post("/chat/completions", body, headers)
        choice = data["choices"][0]
        msg = choice["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc["function"].get("arguments")}
            calls.append(ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=args))
        u = data.get("usage") or {}
        return CompletionResponse(
            id=data.get("id", ""), model=data.get("model", model), provider=self.name,
            content=msg.get("content") or "", tool_calls=calls,
            stop_reason=choice.get("finish_reason") or "stop",
            usage=Usage(input_tokens=u.get("prompt_tokens", 0),
                        output_tokens=u.get("completion_tokens", 0)),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
