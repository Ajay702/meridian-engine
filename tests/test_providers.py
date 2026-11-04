import json

import httpx
import pytest

from meridian.core.errors import ProviderError, RetryableProviderError
from meridian.core.types import CompletionRequest, Message, ToolCall, ToolSpec
from meridian.gateway.providers.http import AnthropicProvider, OpenAICompatibleProvider


def transport(handler):
    return httpx.MockTransport(handler)


async def test_anthropic_translates_tools_and_tool_results():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["headers"] = request.headers
        return httpx.Response(200, json={
            "id": "msg_1", "model": "claude-x", "stop_reason": "tool_use",
            "content": [{"type": "text", "text": "checking"},
                        {"type": "tool_use", "id": "tu_2", "name": "lookup",
                         "input": {"q": "b"}}],
            "usage": {"input_tokens": 42, "output_tokens": 7}})

    p = AnthropicProvider(api_key="k", transport=transport(handler))
    req = CompletionRequest(
        messages=[
            Message(role="system", content="be brief"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="", tool_calls=[
                ToolCall(id="tu_1", name="lookup", arguments={"q": "a"}),
                ToolCall(id="tu_9", name="lookup", arguments={"q": "z"})]),
            Message(role="tool", content="A", tool_call_id="tu_1"),
            Message(role="tool", content="Z", tool_call_id="tu_9"),
        ],
        tools=[ToolSpec(name="lookup", description="d", parameters={"type": "object"})])
    resp = await p.complete(req, "claude-x")
    body = seen["body"]
    assert body["system"] == "be brief"
    assert body["tools"][0]["input_schema"] == {"type": "object"}
    assert body["messages"][1]["content"][0]["type"] == "tool_use"
    results = body["messages"][2]["content"]  # both results merged into one user turn
    assert [b["tool_use_id"] for b in results] == ["tu_1", "tu_9"]
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert resp.tool_calls[0].name == "lookup" and resp.usage.input_tokens == 42
    await p.aclose()


async def test_openai_compatible_parses_tool_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer sk"
        return httpx.Response(200, json={
            "id": "c1", "model": "llama",
            "choices": [{"finish_reason": "tool_calls", "message": {
                "content": None, "tool_calls": [{"id": "t1", "type": "function", "function": {
                    "name": "f", "arguments": "{\"x\": 1}"}}]}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}})

    p = OpenAICompatibleProvider("vllm", base_url="http://x/v1", api_key="sk",
                                 transport=transport(handler))
    resp = await p.complete(CompletionRequest(messages=[Message(role="user", content="q")]), "m")
    assert resp.tool_calls[0].arguments == {"x": 1} and resp.content == ""
    await p.aclose()


@pytest.mark.parametrize(("status", "exc"), [(429, RetryableProviderError),
                                            (503, RetryableProviderError),
                                            (400, ProviderError)])
async def test_status_mapping(status, exc):
    p = AnthropicProvider(api_key="k",
                          transport=transport(lambda r: httpx.Response(status, text="err")))
    with pytest.raises(exc) as ei:
        await p.complete(CompletionRequest(messages=[Message(role="user", content="q")]), "m")
    assert ei.value.retryable == (status != 400)
    await p.aclose()


async def test_transport_error_is_retryable():
    def boom(request):
        raise httpx.ConnectError("refused")

    p = OpenAICompatibleProvider("x", base_url="http://x", transport=transport(boom))
    with pytest.raises(RetryableProviderError):
        await p.complete(CompletionRequest(messages=[Message(role="user", content="q")]), "m")
    await p.aclose()
