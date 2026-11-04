from typing import Literal

from meridian.agents.runtime import Agent, AgentLimits
from meridian.agents.tools import ToolRegistry
from meridian.core.config import ModelConfig
from meridian.core.types import ToolCall
from meridian.gateway.gateway import ModelGateway
from meridian.gateway.providers.mock import ScriptedProvider


def make(script, **limits):
    reg = ToolRegistry()

    @reg.tool
    async def convert(amount: float, to: Literal["EUR", "GBP"] = "EUR") -> dict:
        """Convert USD to another currency."""
        return {"amount": round(amount * {"EUR": 0.9, "GBP": 0.8}[to], 2), "currency": to}

    @reg.tool
    def explode() -> str:
        """Always fails."""
        raise RuntimeError("kaboom")

    p = ScriptedProvider(script)
    g = ModelGateway([ModelConfig(name="m", provider=p.name)])
    g.register_provider(p)
    return Agent(g, reg, limits=AgentLimits(**limits)), p, reg


def test_schema_generated_from_type_hints():
    _, _, reg = make([])
    spec = {s.name: s for s in reg.specs}["convert"]
    assert spec.parameters["properties"]["to"] == {"type": "string", "enum": ["EUR", "GBP"]}
    assert spec.parameters["required"] == ["amount"]
    assert spec.description == "Convert USD to another currency."


async def test_tool_loop_parallel_calls_and_final_answer():
    agent, provider, _ = make([
        [ToolCall(name="convert", arguments={"amount": 100}),
         ToolCall(name="convert", arguments={"amount": 100, "to": "GBP"})],
        "100 USD is 90 EUR or 80 GBP.",
    ])
    res = await agent.run("convert 100 usd")
    assert res.stop_reason == "completed" and "90 EUR" in res.output
    obs = res.steps[0].observations
    assert '"amount": 90.0' in obs[0]["output"] and '"GBP"' in obs[1]["output"]
    tool_msgs = [m for m in provider.requests[1].messages if m.role == "tool"]
    assert len(tool_msgs) == 2


async def test_tool_errors_become_observations():
    agent, _, _ = make([
        [ToolCall(name="explode"), ToolCall(name="nope"),
         ToolCall(name="convert", arguments={"amount": "lots"})],
        "recovered",
    ])
    res = await agent.run("x")
    errs = [o for o in res.steps[0].observations if o["error"]]
    assert len(errs) == 3
    assert "kaboom" in errs[0]["output"] and "unknown tool" in errs[1]["output"]
    assert "invalid arguments" in errs[2]["output"]
    assert res.output == "recovered"


async def test_loop_detection_and_step_limit():
    same = [ToolCall(name="convert", arguments={"amount": 1})]
    agent, _, _ = make([same, same, same, "never"], max_identical_calls=2)
    assert (await agent.run("x")).stop_reason == "loop_detected"

    agent, _, _ = make([[ToolCall(name="convert", arguments={"amount": i})] for i in range(5)],
                       max_steps=3)
    res = await agent.run("x")
    assert res.stop_reason == "max_steps" and len(res.steps) == 3
