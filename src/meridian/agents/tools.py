"""Tool registry: plain Python functions → JSON-Schema tool specs, with validated calls.

    registry = ToolRegistry()

    @registry.tool
    async def get_weather(city: str, unit: Literal["c", "f"] = "c") -> str:
        '''Current weather for a city.'''

The schema is derived from type hints, so the spec the model sees and the
validation applied to its arguments can never drift apart.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import types
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, get_args, get_origin

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict  # pydantic requires this one on Python < 3.12

from meridian.core.types import ToolSpec

_PRIMITIVES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _schema_for(tp: Any) -> dict[str, Any]:
    origin = get_origin(tp)
    if tp in _PRIMITIVES:
        return {"type": _PRIMITIVES[tp]}
    if origin is Literal:
        vals = list(get_args(tp))
        return {"type": _PRIMITIVES.get(type(vals[0]), "string"), "enum": vals}
    if origin in (list, tuple, set):
        args = get_args(tp)
        return {"type": "array", "items": _schema_for(args[0]) if args else {}}
    if origin is dict:
        return {"type": "object"}
    if origin in (typing.Union, types.UnionType):
        non_null = [a for a in get_args(tp) if a is not type(None)]
        if len(non_null) == 1:
            return _schema_for(non_null[0])
        return {"anyOf": [_schema_for(a) for a in non_null]}
    return {}


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    spec: ToolSpec
    timeout_s: float = 30.0
    _adapter: TypeAdapter[Any] | None = None

    async def invoke(self, arguments: dict[str, Any]) -> str:
        if self._adapter is not None:
            arguments = self._adapter.validate_python(arguments)
        result = self.fn(**arguments)
        if inspect.isawaitable(result):
            result = await asyncio.wait_for(result, timeout=self.timeout_s)
        return result if isinstance(result, str) else json.dumps(result, default=str)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(self, fn: Callable[..., Any] | None = None, *, name: str | None = None,
             timeout_s: float = 30.0):
        def register(f: Callable[..., Any]) -> Callable[..., Any]:
            self.register(f, name=name, timeout_s=timeout_s)
            return f
        return register(fn) if fn is not None else register

    def register(self, fn: Callable[..., Any], *, name: str | None = None,
                 timeout_s: float = 30.0) -> Tool:
        sig = inspect.signature(fn)
        hints = typing.get_type_hints(fn)
        props: dict[str, Any] = {}
        required: list[str] = []
        fields: dict[str, Any] = {}
        for pname, param in sig.parameters.items():
            tp = hints.get(pname, str)
            props[pname] = _schema_for(tp)
            if param.default is inspect.Parameter.empty:
                required.append(pname)
                fields[pname] = tp
            else:
                fields[pname] = NotRequired[tp]
        schema = {"type": "object", "properties": props, "required": required,
                  "additionalProperties": False}
        td = TypedDict(f"{fn.__name__}_args", fields)  # type: ignore[misc]
        tool_name = name or fn.__name__
        desc = inspect.getdoc(fn) or tool_name
        tool = Tool(tool_name, desc, fn, ToolSpec(name=tool_name, description=desc,
                                                  parameters=schema),
                    timeout_s, TypeAdapter(td))
        self._tools[tool_name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        """Run a tool; returns (output, is_error). Errors are returned to the model
        as observations rather than raised, so the agent can self-correct."""
        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown tool '{name}'. available: {sorted(self._tools)}", True
        try:
            return await tool.invoke(arguments), False
        except ValidationError as exc:
            return f"error: invalid arguments for '{name}': {exc.errors()}", True
        except TimeoutError:
            return f"error: tool '{name}' timed out after {tool.timeout_s}s", True
        except Exception as exc:
            return f"error: tool '{name}' raised {type(exc).__name__}: {exc}", True
