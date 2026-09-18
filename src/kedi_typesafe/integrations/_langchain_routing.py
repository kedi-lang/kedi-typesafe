from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from ..core.evaluation import EvaluationResult


class ToolCallProposed(RuntimeError):  # noqa: N818 - matches the upstream Pydantic AI proposal name
    """An explicit hand-off proposal; no arguments or execution are invented."""

    def __init__(self, tool_name: str, probability: float) -> None:
        self.tool_name = tool_name
        self.probability = probability
        super().__init__(
            f"Jev proposed {tool_name!r} (probability {probability:.3f}); an explicit argument-producing handler is required"
        )


@dataclass(frozen=True)
class RoutingRequest:
    schema: Mapping[str, Any]
    route_key: str | None
    tools: tuple[dict[str, Any], ...]


def prepare_routing(
    schema: Mapping[str, Any], tools: Sequence[dict[str, Any]], messages: list[BaseMessage]
) -> RoutingRequest:
    completed: set[str] = set()
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage) and message.name:
            completed.add(message.name)
    available = tuple(tool for tool in tools if tool["name"] not in completed)
    if not available:
        return RoutingRequest(schema, None, ())
    properties = dict(schema.get("properties", {}))
    route_key = "__typesafe_route"
    while route_key in properties:
        route_key += "_"
    final = "__final_output"
    while final in {tool["name"] for tool in available}:
        final += "_"
    criteria = {
        final: schema.get("description")
        or "Return the requested structured assessment using the available evidence."
    }
    criteria.update({tool["name"]: tool.get("description") or tool["name"] for tool in available})
    properties[route_key] = {
        "type": "string",
        "enum": list(criteria),
        "description": "Which action does this request call for?",
        "x-typesafe": {"kind": "choice", "criteria": criteria},
    }
    return RoutingRequest(
        {**schema, "properties": properties, "required": [*schema.get("required", []), route_key]},
        route_key,
        available,
    )


def resolve_routing(
    result: EvaluationResult, request: RoutingRequest, threshold: float
) -> tuple[EvaluationResult, AIMessage | None]:
    key = request.route_key
    if key is None:
        return result, None
    selected = result.values[key]
    evidence = result.metadata["answers"][key]
    probability = evidence["probabilities"][selected]
    values = {name: value for name, value in result.values.items() if name != key}
    normalized = replace(result, values=values)
    tool = next((tool for tool in request.tools if tool["name"] == selected), None)
    if tool is None or probability < threshold:
        return normalized, None
    if tool.get("parameters", {}).get("properties"):
        raise ToolCallProposed(selected, probability)
    return normalized, AIMessage(
        content="",
        tool_calls=[{"name": selected, "args": {}, "id": str(uuid4()), "type": "tool_call"}],
    )
