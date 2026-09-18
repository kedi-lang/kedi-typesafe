from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.output import OutputObjectDefinition

from kedi_typesafe import TypeSafeModel
from kedi_typesafe.core.evaluation import EvaluationResult
from kedi_typesafe.integrations._langchain_routing import prepare_routing, resolve_routing
from kedi_typesafe.integrations._pydantic_stream import ExtendedTypeSafeStream
from kedi_typesafe.integrations.langchain import messages_to_state


def test_route_keys_do_not_collide_and_history_is_scoped_to_latest_user() -> None:
    schema = {
        "type": "object",
        "properties": {"__typesafe_route": {"type": "boolean"}},
        "required": ["__typesafe_route"],
    }
    tools = [{"name": "__final_output"}, {"name": "old"}, {"name": "done"}]
    request = prepare_routing(
        schema,
        tools,
        [
            ToolMessage("older", name="old", tool_call_id="1"),
            HumanMessage("new"),
            AIMessage("thinking"),
            ToolMessage("ok", name="done", tool_call_id="2"),
        ],
    )
    assert request.route_key == "__typesafe_route_"
    assert {tool["name"] for tool in request.tools} == {"old", "__final_output"}
    assert "__final_output_" in request.schema["properties"][request.route_key]["enum"]
    assert len(schema["properties"]) == 1


@pytest.mark.parametrize("selected,probability", [("lookup", 0.5), ("__final_output", 1.0)])
def test_route_abstention_does_not_execute(selected: str, probability: float) -> None:
    request = prepare_routing({"properties": {}, "required": []}, [{"name": "lookup"}], [])
    key = request.route_key
    assert key is not None
    result = EvaluationResult(
        values={key: selected},
        model="test",
        input_tokens=1,
        output_tokens=1,
        metadata={"answers": {key: {"probabilities": {selected: probability}}}},
    )
    normalized, call = resolve_routing(result, request, 0.6)
    assert normalized.values == {} and call is None


def test_invalid_langchain_tool_history_is_rejected() -> None:
    message = AIMessage(
        "", invalid_tool_calls=[{"name": "lookup", "args": "{", "id": "1", "error": "bad"}]
    )
    with pytest.raises(ValueError, match="invalid assistant"):
        messages_to_state([message])


@pytest.mark.asyncio
async def test_stream_rejects_unexpected_provider_parts() -> None:
    stream = ExtendedTypeSafeStream(
        ModelRequestParameters(), ModelResponse(parts=[ThinkingPart("unexpected")])
    )
    with pytest.raises(UnexpectedModelBehavior):
        async for _ in stream:
            pass


class NoCallClient:
    async def system_one(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("No provider call expected")

    async def aclose(self) -> None:
        raise AssertionError("Borrowed transport")


@pytest.mark.asyncio
async def test_native_tools_are_rejected_before_network() -> None:
    with pytest.raises(UserError, match="Native tool"):
        await TypeSafeModel(client=NoCallClient()).request(
            [], None, ModelRequestParameters(native_tools=[WebSearchTool()])
        )


@pytest.mark.asyncio
async def test_empty_output_is_local_and_cannot_be_free_text() -> None:
    model = TypeSafeModel(client=NoCallClient())
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("Assess")])]
    response = await model.request(
        messages,
        None,
        ModelRequestParameters(
            output_mode="native",
            output_object=OutputObjectDefinition(
                json_schema={
                    "type": "object",
                    "properties": {
                        "email": {
                            "anyOf": [{"type": "string", "format": "email"}, {"type": "null"}]
                        }
                    },
                    "required": ["email"],
                }
            ),
            allow_text_output=False,
        ),
    )
    assert response.usage.input_tokens == 0
    with pytest.raises(UserError):
        await model.request(messages, None, ModelRequestParameters(allow_text_output=False))
