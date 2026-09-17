from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

import pytest
from langchain.agents import create_agent  # pyright: ignore[reportUnknownVariableType]
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field, model_validator
from typesafe_sdk import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    SystemOneResponse,
    Usage,
)

from kedi_typesafe.core.evaluation import JSONValue
from kedi_typesafe.integrations.langchain import TypeSafeChatModel, messages_to_state


class Decision(BaseModel):
    approved: bool = Field(description="The request is permitted by policy")
    route: Literal["allow", "deny"] = Field(description="Select the policy route")


class FakeAsyncClient:
    def __init__(self, responses: list[SystemOneResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[JSONValue, Mapping[str, Question], str | None]] = []
        self.closed = False

    async def system_one(
        self,
        state: JSONValue,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
    ) -> SystemOneResponse:
        self.calls.append((state, questions, model))
        return self.responses.pop(0)

    async def aclose(self) -> None:
        self.closed = True


class FakeSyncClient:
    def __init__(self, responses: list[SystemOneResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[JSONValue, Mapping[str, Question], str | None]] = []
        self.closed = False

    def system_one(
        self,
        state: JSONValue,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
    ) -> SystemOneResponse:
        self.calls.append((state, questions, model))
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True


def _response(
    *,
    usage: Usage | None = None,
    **answers: Answer,
) -> SystemOneResponse:
    return SystemOneResponse(
        model="jev-test",
        usage=usage or Usage(input_tokens=19, output_tokens=3),
        answers=answers,
    )


def _decision_response() -> SystemOneResponse:
    return _response(
        approved=NoulAnswer(noul=0.9),
        route=ChoiceAnswer(
            choice="allow",
            confidence=0.9,
            probabilities={"allow": 0.9, "deny": 0.1},
        ),
    )


def _response_format(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": "Decision", "schema": schema},
    }


def test_plain_langchain_agent_returns_validated_model_and_usage() -> None:
    client = FakeSyncClient([_decision_response()])
    model = TypeSafeChatModel("jev-latest", sync_client=client)
    agent = cast(
        Any,
        create_agent(
            model,
            response_format=Decision,
            system_prompt="Apply the supplied policy.",
        ),
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "This request follows policy."}]}
    )

    assert result["structured_response"] == Decision(approved=True, route="allow")
    assert len(client.calls) == 1
    state, questions, selected_model = client.calls[0]
    assert state == {
        "messages": [
            {"role": "system", "content": "Apply the supplied policy."},
            {"role": "user", "content": "This request follows policy."},
        ]
    }
    assert selected_model == "jev-latest"
    assert isinstance(questions["approved"], Noul)
    assert isinstance(questions["route"], Choice)
    message = result["messages"][-1]
    assert message.usage_metadata == {
        "input_tokens": 19,
        "output_tokens": 3,
        "total_tokens": 22,
    }
    assert message.response_metadata["provider_name"] == "typesafe"
    assert model.asdict() == {
        "model_name": "jev-latest",
        "threshold": 0.5,
        "_type": "typesafe-jev",
    }
    assert model.profile is not None and model.profile.get("structured_output") is True


@pytest.mark.asyncio
async def test_plain_langchain_agent_supports_native_async_calls() -> None:
    client = FakeAsyncClient([_decision_response()])
    model = TypeSafeChatModel(client=client)

    agent = cast(Any, create_agent(model, response_format=Decision))
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Evaluate this request."}]}
    )

    assert result["structured_response"] == Decision(approved=True, route="allow")
    assert len(client.calls) == 1


def test_with_structured_output_supports_pydantic_and_raw_schemas() -> None:
    client = FakeSyncClient(
        [
            _decision_response(),
            _response(response=NoulAnswer(noul=0.8)),
            _response(response=NoulAnswer(noul=0.8)),
        ]
    )
    model = TypeSafeChatModel(sync_client=client)

    assert model.with_structured_output(Decision).invoke("Evaluate") == Decision(
        approved=True,
        route="allow",
    )
    raw_schema = {
        "title": "BooleanResponse",
        "type": "object",
        "properties": {"response": {"type": "boolean"}},
        "required": ["response"],
    }
    assert model.with_structured_output(raw_schema).invoke("Evaluate") == {"response": True}
    openai_schema = {
        "type": "function",
        "function": {"name": "BooleanResponse", "parameters": raw_schema},
    }
    included = model.with_structured_output(openai_schema, include_raw=True).invoke("Evaluate")
    assert included["parsed"] == {"response": True}
    assert isinstance(included["raw"], AIMessage)
    assert included["parsing_error"] is None


def test_with_structured_output_can_return_parsing_error_with_raw_message() -> None:
    class RejectTrue(BaseModel):
        response: bool

        @model_validator(mode="after")
        def reject_true(self) -> RejectTrue:
            if self.response:
                raise ValueError("true is rejected after schema validation")
            return self

    client = FakeSyncClient([_response(response=NoulAnswer(noul=0.8))])
    output = (
        TypeSafeChatModel(sync_client=client)
        .with_structured_output(
            RejectTrue,
            include_raw=True,
        )
        .invoke("Evaluate")
    )

    assert output["parsed"] is None
    assert isinstance(output["raw"], AIMessage)
    assert isinstance(output["parsing_error"], ValueError)

    failing_model = TypeSafeChatModel(
        sync_client=FakeSyncClient([_response(response=NoulAnswer(noul=0.8))])
    )
    with pytest.raises(ValueError, match="true is rejected"):
        failing_model.with_structured_output(RejectTrue).invoke("Evaluate")


def test_messages_to_state_accepts_text_history_and_rejects_unsupported_parts() -> None:
    assert messages_to_state(
        [
            ChatMessage(role="system", content="policy"),
            HumanMessage(content=["first", {"type": "text", "text": "second"}]),
            AIMessage(content="answer"),
        ]
    ) == {
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "first\nsecond"},
            {"role": "assistant", "content": "answer"},
        ]
    }

    with pytest.raises(ValueError, match="at least one"):
        messages_to_state([])
    with pytest.raises(ValueError, match="tool"):
        messages_to_state([ToolMessage(content="result", tool_call_id="call-1")])
    with pytest.raises(ValueError, match="tool-call"):
        messages_to_state(
            [AIMessage(content="", tool_calls=[{"name": "lookup", "args": {}, "id": "1"}])]
        )
    with pytest.raises(ValueError, match="image"):
        messages_to_state(
            [HumanMessage(content=[{"type": "image_url", "image_url": "https://example.com"}])]
        )
    with pytest.raises(ValueError, match="non-string"):
        messages_to_state([HumanMessage(content=[{"type": "text", "text": 1}])])
    with pytest.raises(ValueError, match="chat"):
        messages_to_state([ChatMessage(role="critic", content="unsupported")])


def test_rejects_tools_raw_calls_and_model_settings_before_provider() -> None:
    client = FakeSyncClient([])
    model = TypeSafeChatModel(sync_client=client)
    schema = Decision.model_json_schema()
    response_format = _response_format(schema)

    with pytest.raises(ValueError, match="does not support tools"):
        model.bind_tools([lambda: None], response_format=response_format)
    with pytest.raises(ValueError, match="tool choice"):
        model.bind_tools([], tool_choice="any", response_format=response_format)
    with pytest.raises(ValueError, match="requires native structured output"):
        model.bind_tools([])
    with pytest.raises(ValueError, match="temperature"):
        model.bind_tools([], response_format=response_format, temperature=0)
    with pytest.raises(ValueError, match="requires native structured output"):
        model.invoke("Write free-form text")
    with pytest.raises(ValueError, match="temperature"):
        model.bind(response_format=response_format, temperature=0).invoke("Evaluate")
    with pytest.raises(ValueError, match="stop sequences"):
        model.bind(response_format=response_format).invoke("Evaluate", stop=["done"])
    with pytest.raises(ValueError, match="model settings"):
        model.with_structured_output(Decision, temperature=0)

    assert client.calls == []


@pytest.mark.parametrize(
    "response_format",
    [
        {"type": "text"},
        {"type": "json_schema"},
        {"type": "json_schema", "json_schema": {}},
    ],
)
def test_rejects_invalid_native_response_formats(response_format: dict[str, Any]) -> None:
    model = TypeSafeChatModel(sync_client=FakeSyncClient([]))
    with pytest.raises(ValueError, match="JSON Schema|response schema"):
        model.bind(response_format=response_format).invoke("Evaluate")


def test_rejects_invalid_direct_schema() -> None:
    model = TypeSafeChatModel(sync_client=FakeSyncClient([]))
    with pytest.raises(ValueError, match="invalid structured output schema"):
        model.with_structured_output({"type": "function", "function": {"name": "Broken"}})


@pytest.mark.asyncio
async def test_context_managers_close_only_owned_clients() -> None:
    async_client = FakeAsyncClient([])
    sync_client = FakeSyncClient([])
    async with TypeSafeChatModel(client=async_client, sync_client=sync_client) as model:
        assert model.model_name == "jev-latest"
        await model.aclose_current()
    assert async_client.closed is False
    assert sync_client.closed is False

    with TypeSafeChatModel(client=async_client, sync_client=sync_client):
        pass
    assert async_client.closed is False
    assert sync_client.closed is False


def test_unknown_usage_remains_in_metadata_without_usage_counters() -> None:
    client = FakeSyncClient(
        [
            _response(
                usage=Usage(),
                response=NoulAnswer(noul=0.8),
            )
        ]
    )
    schema = {
        "type": "object",
        "properties": {"response": {"type": "boolean"}},
        "required": ["response"],
    }
    output = (
        TypeSafeChatModel(sync_client=client)
        .bind(response_format=_response_format(schema))
        .invoke("Evaluate")
    )

    assert output.usage_metadata is None
    assert output.response_metadata["typesafe"]["usage"] == {
        "input_tokens": None,
        "output_tokens": None,
    }
    assert client.calls
