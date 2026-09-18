from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import httpx2
import pytest
from langchain.agents import (
    create_agent,  # pyright: ignore[reportUnknownVariableType] -- upstream graph generics
)
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_typesafe import TypeSafeClassifier
from pydantic import BaseModel
from typesafe_sdk import (
    Answer,
    Choice,
    ChoiceAnswer,
    NoulAnswer,
    Question,
    SystemOneResponse,
    Usage,
)
from typing_extensions import TypedDict

from kedi_typesafe.core.evaluation import JSONValue
from kedi_typesafe.integrations import _langchain_transport as transport_module
from kedi_typesafe.integrations.langchain import ToolCallProposed, TypeSafeChatModel


class Output(BaseModel):
    """Assess whether the claim is established."""

    supported: bool


class Context(TypedDict):
    pass


class Client:
    def __init__(self) -> None:
        self.calls: list[tuple[JSONValue, Mapping[str, Question]]] = []

    async def system_one(
        self, state: JSONValue, questions: Mapping[str, Question], *, model: str | None = None
    ) -> SystemOneResponse:
        self.calls.append((state, questions))
        answers: dict[str, Answer] = {}
        for key, question in questions.items():
            if isinstance(question, Choice):
                answers[key] = ChoiceAnswer(
                    choice="lookup",
                    confidence=1,
                    probabilities={
                        option: float(option == "lookup") for option in question.criteria
                    },
                )
            else:
                answers[key] = NoulAnswer(noul=0.85)
        return SystemOneResponse(
            model="test", answers=answers, usage=Usage(input_tokens=10, output_tokens=2)
        )

    async def aclose(self) -> None:
        raise AssertionError("borrowed client must not be closed")


@pytest.mark.asyncio
async def test_threshold_overrides_are_concurrent_and_isolated():
    model = TypeSafeChatModel(client=Client())
    high = model.with_structured_output(Output)
    low = model.model_copy(update={"typesafe_threshold": 0.8}).with_structured_output(Output)
    first, second = await asyncio.gather(high.ainvoke("claim"), low.ainvoke("claim"))
    assert first.supported is False and second.supported is True
    assert model.threshold == 0.85


@pytest.mark.asyncio
async def test_agent_runs_zero_arg_tool_and_then_returns_typed_output():
    client = Client()
    invoked: list[bool] = []

    def lookup() -> str:
        """Look up supporting evidence."""
        invoked.append(True)
        return "Evidence found"

    agent = create_agent(
        TypeSafeChatModel(client=client),
        tools=[lookup],
        response_format=Output,
        context_schema=Context,
    )
    result = await agent.ainvoke({"messages": [HumanMessage("Check the claim")]})  # pyright: ignore[reportUnknownMemberType] -- upstream Command generic is unrelated to this message input
    assert result["structured_response"] == Output(supported=False)
    assert invoked == [True]
    assert len(client.calls) == 2
    assert "Evidence found" in str(client.calls[-1][0])


@pytest.mark.asyncio
async def test_argument_tool_proposes_without_execution():
    def lookup(query: str) -> str:
        """Look up evidence for a query."""
        raise AssertionError("must not execute fabricated arguments")

    agent = create_agent(
        TypeSafeChatModel(client=Client()),
        tools=[lookup],
        response_format=Output,
        context_schema=Context,
    )
    with pytest.raises(ToolCallProposed) as error:
        await agent.ainvoke({"messages": [HumanMessage("Check claim")]})  # pyright: ignore[reportUnknownMemberType] -- upstream Command generic
    assert error.value.tool_name == "lookup"


class Calls(BaseCallbackHandler):
    def __init__(self) -> None:
        self.starts = 0

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        self.starts += 1


@pytest.mark.asyncio
async def test_official_transport_is_pooled_reports_usage_once_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    clients: list[tuple[httpx2.Client, httpx2.AsyncClient]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx2.Response(
            200,
            json={
                "model": "test",
                "answers": {key: {"type": "noul", "noul": 0.95} for key in payload["questions"]},
                "usage": {"input_tokens": 12, "output_tokens": 3},
            },
        )

    def classifier(**kwargs: Any) -> TypeSafeClassifier:
        sync = httpx2.Client(transport=httpx2.MockTransport(handler))
        asynchronous = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
        clients.append((sync, asynchronous))
        return TypeSafeClassifier(**kwargs, client=sync, async_client=asynchronous)

    monkeypatch.setattr(transport_module, "TypeSafeClassifier", classifier)
    callbacks = Calls()
    async with TypeSafeChatModel(api_key="test") as model:
        runnable = model.with_structured_output(Output, include_raw=True)
        for _ in range(2):
            result = await runnable.ainvoke("claim", config={"callbacks": [callbacks]})
            assert result["parsed"].supported is True
            assert result["raw"].usage_metadata == {
                "input_tokens": 12,
                "output_tokens": 3,
                "total_tokens": 15,
            }
    assert len(requests) == 2 and len(clients) == 1 and callbacks.starts == 2
    assert all(sync.is_closed and asynchronous.is_closed for sync, asynchronous in clients)
    await model.aclose()
    with TypeSafeChatModel(api_key="test") as sync_model:
        runnable = sync_model.with_structured_output(Output)
        for _ in range(2):
            assert runnable.invoke("claim").supported is True
    sync_model.close()
    assert len(requests) == 4 and len(clients) == 2
    assert clients[-1][0].is_closed
    # The paired async client never opened a connection; dispose the test fixture.
    await clients[-1][1].aclose()
    transport_module.SyncClassifierTransport(
        transport_module.ClassifierTransport(api_key="unused", timeout=None)
    ).close()
