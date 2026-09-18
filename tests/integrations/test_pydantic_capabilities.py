from __future__ import annotations

# Shared-model state and registered decorators are inspected by these contract tests.
# pyright: reportPrivateUsage=false
import asyncio
from collections.abc import Mapping
from typing import Annotated

import pytest
from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.typesafe import ToolCallProposed
from pydantic_ai.models.typesafe import TypeSafeModel as UpstreamModel
from typesafe_sdk import (
    Answer,
    Choice,
    ChoiceAnswer,
    NoulAnswer,
    Question,
    SystemOneResponse,
    Usage,
)

from kedi_typesafe import TypeSafeModel, TypeSafeModelSettings
from kedi_typesafe.core.evaluation import JSONValue


class Client:
    def __init__(self) -> None:
        self.calls: list[tuple[JSONValue, Mapping[str, Question]]] = []

    async def system_one(
        self, state: JSONValue, questions: Mapping[str, Question], *, model: str | None = None
    ) -> SystemOneResponse:
        self.calls.append((state, questions))
        answers: dict[str, Answer] = {}
        for name, question in questions.items():
            if isinstance(question, Choice):
                selected = "lookup"
                answers[name] = ChoiceAnswer(
                    choice=selected,
                    confidence=1,
                    probabilities={key: float(key == selected) for key in question.criteria},
                )
            else:
                answers[name] = NoulAnswer(noul=0.85)
        return SystemOneResponse(
            model="test", answers=answers, usage=Usage(input_tokens=10, output_tokens=2)
        )

    async def aclose(self) -> None:
        raise AssertionError("borrowed client must not be closed")


@pytest.mark.asyncio
async def test_request_threshold_does_not_mutate_shared_model() -> None:
    client = Client()
    model = TypeSafeModel(client=client)
    assert isinstance(model, UpstreamModel)
    agent = Agent(model, output_type=bool)
    low, default = await asyncio.gather(
        agent.run("claim", model_settings=TypeSafeModelSettings(typesafe_threshold=0.8)),
        agent.run("claim"),
    )
    assert low.output is True and default.output is False
    assert model._evaluator.threshold == 0.85
    assert len(client.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
async def test_streamed_output_preserves_probability(native: bool) -> None:
    class Output(BaseModel):
        probability: Annotated[float, Field(ge=0, le=1)]

    model = TypeSafeModel(client=Client())
    agent = Agent(model, output_type=NativeOutput(Output) if native else Output)
    async with agent.run_stream("claim") as result:
        assert (await result.get_output()).probability == 0.85


@pytest.mark.asyncio
async def test_zero_arg_tool_is_executed_once_and_history_is_retained() -> None:
    client = Client()
    agent = Agent(
        TypeSafeModel(client=client), output_type=bool, instructions="Check the evidence."
    )
    invoked: list[bool] = []

    @agent.tool_plain
    def lookup() -> str:
        """Look up the evidence."""
        invoked.append(True)
        return "Evidence is established"

    result = await agent.run("Check the claim")
    assert lookup.__name__ == "lookup"
    assert result.output is False
    assert invoked == [True]
    assert len(client.calls) == 2
    assert "Evidence is established" in str(client.calls[1][0])


@pytest.mark.asyncio
async def test_argument_tool_proposes_without_executing() -> None:
    client = Client()
    agent = Agent(
        TypeSafeModel(client=client), output_type=bool, instructions="Check the evidence."
    )

    @agent.tool_plain
    def lookup(query: str) -> str:
        raise AssertionError("Jev cannot invent arguments")

    with pytest.raises(ToolCallProposed) as error:
        await agent.run("Check the claim")
    assert lookup.__name__ == "lookup"
    assert error.value.tool_name == "lookup"
    assert error.value.probability == 1
