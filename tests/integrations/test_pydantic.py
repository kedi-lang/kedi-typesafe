from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Literal

import pytest
from pydantic import BaseModel, EmailStr, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ImageUrl,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_extra_types.phone_numbers import PhoneNumber
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
from kedi_typesafe.integrations.pydantic import TypeSafeModel, messages_to_state


class Route(str, Enum):
    ALLOW = "allow"
    REVIEW = "review"
    DENY = "deny"


class Decision(BaseModel):
    approved: bool = Field(description="The request is permitted by policy")
    route: Route = Field(description="Which route should handle the request?")


class Contact(BaseModel):
    email: EmailStr = Field(description="Which email address should receive the reply?")
    phone: PhoneNumber = Field(description="Which phone number should receive the call?")


class FakeClient:
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


def _response(**answers: Answer) -> SystemOneResponse:
    return SystemOneResponse(
        model="jev-2026-09-15",
        usage=Usage(input_tokens=27, output_tokens=6),
        answers=answers,
    )


def test_plain_pydantic_agent_returns_validated_model_and_one_provider_call() -> None:
    client = FakeClient(
        [
            _response(
                approved=NoulAnswer(noul=0.92),
                route=ChoiceAnswer(
                    choice="review",
                    confidence=0.7,
                    probabilities={"allow": 0.1, "review": 0.8, "deny": 0.1},
                ),
            )
        ]
    )
    model = TypeSafeModel("jev-latest", client=client)
    agent = Agent(model, instructions="Apply the supplied policy.", output_type=Decision)

    result = agent.run_sync("The account has two conflicting verification records.")

    assert result.output == Decision(approved=True, route=Route.REVIEW)
    assert len(client.calls) == 1
    state, questions, selected_model = client.calls[0]
    assert state == {
        "state": "The account has two conflicting verification records.",
        "instructions": "Apply the supplied policy.",
    }
    assert selected_model == "jev-latest"
    assert isinstance(questions["approved"], Noul)
    assert questions["approved"].instructions == "The request is permitted by policy"
    assert isinstance(questions["route"], Choice)
    assert questions["route"].criteria == {"allow": None, "review": None, "deny": None}

    final_message = next(
        message for message in reversed(result.all_messages()) if isinstance(message, ModelResponse)
    )
    assert isinstance(final_message, ModelResponse)
    assert final_message.usage.input_tokens == 27
    assert final_message.usage.output_tokens == 6
    assert final_message.model_name == "jev-2026-09-15"
    assert final_message.provider_name == "typesafe"
    assert final_message.metadata == {
        "typesafe": {
            "answers": {
                "approved": {"type": "noul", "probability": 0.92},
                "route": {
                    "type": "choice",
                    "choice": "review",
                    "confidence": 0.7,
                    "probabilities": {"allow": 0.1, "review": 0.8, "deny": 0.1},
                },
            },
            "boolean_threshold": 0.85,
            "boolean_comparator": ">",
            "usage": {"input_tokens": 27, "output_tokens": 6},
        }
    }


def test_plain_pydantic_agent_extracts_constrained_text_fields() -> None:
    no_match = "__typesafe_no_matching_candidate__"
    client = FakeClient(
        [
            _response(
                email=ChoiceAnswer(
                    choice="new@example.com",
                    confidence=0.9,
                    probabilities={
                        "old@example.com": 0.05,
                        "new@example.com": 0.9,
                        no_match: 0.05,
                    },
                ),
                phone=ChoiceAnswer(
                    choice="+90 555 111 22 33",
                    confidence=0.9,
                    probabilities={
                        "+90 555 111 22 33": 0.9,
                        no_match: 0.1,
                    },
                ),
            )
        ]
    )
    agent = Agent(TypeSafeModel(client=client), output_type=Contact)

    result = agent.run_sync(
        "Ignore old@example.com. Reply to new@example.com and call +90 555 111 22 33."
    )

    assert result.output.email == "new@example.com"
    assert result.output.phone == "tel:+90-555-111-22-33"
    _, questions, _ = client.calls[0]
    assert isinstance(questions["email"], Choice)
    assert list(questions["email"].criteria) == [
        "old@example.com",
        "new@example.com",
        no_match,
    ]
    assert isinstance(questions["phone"], Choice)
    assert list(questions["phone"].criteria) == [
        "+90 555 111 22 33",
        no_match,
    ]


def test_plain_string_uses_explicit_field_extractor() -> None:
    class Person(BaseModel):
        name: str = Field(description="Which person owns the account?")

    no_match = "__typesafe_no_matching_candidate__"
    client = FakeClient(
        [
            _response(
                name=ChoiceAnswer(
                    choice="Ada",
                    confidence=0.8,
                    probabilities={"Ada": 0.8, "Grace": 0.1, no_match: 0.1},
                )
            )
        ]
    )
    model = TypeSafeModel(
        client=client,
        text_extractors={"name": lambda _state: ["Ada", "Grace"]},
    )

    result = Agent(model, output_type=Person).run_sync("The account belongs to Ada, not Grace.")

    assert result.output == Person(name="Ada")


@pytest.mark.parametrize(
    ("output_type", "answer", "expected"),
    [
        (bool, NoulAnswer(noul=0.51), False),
        (
            Literal["yes", "no"],
            ChoiceAnswer(choice="yes", confidence=1, probabilities={"yes": 1, "no": 0}),
            "yes",
        ),
    ],
)
def test_supports_root_bool_and_literal(
    output_type: object,
    answer: NoulAnswer | ChoiceAnswer,
    expected: object,
) -> None:
    client = FakeClient([_response(response=answer)])
    model = TypeSafeModel(client=client)
    agent = Agent(model, output_type=output_type)  # type: ignore[arg-type]
    assert agent.run_sync("Evaluate this request").output == expected


def test_rejects_text_generation_before_calling_provider() -> None:
    client = FakeClient([])
    model = TypeSafeModel(client=client)
    with pytest.raises(UserError, match="Text output is not supported"):
        Agent(model).run_sync("Write a story")
    assert client.calls == []


def test_tools_require_an_explicit_purpose_before_calling_provider() -> None:
    client = FakeClient([])
    agent = Agent(TypeSafeModel(client=client), output_type=bool)

    @agent.tool_plain
    def lookup() -> str:
        return "evidence"

    _ = lookup

    with pytest.raises(UserError, match="docstring"):
        agent.run_sync("Is evidence available?")
    assert client.calls == []


def test_rejects_model_settings_before_calling_provider() -> None:
    client = FakeClient([])
    model = TypeSafeModel(client=client)
    agent = Agent(model, output_type=bool, model_settings={"temperature": 0})
    with pytest.raises(UserError, match="model settings: temperature"):
        agent.run_sync("Is this true?")
    assert client.calls == []


@pytest.mark.asyncio
async def test_context_manager_does_not_close_borrowed_client() -> None:
    client = FakeClient([])
    async with TypeSafeModel("jev-special", client=client) as model:
        assert model.model_name == "jev-special"
        assert model.system == "typesafe"
        assert model.profile.get("supports_tools") is True
        assert model.profile.get("supports_json_schema_output") is True
        await model.aclose_current()
    assert client.closed is False


def test_unknown_usage_is_not_converted_to_zero() -> None:
    client = FakeClient(
        [
            SystemOneResponse(
                model="jev-latest",
                usage=Usage(),
                answers={"response": NoulAnswer(noul=0.8)},
            )
        ]
    )
    result = Agent(TypeSafeModel(client=client), output_type=bool).run_sync("Evaluate")
    final_message = next(
        message for message in reversed(result.all_messages()) if isinstance(message, ModelResponse)
    )
    assert isinstance(final_message, ModelResponse)
    assert final_message.usage.input_tokens == 0
    assert final_message.usage.output_tokens == 0
    assert final_message.usage.details == {}
    assert final_message.metadata is not None
    assert final_message.metadata["typesafe"]["usage"] == {
        "input_tokens": None,
        "output_tokens": None,
    }


def test_messages_to_state_preserves_text_history_and_latest_instructions() -> None:
    state = messages_to_state(
        [
            ModelRequest(
                parts=[UserPromptPart(["first", TextContent(" second")])],
                instructions="old",
            ),
            ModelResponse(parts=[TextPart('{"response":true}')]),
            ModelRequest(
                parts=[
                    SystemPromptPart("runtime system context"),
                    RetryPromptPart("invalid previous value"),
                    UserPromptPart("next"),
                ],
                instructions="new",
            ),
        ]
    )
    assert state == {
        "history": [
            {"user": "first\n\n second"},
            {"assistant": '{"response":true}'},
            {"system": "runtime system context"},
            {
                "retry": (
                    "Validation feedback:\ninvalid previous value\n\nFix the errors and try again."
                ),
            },
        ],
        "text": "next",
    }


def test_messages_to_state_rejects_empty_and_media() -> None:
    with pytest.raises(UserError, match="without user text"):
        messages_to_state([])

    media_request = ModelRequest(
        parts=[UserPromptPart([ImageUrl("https://example.com/evidence.png")])]
    )
    with pytest.raises(UserError):
        messages_to_state([media_request])


@pytest.mark.asyncio
async def test_request_rejects_image_output_contract() -> None:
    client = FakeClient([])
    model = TypeSafeModel(client=client)
    parameters = ModelRequestParameters(
        output_mode="native",
        output_object=None,
        allow_image_output=True,
    )
    with pytest.raises(UserError, match="Image output is not supported"):
        await model.request([ModelRequest.user_text_prompt("state")], None, parameters)


@pytest.mark.asyncio
async def test_model_default_settings_are_validated() -> None:
    client = FakeClient([])
    settings = ModelSettings(temperature=0.2)
    model = TypeSafeModel(client=client, settings=settings)
    parameters = ModelRequestParameters()
    with pytest.raises(UserError, match="temperature"):
        await model.request([ModelRequest.user_text_prompt("state")], None, parameters)
