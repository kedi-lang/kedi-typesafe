from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from typesafe_sdk import (
    Answer,
    Choice,
    ChoiceAnswer,
    NoulAnswer,
    Question,
    SystemOneResponse,
    Usage,
)

import kedi_typesafe.core.evaluation as evaluation_module
from kedi_typesafe.core import (
    TypeSafeEvaluator,
    TypeSafeExtractionError,
    TypeSafeResponseError,
)
from kedi_typesafe.core.evaluation import JSONValue


class FakeClient:
    def __init__(self, response: SystemOneResponse) -> None:
        self.response = response
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
        return self.response

    async def aclose(self) -> None:
        self.closed = True


class FakeSyncClient:
    def __init__(self, response: SystemOneResponse) -> None:
        self.response = response
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
        return self.response

    def close(self) -> None:
        self.closed = True


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean", "description": "The request is allowed"},
        "route": {
            "type": "string",
            "enum": ["allow", "deny"],
            "description": "Select the route",
        },
    },
    "required": ["approved", "route"],
}


@pytest.mark.asyncio
async def test_evaluates_all_questions_in_one_call_and_preserves_evidence() -> None:
    client = FakeClient(
        SystemOneResponse(
            model="jev-2026-09-15",
            usage=Usage(input_tokens=42, output_tokens=9),
            answers={
                "approved": NoulAnswer(noul=0.75),
                "route": ChoiceAnswer(
                    choice="allow",
                    confidence=0.8,
                    probabilities={"allow": 0.9, "deny": 0.1},
                ),
            },
        )
    )
    evaluator = TypeSafeEvaluator("jev-latest", client=client, threshold=0.6)

    result = await evaluator.evaluate(state={"message": "please"}, schema=SCHEMA)

    assert result.values == {"approved": True, "route": "allow"}
    assert result.model == "jev-2026-09-15"
    assert result.input_tokens == 42
    assert result.output_tokens == 9
    assert result.metadata == {
        "answers": {
            "approved": {"type": "noul", "probability": 0.75},
            "route": {
                "type": "choice",
                "choice": "allow",
                "confidence": 0.8,
                "probabilities": {"allow": 0.9, "deny": 0.1},
            },
        },
        "boolean_threshold": 0.6,
        "boolean_comparator": ">",
        "usage": {"input_tokens": 42, "output_tokens": 9},
    }
    assert len(client.calls) == 1
    assert client.calls[0][0] == {"message": "please"}
    assert set(client.calls[0][1]) == {"approved", "route"}
    assert client.calls[0][2] == "jev-latest"


@pytest.mark.asyncio
async def test_threshold_is_strict_and_unknown_usage_stays_unknown() -> None:
    client = FakeClient(
        SystemOneResponse(
            model="jev-latest",
            usage=Usage(),
            answers={"response": NoulAnswer(noul=0.5)},
        )
    )
    evaluator = TypeSafeEvaluator(client=client, threshold=0.5)
    result = await evaluator.evaluate(
        state="claim",
        schema={
            "type": "object",
            "properties": {"response": {"type": "boolean"}},
            "required": ["response"],
        },
    )
    assert result.values == {"response": False}
    assert result.input_tokens is None
    assert result.output_tokens is None


def test_sync_evaluation_uses_the_same_plan_and_decoder() -> None:
    client = FakeSyncClient(
        SystemOneResponse(
            model="jev-sync",
            usage=Usage(input_tokens=4, output_tokens=1),
            answers={"response": NoulAnswer(noul=0.8)},
        )
    )
    evaluator = TypeSafeEvaluator(sync_client=client)

    result = evaluator.evaluate_sync(
        state="claim",
        schema={
            "type": "object",
            "properties": {"response": {"type": "boolean"}},
            "required": ["response"],
        },
    )

    assert result.values == {"response": True}
    assert client.calls[0][0] == "claim"
    assert client.calls[0][2] == "jev-latest"


@pytest.mark.asyncio
async def test_constrained_text_is_selected_from_state_candidates() -> None:
    no_match = "__typesafe_no_matching_candidate__"
    client = FakeClient(
        SystemOneResponse(
            model="jev-latest",
            usage=Usage(input_tokens=18, output_tokens=3),
            answers={
                "email": ChoiceAnswer(
                    choice="new@example.com",
                    confidence=0.9,
                    probabilities={
                        "old@example.com": 0.05,
                        "new@example.com": 0.9,
                        no_match: 0.05,
                    },
                )
            },
        )
    )
    evaluator = TypeSafeEvaluator(client=client)
    result = await evaluator.evaluate(
        state="Old: old@example.com. Preferred: new@example.com.",
        schema={
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "format": "email",
                    "description": "Which email address is preferred?",
                }
            },
            "required": ["email"],
        },
    )

    assert result.values == {"email": "new@example.com"}
    assert result.metadata["answers"]["email"] == {
        "type": "extraction",
        "choice": "new@example.com",
        "confidence": 0.9,
        "probabilities": {
            "old@example.com": 0.05,
            "new@example.com": 0.9,
            no_match: 0.05,
        },
    }
    question = client.calls[0][1]["email"]
    assert isinstance(question, Choice)
    assert question.criteria == {
        "old@example.com": None,
        "new@example.com": None,
        no_match: "None of the candidate values satisfies the requested field.",
    }


@pytest.mark.asyncio
async def test_constrained_text_fails_before_provider_when_no_candidates_exist() -> None:
    client = FakeClient(SystemOneResponse(model="jev", usage=Usage(), answers={}))
    evaluator = TypeSafeEvaluator(client=client)

    with pytest.raises(TypeSafeExtractionError, match="found no candidates"):
        await evaluator.evaluate(
            state="No contact details are present.",
            schema={
                "type": "object",
                "properties": {"email": {"type": "string", "format": "email"}},
                "required": ["email"],
            },
        )

    assert client.calls == []


@pytest.mark.asyncio
async def test_constrained_text_rejects_no_match_answer() -> None:
    no_match = "__typesafe_no_matching_candidate__"
    client = FakeClient(
        SystemOneResponse(
            model="jev",
            usage=Usage(),
            answers={
                "email": ChoiceAnswer(
                    choice=no_match,
                    confidence=0.8,
                    probabilities={"only@example.com": 0.2, no_match: 0.8},
                )
            },
        )
    )

    with pytest.raises(TypeSafeResponseError, match="could not extract"):
        await TypeSafeEvaluator(client=client).evaluate(
            state="only@example.com",
            schema={
                "type": "object",
                "properties": {"email": {"type": "string", "format": "email"}},
                "required": ["email"],
            },
        )


@pytest.mark.parametrize("threshold", [True, float("nan"), float("inf"), -0.1, 1.1])
def test_rejects_invalid_threshold(threshold: float) -> None:
    response = SystemOneResponse(model="jev", usage=Usage(), answers={})
    with pytest.raises(ValueError, match="threshold"):
        TypeSafeEvaluator(client=FakeClient(response), threshold=threshold)


def test_rejects_empty_model_name() -> None:
    response = SystemOneResponse(model="jev", usage=Usage(), answers={})
    with pytest.raises(ValueError, match="model name"):
        TypeSafeEvaluator("  ", client=FakeClient(response))


@pytest.mark.asyncio
async def test_only_closes_owned_client(monkeypatch: pytest.MonkeyPatch) -> None:
    response = SystemOneResponse(model="jev", usage=Usage(), answers={})
    borrowed = FakeClient(response)
    evaluator = TypeSafeEvaluator(client=borrowed)
    await evaluator.aclose()
    assert borrowed.closed is False

    owned = FakeClient(
        SystemOneResponse(
            model="jev",
            usage=Usage(),
            answers={"response": NoulAnswer(noul=0.8)},
        )
    )
    constructor_options: dict[str, object] = {}

    def build_client(**kwargs: object) -> FakeClient:
        constructor_options.update(kwargs)
        return owned

    monkeypatch.setattr(evaluation_module, "AsyncTypeSafeClient", build_client)
    evaluator = TypeSafeEvaluator("jev-test", api_key="secret", timeout=3.0)
    await evaluator.evaluate(
        state="claim",
        schema={
            "type": "object",
            "properties": {"response": {"type": "boolean"}},
            "required": ["response"],
        },
    )
    await evaluator.evaluate(
        state="claim",
        schema={
            "type": "object",
            "properties": {"response": {"type": "boolean"}},
            "required": ["response"],
        },
    )
    await evaluator.aclose()
    await evaluator.aclose_current()
    assert owned.closed is True
    assert constructor_options == {
        "api_key": "secret",
        "model": "jev-test",
        "timeout": 3.0,
    }


def test_only_closes_owned_sync_client(monkeypatch: pytest.MonkeyPatch) -> None:
    response = SystemOneResponse(
        model="jev",
        usage=Usage(),
        answers={"response": NoulAnswer(noul=0.8)},
    )
    borrowed = FakeSyncClient(response)
    evaluator = TypeSafeEvaluator(sync_client=borrowed)
    evaluator.close()
    assert borrowed.closed is False

    owned = FakeSyncClient(response)
    constructor_options: dict[str, object] = {}

    def build_client(**kwargs: object) -> FakeSyncClient:
        constructor_options.update(kwargs)
        return owned

    monkeypatch.setattr(evaluation_module, "TypeSafeClient", build_client)
    evaluator = TypeSafeEvaluator("jev-test", api_key="secret", timeout=3.0)
    evaluator.evaluate_sync(
        state="claim",
        schema={
            "type": "object",
            "properties": {"response": {"type": "boolean"}},
            "required": ["response"],
        },
    )
    evaluator.close()
    evaluator.close()

    assert owned.closed is True
    assert constructor_options == {
        "api_key": "secret",
        "model": "jev-test",
        "timeout": 3.0,
    }


def _single_bool_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"approved": {"type": "boolean"}},
        "required": ["approved"],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answers", "message"),
    [
        ({}, "answer keys do not match"),
        ({"approved": ChoiceAnswer(choice="a", confidence=1, probabilities={"a": 1})}, "Noul"),
        ({"approved": NoulAnswer(noul=-0.1)}, "invalid probability"),
    ],
)
async def test_rejects_invalid_noul_responses(
    answers: dict[str, Answer],
    message: str,
) -> None:
    client = FakeClient(SystemOneResponse(model="jev", usage=Usage(), answers=answers))
    with pytest.raises(TypeSafeResponseError, match=message):
        await TypeSafeEvaluator(client=client).evaluate(
            state="state",
            schema=_single_bool_schema(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "message"),
    [
        (NoulAnswer(noul=0.8), "Choice"),
        (
            ChoiceAnswer(choice="other", confidence=1, probabilities={"allow": 0, "deny": 1}),
            "unsupported value",
        ),
        (
            ChoiceAnswer(choice="allow", confidence=1, probabilities={"allow": 1}),
            "do not match",
        ),
        (
            ChoiceAnswer(
                choice="allow",
                confidence=1,
                probabilities={"allow": float("nan"), "deny": 0},
            ),
            "invalid probabilities",
        ),
        (
            ChoiceAnswer(choice="allow", confidence=1, probabilities={"allow": 0.8, "deny": 0.3}),
            "sum to 1",
        ),
        (
            ChoiceAnswer(
                choice="allow",
                confidence=1.1,
                probabilities={"allow": 0.8, "deny": 0.2},
            ),
            "invalid confidence",
        ),
    ],
)
async def test_rejects_invalid_choice_responses(
    answer: NoulAnswer | ChoiceAnswer,
    message: str,
) -> None:
    schema = {
        "type": "object",
        "properties": {"route": {"type": "string", "enum": ["allow", "deny"]}},
        "required": ["route"],
    }
    client = FakeClient(SystemOneResponse(model="jev", usage=Usage(), answers={"route": answer}))
    with pytest.raises(TypeSafeResponseError, match=message):
        await TypeSafeEvaluator(client=client).evaluate(state="state", schema=schema)
