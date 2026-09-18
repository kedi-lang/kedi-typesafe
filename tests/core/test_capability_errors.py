from __future__ import annotations

from collections.abc import Callable

# Failure injection verifies internal decoder and client-ownership boundaries.
# pyright: reportPrivateUsage=false
from typing import Annotated, Any

import msgspec
import pytest
from pydantic import BaseModel, EmailStr
from typesafe_sdk import NoulAnswer, ScoreAnswer, SystemOneResponse, Usage

from kedi_typesafe import (
    BooleanCriteria,
    ChoiceCriteria,
    Rubric,
    TypeSafeEvaluator,
    TypeSafeResponseError,
    TypeSafeSchemaError,
    build_evaluation_plan,
)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Rubric([None, "good"]),
        lambda: Rubric(["bad", float("nan")]),
        lambda: ChoiceCriteria({}),
        lambda: ChoiceCriteria({1: "value"}),  # pyright: ignore[reportArgumentType] -- invalid input fixture
        lambda: BooleanCriteria(true=object(), false="no"),
    ],
)
def test_metadata_rejects_invalid_criteria(factory: Callable[[], object]) -> None:
    with pytest.raises(TypeSafeSchemaError):
        factory()


def test_conflicting_semantic_metadata_is_rejected():
    class Output(BaseModel):
        value: Annotated[
            bool, BooleanCriteria(true="yes", false="no"), BooleanCriteria(true="other", false="no")
        ]

    with pytest.raises(TypeSafeSchemaError, match="conflicting"):
        Output.model_json_schema()


@pytest.mark.parametrize(
    "field",
    [
        {"type": "object", "properties": {}},
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "boolean"}},
        {
            "type": "array",
            "items": {"type": "string", "enum": ["a", "b"]},
            "x-typesafe": {"kind": "noul"},
        },
        {"type": "boolean", "x-typesafe": "bad"},
        {"type": "boolean", "x-typesafe": {"kind": "unknown"}},
        {"type": "boolean", "x-typesafe": {"kind": "noul", "criteria": {"true": "yes"}}},
        {
            "type": "string",
            "enum": ["a", "b"],
            "x-typesafe": {"kind": "choice", "criteria": {"a": "one"}},
        },
        {
            "type": "string",
            "format": "email",
            "x-typesafe": {"kind": "noul", "criteria": {"true": "yes", "false": "no"}},
        },
        {"type": "number", "x-typesafe": {"kind": "score", "criteria": [None, "good"]}},
        {"type": "string", "x-typesafe": {"kind": "score", "criteria": ["bad", "good"]}},
        {"type": "number", "x-typesafe": {"kind": "score", "criteria": ["good"]}},
        {
            "type": "number",
            "minimum": "bad",
            "x-typesafe": {"kind": "score", "criteria": ["bad", "good"]},
        },
        {
            "type": "number",
            "minimum": 0.5,
            "x-typesafe": {"kind": "score", "criteria": ["bad", "good"]},
        },
        {"anyOf": [{"const": 1, "description": "bad"}, {"const": 2, "description": "good"}]},
    ],
)
def test_invalid_field_contract_is_rejected(field: dict[str, Any]) -> None:
    with pytest.raises(TypeSafeSchemaError):
        build_evaluation_plan(
            {"type": "object", "properties": {"field": field}, "required": ["field"]}
        )


def test_integer_rubric_rounding_and_original_evidence():
    schema = {
        "type": "object",
        "properties": {
            "score": {
                "anyOf": [{"const": 0, "description": "bad"}, {"const": 1, "description": "good"}]
            }
        },
        "required": ["score"],
    }
    plan = build_evaluation_plan(schema)
    response = SystemOneResponse(
        model="test",
        usage=Usage(),
        answers={
            "score": ScoreAnswer(
                score=0.5,
                confidence=0,
                probabilities={0: 0.5, 1: 0.5},
                legend={0: "bad", 1: "good"},
            )
        },
    )
    result = TypeSafeEvaluator()._result(response, plan)
    assert result.values == {"score": 1}
    assert result.metadata["answers"]["score"]["score"] == 0.5


@pytest.mark.parametrize(
    "changes",
    [
        {"score": -1},
        {"score": float("nan")},
        {"confidence": 2},
        {"probabilities": {0: 0.2, 1: 0.2}},
        {"probabilities": {0: float("nan"), 1: 0.5}},
        {"legend": {0: "bad"}},
    ],
)
def test_invalid_rubric_answers_are_errors(changes: dict[str, Any]) -> None:
    class Output(BaseModel):
        score: Annotated[float, Rubric(["bad", "good"])]

    plan = build_evaluation_plan(Output.model_json_schema())
    answer = ScoreAnswer(
        score=0.5, confidence=0, probabilities={0: 0.5, 1: 0.5}, legend={0: "bad", 1: "good"}
    )
    response = SystemOneResponse(
        model="test", usage=Usage(), answers={"score": msgspec.structs.replace(answer, **changes)}
    )
    with pytest.raises(TypeSafeResponseError):
        TypeSafeEvaluator()._result(response, plan)


@pytest.mark.asyncio
async def test_all_nullable_extractions_resolve_locally_without_client():
    class Output(BaseModel):
        email: EmailStr | None

    evaluator = TypeSafeEvaluator(api_key="not-used")
    result = await evaluator.evaluate(state="No contact", schema=Output.model_json_schema())
    assert result.values == {"email": None}
    assert result.input_tokens == 0 and evaluator._clients == {}
    assert evaluator.evaluate_sync(
        state="No contact", schema=Output.model_json_schema()
    ).values == {"email": None}
    assert evaluator._sync_client is None


def test_schema_constraints_remain_errors_instead_of_repair():
    schema = {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "items": {"type": "string", "enum": ["a", "b"]},
                "minItems": 1,
            }
        },
        "required": ["values"],
    }
    plan = build_evaluation_plan(schema)
    response = SystemOneResponse(
        model="test",
        usage=Usage(),
        answers={question.key: NoulAnswer(noul=0) for question in plan.questions},
    )
    with pytest.raises(TypeSafeResponseError, match="schema validation"):
        TypeSafeEvaluator()._result(response, plan)
