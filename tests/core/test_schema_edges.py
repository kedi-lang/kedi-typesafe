from __future__ import annotations

# Malformed wire schemas and direct decoding are internal contract tests.
# pyright: reportPrivateUsage=false
import pytest
from typesafe_sdk import NoulAnswer, SystemOneResponse, Usage

from kedi_typesafe import (
    TypeSafeEvaluator,
    TypeSafeResponseError,
    TypeSafeSchemaError,
    build_evaluation_plan,
)
from kedi_typesafe.core.schema import EvaluationPlan, QuestionSpec


def test_recursive_objects_are_rejected() -> None:
    schema = {
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/Node"}},
        "required": ["node"],
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Node"}},
                "required": ["child"],
            }
        },
    }
    with pytest.raises(TypeSafeSchemaError, match="recursive"):
        build_evaluation_plan(schema)


def test_recursive_alias_is_rejected() -> None:
    schema = {
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/Node"}},
        "required": ["node"],
        "$defs": {"Node": {"$ref": "#/$defs/Node"}},
    }
    with pytest.raises(TypeSafeSchemaError, match="recursive"):
        build_evaluation_plan(schema)


def test_unknown_nested_extractor_is_rejected() -> None:
    schema = {
        "type": "object",
        "properties": {
            "node": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            }
        },
        "required": ["node"],
    }
    with pytest.raises(TypeSafeSchemaError, match="unknown output"):
        build_evaluation_plan(schema, text_extractors={"node.missing": lambda _: []})


def test_direct_plan_without_schema_and_wrong_score_type() -> None:
    evaluator = TypeSafeEvaluator()
    response = SystemOneResponse(model="test", answers={"ok": NoulAnswer(noul=0.95)}, usage=Usage())
    plan = EvaluationPlan(
        questions=(QuestionSpec(key="ok", kind="noul", instructions="Is it established?"),)
    )
    assert evaluator._result(response, plan).values == {"ok": True}
    score = QuestionSpec(
        key="score", kind="score", instructions="Quality", criteria=["bad", "good"]
    )
    with pytest.raises(TypeSafeResponseError, match="Score answer"):
        evaluator._decode_score(score, NoulAnswer(noul=0.5))
