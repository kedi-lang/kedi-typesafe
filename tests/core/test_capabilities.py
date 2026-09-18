from __future__ import annotations

# Direct decoder assertions exercise the shared internal wire contract.
# pyright: reportPrivateUsage=false
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, EmailStr, Field
from typesafe_sdk import ChoiceAnswer, Noul, NoulAnswer, ScoreAnswer, SystemOneResponse, Usage

from kedi_typesafe import BooleanCriteria, ChoiceCriteria, Rubric
from kedi_typesafe.core import TypeSafeEvaluator, TypeSafeSchemaError, build_evaluation_plan
from kedi_typesafe.core.evaluation import validate_threshold


class Evidence(BaseModel):
    probability: Annotated[float, Field(ge=0, le=1)]
    accepted: Annotated[
        bool, BooleanCriteria(true="Evidence supports the claim", false="Insufficient evidence")
    ]


class Review(BaseModel):
    evidence: Evidence
    quality: Annotated[float, Rubric(["poor", "partial", "complete"])]
    labels: list[
        Annotated[
            Literal["urgent", "billing"],
            ChoiceCriteria({"urgent": "Time sensitive", "billing": "Payments"}),
        ]
    ]
    route: Literal["allow", "deny"] | None
    email: EmailStr | None


def test_full_plan_preserves_metadata_and_nested_fields() -> None:
    plan = build_evaluation_plan(Review.model_json_schema()).resolve("No email available")
    questions = plan.native_questions()
    assert set(questions) == {
        "evidence.probability",
        "evidence.accepted",
        "quality",
        "labels.label0",
        "labels.label1",
        "route",
    }
    assert questions["quality"].criteria == ["poor", "partial", "complete"]
    accepted, label = questions["evidence.accepted"], questions["labels.label0"]
    assert isinstance(accepted, Noul) and accepted.criteria is not None
    assert isinstance(label, Noul) and label.criteria is not None
    assert accepted.criteria.get("true") == "Evidence supports the claim"
    assert label.criteria.get("true") == "Time sensitive"
    evaluator = TypeSafeEvaluator()
    no_match = plan.questions[-2].rejected_options[0]
    response = SystemOneResponse(
        model="test",
        usage=Usage(input_tokens=30, output_tokens=8),
        answers={
            "evidence.probability": NoulAnswer(noul=0.85),
            "evidence.accepted": NoulAnswer(noul=0.85),
            "quality": ScoreAnswer(
                score=1.75,
                confidence=0.8,
                probabilities={0: 0.0, 1: 0.25, 2: 0.75},
                legend={0: "poor", 1: "partial", 2: "complete"},
            ),
            "labels.label0": NoulAnswer(noul=0.86),
            "labels.label1": NoulAnswer(noul=0.85),
            "route": ChoiceAnswer(
                choice=no_match,
                confidence=0.9,
                probabilities={"allow": 0.05, "deny": 0.05, no_match: 0.9},
            ),
        },
    )
    result = evaluator._result(response, plan)
    assert Review.model_validate(result.values) == Review(
        evidence=Evidence(probability=0.85, accepted=False),
        quality=1.75,
        labels=["urgent"],
        route=None,
        email=None,
    )
    assert result.metadata["answers"]["quality"]["score"] == 1.75
    assert result.input_tokens == 30
    assert result.metadata["answers"]["email"]["source"] == "no_candidates"


@pytest.mark.parametrize(
    "value", [None, True, False, "0.85", -0.1, 1.1, float("nan"), float("inf")]
)
def test_threshold_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        validate_threshold(value)


@pytest.mark.parametrize("value", [0, 0.85, 1])
def test_threshold_accepts_boundaries(value: float) -> None:
    assert validate_threshold(value) == value


def test_invalid_metadata_is_rejected_before_network() -> None:
    with pytest.raises(TypeSafeSchemaError):
        Rubric(["single"])

    class Invalid(BaseModel):
        answer: Annotated[bool, ChoiceCriteria({"yes": "yes", "no": "no"})]

    with pytest.raises(TypeSafeSchemaError):
        build_evaluation_plan(Invalid.model_json_schema())
