from __future__ import annotations

from typing import Any

import pytest
from typesafe_sdk import Choice, JSONContent, Noul

from kedi_typesafe.core import (
    RegexExtractor,
    TypeSafeExtractionError,
    TypeSafeSchemaError,
    build_evaluation_plan,
)


def _value_candidates(_state: JSONContent) -> list[str]:
    return ["value"]


def test_builds_noul_and_choice_questions_with_descriptions() -> None:
    plan = build_evaluation_plan(
        {
            "type": "object",
            "properties": {
                "approved": {
                    "type": "boolean",
                    "description": "The request is permitted by policy",
                },
                "route": {
                    "type": "string",
                    "enum": ["allow", "review", "deny"],
                    "description": "Which route should handle the request?",
                },
            },
            "required": ["approved", "route"],
        }
    )

    assert [question.kind for question in plan.questions] == ["noul", "choice"]
    native = plan.native_questions()
    assert isinstance(native["approved"], Noul)
    assert native["approved"].instructions == "The request is permitted by policy"
    criteria = native["approved"].criteria
    assert criteria is not None
    true_criterion = criteria.get("true")
    assert isinstance(true_criterion, str)
    assert true_criterion.startswith("The proposition is established")
    assert isinstance(native["route"], Choice)
    assert native["route"].instructions == "Which route should handle the request?"
    assert native["route"].criteria == {"allow": None, "review": None, "deny": None}


def test_resolves_enum_definition_and_uses_fallback_instructions() -> None:
    plan = build_evaluation_plan(
        {
            "type": "object",
            "properties": {
                "response": {"type": "boolean"},
                "review_route": {"$ref": "#/$defs/Route"},
            },
            "required": ["response", "review_route"],
            "$defs": {
                "Route": {
                    "type": "string",
                    "enum": ["manual", "automatic"],
                }
            },
        }
    )

    assert plan.questions[0].instructions == (
        "Is the proposition or yes/no request in the provided state true?"
    )
    assert plan.questions[1].instructions == (
        "Which allowed value best describes 'review route' in the provided state?"
    )


def test_resolves_format_pattern_and_custom_text_extractors() -> None:
    plan = build_evaluation_plan(
        {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "format": "email",
                    "description": "Which address is current?",
                },
                "phone": {"type": "string", "format": "phone"},
                "ticket": {"type": "string", "pattern": r"CASE-\d+"},
                "name": {"type": "string"},
            },
            "required": ["email", "phone", "ticket", "name"],
        },
        text_extractors={"name": RegexExtractor(pattern=r"(?:Dr\. )?[A-Z][a-z]+")},
    ).resolve(
        {
            "text": (
                "Old old@example.com; use new@example.com. "
                "Call +44 20 7946 0958 about CASE-42. Ask Dr. Ada."
            )
        }
    )

    assert [question.kind for question in plan.questions] == [
        "extract",
        "extract",
        "extract",
        "extract",
    ]
    assert plan.questions[0].options == ("old@example.com", "new@example.com")
    assert plan.questions[1].options == ("+44 20 7946 0958",)
    assert plan.questions[2].options == ("CASE-42",)
    assert plan.questions[3].options == ("Old", "Call", "Ask", "Dr. Ada")

    native = plan.native_questions()
    assert isinstance(native["email"], Choice)
    assert native["email"].instructions == "Which address is current?"
    assert native["phone"].instructions == (
        "Which candidate value should be extracted for 'phone'?"
    )
    assert native["email"].criteria is not None
    assert list(native["email"].criteria)[:2] == ["old@example.com", "new@example.com"]
    assert list(native["email"].criteria)[-1].endswith("no_matching_candidate__")


def test_extraction_question_requires_resolution() -> None:
    plan = build_evaluation_plan(
        {
            "type": "object",
            "properties": {"response": {"type": "string", "format": "email"}},
            "required": ["response"],
        }
    )

    with pytest.raises(TypeSafeExtractionError, match="resolve text candidates"):
        plan.native_questions()


def test_no_match_option_cannot_collide_with_candidate() -> None:
    plan = build_evaluation_plan(
        {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        text_extractors={"value": lambda _state: ["__typesafe_no_matching_candidate__"]},
    ).resolve("state")

    assert plan.questions[0].native_options == (
        "__typesafe_no_matching_candidate__",
        "___typesafe_no_matching_candidate__",
    )


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({"type": "string"}, "must be an object"),
        ({"type": "object", "properties": {}}, "at least one property"),
        (
            {"type": "object", "properties": {1: {"type": "boolean"}}},
            "names must be strings",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "boolean"}},
                "required": "value",
            },
            "must declare required",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "boolean"}},
                "required": ["value", 3],
            },
            "must declare required",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "boolean"}},
                "required": [],
            },
            "optional properties are unsupported",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "boolean"}},
                "required": ["value", "missing"],
            },
            "unknown required properties",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "boolean"}},
                "required": ["value"],
                "$defs": [],
            },
            "defs.*must be an object",
        ),
        (
            {
                "type": "object",
                "properties": {"value": "bad"},
                "required": ["value"],
            },
            "must contain a JSON schema object",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"$ref": "external.json"}},
                "required": ["value"],
            },
            "unsupported JSON reference",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"$ref": "#/$defs/Missing"}},
                "required": ["value"],
            },
            "references missing definition",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"$ref": "#/$defs/Loop"}},
                "required": ["value"],
                "$defs": {"Loop": {"$ref": "#/$defs/Loop"}},
            },
            "recursive JSON reference",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "boolean", "description": 3}},
                "required": ["value"],
            },
            "non-string description",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            "candidate extractor",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
            },
            "candidate extractor",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "string", "pattern": ""}},
                "required": ["value"],
            },
            "non-empty string extraction pattern",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "string", "pattern": "["}},
                "required": ["value"],
            },
            "invalid extraction pattern",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "string", "format": 3}},
                "required": ["value"],
            },
            "non-string format",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "string", "enum": ["only"]}},
                "required": ["value"],
            },
            "at least two choice values",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "string", "enum": ["a", "a"]}},
                "required": ["value"],
            },
            "duplicate choice values",
        ),
        (
            {
                "type": "object",
                "properties": {"value": {"type": "string", "enum": ["a", 2]}},
                "required": ["value"],
            },
            "non-empty string choice values",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "enum": [str(index) for index in range(256)]}
                },
                "required": ["value"],
            },
            "255-option",
        ),
    ],
    ids=lambda value: str(value)[:50],
)
def test_rejects_unsupported_schemas(schema: dict[Any, Any], message: str) -> None:
    with pytest.raises(TypeSafeSchemaError, match=message):
        build_evaluation_plan(schema)


def test_ref_siblings_override_definition_metadata() -> None:
    plan = build_evaluation_plan(
        {
            "type": "object",
            "properties": {
                "route": {
                    "$ref": "#/$defs/Route",
                    "description": "Pick the final route",
                }
            },
            "required": ["route"],
            "$defs": {"Route": {"type": "string", "enum": ["a", "b"]}},
        }
    )
    assert plan.questions[0].instructions == "Pick the final route"


@pytest.mark.parametrize(
    ("properties", "extractors", "message"),
    [
        (
            {"value": {"type": "string"}},
            {"missing": _value_candidates},
            "unknown output properties: missing",
        ),
        (
            {"value": {"type": "string"}},
            {"value": 3},
            "must be callable",
        ),
        (
            {"value": {"type": "boolean"}},
            {"value": _value_candidates},
            "requires a string field",
        ),
        (
            {"value": {"type": "string", "enum": ["a", "b"]}},
            {"value": _value_candidates},
            "cannot combine enum values",
        ),
    ],
)
def test_rejects_invalid_text_extractor_configuration(
    properties: dict[str, Any],
    extractors: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(TypeSafeSchemaError, match=message):
        build_evaluation_plan(
            {
                "type": "object",
                "properties": properties,
                "required": list(properties),
            },
            text_extractors=extractors,
        )
