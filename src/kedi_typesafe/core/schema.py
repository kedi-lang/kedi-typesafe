from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from typesafe_sdk import Choice, JSONContent, Noul, NoulCriteria

from .errors import TypeSafeExtractionError, TypeSafeSchemaError
from .extraction import (
    BUILTIN_FORMAT_EXTRACTORS,
    CandidateExtractor,
    RegexExtractor,
    resolve_candidates,
)

QuestionKind = Literal["noul", "choice", "extract"]
TypeSafeQuestion = Noul | Choice
MAX_CHOICE_OPTIONS = 255


@dataclass(frozen=True, slots=True, kw_only=True)
class QuestionSpec:
    """One validated mapping from an output property to a Jev question."""

    key: str
    kind: QuestionKind
    instructions: str
    options: tuple[str, ...] = ()
    extractor: CandidateExtractor | None = None
    rejected_options: tuple[str, ...] = ()

    def resolve(self, state: JSONContent) -> QuestionSpec:
        if self.kind != "extract":
            return self
        extractor = cast(CandidateExtractor, self.extractor)
        candidates = resolve_candidates(
            extractor,
            state,
            field_name=self.key,
            limit=MAX_CHOICE_OPTIONS - 1,
        )
        no_match = _no_match_option(candidates)
        return replace(
            self,
            options=candidates,
            rejected_options=(no_match,),
        )

    @property
    def native_options(self) -> tuple[str, ...]:
        return self.options + self.rejected_options

    def to_native(self) -> TypeSafeQuestion:
        if self.kind == "noul":
            return Noul(
                instructions=self.instructions,
                criteria=NoulCriteria(
                    true="The proposition is established by the provided state.",
                    false="The proposition is false or is not established by the provided state.",
                ),
            )
        if self.kind == "extract" and not self.rejected_options:
            raise TypeSafeExtractionError(
                f"Output property {self.key!r} must resolve text candidates before evaluation"
            )
        criteria: dict[str, str | None] = dict.fromkeys(self.options)
        criteria.update(
            dict.fromkeys(
                self.rejected_options, "None of the candidate values satisfies the requested field."
            )
        )
        return Choice(instructions=self.instructions, criteria=criteria)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationPlan:
    """A complete, provider-ready interpretation of one Pydantic output schema."""

    questions: tuple[QuestionSpec, ...]

    def resolve(self, state: JSONContent) -> EvaluationPlan:
        return EvaluationPlan(
            questions=tuple(question.resolve(state) for question in self.questions)
        )

    def native_questions(self) -> dict[str, TypeSafeQuestion]:
        return {question.key: question.to_native() for question in self.questions}


def build_evaluation_plan(
    schema: Mapping[str, Any],
    *,
    text_extractors: Mapping[str, CandidateExtractor] | None = None,
) -> EvaluationPlan:
    """Convert Pydantic AI's native output object schema into Jev questions."""

    if schema.get("type") != "object":
        raise TypeSafeSchemaError("Jev output schema must be an object")

    raw_properties: object = schema.get("properties")
    if not isinstance(raw_properties, Mapping) or not raw_properties:
        raise TypeSafeSchemaError("Jev output schema must define at least one property")
    properties = cast(Mapping[object, object], raw_properties)
    if not all(isinstance(key, str) for key in properties):
        raise TypeSafeSchemaError("Jev output property names must be strings")
    typed_properties: dict[str, Any] = {cast(str, key): value for key, value in properties.items()}

    raw_required: object = schema.get("required")
    if not isinstance(raw_required, list):
        raise TypeSafeSchemaError("Jev output schema must declare required properties")
    required = cast(list[object], raw_required)
    if not all(isinstance(key, str) for key in required):
        raise TypeSafeSchemaError("Jev output schema must declare required properties")
    typed_required = [cast(str, key) for key in required]
    required_names = set(typed_required)
    property_names = set(typed_properties)
    extractors = dict(text_extractors or {})
    unknown_extractors = sorted(set(extractors) - property_names)
    if unknown_extractors:
        raise TypeSafeSchemaError(
            f"Text extractors reference unknown output properties: {', '.join(unknown_extractors)}"
        )
    invalid_extractors = sorted(
        name for name, extractor in extractors.items() if not callable(extractor)
    )
    if invalid_extractors:
        raise TypeSafeSchemaError(
            f"Text extractors must be callable for: {', '.join(invalid_extractors)}"
        )
    if required_names != property_names:
        missing = sorted(property_names - required_names)
        unknown = sorted(required_names - property_names)
        details: list[str] = []
        if missing:
            details.append(f"optional properties are unsupported: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown required properties: {', '.join(unknown)}")
        raise TypeSafeSchemaError("; ".join(details))

    definitions = schema.get("$defs", {})
    if not isinstance(definitions, Mapping):
        raise TypeSafeSchemaError("Jev output schema `$defs` must be an object")
    typed_definitions = cast(Mapping[str, Any], definitions)

    questions = tuple(
        _question_for_property(
            key,
            _resolve_schema(property_schema, typed_definitions, path=key),
            extractor=extractors.get(key),
        )
        for key, property_schema in typed_properties.items()
    )
    return EvaluationPlan(questions=questions)


def _question_for_property(
    key: str,
    schema: Mapping[str, Any],
    *,
    extractor: CandidateExtractor | None,
) -> QuestionSpec:
    description = schema.get("description")
    if description is not None and not isinstance(description, str):
        raise TypeSafeSchemaError(f"Output property {key!r} has a non-string description")
    description = description.strip() if description else ""

    if schema.get("type") == "boolean":
        if extractor is not None:
            raise TypeSafeSchemaError(
                f"Text extractor for output property {key!r} requires a string field"
            )
        instructions = description or _fallback_bool_instructions(key)
        return QuestionSpec(key=key, kind="noul", instructions=instructions)

    raw_enum: object = schema.get("enum")
    if schema.get("type") == "string" and isinstance(raw_enum, list):
        if extractor is not None:
            raise TypeSafeSchemaError(
                f"Output property {key!r} cannot combine enum values with a text extractor"
            )
        enum = cast(list[object], raw_enum)
        if not all(isinstance(option, str) and option for option in enum):
            raise TypeSafeSchemaError(
                f"Output property {key!r} must use non-empty string choice values"
            )
        options = tuple(cast(str, option) for option in enum)
        if len(options) < 2:
            raise TypeSafeSchemaError(
                f"Output property {key!r} must define at least two choice values"
            )
        if len(options) > 255:
            raise TypeSafeSchemaError(
                f"Output property {key!r} exceeds Jev's 255-option Choice limit"
            )
        if len(set(options)) != len(options):
            raise TypeSafeSchemaError(f"Output property {key!r} has duplicate choice values")
        instructions = description or _fallback_choice_instructions(key)
        return QuestionSpec(
            key=key,
            kind="choice",
            instructions=instructions,
            options=options,
        )

    if schema.get("type") == "string":
        resolved_extractor = extractor if extractor is not None else _schema_extractor(key, schema)
        if resolved_extractor is not None:
            instructions = description or _fallback_extraction_instructions(key)
            return QuestionSpec(
                key=key,
                kind="extract",
                instructions=instructions,
                extractor=resolved_extractor,
            )

    raise TypeSafeSchemaError(
        f"Output property {key!r} is unsupported; expected bool, a finite string "
        "Literal/Enum, or a string with a candidate extractor"
    )


def _schema_extractor(
    key: str,
    schema: Mapping[str, Any],
) -> CandidateExtractor | None:
    pattern = schema.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, str) or not pattern:
            raise TypeSafeSchemaError(
                f"Output property {key!r} must use a non-empty string extraction pattern"
            )
        try:
            return RegexExtractor(pattern=pattern)
        except re.error as exc:
            raise TypeSafeSchemaError(
                f"Output property {key!r} has an invalid extraction pattern: {exc}"
            ) from exc

    format_name = schema.get("format")
    if format_name is None:
        return None
    if not isinstance(format_name, str):
        raise TypeSafeSchemaError(f"Output property {key!r} has a non-string format")
    return BUILTIN_FORMAT_EXTRACTORS.get(format_name)


def _resolve_schema(
    value: Any,
    definitions: Mapping[str, Any],
    *,
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeSafeSchemaError(f"Output property {path!r} must contain a JSON schema object")
    typed_value = cast(Mapping[str, Any], value)
    reference = typed_value.get("$ref")
    if reference is None:
        return typed_value
    if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
        raise TypeSafeSchemaError(f"Output property {path!r} has an unsupported JSON reference")

    definition_name = reference.removeprefix("#/$defs/").replace("~1", "/").replace("~0", "~")
    target = definitions.get(definition_name)
    if not isinstance(target, Mapping):
        raise TypeSafeSchemaError(
            f"Output property {path!r} references missing definition {definition_name!r}"
        )
    typed_target = cast(Mapping[str, Any], target)
    if typed_target.get("$ref") == reference:
        raise TypeSafeSchemaError(f"Output property {path!r} contains a recursive JSON reference")
    resolved = dict(_resolve_schema(typed_target, definitions, path=path))
    resolved.update({key: item for key, item in typed_value.items() if key != "$ref"})
    return resolved


def _fallback_bool_instructions(key: str) -> str:
    if key == "response":
        return "Is the proposition or yes/no request in the provided state true?"
    return f"Is {_humanize(key)!r} true based on the provided state?"


def _fallback_choice_instructions(key: str) -> str:
    if key == "response":
        return "Which allowed value best answers the request in the provided state?"
    return f"Which allowed value best describes {_humanize(key)!r} in the provided state?"


def _fallback_extraction_instructions(key: str) -> str:
    if key == "response":
        return "Which candidate value should be extracted from the provided state?"
    return f"Which candidate value should be extracted for {_humanize(key)!r}?"


def _no_match_option(candidates: tuple[str, ...]) -> str:
    option = "__typesafe_no_matching_candidate__"
    while option in candidates:
        option = f"_{option}"
    return option


def _humanize(value: str) -> str:
    return value.replace("_", " ").strip()


__all__ = ["EvaluationPlan", "QuestionSpec", "build_evaluation_plan"]
