"""Pydantic schema metadata for Jev questions."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import Field, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema

from ._json_types import is_list
from .errors import TypeSafeSchemaError

SCHEMA_KEY = "x-typesafe"
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


def _content(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeSafeSchemaError("TypeSafe criteria must be finite JSON values") from exc


def _extend(
    schema: CoreSchema, handler: GetJsonSchemaHandler, kind: str, criteria: Any
) -> JsonSchemaValue:
    result = dict(handler.resolve_ref_schema(handler(schema)))
    if SCHEMA_KEY in result:
        raise TypeSafeSchemaError("An output field cannot have conflicting TypeSafe criteria")
    result[SCHEMA_KEY] = {"kind": kind, "criteria": _content(criteria)}
    return result


@dataclass(frozen=True)
class Rubric:
    """Ordered described levels; float outputs retain the fractional score."""

    levels: Sequence[Any]

    def __post_init__(self) -> None:
        if isinstance(self.levels, (str, bytes)) or not 2 <= len(self.levels) <= 10:
            raise TypeSafeSchemaError("A rubric requires 2-10 described levels")
        if any(level is None or level == "" for level in self.levels):
            raise TypeSafeSchemaError("Every rubric level needs a description")
        object.__setattr__(self, "levels", tuple(_content(list(self.levels))))

    def _validate_score(self, value: Any) -> int | float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= len(self.levels) - 1
        ):
            raise ValueError(f"Rubric score must be a finite number in [0, {len(self.levels) - 1}]")
        return value

    def __get_pydantic_core_schema__(
        self, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            self._validate_score, handler(source_type)
        )

    def __get_pydantic_json_schema__(
        self, schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        result = _extend(schema, handler, "score", list(self.levels))
        result["minimum"] = max(result.get("minimum", 0), 0)
        result["maximum"] = min(result.get("maximum", len(self.levels) - 1), len(self.levels) - 1)
        return result


@dataclass(frozen=True)
class ChoiceCriteria:
    """Descriptions keyed by exact declared Literal or Enum values."""

    criteria: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.criteria or any(type(key) is not str for key in self.criteria):
            raise TypeSafeSchemaError("Choice criteria require string option names")
        object.__setattr__(self, "criteria", _content(dict(self.criteria)))

    def __get_pydantic_json_schema__(
        self, schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        result = _extend(schema, handler, "choice", dict(self.criteria))
        options = result.get("enum")
        if (
            is_list(options)
            and set(options) == set(self.criteria)
            and all(isinstance(value, str) for value in self.criteria.values())
        ):
            result.pop("enum")
            result["anyOf"] = [
                {"const": option, "description": self.criteria[option]} for option in options
            ]
        return result


@dataclass(frozen=True, kw_only=True)
class BooleanCriteria:
    """Meanings of positive and negative outcomes, independent of threshold."""

    true: Any
    false: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "true", _content(self.true))
        object.__setattr__(self, "false", _content(self.false))

    def __get_pydantic_json_schema__(
        self, schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return _extend(schema, handler, "noul", {"true": self.true, "false": self.false})
