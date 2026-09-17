from __future__ import annotations

import re
from collections.abc import Callable
from typing import cast

import pytest
from typesafe_sdk import JSONContent

from kedi_typesafe.core import CandidateExtractor, RegexExtractor, TypeSafeExtractionError
from kedi_typesafe.core.extraction import resolve_candidates


def _returns_string(_state: JSONContent) -> str:
    return "one"


def _returns_non_string(_state: JSONContent) -> list[object]:
    return ["one", 2]


def _returns_nothing(_state: JSONContent) -> list[str]:
    return []


def _returns_too_many(_state: JSONContent) -> list[str]:
    return ["one", "two", "three"]


def test_regex_extractor_walks_nested_state_and_preserves_unique_match_order() -> None:
    extractor = RegexExtractor(pattern=r"CASE-(\d+)", group=1)
    state: JSONContent = {
        "messages": [
            {"content": "Use CASE-17, not CASE-18."},
            {"content": "CASE-17 appears again."},
        ],
        "attempt": 2,
    }

    assert extractor(state) == ("17", "18")


def test_regex_extractor_rejects_invalid_pattern() -> None:
    with pytest.raises(re.error):
        RegexExtractor(pattern="[")


@pytest.mark.parametrize(
    ("extractor", "message"),
    [
        (_returns_string, "not str"),
        (_returns_non_string, "must be strings"),
        (_returns_nothing, "found no candidates"),
        (_returns_too_many, "at most 2"),
    ],
)
def test_resolve_candidates_rejects_invalid_results(
    extractor: Callable[[JSONContent], object],
    message: str,
) -> None:
    with pytest.raises(TypeSafeExtractionError, match=message):
        resolve_candidates(
            cast(CandidateExtractor, extractor),
            "state",
            field_name="value",
            limit=2,
        )


def test_resolve_candidates_wraps_extractor_failure() -> None:
    def fail(_state: JSONContent) -> list[str]:
        raise RuntimeError("parser unavailable")

    with pytest.raises(TypeSafeExtractionError, match="parser unavailable"):
        resolve_candidates(fail, "state", field_name="value", limit=2)


def test_resolve_candidates_preserves_integration_errors() -> None:
    error = TypeSafeExtractionError("specific failure")

    def fail(_state: JSONContent) -> list[str]:
        raise error

    with pytest.raises(TypeSafeExtractionError) as captured:
        resolve_candidates(fail, "state", field_name="value", limit=2)

    assert captured.value is error


def test_resolve_candidates_deduplicates_custom_extractor_values() -> None:
    assert resolve_candidates(
        lambda _state: ["first", "first", "second", ""],
        "state",
        field_name="value",
        limit=2,
    ) == ("first", "second")
