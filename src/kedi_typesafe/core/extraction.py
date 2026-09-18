from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from re import Pattern
from types import MappingProxyType
from typing import TypeAlias, cast

from typesafe_sdk import JSONContent

from .errors import TypeSafeExtractionError

CandidateExtractor: TypeAlias = Callable[[JSONContent], Iterable[str]]


@dataclass(frozen=True, slots=True, kw_only=True)
class RegexExtractor:
    """Extract regex matches from every string value in TypeSafe state."""

    pattern: str
    flags: int = 0
    group: int | str = 0
    _compiled: Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_compiled", re.compile(self.pattern, self.flags))

    def __call__(self, state: JSONContent) -> tuple[str, ...]:
        return _deduplicate(
            match.group(self.group).strip()
            for text in _iter_strings(state)
            for match in self._compiled.finditer(text)
        )


_EMAIL_EXTRACTOR = RegexExtractor(pattern=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_EXTRACTOR = RegexExtractor(pattern=r"\(?\+?\d[\d\s()\-.]{6,}\d")

BUILTIN_FORMAT_EXTRACTORS: Mapping[str, CandidateExtractor] = MappingProxyType(
    {
        "email": _EMAIL_EXTRACTOR,
        "phone": _PHONE_EXTRACTOR,
        "phone-number": _PHONE_EXTRACTOR,
        "tel": _PHONE_EXTRACTOR,
    }
)


def resolve_candidates(
    extractor: CandidateExtractor,
    state: JSONContent,
    *,
    field_name: str,
    limit: int,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Run and validate one extractor before its candidates become model options."""

    try:
        extracted = extractor(state)
        if isinstance(extracted, str):
            raise TypeError("an extractor must return an iterable of candidate strings, not str")
        candidates = _deduplicate(extracted)
    except TypeSafeExtractionError:
        raise
    except Exception as exc:
        raise TypeSafeExtractionError(
            f"Text extractor for output property {field_name!r} failed: {exc}"
        ) from exc

    if not candidates and not allow_empty:
        raise TypeSafeExtractionError(
            f"Text extractor for output property {field_name!r} found no candidates"
        )
    if len(candidates) > limit:
        raise TypeSafeExtractionError(
            f"Text extractor for output property {field_name!r} found {len(candidates)} "
            f"candidates; Jev supports at most {limit} plus the no-match option"
        )
    return candidates


def _deduplicate(values: Iterable[object]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"candidate values must be strings, received {type(value).__name__}")
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _iter_strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        for nested in mapping.values():
            yield from _iter_strings(nested)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        sequence = cast(Sequence[object], value)
        for nested in sequence:
            yield from _iter_strings(nested)


__all__ = [
    "BUILTIN_FORMAT_EXTRACTORS",
    "CandidateExtractor",
    "RegexExtractor",
    "resolve_candidates",
]
