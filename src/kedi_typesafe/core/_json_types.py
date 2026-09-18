"""Narrow dynamically received JSON containers without discarding their values."""

from collections.abc import Mapping, Sequence
from typing import Any, TypeGuard


def is_mapping(value: object) -> TypeGuard[Mapping[Any, Any]]:
    return isinstance(value, Mapping)


def is_list(value: object) -> TypeGuard[list[Any]]:
    return isinstance(value, list)


def is_sequence(value: object) -> TypeGuard[Sequence[Any]]:
    return isinstance(value, (list, tuple))
