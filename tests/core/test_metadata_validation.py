from typing import Annotated, Any

import pytest
from pydantic import Field, TypeAdapter, ValidationError

from kedi_typesafe import Probability, Rubric


def test_probability_is_a_bounded_float():
    adapter: TypeAdapter[float] = TypeAdapter(Probability)
    for value in (0, 0.85, 1):
        assert isinstance(adapter.validate_python(value), float)
    for value in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            adapter.validate_python(value)
    assert adapter.json_schema() == {"type": "number", "minimum": 0, "maximum": 1}


@pytest.mark.parametrize("base", [float, int])
def test_rubric_enforces_its_range_without_field(base: type[int] | type[float]) -> None:
    adapter: TypeAdapter[Any] = TypeAdapter(
        Annotated[base, Rubric(["Wrong", "Partial", "Correct"])]
    )
    assert adapter.validate_python(2) == 2
    assert adapter.json_schema()["minimum"] == 0
    assert adapter.json_schema()["maximum"] == 2
    for value in (-1, 3, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            adapter.validate_python(value)


def test_rubric_preserves_tighter_constraints():
    adapter: TypeAdapter[float] = TypeAdapter(
        Annotated[float, Field(ge=0.5, le=1.5), Rubric(["A", "B", "C"])]
    )
    assert adapter.json_schema()["minimum"] == 0.5
    assert adapter.json_schema()["maximum"] == 1.5
    with pytest.raises(ValidationError):
        adapter.validate_python(0)


@pytest.mark.parametrize("base,value", [(str, "1"), (bool, True)])
def test_rubric_rejects_non_numeric_values(base: Any, value: Any) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Annotated[base, Rubric(["A", "B"])]).validate_python(value)
