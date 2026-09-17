from __future__ import annotations

import os

import pytest
from pydantic import BaseModel, EmailStr, Field
from pydantic_ai import Agent
from pydantic_extra_types.phone_numbers import PhoneNumber

from kedi_typesafe.integrations.pydantic import TypeSafeModel


class LiveDecision(BaseModel):
    capital_is_ankara: bool = Field(description="Ankara is the capital of Turkey")
    email: EmailStr = Field(description="The active support email, not the archived address")
    phone: PhoneNumber = Field(description="The active support phone number")


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_plain_pydantic_agent() -> None:
    if os.environ.get("TYPESAFE_LIVE_TEST") != "1":
        pytest.skip("Set TYPESAFE_LIVE_TEST=1 to call the real TypeSafe API")
    async with TypeSafeModel() as model:
        result = await Agent(model, output_type=LiveDecision).run(
            "Use established geographic facts to evaluate the proposition. "
            "The archived email is old@example.com. The active support email is "
            "new@example.com and the active support phone is +1 202-555-0100."
        )
    assert result.output.capital_is_ankara is True
    assert result.output.email == "new@example.com"
    assert result.output.phone == "tel:+1-202-555-0100"
