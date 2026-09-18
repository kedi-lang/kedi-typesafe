"""One batched Jev request per integration, using synthetic evidence only."""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from pydantic import BaseModel, EmailStr, Field
from pydantic_ai import Agent

from kedi_typesafe import BooleanCriteria, ChoiceCriteria, Rubric, TypeSafeModel


class Evidence(BaseModel):
    supported: Annotated[
        bool,
        BooleanCriteria(
            true="The customer explicitly asks for a refund", false="No explicit refund request"
        ),
    ]
    probability: Annotated[
        float,
        Field(ge=0, le=1, description="Probability the customer explicitly requests a refund"),
    ]


class Review(BaseModel):
    evidence: Evidence
    urgency: Annotated[
        float, Rubric(["No time pressure", "Some time pressure", "Immediate deadline"])
    ]
    route: Annotated[
        Literal["billing", "technical"],
        ChoiceCriteria({"billing": "Payments and refunds", "technical": "Software defects"}),
    ]
    labels: list[Literal["refund", "bug"]]
    email: Annotated[EmailStr | None, Field(description="Customer email, if present")]


STATE = "I was charged twice. Please refund the duplicate charge today. Contact ada@example.com."


async def main() -> None:
    async with TypeSafeModel() as model:
        result = await Agent(model, output_type=Review).run(STATE)
        print("pydantic", result.output.model_dump(mode="json"))
        print("pydantic_usage", result.usage)
    from kedi_typesafe.integrations.langchain import TypeSafeChatModel

    async with TypeSafeChatModel() as model:
        result = await model.with_structured_output(Review, include_raw=True).ainvoke(STATE)
        if result["parsing_error"] is not None:
            raise result["parsing_error"]
        print("langchain", result["parsed"].model_dump(mode="json"))
        print("langchain_usage", result["raw"].usage_metadata)


if __name__ == "__main__":
    asyncio.run(main())
