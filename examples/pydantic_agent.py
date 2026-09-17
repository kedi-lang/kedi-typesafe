from __future__ import annotations

import asyncio
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from kedi_typesafe.integrations.pydantic import TypeSafeModel


class Triage(BaseModel):
    urgent: bool = Field(description="The request needs action today")
    route: Literal["billing", "support", "other"] = Field(
        description="Which team should handle the request?"
    )


async def main() -> None:
    async with TypeSafeModel("jev-latest") as model:
        agent = Agent(model, output_type=Triage)
        result = await agent.run("I was charged twice. Please fix this today.")
        print(result.output)


if __name__ == "__main__":
    asyncio.run(main())
