import asyncio

from pydantic import BaseModel, EmailStr, Field
from pydantic_ai import Agent
from pydantic_extra_types.phone_numbers import PhoneNumber

from kedi_typesafe.integrations.pydantic import TypeSafeModel


class Contact(BaseModel):
    email: EmailStr = Field(description="The current support email, not the archived address")
    phone: PhoneNumber = Field(description="The current support phone number")


async def main() -> None:
    async with TypeSafeModel() as model:
        agent = Agent(model, output_type=Contact)
        result = await agent.run(
            "Archived contact: old@example.com. Current support: help@example.com, +1 202-555-0100."
        )
        print(result.output.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
