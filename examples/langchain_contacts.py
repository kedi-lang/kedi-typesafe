from langchain.agents import create_agent
from pydantic import BaseModel, EmailStr, Field
from pydantic_extra_types.phone_numbers import PhoneNumber

from kedi_typesafe.integrations.langchain import TypeSafeChatModel


class Contact(BaseModel):
    email: EmailStr = Field(description="The current support email, not the archived address")
    phone: PhoneNumber = Field(description="The current support phone number")


with TypeSafeChatModel() as model:
    agent = create_agent(model, response_format=Contact)
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Archived contact: old@example.com. "
                        "Current support: help@example.com, +1 202-555-0100."
                    ),
                }
            ]
        }
    )
    contact = Contact.model_validate(result["structured_response"])
    print(contact.model_dump_json(indent=2))
