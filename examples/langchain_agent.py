from typing import Literal

from langchain.agents import create_agent
from pydantic import BaseModel, Field

from kedi_typesafe.integrations.langchain import TypeSafeChatModel


class Triage(BaseModel):
    urgent: bool = Field(description="The request needs action today")
    route: Literal["billing", "support", "other"] = Field(
        description="Which team should handle the request?"
    )


with TypeSafeChatModel("jev-latest") as model:
    agent = create_agent(model, response_format=Triage)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "I was charged twice. Fix this today."}]}
    )

print(result["structured_response"])
