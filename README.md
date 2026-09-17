# kedi-typesafe

Framework-native integrations for using TypeSafe AI's Jev models with Kedi's supported agent
frameworks. Core Jev schema translation and the Pydantic AI and LangChain integrations ship as
one `kedi-typesafe` package.

## Pydantic AI

Install the Pydantic package from this workspace and provide `TYPESAFE_API_KEY`:

```python
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from kedi_typesafe.integrations.pydantic import TypeSafeModel


class Triage(BaseModel):
    urgent: bool = Field(description="The request needs action today")
    route: Literal["billing", "support", "other"] = Field(
        description="Which team should handle the request?"
    )


model = TypeSafeModel("jev-latest")
agent = Agent(model, output_type=Triage)
result = agent.run_sync("I was charged twice. Please fix this today.")
print(result.output)
```

The initial surface supports booleans, finite string `Literal`/`Enum` values, constrained text
extraction, and flat Pydantic models composed of those fields. Pydantic's native `EmailStr` and
`pydantic-extra-types` `PhoneNumber` schemas are recognized automatically. A JSON Schema `pattern`
is treated as a regex candidate extractor; other string fields can provide an explicit field
extractor:

Install `kedi-typesafe[email]`, `kedi-typesafe[phone]`, or `kedi-typesafe[all]` when those optional
Pydantic field types are used.

```python
from kedi_typesafe import RegexExtractor
from kedi_typesafe.integrations.pydantic import TypeSafeModel

model = TypeSafeModel(
    text_extractors={"ticket_id": RegexExtractor(pattern=r"CASE-\d+")},
)
```

Candidate extraction is deterministic and Jev only selects among values found in state, following
TypeSafe's pre-parsed extraction pattern. Jev is not a free-form text-generation model, so
unconstrained strings, tool calling, streaming, arbitrary numbers, nested models, and optional
fields are rejected before a provider request is made.

Boolean answers use Jev's Noul probability and a configurable threshold. The default decision is
strictly `probability > 0.5`; the full probability remains available in model response metadata.

## LangChain

Install `kedi-typesafe[langchain]` and use the native structured-output surface:

```python
from typing import Literal

from langchain.agents import create_agent
from pydantic import BaseModel, Field

from kedi_typesafe.integrations.langchain import TypeSafeChatModel


class Triage(BaseModel):
    urgent: bool = Field(description="The request needs action today")
    route: Literal["billing", "support", "other"] = Field(
        description="Which team should handle the request?"
    )


agent = create_agent(TypeSafeChatModel("jev-latest"), response_format=Triage)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "I was charged twice. Fix this today."}]}
)
print(result["structured_response"])
```

`TypeSafeChatModel.with_structured_output(...)` is also supported for direct LangChain model use.
The same Jev schema restrictions described above apply. Free-form invocation, function tools,
media/tool history, stop sequences, and arbitrary model settings fail before a provider request.

## Development

```bash
uv sync --all-extras --group dev
make prod
```

Normal tests use deterministic typed fakes and never call TypeSafe. Live API validation is kept
separate so CI and local quality gates do not consume credentials or incur cost.
