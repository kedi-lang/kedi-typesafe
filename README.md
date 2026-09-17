# kedi-typesafe

Framework-native integrations for using TypeSafe AI's Jev models with Kedi's supported agent
frameworks. Core Jev schema translation and the Pydantic AI and LangChain integrations ship as
one `kedi-typesafe` package.

## Pydantic AI

Install the package and provide `TYPESAFE_API_KEY` in your environment:

```bash
pip install kedi-typesafe
export TYPESAFE_API_KEY="your-key"
```

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

## Email and Phone Extraction

Install the field validators:

```bash
pip install 'kedi-typesafe[email,phone]'
```

```python
from pydantic import BaseModel, EmailStr, Field
from pydantic_ai import Agent
from pydantic_extra_types.phone_numbers import PhoneNumber

from kedi_typesafe.integrations.pydantic import TypeSafeModel


class Contact(BaseModel):
    email: EmailStr = Field(description="The current support email, not the archived address")
    phone: PhoneNumber = Field(description="The current support phone number")


agent = Agent(TypeSafeModel(), output_type=Contact)
result = agent.run_sync(
    "Archived contact: old@example.com. Current support: help@example.com, +1 202-555-0100."
)
print(result.output)
```

Email and phone candidates are extracted from the supplied text automatically. Jev selects the
candidate matching the field description; Pydantic then validates the result. Phone numbers use
Pydantic's normalized format, for example `tel:+1-202-555-0100`. Provide international numbers
with a country code. Missing candidates or a no-match answer raise an extraction error rather
than inventing a value. All fields in this initial integration must be required.

For LangChain, use the same `Contact` schema with
`create_agent(TypeSafeChatModel(), response_format=Contact)` and install
`kedi-typesafe[langchain,email,phone]`.

## Runnable Examples

After cloning this repository, run `uv sync --all-extras` and set `TYPESAFE_API_KEY`.

| Example | Command |
| --- | --- |
| Pydantic classification | `uv run examples/pydantic_agent.py` |
| LangChain classification | `uv run examples/langchain_agent.py` |
| Pydantic email and phone | `uv run examples/pydantic_contacts.py` |
| LangChain email and phone | `uv run examples/langchain_contacts.py` |

These examples call the real TypeSafe API. An async context manager closes owned connections
after use; it is optional for constructing the model. Reusing a model within the same event
loop allows its HTTP connection to be reused.

## Development

```bash
uv sync --all-extras --group dev
make prod
```

Normal tests use deterministic typed fakes and never call TypeSafe. Live API validation is kept
separate so CI and local quality gates do not consume credentials or incur cost.
