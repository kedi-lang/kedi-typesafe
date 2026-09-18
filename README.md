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

The surface supports booleans, finite string `Literal`/`Enum` values, constrained text
extraction, and nested Pydantic models composed of supported fields. Pydantic's native `EmailStr` and
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
unconstrained strings, arbitrary numeric extraction, recursive schemas, media, and omitted/defaulted
fields are rejected before a provider request is made. Nullable choices and extraction are supported:
no-match is an explicit option, not a confidence cutoff. Missing candidates resolve locally to `None`
only for nullable extraction. Provider failures remain exceptions.

Boolean answers use Jev's Noul probability and a configurable threshold. The default decision is
strictly `probability > 0.85`; equality is false. Pass `threshold=...` to either model constructor,
or `typesafe_threshold` per request to override it without mutating the model. This is a decision
policy, not a guarantee of 85% empirical accuracy. Version 0.2 changes the legacy default of 0.5.

```python
agent = Agent(TypeSafeModel(threshold=0.9), output_type=Triage)
result = agent.run_sync("Please refund the duplicate charge", model_settings={"typesafe_threshold": 0.95})
```

## Typed Questions

```python
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from kedi_typesafe import BooleanCriteria, ChoiceCriteria, Rubric

class Review(BaseModel):
    probability: Annotated[
        float,
        Field(ge=0, le=1, description="Is a refund requested?"),
    ]
    quality: Annotated[
        float,
        Rubric(["Incorrect", "Partially correct", "Correct"]),
    ]
    urgent: Annotated[
        bool,
        BooleanCriteria(
            true="Immediate harm",
            false="No established immediate harm",
        ),
    ]
    route: Annotated[
        Literal["billing", "support"],
        ChoiceCriteria({"billing": "Payments", "support": "Product help"}),
    ]
    labels: list[Literal["refund", "bug"]]
```

All independent fields share one provider request. Raw bounded probabilities are not thresholded.
Rubric floats retain fractional positions; integer rubric fields round to the nearest level, with
ties rounded upward. The original score remains in metadata. Each multi-label option gets an
independent Noul question and is included only above the configured threshold, in declaration order.
Questions that depend on another answer belong in subsequent calls.

`Rubric` requires 2-10 described levels. `ChoiceCriteria` keys must match the declared options.
Criteria may contain JSON objects or arrays; they are not flattened into strings. Unsupported or
conflicting metadata is rejected. List/numeric constraints are validated, never silently repaired.

For both a probability and its boolean decision, request the probability once and compare it in
Python. A second model call is unnecessary.

## Evidence and Lifecycle

Pydantic responses expose `provider_details["typesafe"]` and `metadata["typesafe"]`.
LangChain messages expose `response_metadata["typesafe"]`; use `include_raw=True` with
`with_structured_output` to retain them. Evidence includes raw probabilities, Choice/Score
distributions and confidence, rubric legends, thresholds, and reported usage. Noul has no separate
provider confidence. Unknown usage remains `None` in evidence even when framework counters need zero.

Reuse the model for warm connections. `async with TypeSafeModel()` and
`async with TypeSafeChatModel()` close owned connections; borrowed SDK clients remain caller-owned.
Async clients are event-loop-local. Pydantic AI streaming emits the complete answer rather than
pretending Jev generates incremental tokens. No evidence is stored in a global last-result slot.

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
The same Jev schema restrictions described above apply. Free-form invocation, media,
stop sequences, and unsupported model settings fail before a provider request. Tool and retry
history are retained. Set `typesafe_threshold` in `.bind(...)` or a request-local model copy.

## Explicit Tool Routing

Attach tools only when the application intends routing. Jev can select a zero-argument tool;
the framework executes it through its normal hooks and approval policy. Jev does not fabricate
arguments. An argument-requiring tool raises `ToolCallProposed`, with its name and probability.
Use an explicit downstream handler or Pydantic AI `FallbackModel` if desired; this package does
not silently call another model. LangChain exports its proposal from
`kedi_typesafe.integrations.langchain`.

`typesafe_tool_call_threshold` controls routing independently (default 0.6, upstream semantics).
Provide a meaningful output model docstring or agent instructions for Pydantic routing.
The routing decision is another question in the same batch, not another provider request.
Routing is not authorization: never treat a positive Jev assessment as permission to bypass
deterministic tool approval or workspace policy.

## Integration Boundaries

The Pydantic model extends the official implementation in Pydantic AI 2.45.0. The LangChain
chat model delegates provider transport to `langchain-typesafe==0.0.1a2`; a narrow pinned
private transport bridge avoids nesting a second LLM trace under the chat-model trace. Shared
schema lowering and decoding provide the additional thresholds, criteria, and extraction.
The beta dependency and private upstream bridges are covered by contract tests.

Sampling settings do not apply. Configure connection timeout in the constructor. Client-level
retry behavior belongs to the chosen transport; the wrappers do not add a second retry loop.
Install `kedi-typesafe[all]` and run `examples/capabilities.py` for both standalone integrations.
The seven programs under `examples/kedi/` require Kedi; `examples/validate_kedi.py` runs them
on both adapters. Their semantic outputs are model judgments, not guaranteed expected answers.

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
than inventing a value, unless the declared type is nullable. Fields must be
required; a required nullable field may contain `None` but may not be omitted.

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

## Probability and Rubric Bounds

`Probability` is a reusable finite float annotation in [0, 1]:

```python
from typing import Annotated
from pydantic import BaseModel
from kedi_typesafe import Probability, Rubric

class Assessment(BaseModel):
    supported_probability: Probability
    quality: Annotated[
        float,
        Rubric(["Incorrect", "Partly correct", "Correct"]),
    ]
```

The rubric contributes and validates [0, N-1] bounds automatically. Fractional
float scores are retained. You do not need another `Field(ge=..., le=...)` for
those same bounds. General constraints remain Pydantic-compatible. When using
Kedi, `from kedi import Constraints` offers a validation-only helper and
`from kedi.typesafe import Probability, Rubric` exposes these Jev primitives.
The standalone package does not depend on Kedi.

See [release validation](validation/README.md) for reproducible synthetic batching
checks and their limitations. Version 0.2 changes the default boolean threshold
from 0.5 to 0.85 while retaining the strict `>` comparator; explicitly set the old
threshold if an application requires that policy.

## Development

```bash
uv sync --all-extras --group dev
make prod
```

Normal tests use deterministic typed fakes and never call TypeSafe. Live API validation is kept
separate so CI and local quality gates do not consume credentials or incur cost.
