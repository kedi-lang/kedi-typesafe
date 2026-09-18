"""Bounded synthetic, paired batch/separate validation; prints secret-free JSON."""

from __future__ import annotations

import asyncio
import json
from importlib.metadata import version
from time import perf_counter
from typing import Annotated, Any

from pydantic import Field, create_model
from pydantic_ai import Agent

from kedi_typesafe import Probability, Rubric, TypeSafeModel
from kedi_typesafe.integrations.langchain import TypeSafeChatModel

CASES = [
    ("supported", "Reference: ticket A is closed. Answer: ticket A is closed.", True),
    ("contradicted", "Reference: ticket A is open. Answer: ticket A is closed.", False),
    ("unknown", "Reference: ticket B is closed. Answer: ticket A is closed.", False),
]
FIELDS = {
    "supported": (
        bool,
        Field(description="Is the answer established by the supplied reference only?"),
    ),
    "probability": (
        Probability,
        Field(description="Probability the answer is established by the supplied reference only"),
    ),
    "quality": (
        Annotated[
            float, Rubric(["Unsupported or contradicted", "Partially supported", "Fully supported"])
        ],
        Field(description="How well is the answer supported by the supplied reference only?"),
    ),
}


async def main() -> None:
    records: list[dict[str, Any]] = []
    for adapter, model in (("pydantic", TypeSafeModel()), ("langchain", TypeSafeChatModel())):
        async with model:
            for case, state, expected in CASES:
                for repetition in range(2):
                    for mode in ("batch", "separate") if repetition == 0 else ("separate", "batch"):
                        groups = (
                            [FIELDS]
                            if mode == "batch"
                            else [{key: field} for key, field in FIELDS.items()]
                        )
                        values: dict[str, Any] = {}
                        input_tokens = output_tokens = 0
                        started = perf_counter()
                        for fields in groups:
                            output = create_model("Assessment", **fields)
                            if isinstance(model, TypeSafeModel):
                                response = await Agent(model, output_type=output).run(state)
                                values.update(response.output.model_dump())
                                input_tokens += response.usage.input_tokens
                                output_tokens += response.usage.output_tokens
                            else:
                                response = await model.with_structured_output(
                                    output, include_raw=True
                                ).ainvoke(state)
                                values.update(response["parsed"].model_dump())
                                usage = response["raw"].usage_metadata
                                input_tokens += usage["input_tokens"]
                                output_tokens += usage["output_tokens"]
                        records.append(
                            {
                                "adapter": adapter,
                                "case": case,
                                "mode": mode,
                                "repetition": repetition,
                                "expected_supported": expected,
                                "values": values,
                                "correct": values["supported"] == expected,
                                "requests": len(groups),
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "seconds": round(perf_counter() - started, 4),
                            }
                        )
    print(
        json.dumps(
            {
                "versions": {
                    name: version(name)
                    for name in (
                        "kedi-typesafe",
                        "pydantic-ai-slim",
                        "langchain",
                        "langchain-typesafe",
                        "typesafe-sdk",
                    )
                },
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
