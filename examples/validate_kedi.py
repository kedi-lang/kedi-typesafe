"""Run the seven capability programs on both Kedi adapters (requires Kedi)."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

from kedi.lang import compile_program, parse_program


def main() -> None:
    root = Path(__file__).with_name("kedi")
    for adapter in ("pydantic", "langchain"):
        for path in sorted(root.glob("0*.kedi")):
            source = path.read_text().replace("> adapter: pydantic", f"> adapter: {adapter}")
            started = perf_counter()
            result = compile_program(parse_program(source)).run_main()
            print(
                json.dumps(
                    {
                        "adapter": adapter,
                        "example": path.name,
                        "output": result,
                        "seconds": round(perf_counter() - started, 3),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
