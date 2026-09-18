"""Run the pitch example and its decision/repair controls with live providers."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from kedi import InteractiveSession
from kedi.agent_adapter.invocation import observe_invocations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, help="Optional dotenv file to load.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    load_dotenv(args.env_file or root / ".env", override=False)
    source_path = Path(__file__).with_name("main.kedi")
    source = source_path.read_text()
    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256(source.encode()).hexdigest(),
        "mode": "live",
        "versions": {
            package: version(package)
            for package in ("kedi", "kedi-typesafe", "pydantic-ai-slim", "typesafe-sdk")
        },
        "runs": [],
    }
    session = InteractiveSession(cwd=source_path.parent)

    def execute(label: str, program: str, *, expected: bool | None = None) -> object:
        stdout = io.StringIO()
        usage = []
        started = perf_counter()
        with redirect_stdout(stdout), observe_invocations(usage.append):
            result = session.execute(program)
        entry = {
            "label": label,
            "elapsed_seconds": round(perf_counter() - started, 3),
            "stdout": stdout.getvalue(),
            "result": result,
            "invocations": [asdict(item) for item in usage],
        }
        if expected is not None:
            entry["expected"] = expected
            entry["matches_expected"] = result is expected
        record["runs"].append(entry)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, default=str) + "\n")
        print(f"{label} ({entry['elapsed_seconds']}s)", flush=True)
        print(stdout.getvalue(), end="")
        print(f"Result: {result}\n", flush=True)
        return result

    try:
        execute("generate_and_review", source)
        execute(
            "negative_control",
            "[unsafe] = Your refund has been approved and will arrive tomorrow.\n"
            "= `supported(unsafe, policy)`\n",
            expected=False,
        )
        execute(
            "invented_action_control",
            "[invented] = I have already forwarded your case to billing and opened a refund review.\n"
            "= `supported(invented, policy)`\n",
            expected=False,
        )
        execute(
            "positive_control",
            "[grounded] = Please share your invoice number for billing review. "
            "A refund has not been approved, and we cannot promise a refund date.\n"
            "= `supported(grounded, policy)`\n",
            expected=True,
        )
        execute(
            "repair_negative_control",
            "[repaired] = `write_reply(customer, policy, unsafe)`\n"
            '`print(f"Repaired: {repaired}")`\n'
            "= `supported(repaired, policy)`\n",
            expected=True,
        )
    finally:
        session.close()
    if any(item.get("matches_expected") is False for item in record["runs"]):
        raise SystemExit("A live control did not match its expectation; see the saved record.")


if __name__ == "__main__":
    main()
