# Jev 0.2 validation

`jev-0.2-release.json` contains synthetic release checks collected on 2026-09-18.
Reproduce with `PYDANTIC_AI_NO_BANNER=1 python examples/validate_release.py` after
setting `TYPESAFE_API_KEY`. No credentials, customer data, or private prompts are
included. Framework versions are recorded in the JSON.

Each adapter evaluates three fixed reference/answer pairs (supported, contradicted,
insufficient evidence), twice. Each evaluation asks for a boolean, raw probability,
and rubric score. Batch mode asks all three questions in one request; separate mode
asks each question in its own request. Mode order alternates across repetitions.
Model instances are reused, so these are not isolated cold-start latency estimates.

| Adapter | Mode | Boolean matches | Requests | Input tokens | Output tokens | Total seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Pydantic AI | Batch | 6/6 | 6 | 2,610 | 306 | 2.537 |
| Pydantic AI | Separate | 6/6 | 18 | 5,886 | 354 | 6.522 |
| LangChain | Batch | 6/6 | 6 | 2,742 | 306 | 2.643 |
| LangChain | Separate | 6/6 | 18 | 6,282 | 354 | 7.015 |

Across adapters, batch mode uses 12 requests versus 36 separate requests. Every
output passed its declared schema. Raw probability and rubric values remain in
the JSON; they are not rounded into a claim of perfect semantic agreement.

This tiny smoke matrix demonstrates execution, usage accounting, and batching. It
does **not** estimate general precision/recall, calibrated confidence, or production
latency. Earlier seven-capability live checks included missed labels and unexpected
urgency decisions; those are model-quality limitations, not erased by this check.

## Executed Kedi Programs

`kedi-capabilities-0.2.json` records a separate final execution of all seven
capability programs on both adapters. All 14 executions completed with valid
typed output. This is not 14/14 semantic accuracy: LangChain selected only billing
in the multi-label example while Pydantic selected billing and login; both chose
the review branch in the probability example. The rubric values were 1.17 and
1.12, respectively. No threshold or prompt was changed to force agreement.

Reproduce with `python examples/validate_kedi.py` in an environment containing
the matching Kedi integration and the optional dependencies. These programs use
the moving `jev-latest` alias, so neither exact values nor latency are guaranteed
to reproduce. No statistical comparison between adapters is justified by one
sample per example.

## Local Quality Gates

- `make prod`: lint, formatting, ty, strict basedpyright, 165 tests,
  100% statement/branch coverage, wheel and source distribution build.
- Final wheel: independently installed on Python 3.10 with base, email, phone,
  langchain and all extras. Base/email/phone imports did not require LangChain.
- Kedi integration: full suite 3,836 passed, one skipped; fixture programs also
  execute without an ignored TypeSafe source checkout.

These gates do not establish combined Kedi release installability. Its pinned
Codex helper currently declares an incompatible Pydantic maximum. A compatible
helper release and publication of this package must precede the final Kedi lock.
