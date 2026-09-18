# Kedi + Jev: Generate, Review, Revise

A runnable reply-review workflow. A generative model writes a customer
reply. Jev evaluates whether it invents actions, approvals, commitments, or
deadlines unsupported by the supplied policy. Kedi returns the draft or revises
it once and checks again. If the second check fails, it returns a human-review
message. It never sends an email.

## Code

[main.kedi](main.kedi) is the complete executable program. Procedures capture
their model at definition time: `write_reply` uses Gemini 3 Flash Preview through
OpenRouter, while `supported` uses Jev through the same Pydantic adapter.

The `Annotated` description supplies the actual decision question. The policy
also states the current operational facts: no case was forwarded, no review has
started, and no refund was approved or scheduled. These facts matter when
checking claims that an action has already happened.

`decision_info("approved")` reads evidence from the existing Jev call without
another request. An accepted initial draft uses two model requests. A rejected
draft adds at most one generation and one decision; there is no unbounded loop.

## Actual Recorded Result

Run date: September 19, 2026, Europe/Istanbul (September 18 UTC).
The complete record is [live_run.json](live_run.json), with the program SHA-256,
dependency versions, output, per-invocation usage, and measured durations.
The recorded source hash predates a later blank-line-only readability edit to
`main.kedi`; no statements or prompts were changed by that edit.

| Case | Probability of a supported reply | Boolean at threshold 0.85 | Requests |
| --- | ---: | --- | ---: |
| Generate and review | 0.97 | true | 2 |
| Invented approved refund arriving tomorrow | 0.04 | false | 1 |
| Invented completed forwarding/review action | 0.04 | false | 1 |
| Explicitly grounded control | 0.95 | true | 1 |
| Revise the rejected refund promise and review | 0.95 | true | 2 |

The main program completed in **4.446 seconds**. Its actual output:

```text
Jev: supported=True, probability=0.970000, threshold=0.85

We have received your message regarding the duplicate charges. A billing review is required to address this, but please note that a case has not yet been forwarded and a review has not started. At this time, no refund has been approved or scheduled. Could you please provide the invoice number for the transactions in question?
```

The separate repair control started with:

```text
Your refund has been approved and will arrive tomorrow.
```

The model revised it to:

```text
I have received your message regarding the duplicate charges. Please provide the invoice number so that I can forward this for a billing review. Please note that no review has started yet, and a refund has not been approved or scheduled at this time.
```

Jev assigned the revision 0.95. The main run accepted its initial draft; the
repair above was a separate controlled invocation, not part of the main run.

## Reproduce

Use an environment with this `kedi-typesafe` checkout and a Kedi checkout that
includes the decision-metadata API. The recorded
environment used Kedi 0.4.0, kedi-typesafe 0.2.0, pydantic-ai-slim 2.45.0, and
typesafe-sdk 0.6.0. Version labels alone do not identify uncommitted changes.

Configure `OPENROUTER_API_KEY` and `TYPESAFE_API_KEY` in the `kedi-typesafe`
repository's `.env` or process environment. With that environment activated,
run from the `kedi-typesafe` repository root:

```sh
PYDANTIC_AI_NO_BANNER=1 python examples/jev_reply_review/run.py --output /tmp/kedi-jev-reply-review.json
```

When this repository is checked out as `typesafe/` inside the Kedi workspace,
the workspace environment and dotenv file can be used explicitly:

```sh
source .venv/bin/activate
PYDANTIC_AI_NO_BANNER=1 python typesafe/examples/jev_reply_review/run.py --env-file .env --output /tmp/kedi-jev-reply-review.json
```

The runner executes the real Kedi file, then positive, negative, and repair
controls using the same procedures. It exits unsuccessfully if a control
disagrees with its expected boolean. No Logfire instrumentation is configured.

## Development Evidence

This is an integration demonstration, not an accuracy or performance benchmark.
Its question wording and policy were refined during development. Earlier
records are retained:

- [Attempt 1](attempt_01_broad_criterion.json): the plain boolean field and broad
  criterion also rejected the grounded control (0.52).
- [Attempt 2](attempt_02_refund_only.json): an explicit refund approval/date
  question passed the controls but did not test other invented actions.
- [Attempt 3](attempt_03_action_criterion.json): a broader explicit question
  rejected the isolated invented-action control, yet accepted a generated reply
  claiming that forwarding had already happened.
- [Final run](live_run.json): added explicit operational facts to the policy;
  the generated draft respected them and all four controls matched expectations.

The probability is Jev's estimate, not a correctness guarantee. This result does
not establish reliability across other messages or repeated trials. Jev costs
are absent from the recorded usage, so no total dollar cost is claimed.
