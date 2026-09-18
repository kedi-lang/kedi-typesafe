# Examples

Set `TYPESAFE_API_KEY`; examples do not print secrets. Install
`kedi-typesafe[all]` for the optional LangChain and contact field dependencies.

- `capabilities.py`: batched nested criteria, raw probability, fractional rubric, labels,
  and nullable email using standalone Pydantic AI and LangChain.
- `kedi/01_rubric.kedi` through `kedi/07_nullable.kedi`: the same capabilities in Kedi.
- `validate_kedi.py`: executes the seven Kedi examples on both adapters and prints outcomes
  and elapsed time. Requires an installed Kedi checkout with the TypeSafe integration.
- `kedi/judge.kedi`: a rubric-backed procedure used by Kedi's existing evaluation surface.
- `kedi/binary_judge.kedi` and `kedi/weighted_judge.kedi`: boolean and weighted
  multi-dimensional evaluation through the same surface.
- `kedi/routing.kedi`: explicit classification with a review path; no implicit
  downstream LLM invocation.
- `validate_release.py`: matched batch/separate synthetic requests on both adapters;
  writes JSON to stdout. See `../validation/` for the captured smoke matrix.
- `risk_review.py`: optional Kedi pre-tool hook with an explicit allowlist, a
  separate tool-free evaluator, and review usage records. Approval policy remains
  authoritative; this is an additional model-based screen, not a security guarantee.
- [jev_reply_review/](jev_reply_review/README.md): a generative model drafts a
  reply, Jev reviews it, and Kedi can revise it once. Includes live records and
  controls. Requires Kedi's decision-metadata API and `OPENROUTER_API_KEY`;
  its runner loads dotenv and accepts an explicit `--env-file`.

For a probability and a boolean decision, return the probability and compare locally.
Each template batches its independent fields; dependencies require separate statements.
Changing the threshold changes decisions, not provider probabilities or calibration.
