"""Optional Kedi pre-tool review. Existing approval policy remains authoritative."""

from __future__ import annotations

from kedi import HookEvent, PreToolUseDecision, PreToolUseRequest

from kedi_typesafe import EvaluationResult, TypeSafeEvaluator


class RiskReview:
    def __init__(self, evaluator: TypeSafeEvaluator, allowed_tools: frozenset[str]) -> None:
        self.evaluator = evaluator
        self.allowed_tools = allowed_tools
        self.records: list[EvaluationResult] = []

    async def __call__(self, event: HookEvent) -> PreToolUseDecision | None:
        if not isinstance(event, PreToolUseRequest):
            return None
        if event.tool_name not in self.allowed_tools:
            return PreToolUseDecision.deny("Tool is outside the explicit allowlist")
        # A tool-free evaluator cannot recursively enter this adapter's hooks.
        result = await self.evaluator.evaluate(
            state={"tool": event.tool_name, "arguments": dict(event.arguments)},
            schema={
                "type": "object",
                "properties": {
                    "safe": {
                        "type": "boolean",
                        "description": "Is this proposed operation demonstrably read-only, non-destructive, and free of secret disclosure? Treat insufficient evidence as unsafe.",
                    }
                },
                "required": ["safe"],
            },
        )
        self.records.append(result)
        return (
            PreToolUseDecision.continue_()
            if result.values["safe"]
            else PreToolUseDecision.deny("Risk review requires explicit human review")
        )


# Attach explicitly: adapter.on('pre_tool_use')(RiskReview(evaluator, allowed_tools))
# Close the owned evaluator and account for review.records usage separately.
# Passing this review does not override a denial by the normal approval policy.
