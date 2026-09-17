from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

from typesafe_sdk import (
    Answer,
    AsyncTypeSafeClient,
    ChoiceAnswer,
    JSONContent,
    NoulAnswer,
    Question,
    SystemOneResponse,
    TypeSafeClient,
)

from .errors import TypeSafeResponseError
from .extraction import CandidateExtractor
from .schema import EvaluationPlan, QuestionSpec, build_evaluation_plan

JSONValue: TypeAlias = JSONContent


class AsyncSystemOneClient(Protocol):
    async def system_one(
        self,
        state: JSONValue,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
    ) -> SystemOneResponse: ...

    async def aclose(self) -> None: ...


class SystemOneClient(Protocol):
    def system_one(
        self,
        state: JSONValue,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
    ) -> SystemOneResponse: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationResult:
    values: dict[str, bool | str]
    model: str
    input_tokens: int | None
    output_tokens: int | None
    metadata: dict[str, Any]


class TypeSafeEvaluator:
    """Execute validated Jev question plans and normalize their typed answers."""

    def __init__(
        self,
        model_name: str = "jev-latest",
        *,
        api_key: str | None = None,
        threshold: float = 0.5,
        timeout: float | None = None,
        client: AsyncSystemOneClient | None = None,
        sync_client: SystemOneClient | None = None,
        text_extractors: Mapping[str, CandidateExtractor] | None = None,
    ) -> None:
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("TypeSafe model name must not be empty")
        if isinstance(threshold, bool) or not math.isfinite(threshold):
            raise ValueError("TypeSafe boolean threshold must be a finite number")
        if not 0 <= threshold <= 1:
            raise ValueError("TypeSafe boolean threshold must be between 0 and 1")

        self.model_name = model_name
        self.threshold = float(threshold)
        self.text_extractors = dict(text_extractors or {})
        self._api_key = api_key
        self._timeout = timeout
        self._borrowed_client = client
        self._clients: dict[asyncio.AbstractEventLoop, AsyncSystemOneClient] = {}
        self._owns_sync_client = sync_client is None
        self._sync_client = sync_client

    async def evaluate(
        self,
        *,
        state: JSONValue,
        schema: Mapping[str, Any],
    ) -> EvaluationResult:
        plan = self._plan(state, schema)
        client = self._borrowed_client
        if client is None:
            loop = asyncio.get_running_loop()
            client = self._clients.get(loop)
            if client is None:
                client = AsyncTypeSafeClient(
                    api_key=self._api_key,
                    model=self.model_name,
                    timeout=self._timeout,
                )
                self._clients[loop] = client
        response = await client.system_one(
            state,
            plan.native_questions(),
            model=self.model_name,
        )
        return self._result(response, plan)

    def evaluate_sync(
        self,
        *,
        state: JSONValue,
        schema: Mapping[str, Any],
    ) -> EvaluationResult:
        plan = self._plan(state, schema)
        client = self._sync_client
        if client is None:
            client = TypeSafeClient(
                api_key=self._api_key,
                model=self.model_name,
                timeout=self._timeout,
            )
            self._sync_client = client
        response = client.system_one(
            state,
            plan.native_questions(),
            model=self.model_name,
        )
        return self._result(response, plan)

    def close(self) -> None:
        if self._owns_sync_client and self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    def _plan(
        self,
        state: JSONValue,
        schema: Mapping[str, Any],
    ) -> EvaluationPlan:
        return build_evaluation_plan(
            schema,
            text_extractors=self.text_extractors,
        ).resolve(state)

    def _result(
        self,
        response: SystemOneResponse,
        plan: EvaluationPlan,
    ) -> EvaluationResult:
        values, answer_metadata = self._decode(response, plan)
        return EvaluationResult(
            values=values,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            metadata={
                "answers": answer_metadata,
                "boolean_threshold": self.threshold,
                "boolean_comparator": ">",
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            },
        )

    async def aclose(self) -> None:
        await self.aclose_current()
        self.close()

    async def aclose_current(self) -> None:
        """Close the owned async client bound to the current event loop."""

        if self._borrowed_client is not None:
            return
        loop = asyncio.get_running_loop()
        client = self._clients.pop(loop, None)
        if client is not None:
            await client.aclose()

    def _decode(
        self,
        response: SystemOneResponse,
        plan: EvaluationPlan,
    ) -> tuple[dict[str, bool | str], dict[str, Any]]:
        expected_keys = {question.key for question in plan.questions}
        actual_keys = set(response.answers)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            raise TypeSafeResponseError(
                f"TypeSafe response answer keys do not match the request; "
                f"missing={missing}, unexpected={unexpected}"
            )

        values: dict[str, bool | str] = {}
        metadata: dict[str, Any] = {}
        for question in plan.questions:
            answer = response.answers[question.key]
            if question.kind == "noul":
                value, evidence = self._decode_noul(question, answer)
            else:
                value, evidence = self._decode_choice(question, answer)
            values[question.key] = value
            metadata[question.key] = evidence
        return values, metadata

    def _decode_noul(
        self,
        question: QuestionSpec,
        answer: Answer,
    ) -> tuple[bool, dict[str, Any]]:
        if not isinstance(answer, NoulAnswer):
            raise TypeSafeResponseError(f"TypeSafe answer {question.key!r} must be a Noul answer")
        probability = answer.noul
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise TypeSafeResponseError(
                f"TypeSafe Noul answer {question.key!r} returned invalid probability {probability!r}"
            )
        return probability > self.threshold, {
            "type": "noul",
            "probability": probability,
        }

    @staticmethod
    def _decode_choice(
        question: QuestionSpec,
        answer: Answer,
    ) -> tuple[str, dict[str, Any]]:
        if not isinstance(answer, ChoiceAnswer):
            raise TypeSafeResponseError(f"TypeSafe answer {question.key!r} must be a Choice answer")
        if answer.choice in question.rejected_options:
            raise TypeSafeResponseError(
                f"TypeSafe could not extract a matching candidate for {question.key!r}"
            )
        if answer.choice not in question.options:
            raise TypeSafeResponseError(
                f"TypeSafe Choice answer {question.key!r} returned unsupported value "
                f"{answer.choice!r}"
            )
        expected = set(question.native_options)
        if set(answer.probabilities) != expected:
            raise TypeSafeResponseError(
                f"TypeSafe Choice probabilities {question.key!r} do not match its options"
            )
        probabilities = dict(answer.probabilities)
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values()):
            raise TypeSafeResponseError(
                f"TypeSafe Choice answer {question.key!r} returned invalid probabilities"
            )
        if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise TypeSafeResponseError(
                f"TypeSafe Choice probabilities {question.key!r} must sum to 1"
            )
        if not math.isfinite(answer.confidence) or not 0 <= answer.confidence <= 1:
            raise TypeSafeResponseError(
                f"TypeSafe Choice answer {question.key!r} returned invalid confidence"
            )
        return answer.choice, {
            "type": "extraction" if question.kind == "extract" else "choice",
            "choice": answer.choice,
            "confidence": answer.confidence,
            "probabilities": probabilities,
        }


__all__ = [
    "AsyncSystemOneClient",
    "EvaluationResult",
    "JSONValue",
    "SystemOneClient",
    "TypeSafeEvaluator",
]
