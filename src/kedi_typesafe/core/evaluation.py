from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

from jsonschema import Draft202012Validator, ValidationError
from typesafe_sdk import (
    Answer,
    AsyncTypeSafeClient,
    ChoiceAnswer,
    JSONContent,
    NoulAnswer,
    Question,
    ScoreAnswer,
    SystemOneResponse,
    TypeSafeClient,
    Usage,
)

from .errors import TypeSafeResponseError
from .extraction import CandidateExtractor
from .schema import EvaluationPlan, QuestionSpec, build_evaluation_plan

JSONValue: TypeAlias = JSONContent
DEFAULT_THRESHOLD = 0.85


def _request_metadata(
    state: JSONValue,
    plan: EvaluationPlan,
    *,
    model: str,
    threshold: float,
) -> dict[str, str]:
    questions = [
        {
            "key": question.key,
            "kind": question.kind,
            "instructions": question.instructions,
            "options": list(question.native_options),
            "path": list(question.path),
            "probability": question.probability,
            "nullable": question.nullable,
            "local_none": question.local_none,
            "label": question.label,
            "criteria": question.criteria,
            "integer_score": question.integer_score,
        }
        for question in plan.questions
    ]
    state_fingerprint = _json_fingerprint(state)
    questions_fingerprint = _json_fingerprint(questions)
    config_fingerprint = _json_fingerprint(
        {
            "model": model,
            "boolean_threshold": threshold,
            "boolean_comparator": ">",
        }
    )
    return {
        "state_fingerprint": state_fingerprint,
        "questions_fingerprint": questions_fingerprint,
        "config_fingerprint": config_fingerprint,
        "request_fingerprint": _json_fingerprint(
            {
                "state_fingerprint": state_fingerprint,
                "questions_fingerprint": questions_fingerprint,
                "config_fingerprint": config_fingerprint,
            }
        ),
    }


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"kedi-typesafe-request-v1:sha256:{hashlib.sha256(payload).hexdigest()}"


def validate_threshold(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("TypeSafe boolean threshold must be a finite number")
    if not 0 <= value <= 1:
        raise ValueError("TypeSafe boolean threshold must be between 0 and 1")
    return float(value)


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
    values: dict[str, Any]
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
        threshold: float = DEFAULT_THRESHOLD,
        timeout: float | None = None,
        client: AsyncSystemOneClient | None = None,
        sync_client: SystemOneClient | None = None,
        text_extractors: Mapping[str, CandidateExtractor] | None = None,
    ) -> None:
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("TypeSafe model name must not be empty")
        self.model_name = model_name
        self.threshold = validate_threshold(threshold)
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
        threshold: float | None = None,
    ) -> EvaluationResult:
        plan = self._plan(state, schema)
        effective_threshold = self.threshold if threshold is None else validate_threshold(threshold)
        request_metadata = _request_metadata(
            state,
            plan,
            model=self.model_name,
            threshold=effective_threshold,
        )
        if not plan.native_questions():
            return self._result(
                SystemOneResponse(
                    model=self.model_name, answers={}, usage=Usage(input_tokens=0, output_tokens=0)
                ),
                plan,
                effective_threshold,
                request_metadata=request_metadata,
            )
        client = self.async_client()
        response = await client.system_one(
            state,
            plan.native_questions(),
            model=self.model_name,
        )
        return self._result(
            response,
            plan,
            effective_threshold,
            request_metadata=request_metadata,
        )

    def async_client(self) -> AsyncSystemOneClient:
        """Return the borrowed client or the owned client for this event loop."""
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
        return client

    def evaluate_sync(
        self,
        *,
        state: JSONValue,
        schema: Mapping[str, Any],
        threshold: float | None = None,
    ) -> EvaluationResult:
        plan = self._plan(state, schema)
        effective_threshold = self.threshold if threshold is None else validate_threshold(threshold)
        request_metadata = _request_metadata(
            state,
            plan,
            model=self.model_name,
            threshold=effective_threshold,
        )
        if not plan.native_questions():
            return self._result(
                SystemOneResponse(
                    model=self.model_name, answers={}, usage=Usage(input_tokens=0, output_tokens=0)
                ),
                plan,
                effective_threshold,
                request_metadata=request_metadata,
            )
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
        return self._result(
            response,
            plan,
            effective_threshold,
            request_metadata=request_metadata,
        )

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
        threshold: float | None = None,
        *,
        request_metadata: Mapping[str, str] | None = None,
    ) -> EvaluationResult:
        effective_threshold = self.threshold if threshold is None else threshold
        values, answer_metadata = self._decode(response, plan, effective_threshold)
        if plan.schema is not None:
            try:
                validator: Any = Draft202012Validator(plan.schema)
                validator.validate(values)
            except ValidationError as exc:
                raise TypeSafeResponseError(
                    f"TypeSafe output failed schema validation at {list(exc.path)!r}: {exc.message}"
                ) from exc
        return EvaluationResult(
            values=values,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            metadata={
                "schema_version": 1,
                "answers": answer_metadata,
                "boolean_threshold": effective_threshold,
                "boolean_comparator": ">",
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                **dict(request_metadata or {}),
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
        threshold: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        expected_keys = {question.key for question in plan.questions if not question.local_none}
        actual_keys = set(response.answers)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            raise TypeSafeResponseError(
                f"TypeSafe response answer keys do not match the request; "
                f"missing={missing}, unexpected={unexpected}"
            )

        values: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        for question in plan.questions:
            evidence: dict[str, Any]
            if question.local_none:
                value, evidence = None, {"type": "extraction", "source": "no_candidates"}
            elif question.kind == "noul":
                value, evidence = self._decode_noul(
                    question, response.answers[question.key], threshold
                )
            elif question.kind == "score":
                value, evidence = self._decode_score(question, response.answers[question.key])
            else:
                value, evidence = self._decode_choice(question, response.answers[question.key])
            path = question.path or (question.key,)
            target = values
            for segment in path[:-1]:
                target = target.setdefault(segment, {})
            if question.label is not None:
                labels = target.setdefault(path[-1], [])
                if value:
                    labels.append(question.label)
            else:
                target[path[-1]] = value
            answer_metadata: dict[str, Any] = {**evidence, "path": list(path)}
            if question.label is not None:
                answer_metadata["label"] = question.label
            metadata[question.key] = answer_metadata
        return values, metadata

    def _decode_noul(
        self,
        question: QuestionSpec,
        answer: Answer,
        threshold: float | None = None,
    ) -> tuple[bool | float, dict[str, Any]]:
        if not isinstance(answer, NoulAnswer):
            raise TypeSafeResponseError(f"TypeSafe answer {question.key!r} must be a Noul answer")
        probability = answer.noul
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise TypeSafeResponseError(
                f"TypeSafe Noul answer {question.key!r} returned invalid probability {probability!r}"
            )
        effective_threshold = self.threshold if threshold is None else threshold
        return (probability if question.probability else probability > effective_threshold), {
            "type": "noul",
            "probability": probability,
            "output_kind": "probability" if question.probability else "boolean",
        }

    @staticmethod
    def _decode_choice(
        question: QuestionSpec,
        answer: Answer,
    ) -> tuple[str | None, dict[str, Any]]:
        if not isinstance(answer, ChoiceAnswer):
            raise TypeSafeResponseError(f"TypeSafe answer {question.key!r} must be a Choice answer")
        if answer.choice in question.rejected_options and not question.nullable:
            raise TypeSafeResponseError(
                f"TypeSafe could not extract a matching candidate for {question.key!r}"
            )
        if answer.choice not in question.native_options:
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
        return (None if answer.choice in question.rejected_options else answer.choice), {
            "type": "extraction" if question.kind == "extract" else "choice",
            "choice": answer.choice,
            "confidence": answer.confidence,
            "probabilities": probabilities,
        }

    @staticmethod
    def _decode_score(question: QuestionSpec, answer: Answer) -> tuple[float | int, dict[str, Any]]:
        if not isinstance(answer, ScoreAnswer):
            raise TypeSafeResponseError(f"TypeSafe answer {question.key!r} must be a Score answer")
        expected = set(range(len(question.criteria)))
        probabilities = answer.probabilities
        if set(probabilities) != expected or set(answer.legend) != expected:
            raise TypeSafeResponseError("Score probabilities and legend must match rubric levels")
        if any(
            not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()
        ) or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6):
            raise TypeSafeResponseError("Invalid Score probabilities")
        if not math.isfinite(answer.score) or not 0 <= answer.score <= max(expected):
            raise TypeSafeResponseError("Invalid rubric score")
        if not math.isfinite(answer.confidence) or not 0 <= answer.confidence <= 1:
            raise TypeSafeResponseError("Invalid Score confidence")
        value = int(answer.score + 0.5) if question.integer_score else answer.score
        return value, {
            "type": "score",
            "score": answer.score,
            "confidence": answer.confidence,
            "probabilities": dict(probabilities),
            "legend": dict(answer.legend),
        }


__all__ = [
    "AsyncSystemOneClient",
    "EvaluationResult",
    "JSONValue",
    "SystemOneClient",
    "TypeSafeEvaluator",
]
