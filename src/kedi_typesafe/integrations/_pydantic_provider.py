from __future__ import annotations

from typing import Any

from pydantic_ai.providers import Provider

from ..core.evaluation import AsyncSystemOneClient, TypeSafeEvaluator


class EvaluatorProvider(Provider[Any]):
    """Share the evaluator's event-loop-local client with Pydantic AI."""

    def __init__(self, evaluator: TypeSafeEvaluator) -> None:
        self.evaluator = evaluator

    @property
    def name(self) -> str:
        return "typesafe"

    @property
    def base_url(self) -> str:
        return "https://api.typesafe.ai/v1"

    @property
    def client(self) -> AsyncSystemOneClient:
        return self.evaluator.async_client()
