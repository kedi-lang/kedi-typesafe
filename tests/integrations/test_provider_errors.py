from __future__ import annotations

# Transport ownership assertions deliberately inspect the evaluator boundary.
# pyright: reportPrivateUsage=false
import asyncio
from typing import Any

import httpx2
import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from typesafe_sdk import (
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeError,
)

from kedi_typesafe import TypeSafeModel
from kedi_typesafe.integrations._pydantic_errors import provider_errors
from kedi_typesafe.integrations._pydantic_provider import EvaluatorProvider


@pytest.mark.parametrize(
    "source,target",
    [
        (
            TypeSafeAPIResponseValidationError(200, {}, httpx2.Headers(), "answers"),
            UnexpectedModelBehavior,
        ),
        (
            TypeSafeAPIError(429, {"error": "limited"}, httpx2.Headers({"retry-after": "1"})),
            ModelHTTPError,
        ),
        (TypeSafeAPIConnectionError("offline"), ModelAPIError),
        (TypeSafeError("bad request"), UserError),
    ],
)
def test_provider_error_mapping_preserves_cause(source: Exception, target: type[Exception]) -> None:
    with pytest.raises(target) as error, provider_errors("test"):
        raise source
    assert error.value.__cause__ is source


class FailingClient:
    def __init__(self, failure: Exception) -> None:
        self.failure = failure
        self.calls = 0

    async def system_one(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        raise self.failure

    async def aclose(self) -> None:
        raise AssertionError("Borrowed client cannot be closed")


@pytest.mark.asyncio
async def test_auth_failure_is_not_retried_by_wrapper() -> None:
    client = FailingClient(TypeSafeAPIError(401, {}, httpx2.Headers()))
    model = TypeSafeModel(client=client)
    assert EvaluatorProvider(model._evaluator).client is client
    with pytest.raises(ModelHTTPError):
        await Agent(model, output_type=bool).run("Claim")
    assert client.calls == 1
    await model.aclose()


@pytest.mark.asyncio
async def test_cancellation_propagates_without_retry_or_borrowed_close() -> None:
    started = asyncio.Event()

    class WaitingClient:
        calls = 0

        async def system_one(self, *args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            started.set()
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            raise AssertionError("Borrowed client cannot be closed")

    client = WaitingClient()
    model = TypeSafeModel(client=client)
    task = asyncio.create_task(Agent(model, output_type=bool).run("Claim"))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.calls == 1
    await model.aclose()
