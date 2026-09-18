from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping

import msgspec
from langchain_typesafe import TypeSafeClassifier
from langchain_typesafe.types import Question as ClassifierQuestion
from langchain_typesafe.types import State
from pydantic import JsonValue, TypeAdapter
from typesafe_sdk import Question, SystemOneResponse

from ..core.evaluation import JSONValue

_QUESTIONS = TypeAdapter(dict[str, ClassifierQuestion])
# The bridge accepts only the JSON subset of the classifier's broader State type.
_STATE: TypeAdapter[State] = TypeAdapter(str | list[JsonValue] | dict[str, JsonValue])


class ClassifierTransport:
    """Reuse official transport without nesting another LLM invocation span."""

    def __init__(self, *, api_key: str | None, timeout: float | None) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self._sync: TypeSafeClassifier | None = None
        self._async: dict[asyncio.AbstractEventLoop, TypeSafeClassifier] = {}

    def _request(
        self,
        state: JSONValue,
        questions: Mapping[str, Question],
        model: str | None,
        *,
        asynchronous: bool,
    ) -> TypeSafeClassifier:
        del state
        typed = _QUESTIONS.validate_python(msgspec.to_builtins(dict(questions)))
        loop = asyncio.get_running_loop() if asynchronous else None
        base = self._async.get(loop) if loop is not None else self._sync
        if base is None:
            base = TypeSafeClassifier(
                questions=typed,
                model=model or "jev-latest",
                timeout=self.timeout if self.timeout is not None else 30.0,
                api_key=self.api_key
                if self.api_key is not None
                else os.environ.get("TYPESAFE_API_KEY", ""),
            )
            if loop is None:
                self._sync = base
            else:
                self._async[loop] = base
        return base.model_copy(update={"questions": typed, "model": model or "jev-latest"})

    async def system_one(
        self, state: JSONValue, questions: Mapping[str, Question], *, model: str | None = None
    ) -> SystemOneResponse:
        request = self._request(state, questions, model, asynchronous=True)
        # Pinned classifier transport avoids double-counting a nested LLM span.
        response = await request._aclassify(_STATE.validate_python(state))  # pyright: ignore[reportPrivateUsage]
        return msgspec.json.decode(
            response.model_dump_json(exclude={"request_id"}), type=SystemOneResponse
        )

    def system_one_sync(
        self, state: JSONValue, questions: Mapping[str, Question], *, model: str | None = None
    ) -> SystemOneResponse:
        request = self._request(state, questions, model, asynchronous=False)
        response = request._classify(_STATE.validate_python(state))  # pyright: ignore[reportPrivateUsage]
        return msgspec.json.decode(
            response.model_dump_json(exclude={"request_id"}), type=SystemOneResponse
        )

    async def aclose(self) -> None:
        base = self._async.pop(asyncio.get_running_loop(), None)
        if base is not None:
            assert base.async_client is not None and base.client is not None
            await base.async_client.aclose()
            base.client.close()

    def close(self) -> None:
        if self._sync is not None:
            assert self._sync.client is not None
            self._sync.client.close()
            # The async client paired with the sync classifier was never used.
            self._sync = None


class SyncClassifierTransport:
    def __init__(self, transport: ClassifierTransport) -> None:
        self.transport = transport

    def system_one(
        self, state: JSONValue, questions: Mapping[str, Question], *, model: str | None = None
    ) -> SystemOneResponse:
        return self.transport.system_one_sync(state, questions, model=model)

    def close(self) -> None:
        self.transport.close()
