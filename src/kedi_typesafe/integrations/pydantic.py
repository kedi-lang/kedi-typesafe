from __future__ import annotations

import json
from collections.abc import Mapping
from types import TracebackType
from typing import NoReturn

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, check_allow_model_requests
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage
from typing_extensions import Self

from ..core import CandidateExtractor, TypeSafeEvaluator
from ..core.evaluation import AsyncSystemOneClient, JSONValue

_PROFILE = ModelProfile(
    supports_tools=False,
    supports_json_schema_output=True,
    supports_json_object_output=False,
    default_structured_output_mode="native",
)


def messages_to_state(messages: list[ModelMessage]) -> JSONValue:
    """Project supported Pydantic AI message history into structured Jev state."""

    rendered: list[JSONValue] = []
    latest_instructions: str | None = None

    for message in messages:
        if isinstance(message, ModelRequest):
            if message.instructions is not None:
                latest_instructions = message.instructions
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    rendered.append({"role": "user", "content": _user_content(part)})
                elif isinstance(part, SystemPromptPart):
                    rendered.append({"role": "system", "content": part.content})
                elif isinstance(part, RetryPromptPart):
                    rendered.append({"role": "user", "content": part.model_response()})
                else:
                    _unsupported_part(part.part_kind)
        else:
            for part in message.parts:
                if isinstance(part, TextPart):
                    rendered.append({"role": "assistant", "content": part.content})
                else:
                    _unsupported_part(part.part_kind)

    if not rendered:
        raise ValueError("TypeSafeModel requires at least one supported text message")

    state: dict[str, JSONValue] = {"messages": rendered}
    if latest_instructions:
        state["instructions"] = latest_instructions
    return state


def _user_content(part: UserPromptPart) -> str:
    if isinstance(part.content, str):
        return part.content
    chunks: list[str] = []
    for item in part.content:
        if isinstance(item, str):
            chunks.append(item)
        elif isinstance(item, TextContent):
            chunks.append(item.content)
        else:
            _unsupported_part(item.kind)
    return "\n".join(chunks)


def _unsupported_part(kind: str) -> NoReturn:
    raise ValueError(
        f"TypeSafeModel does not support Pydantic AI message part {kind!r}; "
        "provide text-only state without tool or media history"
    )


class TypeSafeModel(Model):
    """Pydantic AI Model backed by TypeSafe AI's System One API."""

    def __init__(
        self,
        model_name: str = "jev-latest",
        *,
        api_key: str | None = None,
        threshold: float = 0.5,
        timeout: float | None = None,
        client: AsyncSystemOneClient | None = None,
        settings: ModelSettings | None = None,
        text_extractors: Mapping[str, CandidateExtractor] | None = None,
    ) -> None:
        super().__init__(settings=settings, profile=_PROFILE)
        self._evaluator = TypeSafeEvaluator(
            model_name,
            api_key=api_key,
            threshold=threshold,
            timeout=timeout,
            client=client,
            text_extractors=text_extractors,
        )

    @property
    def model_name(self) -> str:
        return self._evaluator.model_name

    @property
    def system(self) -> str:
        return "typesafe"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings, parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        self._validate_request(model_settings, parameters)
        output_object = parameters.output_object
        assert output_object is not None

        result = await self._evaluator.evaluate(
            state=messages_to_state(messages),
            schema=output_object.json_schema,
        )
        return ModelResponse(
            parts=[TextPart(json.dumps(result.values, ensure_ascii=False, separators=(",", ":")))],
            usage=RequestUsage(
                input_tokens=result.input_tokens or 0,
                output_tokens=result.output_tokens or 0,
            ),
            model_name=result.model,
            provider_name=self.system,
            provider_url="https://api.typesafe.ai/v1",
            metadata={"typesafe": result.metadata},
            finish_reason="stop",
        )

    async def aclose(self) -> None:
        await self._evaluator.aclose()

    async def aclose_current(self) -> None:
        await self._evaluator.aclose_current()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        await self.aclose()
        return None

    @staticmethod
    def _validate_request(
        model_settings: ModelSettings | None,
        parameters: ModelRequestParameters,
    ) -> None:
        if model_settings:
            names = ", ".join(sorted(model_settings))
            raise UserError(f"TypeSafeModel does not support model settings: {names}")
        if parameters.function_tools or parameters.native_tools:
            raise UserError("TypeSafeModel does not support function or native tools")
        if parameters.output_mode != "native" or parameters.output_object is None:
            raise UserError(
                "TypeSafeModel requires native structured output; pass `output_type` to Agent"
            )


__all__ = ["TypeSafeModel", "messages_to_state"]
