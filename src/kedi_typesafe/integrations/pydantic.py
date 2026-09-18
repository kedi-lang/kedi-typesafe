from __future__ import annotations

# This pinned integration intentionally shares upstream projection and evaluator internals.
# pyright: reportPrivateUsage=false
import json
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestParameters, check_allow_model_requests
from pydantic_ai.models import typesafe as upstream
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage
from typesafe_sdk import SystemOneResponse, Usage
from typing_extensions import Self

from ..core import CandidateExtractor, TypeSafeEvaluator
from ..core.evaluation import DEFAULT_THRESHOLD, AsyncSystemOneClient, JSONValue, validate_threshold
from ..core.schema import EvaluationPlan
from ._pydantic_errors import provider_errors
from ._pydantic_provider import EvaluatorProvider
from ._pydantic_stream import ExtendedTypeSafeStream

_PROFILE = ModelProfile(
    supports_tools=True,
    supports_json_schema_output=True,
    supports_json_object_output=False,
    default_structured_output_mode="tool",
)


class TypeSafeModelSettings(upstream.TypeSafeModelSettings, total=False):
    typesafe_threshold: float
    """Strict probability threshold for bools and label membership; default 0.85."""


def messages_to_state(messages: list[ModelMessage]) -> JSONValue:
    """Use the pinned upstream projection, including tool and retry history."""
    return upstream._map_messages(messages)


class TypeSafeModel(upstream.TypeSafeModel):
    """Upstream Jev model with explicit decision thresholds and extended schemas."""

    supports_decision_threshold = True

    def __init__(
        self,
        model_name: str = "jev-latest",
        *,
        api_key: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        timeout: float | None = None,
        client: AsyncSystemOneClient | None = None,
        settings: ModelSettings | None = None,
        text_extractors: Mapping[str, CandidateExtractor] | None = None,
    ) -> None:
        self._evaluator = TypeSafeEvaluator(
            model_name,
            api_key=api_key,
            threshold=threshold,
            timeout=timeout,
            client=client,
            text_extractors=text_extractors,
        )
        super().__init__(
            model_name,
            provider=EvaluatorProvider(self._evaluator),
            profile=_PROFILE,
            settings=settings,
        )

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        settings, parameters = self.prepare_request(model_settings, model_request_parameters)
        settings = settings or {}
        unsupported = set(settings) - {"typesafe_threshold", "typesafe_tool_call_threshold"}
        if unsupported:
            raise UserError(
                f"TypeSafeModel does not support model settings: {', '.join(sorted(unsupported))}"
            )
        threshold = validate_threshold(
            settings.get("typesafe_threshold", self._evaluator.threshold)
        )
        tool_threshold = validate_threshold(settings.get("typesafe_tool_call_threshold", 0.6))
        native = parameters.output_mode == "native" and parameters.output_object is not None
        if native:
            output = parameters.output_object
            assert output is not None
            output_tool = ToolDefinition(
                name=output.name or "final_result",
                description=output.description,
                parameters_json_schema=output.json_schema,
            )
            hand_offs: list[ToolDefinition] = []
        else:
            output_tool, hand_offs = upstream._output_tools(parameters)
        tools = upstream._tools_left(
            messages,
            [
                *hand_offs,
                *[
                    tool
                    for tool in parameters.function_tools
                    if parameters.visibility_of(tool.name) != "withheld"
                ],
            ],
        )
        state = messages_to_state(messages)
        instructions = "\n\n".join(
            part.content for part in (self._get_instruction_parts(messages, parameters) or [])
        )
        if instructions:
            state = {"state": state, "instructions": instructions}
        plan = (
            self._evaluator._plan(state, output_tool.parameters_json_schema)
            if output_tool
            else EvaluationPlan(questions=())
        )
        questions = plan.native_questions()
        route_key = upstream._tool_question(questions, output_tool, tools, instructions or None)
        if questions:
            with provider_errors(self.model_name):
                response = await self._evaluator.async_client().system_one(
                    state, questions, model=self.model_name
                )
        else:
            response = SystemOneResponse(
                model=self.model_name, answers={}, usage=Usage(input_tokens=0, output_tokens=0)
            )
        details: dict[str, Any] = {}
        call = (
            upstream._tool_call(
                self.model_name,
                response.answers.get(route_key),
                output_tool,
                tools,
                {tool.name for tool in hand_offs},
                tool_threshold,
                details,
            )
            if route_key
            else None
        )
        parts: list[ModelResponsePart]
        if call is not None:
            parts = [call]
        else:
            field_response = SystemOneResponse(
                model=response.model,
                usage=response.usage,
                answers={key: value for key, value in response.answers.items() if key != route_key},
            )
            result = self._evaluator._result(field_response, plan, threshold)
            details["typesafe"] = result.metadata
            # Without an output schema, upstream routing returns a call or raises.
            assert output_tool is not None
            parts = (
                [TextPart(json.dumps(result.values, ensure_ascii=False))]
                if native
                else [ToolCallPart(output_tool.name, result.values)]
            )
        return ModelResponse(
            parts=parts,
            usage=RequestUsage(
                input_tokens=response.usage.input_tokens or 0,
                output_tokens=response.usage.output_tokens or 0,
            ),
            model_name=response.model,
            provider_name=self.system,
            provider_url=self.base_url,
            provider_details=details,
            metadata=details,
            finish_reason="stop" if native and call is None else "tool_call",
        )

    async def aclose(self) -> None:
        await self._evaluator.aclose()

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[ExtendedTypeSafeStream]:
        response = await self.request(messages, model_settings, model_request_parameters)
        yield ExtendedTypeSafeStream(model_request_parameters, response)

    async def aclose_current(self) -> None:
        await self._evaluator.aclose_current()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


__all__ = ["TypeSafeModel", "TypeSafeModelSettings", "messages_to_state"]
