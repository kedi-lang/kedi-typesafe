from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import Any, NoReturn, cast

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import PrivateAttr, TypeAdapter
from typing_extensions import Self

from ..core import CandidateExtractor, TypeSafeEvaluator
from ..core.evaluation import AsyncSystemOneClient, JSONValue, SystemOneClient

_PROFILE = {
    "name": "TypeSafe Jev",
    "text_inputs": True,
    "text_outputs": True,
    "tool_calling": False,
    "tool_choice": False,
    "structured_output": True,
}


def messages_to_state(messages: list[BaseMessage]) -> JSONValue:
    """Project supported LangChain message history into structured Jev state."""

    rendered: list[JSONValue] = []
    for message in messages:
        if isinstance(message, AIMessage) and (message.tool_calls or message.invalid_tool_calls):
            _unsupported_message("assistant tool-call history")
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, ChatMessage) and message.role in {
            "system",
            "user",
            "assistant",
        }:
            role = message.role
        else:
            _unsupported_message(message.type)
        rendered.append({"role": role, "content": _text_content(message)})

    if not rendered:
        raise ValueError("TypeSafeChatModel requires at least one supported text message")
    return {"messages": rendered}


def _text_content(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    chunks: list[str] = []
    for item in message.content:
        if isinstance(item, str):
            chunks.append(item)
            continue
        content_block = cast(dict[str, Any], item)
        if content_block.get("type") == "text":
            text = content_block.get("text")
            if not isinstance(text, str):
                _unsupported_message("non-string text content")
            chunks.append(text)
        else:
            kind = content_block.get("type", "unknown")
            _unsupported_message(f"content block {kind!r}")
    return "\n".join(chunks)


def _unsupported_message(kind: str) -> NoReturn:
    raise ValueError(
        f"TypeSafeChatModel does not support LangChain {kind}; "
        "provide text-only state without tool or media history"
    )


class TypeSafeChatModel(BaseChatModel):
    """LangChain chat model backed by TypeSafe AI's System One API."""

    model_name: str = "jev-latest"
    threshold: float = 0.5
    _evaluator: TypeSafeEvaluator = PrivateAttr()

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
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("profile", dict(_PROFILE))
        model_values: dict[str, Any] = {
            "model_name": model_name,
            "threshold": threshold,
            **kwargs,
        }
        super().__init__(**model_values)
        self._evaluator = TypeSafeEvaluator(
            model_name,
            api_key=api_key,
            threshold=threshold,
            timeout=timeout,
            client=client,
            sync_client=sync_client,
            text_extractors=text_extractors,
        )

    @property
    def _llm_type(self) -> str:
        return "typesafe-jev"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "threshold": self.threshold}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        schema = self._request_schema(stop=stop, kwargs=kwargs)
        result = self._evaluator.evaluate_sync(
            state=messages_to_state(messages),
            schema=schema,
        )
        return _chat_result(result)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del run_manager
        schema = self._request_schema(stop=stop, kwargs=kwargs)
        result = await self._evaluator.evaluate(
            state=messages_to_state(messages),
            schema=schema,
        )
        return _chat_result(result)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        if tools:
            raise ValueError("TypeSafeChatModel does not support tools")
        if tool_choice is not None:
            raise ValueError("TypeSafeChatModel does not support tool choice")
        response_format = kwargs.pop("response_format", None)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise ValueError(f"TypeSafeChatModel does not support model settings: {names}")
        if response_format is None:
            raise ValueError(
                "TypeSafeChatModel requires native structured output; provide a response schema"
            )
        return self.bind(response_format=response_format)

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise ValueError(f"TypeSafeChatModel does not support model settings: {names}")
        response_format, validator = _structured_contract(schema)
        model = self.bind(response_format=response_format)

        def parse(message: AIMessage) -> Any:
            try:
                parsed = validator.validate_json(_message_text(message))
            except Exception as exc:
                if include_raw:
                    return {"raw": message, "parsed": None, "parsing_error": exc}
                raise
            if include_raw:
                return {"raw": message, "parsed": parsed, "parsing_error": None}
            return parsed

        return model | RunnableLambda(parse)

    def close(self) -> None:
        self._evaluator.close()

    async def aclose(self) -> None:
        await self._evaluator.aclose()

    async def aclose_current(self) -> None:
        await self._evaluator.aclose_current()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        self.close()
        return None

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
    def _request_schema(
        *,
        stop: list[str] | None,
        kwargs: dict[str, Any],
    ) -> Mapping[str, Any]:
        if stop:
            raise ValueError("TypeSafeChatModel does not support stop sequences")
        response_format = kwargs.pop("response_format", None)
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise ValueError(f"TypeSafeChatModel does not support model settings: {names}")
        if not isinstance(response_format, Mapping):
            raise ValueError(
                "TypeSafeChatModel requires native structured output; provide a response schema"
            )
        typed_response_format = cast(Mapping[str, Any], response_format)
        if typed_response_format.get("type") != "json_schema":
            raise ValueError("TypeSafeChatModel requires a JSON Schema response format")
        envelope = typed_response_format.get("json_schema")
        if not isinstance(envelope, Mapping):
            raise ValueError("TypeSafeChatModel received an invalid JSON Schema response format")
        typed_envelope = cast(Mapping[str, Any], envelope)
        schema = typed_envelope.get("schema")
        if not isinstance(schema, Mapping):
            raise ValueError("TypeSafeChatModel received an invalid JSON Schema response schema")
        return cast(Mapping[str, Any], schema)


def _structured_contract(
    schema: dict[str, Any] | type,
) -> tuple[dict[str, Any], TypeAdapter[Any]]:
    if isinstance(schema, dict):
        function = schema.get("function")
        if schema.get("type") == "function" and isinstance(function, Mapping):
            typed_function = cast(Mapping[str, Any], function)
            json_schema = typed_function.get("parameters")
            name: object = typed_function.get("name", "response")
        else:
            json_schema = schema
            name = schema.get("title", "response")
        if not isinstance(json_schema, Mapping):
            raise ValueError("TypeSafeChatModel received an invalid structured output schema")
        json_schema = cast(Mapping[str, Any], json_schema)
        validator = TypeAdapter(dict[str, Any])
    else:
        validator = cast(TypeAdapter[Any], TypeAdapter(schema))
        json_schema = validator.json_schema()
        name = getattr(schema, "__name__", "response")
    return (
        {
            "type": "json_schema",
            "json_schema": {"name": str(name), "schema": dict(json_schema)},
        },
        validator,
    )


def _message_text(message: AIMessage) -> str:
    return cast(str, message.content)


def _chat_result(result: Any) -> ChatResult:
    usage_metadata = None
    if result.input_tokens is not None or result.output_tokens is not None:
        input_tokens = result.input_tokens or 0
        output_tokens = result.output_tokens or 0
        usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    message = AIMessage(
        content=json.dumps(result.values, ensure_ascii=False, separators=(",", ":")),
        response_metadata={
            "model_name": result.model,
            "provider_name": "typesafe",
            "typesafe": result.metadata,
        },
        usage_metadata=usage_metadata,
    )
    return ChatResult(
        generations=[ChatGeneration(message=message)],
        llm_output={"model_name": result.model, "provider_name": "typesafe"},
    )


__all__ = ["TypeSafeChatModel", "messages_to_state"]
