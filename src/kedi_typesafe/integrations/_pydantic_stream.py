from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponseStreamEvent, TextPart, ToolCallPart
from pydantic_ai.models.typesafe import TypeSafeStreamedResponse


class ExtendedTypeSafeStream(TypeSafeStreamedResponse):
    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        for index, part in enumerate(self._response.parts):
            if isinstance(part, TextPart):
                for event in self._parts_manager.handle_text_delta(
                    vendor_part_id=index, content=part.content
                ):
                    yield event
            elif isinstance(part, ToolCallPart):
                yield self._parts_manager.handle_tool_call_part(
                    vendor_part_id=index,
                    tool_name=part.tool_name,
                    args=part.args_as_dict(),
                    tool_call_id=part.tool_call_id,
                )
            else:
                raise UnexpectedModelBehavior(
                    f"Unsupported TypeSafe response part: {part.part_kind}"
                )
