"""A deterministic, scriptable provider for tests, demos and CI.

:class:`MockProvider` replays a script of responses, one per call, and records
every request it receives so tests can assert on prompts (for example that a
repair turn really contains the validation errors). It never touches the
network and produces identical output on every run.

Script items can be:

* ``str`` - returned verbatim as the model output,
* ``dict`` / ``list`` / Pydantic model - serialised to pretty-printed JSON,
* :class:`MockResponse` - full control over text, stop reason and usage,
* an exception instance - raised from the call (simulate provider failures),
* a callable ``(CompletionRequest) -> item`` - compute the answer from the
  request; the returned item is interpreted with these same rules.
"""

from __future__ import annotations

import json
import math
import threading
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, Union

from pydantic import BaseModel

from ..errors import ScriptExhaustedError
from ..types import STOP_END_TURN, Completion, CompletionRequest, StreamChunk, Usage
from .base import BaseProvider

__all__ = ["MockProvider", "MockResponse", "ScriptItem"]


@dataclass(frozen=True, slots=True)
class MockResponse:
    """A fully specified scripted response."""

    text: str
    stop_reason: str = STOP_END_TURN
    #: Defaults to a deterministic estimate (about four characters per token).
    usage: Usage | None = None
    model: str | None = None


ScriptItem: TypeAlias = Union[  # noqa: UP007 - recursive alias needs a runtime Union
    str,
    dict[str, Any],
    list[Any],
    BaseModel,
    MockResponse,
    BaseException,
    Callable[[CompletionRequest], "ScriptItem"],
]


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4)) if text else 0


def _render(item: Any) -> MockResponse:
    if isinstance(item, MockResponse):
        return item
    if isinstance(item, str):
        return MockResponse(text=item)
    if isinstance(item, BaseModel):
        return MockResponse(text=item.model_dump_json(indent=2))
    if isinstance(item, (dict, list)):
        return MockResponse(text=json.dumps(item, indent=2, ensure_ascii=False, default=str))
    raise TypeError(f"Unsupported MockProvider script item: {item!r}")


class MockProvider(BaseProvider):
    """Deterministic provider that replays scripted responses.

    Args:
        script: Responses to return, in order (one per call).
        responder: Used once ``script`` is exhausted; called with the request.
            Handy for rule-based mocks that answer any number of calls.
        on_exhausted: What to do when both are exhausted: ``"raise"``
            :class:`~llm_structured_extract.errors.ScriptExhaustedError` (default)
            or ``"repeat_last"`` to keep returning the last scripted item.
        chunk_size: Characters per chunk when streaming.
        model: Model name reported on completions.

    Example::

        provider = MockProvider([
            '{"name": "Ada", "age": "unknown"}',  # attempt 1: fails validation
            {"name": "Ada", "age": 36},            # attempt 2: repaired
        ])
    """

    name = "mock"

    def __init__(
        self,
        script: Iterable[ScriptItem] = (),
        *,
        responder: Callable[[CompletionRequest], ScriptItem] | None = None,
        on_exhausted: Literal["raise", "repeat_last"] = "raise",
        chunk_size: int = 16,
        model: str = "mock-1",
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        self._script: deque[ScriptItem] = deque(script)
        self._responder = responder
        self._on_exhausted = on_exhausted
        self._last: ScriptItem | None = None
        self._lock = threading.Lock()
        self.chunk_size = chunk_size
        self.model = model
        #: Every request received, in order.
        self.requests: list[CompletionRequest] = []

    @property
    def calls(self) -> int:
        """Number of requests served so far."""
        return len(self.requests)

    @property
    def remaining(self) -> int:
        """Number of scripted items not yet consumed."""
        return len(self._script)

    def _next_item(self, request: CompletionRequest) -> ScriptItem:
        with self._lock:
            self.requests.append(request)
            if self._script:
                item = self._script.popleft()
                self._last = item
                return item
            if self._responder is not None:
                return self._responder
            if self._on_exhausted == "repeat_last" and self._last is not None:
                return self._last
            raise ScriptExhaustedError(
                f"MockProvider script exhausted after {len(self.requests) - 1} call(s); "
                "add more responses or pass responder=..."
            )

    def _respond(self, request: CompletionRequest) -> Completion:
        item: Any = self._next_item(request)
        while callable(item):
            item = item(request)
        if isinstance(item, BaseException):
            raise item
        response = _render(item)
        usage = response.usage or Usage(
            input_tokens=_estimate_tokens(request.system)
            + sum(_estimate_tokens(message.content) for message in request.messages),
            output_tokens=_estimate_tokens(response.text),
        )
        return Completion(
            text=response.text,
            usage=usage,
            stop_reason=response.stop_reason,
            model=response.model or self.model,
        )

    def _chunks(self, completion: Completion) -> Iterator[StreamChunk]:
        text = completion.text
        for start in range(0, len(text), self.chunk_size):
            yield StreamChunk(text=text[start : start + self.chunk_size])
        yield StreamChunk(completion=completion)

    # -- Provider API ------------------------------------------------------ #

    def complete(self, request: CompletionRequest) -> Completion:
        return self._respond(request)

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        yield from self._chunks(self._respond(request))

    async def acomplete(self, request: CompletionRequest) -> Completion:
        return self._respond(request)

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        for chunk in self._chunks(self._respond(request)):
            yield chunk
