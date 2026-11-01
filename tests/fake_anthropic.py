"""An offline stand-in for the Anthropic Messages API.

Instead of faking SDK objects, this serves scripted *HTTP* responses (Server-Sent
Events, exactly as the real API streams them) through an ``httpx2.MockTransport``
plugged into a real ``anthropic`` client. Tests therefore exercise the SDK's own
request serialisation and stream parsing, and can assert on the exact JSON body
and headers that would have been sent - without any network access.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import anthropic
import httpx2

BASE_URL = "https://api.anthropic.test"


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Thinking:
    """A thinking block with omitted text (the default display on current models)."""

    signature: str = "sig-abc"


@dataclass(frozen=True)
class ToolUse:
    #: Raw JSON text of the tool input, streamed as ``input_json_delta`` fragments.
    input_json: str
    name: str
    id: str = "toolu_01"


Block: TypeAlias = Text | Thinking | ToolUse


@dataclass(frozen=True)
class Turn:
    """One scripted assistant response."""

    blocks: tuple[Block, ...]
    stop_reason: str = "end_turn"
    stop_details: dict[str, Any] | None = None
    input_tokens: int = 120
    output_tokens: int = 40
    cache_read_input_tokens: int = 0
    model: str = "claude-opus-5-5"
    chunk_size: int = 7


@dataclass(frozen=True)
class HTTPError:
    status: int
    error_type: str = "api_error"
    message: str = "something went wrong"


@dataclass(frozen=True)
class ConnectionFailure:
    message: str = "connection refused"


Script: TypeAlias = Turn | HTTPError | ConnectionFailure


def text_turn(text: str, **kwargs: Any) -> Turn:
    return Turn(blocks=(Thinking(), Text(text)), **kwargs)


def tool_turn(input_json: str, name: str, **kwargs: Any) -> Turn:
    kwargs.setdefault("stop_reason", "tool_use")
    return Turn(blocks=(Thinking(), ToolUse(input_json=input_json, name=name)), **kwargs)


@dataclass
class RecordedRequest:
    method: str
    url: str
    headers: dict[str, str]
    body: dict[str, Any] = field(default_factory=dict)


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


def _sse(turn: Turn) -> bytes:
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": turn.model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": turn.input_tokens,
                    "output_tokens": 1,
                    "cache_read_input_tokens": turn.cache_read_input_tokens,
                    "cache_creation_input_tokens": 0,
                },
            },
        }
    ]
    for index, block in enumerate(turn.blocks):
        if isinstance(block, Text):
            events.append(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": ""},
                }
            )
            for piece in _chunks(block.text, turn.chunk_size):
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "text_delta", "text": piece},
                    }
                )
        elif isinstance(block, Thinking):
            events.append(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                }
            )
            events.append(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "signature_delta", "signature": block.signature},
                }
            )
        else:
            events.append(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": {},
                    },
                }
            )
            for piece in _chunks(block.input_json, turn.chunk_size):
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": piece},
                    }
                )
        events.append({"type": "content_block_stop", "index": index})
    delta: dict[str, Any] = {"stop_reason": turn.stop_reason, "stop_sequence": None}
    if turn.stop_details is not None:
        delta["stop_details"] = turn.stop_details
    events.append(
        {"type": "message_delta", "delta": delta, "usage": {"output_tokens": turn.output_tokens}}
    )
    events.append({"type": "message_stop"})
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


class FakeAnthropicAPI:
    """Replays a script of responses, one per HTTP request, and records requests."""

    def __init__(self, script: Iterable[Script]) -> None:
        self.script: deque[Script] = deque(script)
        self.requests: list[RecordedRequest] = []

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append(
            RecordedRequest(
                method=request.method,
                url=str(request.url),
                headers=dict(request.headers),
                body=body,
            )
        )
        if not self.script:
            raise AssertionError("FakeAnthropicAPI received more requests than scripted")
        item = self.script.popleft()
        if isinstance(item, ConnectionFailure):
            raise httpx2.ConnectError(item.message, request=request)
        if isinstance(item, HTTPError):
            return httpx2.Response(
                item.status,
                json={"type": "error", "error": {"type": item.error_type, "message": item.message}},
            )
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse(item)
        )

    def client(self) -> anthropic.Anthropic:
        return anthropic.Anthropic(
            api_key="sk-ant-test-not-a-real-key",
            base_url=BASE_URL,
            max_retries=0,
            http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(self._handle)),
        )

    def async_client(self) -> anthropic.AsyncAnthropic:
        return anthropic.AsyncAnthropic(
            api_key="sk-ant-test-not-a-real-key",
            base_url=BASE_URL,
            max_retries=0,
            http_client=anthropic.DefaultAsyncHttpxClient(
                transport=httpx2.MockTransport(self._handle)
            ),
        )

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [request.body for request in self.requests]
