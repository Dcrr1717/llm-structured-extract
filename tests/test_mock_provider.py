from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from llm_structured_extract import (
    BaseProvider,
    Completion,
    CompletionRequest,
    Message,
    MockProvider,
    MockResponse,
    Provider,
    ProviderError,
    ScriptExhaustedError,
    StreamChunk,
    Usage,
)


def _request(attempt: int = 1) -> CompletionRequest:
    return CompletionRequest(
        system="system prompt",
        messages=(Message(role="user", content="document text"),),
        schema_name="Thing",
        json_schema={"type": "object"},
        attempt=attempt,
    )


class Thing(BaseModel):
    name: str


def test_script_item_kinds() -> None:
    provider = MockProvider(
        [
            "raw text",
            {"b": 1, "a": [1, 2]},
            [1, 2],
            Thing(name="x"),
            MockResponse(text="cut", stop_reason="max_tokens", usage=Usage(1, 2), model="m-2"),
            lambda request: {"attempt": request.attempt},
        ]
    )
    texts = [provider.complete(_request(attempt=n)).text for n in range(1, 5)]
    assert texts == [
        "raw text",
        '{\n  "b": 1,\n  "a": [\n    1,\n    2\n  ]\n}',
        "[\n  1,\n  2\n]",
        '{\n  "name": "x"\n}',
    ]
    truncated = provider.complete(_request())
    assert (truncated.stop_reason, truncated.usage, truncated.model) == (
        "max_tokens",
        Usage(1, 2),
        "m-2",
    )
    assert provider.complete(_request(attempt=6)).text == '{\n  "attempt": 6\n}'
    assert provider.calls == 6
    assert provider.remaining == 0


def test_requests_are_recorded() -> None:
    provider = MockProvider(["a", "b"])
    first, second = _request(1), _request(2)
    provider.complete(first)
    provider.complete(second)
    assert provider.requests == [first, second]


def test_exceptions_are_raised() -> None:
    provider = MockProvider([ProviderError("boom", provider="mock", retryable=True)])
    with pytest.raises(ProviderError, match="boom"):
        provider.complete(_request())


def test_exhaustion_raises_by_default() -> None:
    provider = MockProvider(["only one"])
    provider.complete(_request())
    with pytest.raises(ScriptExhaustedError, match="exhausted after 1 call"):
        provider.complete(_request())


def test_repeat_last() -> None:
    provider = MockProvider(["same"], on_exhausted="repeat_last")
    assert [provider.complete(_request()).text for _ in range(3)] == ["same"] * 3


def test_responder_after_script() -> None:
    provider = MockProvider(["scripted"], responder=lambda request: f"computed {request.attempt}")
    assert provider.complete(_request(1)).text == "scripted"
    assert provider.complete(_request(2)).text == "computed 2"
    assert provider.complete(_request(3)).text == "computed 3"


def test_usage_is_deterministic() -> None:
    usage_a = MockProvider(["x" * 40]).complete(_request()).usage
    usage_b = MockProvider(["x" * 40]).complete(_request()).usage
    assert usage_a == usage_b
    assert usage_a.output_tokens == 10
    assert usage_a.input_tokens == 8  # ceil(13 / 4) for "system prompt" and "document text"
    assert MockProvider([""]).complete(_request()).usage.output_tokens == 0


def test_streaming_chunks_reassemble() -> None:
    provider = MockProvider(["abcdefghij"], chunk_size=3)
    chunks = list(provider.stream(_request()))
    assert [chunk.text for chunk in chunks[:-1]] == ["abc", "def", "ghi", "j"]
    final = chunks[-1].completion
    assert final is not None
    assert final.text == "abcdefghij"


def test_async_api() -> None:
    provider = MockProvider(["one", "two"], chunk_size=2)

    async def run() -> tuple[str, list[str]]:
        completion = await provider.acomplete(_request())
        chunks = [chunk.text async for chunk in provider.astream(_request())]
        return completion.text, chunks

    text, chunks = asyncio.run(run())
    assert text == "one"
    assert chunks == ["tw", "o", ""]


def test_invalid_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        MockProvider([], chunk_size=0)


def test_unsupported_script_item() -> None:
    provider = MockProvider([3.14])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="Unsupported MockProvider script item"):
        provider.complete(_request())


def test_mock_satisfies_the_provider_protocol() -> None:
    assert isinstance(MockProvider(), Provider)


class EchoProvider(BaseProvider):
    """Minimal provider: only ``complete`` is implemented."""

    name = "echo"

    def complete(self, request: CompletionRequest) -> Completion:
        return Completion(text=request.messages[-1].content, usage=Usage(1, 1))


def test_base_provider_defaults() -> None:
    provider = EchoProvider()
    assert isinstance(provider, Provider)
    chunks = list(provider.stream(_request()))
    assert chunks == [
        StreamChunk(text="document text", completion=Completion("document text", Usage(1, 1)))
    ]

    async def run() -> tuple[Completion, list[StreamChunk]]:
        return await provider.acomplete(_request()), [c async for c in provider.astream(_request())]

    completion, async_chunks = asyncio.run(run())
    assert completion.text == "document text"
    assert async_chunks == chunks
