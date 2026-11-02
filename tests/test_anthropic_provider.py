"""AnthropicProvider tests: the real SDK over a scripted, offline HTTP transport."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import anthropic
import pytest
from anthropic.lib.streaming import BetaMessageStream
from pydantic import BaseModel, Field

from llm_structured_extract import (
    AnthropicProvider,
    CompletionRequest,
    ExtractionFailedError,
    Extractor,
    Message,
    OutputParseError,
    OutputTruncatedError,
    OutputValidationError,
    PartialEvent,
    ProviderError,
    ProviderNotConfiguredError,
    RefusalError,
    ResultEvent,
    SchemaError,
)
from llm_structured_extract.providers.anthropic import DEFAULT_MODEL, REFUSAL_FALLBACK_BETA

from .fake_anthropic import (
    ConnectionFailure,
    FakeAnthropicAPI,
    HTTPError,
    Text,
    Thinking,
    Turn,
    text_turn,
    tool_turn,
)


class Ticket(BaseModel):
    """A support ticket."""

    subject: str = Field(max_length=80)
    priority: int = Field(ge=1, le=5, description="1 = lowest, 5 = highest")
    tags: list[str] = Field(default_factory=list)


GOOD = {"subject": "Login fails", "priority": 4, "tags": ["auth"]}
GOOD_JSON = json.dumps(GOOD)
DOCUMENT = "Customer says login fails since this morning, quite urgent."


def provider_for(api: FakeAnthropicAPI, **kwargs: Any) -> AnthropicProvider:
    return AnthropicProvider(client=api.client(), async_client=api.async_client(), **kwargs)


# --------------------------------------------------------------------------- #
# Configuration and credential gating
# --------------------------------------------------------------------------- #


def test_requires_an_api_key() -> None:
    with pytest.raises(ProviderNotConfiguredError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider()


def test_reads_the_key_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    provider = AnthropicProvider()
    assert provider._sync_client().api_key == "sk-ant-from-env"
    assert provider._async().api_key == "sk-ant-from-env"


def test_explicit_key_and_client_options() -> None:
    provider = AnthropicProvider(api_key="sk-ant-explicit", timeout=12.5, max_retries=5)
    client = provider._sync_client()
    assert client.api_key == "sk-ant-explicit"
    assert client.max_retries == 5
    assert client.timeout == 12.5


def test_repr_never_leaks_the_key() -> None:
    provider = AnthropicProvider(api_key="sk-ant-secret-value")
    assert "sk-ant" not in repr(provider)
    assert repr(provider) == (
        "AnthropicProvider(model='claude-opus-5-5', mode='json_schema', effort='medium')"
    )


def test_injected_client_skips_the_key_check() -> None:
    api = FakeAnthropicAPI([])
    provider = AnthropicProvider(client=api.client())
    with pytest.raises(ProviderNotConfiguredError, match="async_client"):
        provider._async()
    provider = AnthropicProvider(async_client=api.async_client())
    with pytest.raises(ProviderNotConfiguredError, match="client="):
        provider._sync_client()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"mode": "xml"}, "mode must be one of"), ({"effort": "extreme"}, "effort must be one of")],
)
def test_invalid_options(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AnthropicProvider(api_key="sk-ant-x", **kwargs)


def test_default_model() -> None:
    assert DEFAULT_MODEL == "claude-opus-5-5"


# --------------------------------------------------------------------------- #
# json_schema mode (default)
# --------------------------------------------------------------------------- #


def test_json_schema_mode_request_and_response() -> None:
    api = FakeAnthropicAPI([text_turn(GOOD_JSON, input_tokens=900, cache_read_input_tokens=700)])
    result = Extractor(provider_for(api), max_tokens=4096).extract(DOCUMENT, Ticket)

    assert result.value == Ticket.model_validate(GOOD)
    usage = result.usage
    assert (usage.input_tokens, usage.cache_read_input_tokens, usage.output_tokens) == (
        900,
        700,
        40,
    )
    assert result.model == "claude-opus-5-5"

    (request,) = api.requests
    assert request.url.endswith("/v1/messages")
    assert REFUSAL_FALLBACK_BETA in request.headers["anthropic-beta"]
    body = request.body
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 4096
    assert body["stream"] is True
    assert body["fallbacks"] == "default"
    assert "thinking" not in body  # Claude Opus 5.5 runs adaptive thinking by default
    assert "tools" not in body
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert '<json_schema name="Ticket">' in body["system"][0]["text"]
    (turn,) = body["messages"]
    assert turn["role"] == "user"
    assert f"<document>\n{DOCUMENT}\n</document>" in turn["content"]

    output_config = body["output_config"]
    assert output_config["effort"] == "medium"
    api_schema = output_config["format"]["schema"]
    assert output_config["format"]["type"] == "json_schema"
    assert api_schema["additionalProperties"] is False
    # Constraints the API cannot enforce move into descriptions; validation still enforces them.
    priority = api_schema["properties"]["priority"]
    assert "minimum" not in priority
    assert "minimum: 1" in priority["description"]


def test_semantic_repair_replays_content_blocks_verbatim() -> None:
    api = FakeAnthropicAPI(
        [
            text_turn(json.dumps({**GOOD, "priority": 9})),
            text_turn(GOOD_JSON),
        ]
    )
    result = Extractor(provider_for(api)).extract(DOCUMENT, Ticket)
    assert result.value.priority == 4
    assert isinstance(result.attempts[0].error, OutputValidationError)

    first, second = api.bodies
    # The system prompt and first turn are byte-identical: cache-friendly and append-only.
    assert second["system"] == first["system"]
    assert second["messages"][0] == first["messages"][0]
    assistant = second["messages"][1]
    assert assistant["role"] == "assistant"
    assert [block["type"] for block in assistant["content"]] == ["thinking", "text"]
    assert assistant["content"][0]["signature"] == "sig-abc"
    assert assistant["content"][1]["text"] == json.dumps({**GOOD, "priority": 9})
    feedback = second["messages"][2]
    assert feedback["role"] == "user"
    assert "`priority`: Input should be less than or equal to 5 (got: 9)" in feedback["content"]


def test_refusal_fallback_can_be_disabled() -> None:
    api = FakeAnthropicAPI([text_turn(GOOD_JSON)])
    Extractor(provider_for(api, refusal_fallback=False, prompt_caching=False, effort=None)).extract(
        DOCUMENT, Ticket
    )
    (request,) = api.requests
    assert "anthropic-beta" not in request.headers
    assert "fallbacks" not in request.body
    assert "cache_control" not in request.body["system"][0]
    assert "effort" not in request.body["output_config"]


def test_extra_body_and_model_override() -> None:
    api = FakeAnthropicAPI([text_turn(GOOD_JSON, model="claude-sonnet-5-5")])
    provider = provider_for(
        api,
        model="claude-sonnet-5-5",
        extra_body={"thinking": {"type": "adaptive", "display": "summarized"}},
    )
    result = Extractor(provider).extract(DOCUMENT, Ticket)
    assert api.bodies[0]["model"] == "claude-sonnet-5-5"
    assert api.bodies[0]["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert result.model == "claude-sonnet-5-5"


def test_schemas_the_api_cannot_express_raise_schema_error() -> None:
    class Loose(BaseModel):
        payload: Any

    api = FakeAnthropicAPI([])
    with pytest.raises(SchemaError, match="mode='prompt'"):
        Extractor(provider_for(api)).extract(DOCUMENT, Loose)
    assert api.requests == []


# --------------------------------------------------------------------------- #
# prompt mode
# --------------------------------------------------------------------------- #


def test_prompt_mode_has_no_api_constraints() -> None:
    class Loose(BaseModel):
        payload: Any

    api = FakeAnthropicAPI([text_turn('Here it is:\n```json\n{"payload": [1, "two"]}\n```')])
    result = Extractor(provider_for(api, mode="prompt")).extract(DOCUMENT, Loose)
    assert result.value.payload == [1, "two"]
    body = api.bodies[0]
    assert "format" not in body["output_config"]
    assert body["output_config"]["effort"] == "medium"
    assert "tools" not in body


# --------------------------------------------------------------------------- #
# tool mode
# --------------------------------------------------------------------------- #


def test_tool_mode_request_and_response() -> None:
    api = FakeAnthropicAPI([tool_turn(GOOD_JSON, name="record_Ticket")])
    result = Extractor(provider_for(api, mode="tool")).extract(DOCUMENT, Ticket)
    assert result.value == Ticket.model_validate(GOOD)

    body = api.bodies[0]
    (tool,) = body["tools"]
    assert tool["name"] == "record_Ticket"
    assert tool["strict"] is True
    assert tool["input_schema"]["additionalProperties"] is False
    assert "eager_input_streaming" not in tool  # only set when the caller streams
    assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert "format" not in body["output_config"]
    assert "calling the `record_Ticket` tool exactly once" in body["system"][0]["text"]


def test_tool_mode_repair_answers_the_tool_call_with_an_error_result() -> None:
    api = FakeAnthropicAPI(
        [
            tool_turn(json.dumps({**GOOD, "subject": "x" * 100}), name="record_Ticket"),
            tool_turn(GOOD_JSON, name="record_Ticket"),
        ]
    )
    result = Extractor(provider_for(api, mode="tool")).extract(DOCUMENT, Ticket)
    assert result.value.subject == "Login fails"

    second = api.bodies[1]
    assistant, tool_result_turn = second["messages"][1], second["messages"][2]
    assert [block["type"] for block in assistant["content"]] == ["thinking", "tool_use"]
    (tool_result,) = tool_result_turn["content"]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "toolu_01"
    assert tool_result["is_error"] is True
    assert "`subject`: String should have at most 80 characters" in tool_result["content"]


def test_tool_mode_missing_tool_call_is_repaired() -> None:
    api = FakeAnthropicAPI(
        [
            Turn(blocks=(Thinking(), Text("I think the priority is high."))),
            tool_turn(GOOD_JSON, name="record_Ticket"),
        ]
    )
    result = Extractor(provider_for(api, mode="tool")).extract(DOCUMENT, Ticket)
    assert result.repaired
    second = api.bodies[1]
    assert second["messages"][1]["content"][1]["text"] == "I think the priority is high."
    assert "did not contain the `Ticket` JSON value" in second["messages"][2]["content"]


def test_tool_mode_streams_partial_tool_input() -> None:
    api = FakeAnthropicAPI([tool_turn(GOOD_JSON, name="record_Ticket", chunk_size=6)])
    events = list(Extractor(provider_for(api, mode="tool")).stream(DOCUMENT, Ticket))
    partials = [event.data for event in events if isinstance(event, PartialEvent)]
    assert partials[0] == {}
    assert partials[-1] == GOOD
    assert isinstance(events[-1], ResultEvent)
    assert api.bodies[0]["tools"][0]["eager_input_streaming"] is True


def test_tool_mode_rejects_non_object_roots() -> None:
    api = FakeAnthropicAPI([])
    with pytest.raises(SchemaError, match="object at the root"):
        Extractor(provider_for(api, mode="tool")).extract(DOCUMENT, list[Ticket])


def test_tool_names_are_sanitised() -> None:
    api = FakeAnthropicAPI([tool_turn("[]", name="record_list_Ticket")])
    provider = provider_for(api, mode="tool")
    request = CompletionRequest(
        system="s",
        messages=(Message(role="user", content="u"),),
        schema_name="list[Ticket]",
        json_schema={"type": "object", "properties": {}},
    )
    provider.complete(request)
    assert api.bodies[0]["tools"][0]["name"] == "record_list_Ticket"


@pytest.mark.parametrize(
    ("raw_input", "message", "raw_kept"),
    [
        # The SDK alone would silently turn these into {"subject": "Login fails"} and {}.
        ('{"subject": "Login fails" "priority": 4}', "Expected ','", True),
        ('{"subject": "Login fa', "incomplete", True),
        # The SDK cannot parse this at all and raises on the first fragment, before
        # any text reaches us; that still becomes a repairable error.
        ("not json at all", "tool input was not valid JSON", False),
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_malformed_tool_input_is_reported_and_repaired(
    raw_input: str, message: str, raw_kept: bool, streaming: bool
) -> None:
    api = FakeAnthropicAPI(
        [tool_turn(raw_input, name="record_Ticket"), tool_turn(GOOD_JSON, name="record_Ticket")]
    )
    extractor = Extractor(provider_for(api, mode="tool"))
    if streaming:
        final = list(extractor.stream(DOCUMENT, Ticket))[-1]
        assert isinstance(final, ResultEvent)
        result = final.result
    else:
        result = extractor.extract(DOCUMENT, Ticket)
    assert result.value == Ticket.model_validate(GOOD)
    error = result.attempts[0].error
    assert isinstance(error, OutputParseError)
    assert message in str(error)
    assert error.raw_text == (raw_input if raw_kept else "")


def test_value_errors_outside_tool_mode_are_not_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeAnthropicAPI([text_turn(GOOD_JSON)])

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("a genuine bug")

    monkeypatch.setattr(BetaMessageStream, "get_final_message", explode)
    with pytest.raises(ValueError, match="a genuine bug"):
        Extractor(provider_for(api)).extract(DOCUMENT, Ticket)


# --------------------------------------------------------------------------- #
# Stop reasons and failures
# --------------------------------------------------------------------------- #


def test_refusal_raises_with_category() -> None:
    api = FakeAnthropicAPI(
        [
            Turn(
                blocks=(),
                stop_reason="refusal",
                stop_details={"type": "refusal", "category": "cyber", "explanation": "Declined."},
            )
        ]
    )
    with pytest.raises(RefusalError, match="category: cyber") as info:
        Extractor(provider_for(api)).extract(DOCUMENT, Ticket)
    assert info.value.category == "cyber"
    assert info.value.explanation == "Declined."
    assert len(api.requests) == 1  # never retried


def test_max_tokens_is_reported_as_truncation() -> None:
    api = FakeAnthropicAPI(
        [text_turn(GOOD_JSON[:20], stop_reason="max_tokens"), text_turn(GOOD_JSON)]
    )
    result = Extractor(provider_for(api)).extract(DOCUMENT, Ticket)
    assert isinstance(result.attempts[0].error, OutputTruncatedError)
    assert result.attempts[0].stop_reason == "max_tokens"


@pytest.mark.parametrize(
    ("failure", "status", "retryable"),
    [
        (HTTPError(429, "rate_limit_error", "slow down"), 429, True),
        (HTTPError(529, "overloaded_error", "overloaded"), 529, True),
        (HTTPError(400, "invalid_request_error", "bad schema"), 400, False),
        (HTTPError(401, "authentication_error", "invalid x-api-key"), 401, False),
    ],
)
def test_http_errors_become_provider_errors(
    failure: HTTPError, status: int, retryable: bool
) -> None:
    api = FakeAnthropicAPI([failure])
    with pytest.raises(ProviderError) as info:
        Extractor(provider_for(api)).extract(DOCUMENT, Ticket)
    error = info.value
    assert error.status_code == status
    assert error.retryable is retryable
    assert error.provider == "anthropic"
    assert failure.message in str(error)
    assert isinstance(error.__cause__, anthropic.APIStatusError)


def test_connection_errors_are_retryable_provider_errors() -> None:
    api = FakeAnthropicAPI([ConnectionFailure()])
    with pytest.raises(ProviderError, match="Could not reach the Anthropic API") as info:
        Extractor(provider_for(api)).extract(DOCUMENT, Ticket)
    assert info.value.retryable
    assert info.value.status_code is None


def test_exhaustion_through_the_real_sdk() -> None:
    api = FakeAnthropicAPI([text_turn('{"subject": "a"}')] * 2)
    with pytest.raises(ExtractionFailedError):
        Extractor(provider_for(api), max_retries=1).extract(DOCUMENT, Ticket)
    assert len(api.requests) == 2


# --------------------------------------------------------------------------- #
# Streaming and async
# --------------------------------------------------------------------------- #


def test_sync_stream_json_mode() -> None:
    api = FakeAnthropicAPI([text_turn(GOOD_JSON, chunk_size=4)])
    events = list(Extractor(provider_for(api)).stream(DOCUMENT, Ticket))
    partials = [event for event in events if isinstance(event, PartialEvent)]
    assert len(partials) > 3
    assert partials[-1].data == GOOD
    assert isinstance(events[-1], ResultEvent)


def test_async_extract_and_stream() -> None:
    api = FakeAnthropicAPI(
        [
            text_turn(json.dumps({**GOOD, "priority": 0})),
            text_turn(GOOD_JSON),
            tool_turn(GOOD_JSON, name="record_Ticket"),
        ]
    )
    provider = provider_for(api)
    tool_provider = provider_for(api, mode="tool")

    async def run() -> tuple[Ticket, list[Any]]:
        result = await Extractor(provider).aextract(DOCUMENT, Ticket)
        events = [event async for event in Extractor(tool_provider).astream(DOCUMENT, Ticket)]
        return result.value, events

    value, events = asyncio.run(run())
    assert value == Ticket.model_validate(GOOD)
    assert isinstance(events[-1], ResultEvent)
    assert events[-1].result.value == Ticket.model_validate(GOOD)
    assert len(api.requests) == 3
    assert api.bodies[2]["tools"][0]["eager_input_streaming"] is True


def test_async_errors_are_wrapped() -> None:
    api = FakeAnthropicAPI([HTTPError(500), ConnectionFailure(), HTTPError(503)])
    provider = provider_for(api)

    async def run() -> list[ProviderError]:
        errors = []
        for _ in range(2):
            with pytest.raises(ProviderError) as info:
                await Extractor(provider).aextract(DOCUMENT, Ticket)
            errors.append(info.value)
        with pytest.raises(ProviderError) as info:
            async for _event in Extractor(provider).astream(DOCUMENT, Ticket):
                pass
        errors.append(info.value)
        return errors

    server, connection, streamed = asyncio.run(run())
    assert (server.status_code, server.retryable) == (500, True)
    assert (connection.status_code, connection.retryable) == (None, True)
    assert (streamed.status_code, streamed.retryable) == (503, True)


def test_sync_stream_errors_are_wrapped() -> None:
    api = FakeAnthropicAPI([HTTPError(400, "invalid_request_error", "nope")])
    with pytest.raises(ProviderError, match="HTTP 400"):
        list(Extractor(provider_for(api)).stream(DOCUMENT, Ticket))
