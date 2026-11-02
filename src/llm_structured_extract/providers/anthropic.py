"""Anthropic (Claude) provider built on the official ``anthropic`` SDK.

Install with ``pip install 'llm-structured-extract[anthropic]'`` and set
``ANTHROPIC_API_KEY``. The key is read from the environment (or passed
explicitly); it is never stored in code or logged.

Three output modes are supported:

``"json_schema"`` (default)
    Native structured outputs (``output_config.format``). Claude's decoding is
    constrained to the schema, so syntax errors disappear; the repair loop then
    only has to deal with *semantic* violations (cross-field rules, numeric
    ranges and string lengths that the API does not enforce, ...).
``"tool"``
    Strict tool use: the schema becomes the input schema of a single
    ``record_<Schema>`` tool. When streaming, the tool input is streamed eagerly
    and parsed incrementally; the extractor validates the final input.
``"prompt"``
    Plain instructions with no API-level constraint. Works with any model or
    gateway; relies entirely on tolerant parsing plus repair.

Every call streams under the hood (``get_final_message()`` for non-streaming
use), which avoids HTTP timeouts on long outputs. By default the request opts
into server-side refusal fallback (``fallbacks="default"``), so a request
declined by a safety classifier is retried server-side on a fallback model;
pass ``refusal_fallback=False`` for gateways that reject beta parameters.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from ..errors import (
    OutputParseError,
    ProviderError,
    ProviderNotConfiguredError,
    RefusalError,
    SchemaError,
)
from ..types import STOP_REFUSAL, Completion, CompletionRequest, StreamChunk, Usage
from .base import BaseProvider

try:
    import anthropic
    from anthropic.lib.streaming import BetaInputJsonEvent, BetaTextEvent
    from anthropic.types.beta import BetaTextBlock, BetaToolUseBlock

    _IMPORT_ERROR: ImportError | None = None
except ImportError as _error:  # pragma: no cover - exercised only without the extra
    _IMPORT_ERROR = _error

if TYPE_CHECKING:
    from anthropic.lib.streaming import BetaAsyncMessageStreamManager, BetaMessageStreamManager
    from anthropic.types.beta import (
        BetaContentBlock,
        BetaContentBlockParam,
        BetaMessage,
        BetaMessageParam,
        BetaOutputConfigParam,
        BetaTextBlockParam,
        BetaToolChoiceAutoParam,
        BetaToolParam,
        BetaToolResultBlockParam,
    )
    from anthropic.types.beta.beta_tool_param import InputSchema

__all__ = [
    "API_KEY_ENV",
    "DEFAULT_MODEL",
    "REFUSAL_FALLBACK_BETA",
    "AnthropicProvider",
    "Effort",
    "Mode",
]

DEFAULT_MODEL = "claude-opus-5-5"
API_KEY_ENV = "ANTHROPIC_API_KEY"
#: Beta header for the ``fallbacks="default"`` form of server-side refusal fallback.
REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"

Mode = Literal["json_schema", "tool", "prompt"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]

_MODES: frozenset[str] = frozenset({"json_schema", "tool", "prompt"})
_EFFORTS: frozenset[str] = frozenset({"low", "medium", "high", "xhigh", "max"})
_RETRYABLE_STATUS: frozenset[int] = frozenset({408, 409, 429})
_TOOL_NAME_UNSAFE = re.compile(r"[^a-zA-Z0-9_-]+")


@dataclass(frozen=True, slots=True)
class _Replay:
    """Native payload attached to completions: the exact content blocks Claude returned.

    Replaying them unchanged during a repair round keeps the conversation
    append-only (thinking blocks included), which preserves prompt-cache hits
    and satisfies Claude's preserved-thinking checks.
    """

    content: tuple[BetaContentBlock, ...]
    tool_use_ids: tuple[str, ...]


def _tool_name(schema_name: str) -> str:
    slug = _TOOL_NAME_UNSAFE.sub("_", schema_name).strip("_") or "output"
    return f"record_{slug}"[:64]


class AnthropicProvider(BaseProvider):
    """Claude via the official Anthropic SDK.

    Args:
        model: Model ID. Defaults to ``claude-opus-5-5``.
        mode: ``"json_schema"`` (default), ``"tool"`` or ``"prompt"``; see the
            module documentation.
        effort: Thinking/effort level sent as ``output_config.effort``. ``None``
            leaves the API default. Extraction rarely needs more than ``"medium"``.
        api_key: Explicit key. Defaults to the ``ANTHROPIC_API_KEY`` environment
            variable. Never hardcode keys in source.
        client / async_client: Pre-configured SDK clients (custom base URL, proxy,
            profile or workload-identity auth...). Injecting one skips the
            ``ANTHROPIC_API_KEY`` check.
        refusal_fallback: Opt into server-side refusal fallback (beta).
        prompt_caching: Mark the (schema-bearing, attempt-invariant) system prompt
            as cacheable. Prompts below the model's minimum cacheable length are
            simply not cached.
        timeout / max_retries: Forwarded to clients this provider creates. The
            SDK already retries connection errors, 408/409/429 and 5xx.
        extra_body: Extra request fields merged into every call, e.g.
            ``{"thinking": {"type": "adaptive", "display": "summarized"}}``.

    Raises:
        ProviderNotConfiguredError: the SDK is not installed, or no client was
            injected and no API key is available.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        mode: Mode = "json_schema",
        effort: Effort | None = "medium",
        api_key: str | None = None,
        client: anthropic.Anthropic | None = None,
        async_client: anthropic.AsyncAnthropic | None = None,
        refusal_fallback: bool = True,
        prompt_caching: bool = True,
        timeout: float | None = None,
        max_retries: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise ProviderNotConfiguredError(
                "AnthropicProvider requires the 'anthropic' package: "
                "pip install 'llm-structured-extract[anthropic]'",
                provider=self.name,
            ) from _IMPORT_ERROR
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}, got {mode!r}")
        if effort is not None and effort not in _EFFORTS:
            raise ValueError(f"effort must be one of {sorted(_EFFORTS)} or None, got {effort!r}")

        self._api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        if client is None and async_client is None and not self._api_key:
            raise ProviderNotConfiguredError(
                f"AnthropicProvider needs credentials: set the {API_KEY_ENV} environment "
                "variable, pass api_key=..., or inject a configured anthropic client.",
                provider=self.name,
            )
        self.model = model
        self.mode: Mode = mode
        self.effort: Effort | None = effort
        self.refusal_fallback = refusal_fallback
        self.prompt_caching = prompt_caching
        self.extra_body = dict(extra_body) if extra_body else None
        self._timeout = timeout
        self._max_retries = max_retries
        self._client = client
        self._async_client = async_client
        self._schema_cache: dict[str, dict[str, Any]] = {}

    def __repr__(self) -> str:  # never include the API key
        return (
            f"AnthropicProvider(model={self.model!r}, mode={self.mode!r}, effort={self.effort!r})"
        )

    # -- clients ----------------------------------------------------------- #

    def _client_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"api_key": self._api_key}
        if self._timeout is not None:
            options["timeout"] = self._timeout
        if self._max_retries is not None:
            options["max_retries"] = self._max_retries
        return options

    def _sync_client(self) -> anthropic.Anthropic:
        if self._client is None:
            if not self._api_key:
                raise ProviderNotConfiguredError(
                    "Only an async client was injected; pass client=... for synchronous calls.",
                    provider=self.name,
                )
            self._client = anthropic.Anthropic(**self._client_options())
        return self._client

    def _async(self) -> anthropic.AsyncAnthropic:
        if self._async_client is None:
            if not self._api_key:
                raise ProviderNotConfiguredError(
                    "Only a sync client was injected; pass async_client=... for async calls.",
                    provider=self.name,
                )
            self._async_client = anthropic.AsyncAnthropic(**self._client_options())
        return self._async_client

    # -- request building -------------------------------------------------- #

    def _api_schema(self, request: CompletionRequest) -> dict[str, Any]:
        """The request schema rewritten into the subset the API's constrained decoding accepts.

        Unsupported keywords (numeric ranges, string lengths, ...) are moved into
        descriptions by :func:`anthropic.transform_schema`; they are still
        enforced client-side by the extractor's validation step.
        """
        key = json.dumps(request.json_schema, sort_keys=True, default=str)
        cached = self._schema_cache.get(key)
        if cached is None:
            try:
                cached = anthropic.transform_schema(dict(request.json_schema))
            except (ValueError, TypeError) as error:
                raise SchemaError(
                    f"The `{request.schema_name}` schema cannot be used with "
                    f"mode={self.mode!r}: {error}. Use mode='prompt' instead."
                ) from error
            self._schema_cache[key] = cached
        return cached

    def _system(
        self, request: CompletionRequest, tool_name: str | None
    ) -> list[BetaTextBlockParam]:
        text = request.system
        if tool_name is not None:
            text += (
                f"\n\nDeliver your answer by calling the `{tool_name}` tool exactly once; its "
                f"input is the complete `{request.schema_name}` JSON value. Do not write the "
                "JSON as text."
            )
        block: BetaTextBlockParam = {"type": "text", "text": text}
        if self.prompt_caching:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def _messages(self, request: CompletionRequest) -> list[BetaMessageParam]:
        """Convert provider-neutral turns, replaying Claude's own content blocks verbatim."""
        messages: list[BetaMessageParam] = []
        open_tool_calls: tuple[str, ...] = ()
        for message in request.messages:
            if message.role == "assistant":
                replay = message.native if isinstance(message.native, _Replay) else None
                if replay is not None:
                    content: list[BetaContentBlockParam] = list(replay.content)
                    messages.append({"role": "assistant", "content": content})
                    open_tool_calls = replay.tool_use_ids
                else:
                    messages.append({"role": "assistant", "content": message.content})
                    open_tool_calls = ()
            elif open_tool_calls:
                # A tool call must be answered by tool_result blocks: the repair
                # feedback becomes the (error) result of the rejected call.
                results: list[BetaContentBlockParam] = []
                for index, tool_use_id in enumerate(open_tool_calls):
                    result: BetaToolResultBlockParam = {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "is_error": True,
                        "content": message.content
                        if index == 0
                        else "Ignored: call the tool once.",
                    }
                    results.append(result)
                messages.append({"role": "user", "content": results})
                open_tool_calls = ()
            else:
                messages.append({"role": "user", "content": message.content})
        return messages

    def _output_config(self, request: CompletionRequest) -> BetaOutputConfigParam | None:
        config: BetaOutputConfigParam = {}
        if self.effort is not None:
            config["effort"] = self.effort
        if self.mode == "json_schema":
            config["format"] = {"type": "json_schema", "schema": self._api_schema(request)}
        return config or None

    def _tools(
        self, request: CompletionRequest, tool_name: str | None, *, eager: bool
    ) -> list[BetaToolParam] | None:
        if tool_name is None:
            return None
        schema = self._api_schema(request)
        if schema.get("type") != "object":
            raise SchemaError(
                f"mode='tool' needs an object at the root of the `{request.schema_name}` "
                "schema; use mode='json_schema' or mode='prompt' for other roots."
            )
        tool: BetaToolParam = {
            "name": tool_name,
            "description": (
                f"Record the `{request.schema_name}` extracted from the document. "
                "Call this exactly once, with the complete extracted value."
            ),
            "input_schema": cast("InputSchema", schema),
            "strict": True,
        }
        if eager:
            tool["eager_input_streaming"] = True
        return [tool]

    def _tool_name_for(self, request: CompletionRequest) -> str | None:
        return _tool_name(request.schema_name) if self.mode == "tool" else None

    def _open(self, request: CompletionRequest, *, eager: bool) -> BetaMessageStreamManager:
        tool_name = self._tool_name_for(request)
        tools = self._tools(request, tool_name, eager=eager)
        output_config = self._output_config(request)
        tool_choice: BetaToolChoiceAutoParam = {"type": "auto", "disable_parallel_tool_use": True}
        omit = anthropic.omit
        return self._sync_client().beta.messages.stream(
            model=self.model,
            max_tokens=request.max_tokens,
            system=self._system(request, tool_name),
            messages=self._messages(request),
            output_config=output_config if output_config is not None else omit,
            tools=tools if tools is not None else omit,
            tool_choice=tool_choice if tools is not None else omit,
            betas=[REFUSAL_FALLBACK_BETA] if self.refusal_fallback else omit,
            fallbacks="default" if self.refusal_fallback else omit,
            extra_body=self.extra_body,
        )

    def _aopen(self, request: CompletionRequest, *, eager: bool) -> BetaAsyncMessageStreamManager:
        tool_name = self._tool_name_for(request)
        tools = self._tools(request, tool_name, eager=eager)
        output_config = self._output_config(request)
        tool_choice: BetaToolChoiceAutoParam = {"type": "auto", "disable_parallel_tool_use": True}
        omit = anthropic.omit
        return self._async().beta.messages.stream(
            model=self.model,
            max_tokens=request.max_tokens,
            system=self._system(request, tool_name),
            messages=self._messages(request),
            output_config=output_config if output_config is not None else omit,
            tools=tools if tools is not None else omit,
            tool_choice=tool_choice if tools is not None else omit,
            betas=[REFUSAL_FALLBACK_BETA] if self.refusal_fallback else omit,
            fallbacks="default" if self.refusal_fallback else omit,
            extra_body=self.extra_body,
        )

    # -- response handling ------------------------------------------------- #

    def _completion(
        self, message: BetaMessage, tool_name: str | None, raw_tool_input: str = ""
    ) -> Completion:
        if message.stop_reason == STOP_REFUSAL:
            details = message.stop_details
            category = details.category if details is not None else None
            explanation = details.explanation if details is not None else None
            suffix = f" (category: {category})" if category else ""
            raise RefusalError(
                f"Claude declined the request{suffix}.",
                provider=self.name,
                category=category,
                explanation=explanation,
            )
        tool_calls = [block for block in message.content if isinstance(block, BetaToolUseBlock)]
        if tool_name is not None:
            # Prefer the raw JSON the model streamed: the SDK's tolerant parser can
            # silently truncate malformed input (dropping fields), whereas our own
            # parser reports the defect so it can be repaired.
            call = next((block for block in tool_calls if block.name == tool_name), None)
            if call is None:
                text = ""
            elif raw_tool_input.strip():
                text = raw_tool_input
            else:
                text = json.dumps(call.input, ensure_ascii=False)
        else:
            text = "".join(
                block.text for block in message.content if isinstance(block, BetaTextBlock)
            )
        usage = message.usage
        return Completion(
            text=text,
            usage=Usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens or 0,
                cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            ),
            stop_reason=message.stop_reason,
            model=message.model,
            native=_Replay(
                content=tuple(message.content),
                tool_use_ids=tuple(block.id for block in tool_calls),
            ),
        )

    @staticmethod
    def _delta(event: object, tool_name: str | None) -> str:
        if tool_name is None and isinstance(event, BetaTextEvent):
            return event.text
        if tool_name is not None and isinstance(event, BetaInputJsonEvent):
            return event.partial_json
        return ""

    def _wrap_sdk_error(self, error: anthropic.AnthropicError) -> ProviderError:
        if isinstance(error, anthropic.APIStatusError):
            status = error.status_code
            return ProviderError(
                f"Anthropic API returned HTTP {status}: {error.message}",
                provider=self.name,
                status_code=status,
                retryable=status in _RETRYABLE_STATUS or status >= 500,
            )
        if isinstance(error, anthropic.APIConnectionError):  # includes timeouts
            return ProviderError(
                f"Could not reach the Anthropic API: {error}", provider=self.name, retryable=True
            )
        return ProviderError(f"Anthropic SDK error: {error}", provider=self.name)

    @staticmethod
    def _tool_json_error(error: ValueError, tool_name: str, raw: list[str]) -> OutputParseError:
        return OutputParseError(
            f"The `{tool_name}` tool input was not valid JSON: {error}", raw_text="".join(raw)
        )

    # -- Provider API ------------------------------------------------------ #

    def complete(self, request: CompletionRequest) -> Completion:
        tool_name = self._tool_name_for(request)
        raw: list[str] = []
        try:
            with self._open(request, eager=False) as stream:
                for event in stream:
                    raw.append(self._delta(event, tool_name))
                message = stream.get_final_message()
        except anthropic.AnthropicError as error:
            raise self._wrap_sdk_error(error) from error
        except ValueError as error:  # the SDK could not parse the tool input at all
            if tool_name is None:
                raise
            raise self._tool_json_error(error, tool_name, raw) from error
        return self._completion(message, tool_name, "".join(raw))

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        tool_name = self._tool_name_for(request)
        raw: list[str] = []
        try:
            with self._open(request, eager=tool_name is not None) as stream:
                for event in stream:
                    delta = self._delta(event, tool_name)
                    if delta:
                        raw.append(delta)
                        yield StreamChunk(text=delta)
                message = stream.get_final_message()
        except anthropic.AnthropicError as error:
            raise self._wrap_sdk_error(error) from error
        except ValueError as error:
            if tool_name is None:
                raise
            raise self._tool_json_error(error, tool_name, raw) from error
        yield StreamChunk(completion=self._completion(message, tool_name, "".join(raw)))

    async def acomplete(self, request: CompletionRequest) -> Completion:
        tool_name = self._tool_name_for(request)
        raw: list[str] = []
        try:
            async with self._aopen(request, eager=False) as stream:
                async for event in stream:
                    raw.append(self._delta(event, tool_name))
                message = await stream.get_final_message()
        except anthropic.AnthropicError as error:
            raise self._wrap_sdk_error(error) from error
        except ValueError as error:
            if tool_name is None:
                raise
            raise self._tool_json_error(error, tool_name, raw) from error
        return self._completion(message, tool_name, "".join(raw))

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        tool_name = self._tool_name_for(request)
        raw: list[str] = []
        try:
            async with self._aopen(request, eager=tool_name is not None) as stream:
                async for event in stream:
                    delta = self._delta(event, tool_name)
                    if delta:
                        raw.append(delta)
                        yield StreamChunk(text=delta)
                message = await stream.get_final_message()
        except anthropic.AnthropicError as error:
            raise self._wrap_sdk_error(error) from error
        except ValueError as error:
            if tool_name is None:
                raise
            raise self._tool_json_error(error, tool_name, raw) from error
        yield StreamChunk(completion=self._completion(message, tool_name, "".join(raw)))
