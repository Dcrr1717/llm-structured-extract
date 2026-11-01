"""The provider abstraction.

:class:`Provider` is the structural contract (a :class:`typing.Protocol`): any
object with these methods works, no inheritance required. :class:`BaseProvider`
is a convenience ABC: implement :meth:`~BaseProvider.complete` and you get
working (if non-incremental) streaming and async support for free; override the
other methods when your backend can do better.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from typing import Protocol, runtime_checkable

from ..types import Completion, CompletionRequest, StreamChunk

__all__ = ["BaseProvider", "Provider"]


@runtime_checkable
class Provider(Protocol):
    """What the extractor needs from an LLM backend.

    Contract:

    * Return a :class:`~llm_structured_extract.types.Completion` with the raw
      output text. Do **not** validate it - the extractor does that.
    * Report truncation via ``stop_reason="max_tokens"`` so the extractor can
      tell a cut-off answer from a malformed one.
    * Raise :class:`~llm_structured_extract.errors.ProviderError` for failed
      calls and :class:`~llm_structured_extract.errors.RefusalError` for refusals.
    * Streams must end with a chunk carrying the final ``Completion``.
    """

    @property
    def name(self) -> str: ...

    def complete(self, request: CompletionRequest) -> Completion: ...

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]: ...

    async def acomplete(self, request: CompletionRequest) -> Completion: ...

    def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]: ...


class BaseProvider(ABC):
    """Optional base class implementing everything in terms of :meth:`complete`."""

    name: str = "provider"

    @abstractmethod
    def complete(self, request: CompletionRequest) -> Completion:
        """Produce one completion for ``request``."""

    def stream(self, request: CompletionRequest) -> Iterator[StreamChunk]:
        """Default: a single chunk containing the whole completion."""
        completion = self.complete(request)
        yield StreamChunk(text=completion.text, completion=completion)

    async def acomplete(self, request: CompletionRequest) -> Completion:
        """Default: run :meth:`complete` in a worker thread."""
        return await asyncio.to_thread(self.complete, request)

    async def astream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Default: a single chunk containing the whole (async) completion."""
        completion = await self.acomplete(request)
        yield StreamChunk(text=completion.text, completion=completion)
