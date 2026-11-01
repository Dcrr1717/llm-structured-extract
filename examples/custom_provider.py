"""Bring your own provider: a caching wrapper around any other provider.

Run offline::

    python examples/custom_provider.py

A provider is anything that implements the small ``Provider`` protocol. Subclass
``BaseProvider`` and implement ``complete()`` to get streaming and async support
for free. This one memoises completions, so re-running an extraction over the
same document (in a notebook, a test suite, a re-tried batch job) costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import sys

from pydantic import BaseModel

from llm_structured_extract import (
    BaseProvider,
    Completion,
    CompletionRequest,
    Extractor,
    MockProvider,
    Provider,
)


class CachingProvider(BaseProvider):
    """Memoises completions of an inner provider, keyed by the full request."""

    name = "caching"

    def __init__(self, inner: Provider) -> None:
        self.inner = inner
        self.cache: dict[str, Completion] = {}
        self.hits = 0

    @staticmethod
    def key(request: CompletionRequest) -> str:
        payload = {
            "system": request.system,
            "messages": [[m.role, m.content] for m in request.messages],
            "schema": request.json_schema,
            "max_tokens": request.max_tokens,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def complete(self, request: CompletionRequest) -> Completion:
        key = self.key(request)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        completion = self.inner.complete(request)
        self.cache[key] = completion
        return completion


class Contact(BaseModel):
    name: str
    email: str
    company: str | None = None


def main() -> int:
    inner = MockProvider([{"name": "Grace Hopper", "email": "grace@navy.example", "company": None}])
    provider = CachingProvider(inner)
    extractor = Extractor(provider)
    text = "Reach Grace Hopper at grace@navy.example."

    first = extractor.extract(text, Contact).value
    second = extractor.extract(text, Contact).value  # served from the cache
    print(first)
    print(f"equal: {first == second}, model calls: {inner.calls}, cache hits: {provider.hits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
