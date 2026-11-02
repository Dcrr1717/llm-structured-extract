"""Shared fixtures.

Two guarantees for every test that is not marked ``live``:

* no real network traffic: outbound socket connections and DNS lookups to
  non-loopback hosts raise immediately, so a forgotten mock fails loudly
  instead of silently calling a paid API;
* no credentials: ``ANTHROPIC_API_KEY`` is removed from the environment.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest

_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


class NetworkAccessError(RuntimeError):
    pass


def _host_of(address: Any) -> str:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return str(address)


@pytest.fixture(autouse=True)
def _offline(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    if request.node.get_closest_marker("live") is not None:
        yield
        return

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def guard(sock: socket.socket, address: Any) -> None:
        if sock.family in (socket.AF_INET, socket.AF_INET6) and _host_of(address) not in _LOOPBACK:
            raise NetworkAccessError(f"network access is disabled in tests (tried {address!r})")

    def connect(sock: socket.socket, address: Any) -> None:
        guard(sock, address)
        real_connect(sock, address)

    def connect_ex(sock: socket.socket, address: Any) -> int:
        guard(sock, address)
        return real_connect_ex(sock, address)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host is not None and str(host) not in _LOOPBACK:
            raise NetworkAccessError(f"DNS lookups are disabled in tests (tried {host!r})")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    yield
