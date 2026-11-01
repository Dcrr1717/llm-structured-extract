"""Tolerant JSON parsing for LLM output - complete and partial (streaming).

Models are good at JSON but not perfect. Typical defects, all handled here:

* prose or markdown fences around the value (``Here you go: ```json {...} ``` ``)
* trailing commas, ``//`` and ``/* */`` comments
* single-quoted strings and Python literals (``True`` / ``False`` / ``None``)
* raw control characters (literal newlines) inside strings

:func:`parse_json` returns a complete value or raises
:class:`~llm_structured_extract.errors.OutputParseError` with line/column
information. :func:`parse_partial_json` accepts a *prefix* of a JSON document -
what you have mid-stream - and returns the best-effort value seen so far, closing
any open strings, arrays and objects and dropping dangling keys.

Strict, valid JSON always takes the fast path through :func:`json.loads`; the
lenient parser only runs when that fails.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Final, NoReturn

from .errors import OutputParseError

__all__ = ["PartialJSON", "parse_json", "parse_partial_json"]

_MISSING: Final = object()

_NUMBER_TOKEN = re.compile(r"[-+0-9.eE]+")
_STRICT_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?")
_LITERALS: Final[dict[str, object]] = {
    "true": True,
    "false": False,
    "null": None,
    # Python-style literals, a common slip when models "think in Python".
    "True": True,
    "False": False,
    "None": None,
}
_ESCAPES: Final[dict[str, str]] = {
    '"': '"',
    "'": "'",
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
_FENCE = re.compile(r"```[ \t]*(?:json|JSON|json5)?[ \t]*\r?\n?(.*?)```", re.DOTALL)
_MAX_CANDIDATE_STARTS: Final = 32


@dataclass(frozen=True, slots=True)
class PartialJSON:
    """Result of :func:`parse_partial_json`.

    ``complete`` is ``True`` once the top-level value has been closed.
    """

    value: Any
    complete: bool


class _SyntaxError(Exception):
    def __init__(self, message: str, position: int) -> None:
        super().__init__(message)
        self.message = message
        self.position = position


class _Parser:
    """A small recursive-descent JSON parser that is lenient and EOF-tolerant.

    Every ``_parse_*`` method returns ``(value, done)``. ``done`` is ``False``
    when the input ended inside the value; ``value`` is then the best partial
    value or ``_MISSING`` when nothing usable has been read yet.
    """

    def __init__(self, text: str, start: int = 0) -> None:
        self.text = text
        self.pos = start
        self.length = len(text)

    # -- helpers --------------------------------------------------------- #

    def fail(self, message: str) -> NoReturn:
        raise _SyntaxError(message, self.pos)

    def at_end(self) -> bool:
        return self.pos >= self.length

    def skip_whitespace(self) -> None:
        text, length = self.text, self.length
        while self.pos < length:
            char = text[self.pos]
            if char in " \t\r\n﻿":
                self.pos += 1
            elif text.startswith("//", self.pos):
                newline = text.find("\n", self.pos)
                self.pos = length if newline == -1 else newline + 1
            elif text.startswith("/*", self.pos):
                close = text.find("*/", self.pos + 2)
                self.pos = length if close == -1 else close + 2
            else:
                return

    # -- grammar --------------------------------------------------------- #

    def parse_value(self) -> tuple[Any, bool]:
        self.skip_whitespace()
        if self.at_end():
            return _MISSING, False
        char = self.text[self.pos]
        if char == "{":
            return self.parse_object()
        if char == "[":
            return self.parse_array()
        if char in "\"'":
            return self.parse_string()
        if char == "-" or char.isdigit():
            return self.parse_number()
        if char.isalpha():
            return self.parse_literal()
        self.fail(f"Unexpected character {char!r}")

    def parse_object(self) -> tuple[Any, bool]:
        self.pos += 1  # consume "{"
        result: dict[str, Any] = {}
        while True:
            self.skip_whitespace()
            if self.at_end():
                return result, False
            char = self.text[self.pos]
            if char == "}":
                self.pos += 1
                return result, True
            if char not in "\"'":
                self.fail(f"Expected a quoted property name, found {char!r}")
            key, key_done = self.parse_string()
            if not key_done:
                return result, False  # dangling, half-written key: drop it
            self.skip_whitespace()
            if self.at_end():
                return result, False
            if self.text[self.pos] != ":":
                self.fail(f"Expected ':' after property name {key!r}")
            self.pos += 1
            value, value_done = self.parse_value()
            if value is not _MISSING:
                result[key] = value
            if not value_done:
                return result, False
            self.skip_whitespace()
            if self.at_end():
                return result, False
            char = self.text[self.pos]
            if char == ",":
                self.pos += 1  # a trailing comma before "}" is tolerated by the loop
            elif char == "}":
                self.pos += 1
                return result, True
            else:
                self.fail(f"Expected ',' or '}}' after the value of {key!r}, found {char!r}")

    def parse_array(self) -> tuple[Any, bool]:
        self.pos += 1  # consume "["
        result: list[Any] = []
        while True:
            self.skip_whitespace()
            if self.at_end():
                return result, False
            if self.text[self.pos] == "]":
                self.pos += 1
                return result, True
            value, value_done = self.parse_value()
            if value is not _MISSING:
                result.append(value)
            if not value_done:
                return result, False
            self.skip_whitespace()
            if self.at_end():
                return result, False
            char = self.text[self.pos]
            if char == ",":
                self.pos += 1
            elif char == "]":
                self.pos += 1
                return result, True
            else:
                self.fail(f"Expected ',' or ']' in array, found {char!r}")

    def parse_string(self) -> tuple[str, bool]:
        quote = self.text[self.pos]
        self.pos += 1
        text, length = self.text, self.length
        chunks: list[str] = []
        run_start = self.pos
        while self.pos < length:
            char = text[self.pos]
            if char == quote:
                chunks.append(text[run_start : self.pos])
                self.pos += 1
                return "".join(chunks), True
            if char != "\\":
                self.pos += 1
                continue
            chunks.append(text[run_start : self.pos])
            if self.pos + 1 >= length:  # lone backslash at EOF: wait for more input
                self.pos = length
                return "".join(chunks), False
            escape = text[self.pos + 1]
            if escape == "u":
                decoded, consumed = self._decode_unicode_escape(self.pos)
                if decoded is None:  # truncated \u escape at EOF
                    self.pos = length
                    return "".join(chunks), False
                chunks.append(decoded)
                self.pos += consumed
            else:
                # Unknown escapes (e.g. a Windows path "C:\Users") are kept verbatim.
                chunks.append(_ESCAPES.get(escape, "\\" + escape))
                self.pos += 2
            run_start = self.pos
        chunks.append(text[run_start:])
        return "".join(chunks), False

    def _decode_unicode_escape(self, start: int) -> tuple[str | None, int]:
        """Decode ``\\uXXXX`` (and a following low surrogate) at ``start``."""
        hex_digits = self.text[start + 2 : start + 6]
        if len(hex_digits) < 4:
            return None, 0
        try:
            code = int(hex_digits, 16)
        except ValueError:
            self.pos = start
            self.fail("Invalid \\u escape in string")
        if 0xD800 <= code <= 0xDBFF and self.text.startswith("\\u", start + 6):
            low_digits = self.text[start + 8 : start + 12]
            if len(low_digits) < 4:
                return None, 0
            try:
                low = int(low_digits, 16)
            except ValueError:
                low = -1
            if 0xDC00 <= low <= 0xDFFF:
                combined = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                return chr(combined), 12
        return chr(code), 6

    def parse_number(self) -> tuple[Any, bool]:
        match = _NUMBER_TOKEN.match(self.text, self.pos)
        assert match is not None  # parse_value only dispatches here on "-" or a digit
        token = match.group(0)
        self.pos = match.end()
        reached_end = self.at_end()
        strict = _STRICT_NUMBER.match(token)
        if strict is not None and strict.end() == len(token):
            return _to_number(token), True
        if reached_end:
            # A number that is still being written, e.g. "-", "12.", "1e".
            if strict is None:
                return _MISSING, False
            return _to_number(strict.group(0)), False
        self.pos -= len(token)
        self.fail(f"Invalid number {token!r}")

    def parse_literal(self) -> tuple[Any, bool]:
        start = self.pos
        while self.pos < self.length and self.text[self.pos].isalpha():
            self.pos += 1
        word = self.text[start : self.pos]
        if word in _LITERALS:
            return _LITERALS[word], True
        if self.at_end() and any(literal.startswith(word) for literal in _LITERALS):
            return _MISSING, False  # e.g. "tru" mid-stream
        self.pos = start
        self.fail(f"Unexpected token {word!r}")


def _to_number(token: str) -> int | float:
    if any(char in token for char in ".eE"):
        return float(token)
    return int(token)


def _line_and_column(text: str, position: int) -> tuple[int, int]:
    line = text.count("\n", 0, position) + 1
    last_newline = text.rfind("\n", 0, position)
    return line, position - last_newline


def _candidate_starts(text: str) -> list[int]:
    starts = [index for index, char in enumerate(text) if char in "{["]
    if not starts and text.strip():
        starts = [len(text) - len(text.lstrip())]  # no container: try a bare scalar root
    return starts[:_MAX_CANDIDATE_STARTS]


def _scan(source: str) -> tuple[Any, bool, _SyntaxError | None]:
    """Find the first JSON value in ``source`` that parses, completely or up to EOF.

    Returns ``(value, done, furthest_error)``; ``value`` is ``_MISSING`` when no
    candidate worked. Candidates that start *inside* a value that already failed
    or ran to EOF are skipped, so a truncated object is never "rescued" by
    returning one of its nested arrays.
    """
    furthest: _SyntaxError | None = None
    resume_at = 0
    for start in _candidate_starts(source):
        if start < resume_at:
            continue
        try:
            value, done = _Parser(source, start).parse_value()
        except _SyntaxError as error:
            if furthest is None or error.position > furthest.position:
                furthest = error
            resume_at = max(error.position, start + 1)
            continue
        except RecursionError:
            resume_at = start + 1
            continue
        if value is _MISSING:
            continue
        return value, done, furthest
    return _MISSING, False, furthest


def parse_json(text: str) -> Any:
    """Parse one complete JSON value out of raw model output.

    Raises :class:`~llm_structured_extract.errors.OutputParseError` when no
    complete value can be recovered. A value that starts but runs into the end
    of the text is reported as incomplete; otherwise the error points at the
    furthest position any candidate reached, which is usually the real defect.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        pass

    # Prefer the contents of a fenced code block, then the whole text.
    sources = [(fence.group(1), fence.start(1)) for fence in _FENCE.finditer(text)]
    sources.append((text, 0))

    truncated = False
    furthest: tuple[_SyntaxError, int] | None = None
    for source, offset in sources:
        value, done, error = _scan(source)
        if value is not _MISSING:
            if done:
                return value
            truncated = True
        if error is not None and (
            furthest is None or error.position + offset > furthest[0].position + furthest[1]
        ):
            furthest = (error, offset)

    if truncated:
        line, column = _line_and_column(text, len(text))
        raise OutputParseError(
            "Unexpected end of output: the JSON value is incomplete",
            raw_text=text,
            position=len(text),
            line=line,
            column=column,
        )
    if furthest is None or not any(char in text for char in "{["):
        raise OutputParseError("No JSON value found in the output", raw_text=text)
    error, offset = furthest
    position = error.position + offset
    line, column = _line_and_column(text, position)
    raise OutputParseError(
        error.message, raw_text=text, position=position, line=line, column=column
    )


def parse_partial_json(text: str) -> PartialJSON | None:
    """Best-effort parse of a JSON *prefix*, as received mid-stream.

    Returns ``None`` when nothing usable has arrived yet or the prefix is not
    valid JSON so far. Open strings are closed (keeping the partial text), open
    containers are closed, and keys whose value has not started are dropped::

        >>> parse_partial_json('{"name": "Acme", "items": [{"sku": "A1"}, {"sk')
        PartialJSON(value={'name': 'Acme', 'items': [{'sku': 'A1'}, {}]}, complete=False)
    """
    value, done, _ = _scan(text)
    if value is _MISSING:
        return None
    return PartialJSON(value=value, complete=done)
