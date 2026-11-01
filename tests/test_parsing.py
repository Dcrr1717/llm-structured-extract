from __future__ import annotations

import json
from typing import Any

import pytest

from llm_structured_extract import OutputParseError, PartialJSON, parse_json, parse_partial_json

# --------------------------------------------------------------------------- #
# parse_json: complete values
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param('{"a": 1, "b": [true, null]}', {"a": 1, "b": [True, None]}, id="plain"),
        pytest.param('  \n{"a": 1}\n  ', {"a": 1}, id="whitespace"),
        pytest.param("[1, 2, 3]", [1, 2, 3], id="array-root"),
        pytest.param("42", 42, id="scalar-root"),
        pytest.param('"hello"', "hello", id="string-root"),
        pytest.param('```json\n{"a": 1}\n```', {"a": 1}, id="fenced"),
        pytest.param('```\n{"a": 1}\n```', {"a": 1}, id="fenced-no-language"),
        pytest.param(
            'Sure! Here is the data:\n```json\n{"a": 1}\n```\nLet me know if you need more.',
            {"a": 1},
            id="fenced-with-prose",
        ),
        pytest.param('The result is {"a": 1}. Done.', {"a": 1}, id="inline-prose"),
        pytest.param('{"a": [1, 2,], "b": 3,}', {"a": [1, 2], "b": 3}, id="trailing-commas"),
        pytest.param(
            '{\n  // the id\n  "id": 7, /* inline */ "ok": true\n}',
            {"id": 7, "ok": True},
            id="comments",
        ),
        pytest.param("{'a': 'single', \"b\": 2}", {"a": "single", "b": 2}, id="single-quotes"),
        pytest.param(
            '{"a": True, "b": False, "c": None}',
            {"a": True, "b": False, "c": None},
            id="python-literals",
        ),
        pytest.param(
            '{"text": "line one\nline two"}',
            {"text": "line one\nline two"},
            id="raw-newline-in-string",
        ),
        pytest.param(
            '{"e": "caf\\u00e9 \\ud83d\\ude00 \\n \\" \\/"}', {"e": 'café 😀 \n " /'}, id="escapes"
        ),
        pytest.param(
            '{"x": -1.5e3, "y": 0, "z": 10}', {"x": -1500.0, "y": 0, "z": 10}, id="numbers"
        ),
        pytest.param(
            'Use {curly} braces like this: {"a": 1}', {"a": 1}, id="skips-non-json-braces-in-prose"
        ),
        pytest.param('\ufeff{"bom": true}', {"bom": True}, id="byte-order-mark"),
    ],
)
def test_parse_json_tolerates_common_llm_defects(raw: str, expected: Any) -> None:
    assert parse_json(raw) == expected


def test_strict_json_takes_the_fast_path() -> None:
    document = {"nested": {"list": [1, 2, {"deep": "value"}]}, "unicode": "日本語"}
    assert parse_json(json.dumps(document)) == document


def test_truncated_value_is_reported_as_incomplete() -> None:
    with pytest.raises(OutputParseError, match="incomplete") as info:
        parse_json('{"items": [1, 2], "total": ')
    assert info.value.line == 1
    assert info.value.raw_text == '{"items": [1, 2], "total": '


def test_truncated_object_is_not_rescued_by_a_nested_value() -> None:
    # The inner array is complete, but returning it would silently drop the
    # enclosing (truncated) object.
    with pytest.raises(OutputParseError, match="incomplete"):
        parse_json('{"items": [1, 2], "note": "cut off he')


def test_syntax_error_reports_line_and_column() -> None:
    raw = '{\n  "a": 1,\n  "b": 2 "c": 3\n}'
    with pytest.raises(OutputParseError) as info:
        parse_json(raw)
    error = info.value
    assert error.line == 3
    assert error.column is not None
    assert "line 3" in str(error)
    assert "Expected ','" in error.message


def test_invalid_number_is_rejected_not_truncated() -> None:
    with pytest.raises(OutputParseError, match="Invalid number"):
        parse_json('{"total": 12.5.3}')


def test_no_json_at_all() -> None:
    with pytest.raises(OutputParseError, match="No JSON value found"):
        parse_json("I could not find any invoice in this document.")


def test_empty_output() -> None:
    with pytest.raises(OutputParseError):
        parse_json("   ")


def test_bad_unicode_escape() -> None:
    with pytest.raises(OutputParseError, match="escape"):
        parse_json('{"a": "\\uZZZZ"}')


def test_unquoted_key_is_rejected() -> None:
    with pytest.raises(OutputParseError, match="quoted property name"):
        parse_json("{a: 1}")


def test_unknown_escape_is_kept_verbatim() -> None:
    # Invalid in strict JSON; a model writing a Windows path meant the backslash.
    assert parse_json('{"path": "C:\\Users"}') == {"path": "C:\\Users"}


def test_deeply_nested_input_does_not_crash() -> None:
    with pytest.raises(OutputParseError):
        parse_json("[" * 5000 + "x")


# --------------------------------------------------------------------------- #
# parse_partial_json: streaming prefixes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        pytest.param("{", {}, id="open-object"),
        pytest.param('{"na', {}, id="dangling-key"),
        pytest.param('{"name"', {}, id="key-without-colon"),
        pytest.param('{"name": ', {}, id="key-without-value"),
        pytest.param('{"name": "Ac', {"name": "Ac"}, id="partial-string-value"),
        pytest.param('{"name": "Acme", "n": 12', {"name": "Acme", "n": 12}, id="partial-number"),
        pytest.param('{"n": 12.', {"n": 12}, id="number-with-dangling-dot"),
        pytest.param('{"n": -', {}, id="lone-minus"),
        pytest.param('{"n": 1e', {"n": 1}, id="dangling-exponent"),
        pytest.param('{"ok": tr', {}, id="partial-literal"),
        pytest.param('{"ok": true', {"ok": True}, id="complete-literal"),
        pytest.param('{"s": "a\\', {"s": "a"}, id="dangling-backslash"),
        pytest.param('{"s": "\\u00', {"s": ""}, id="partial-unicode-escape"),
        pytest.param('{"s": "\\ud83d\\ude', {"s": ""}, id="partial-surrogate-pair"),
        pytest.param("[1, 2", [1, 2], id="open-array"),
        pytest.param("[1, 2,", [1, 2], id="array-trailing-comma"),
        pytest.param(
            '{"items": [{"sku": "A1"}, {"sk',
            {"items": [{"sku": "A1"}, {}]},
            id="nested",
        ),
        pytest.param('```json\n{"a": [1', {"a": [1]}, id="opening-fence"),
        pytest.param('Here you go: {"a": 1', {"a": 1}, id="leading-prose"),
    ],
)
def test_partial_prefixes(prefix: str, expected: Any) -> None:
    partial = parse_partial_json(prefix)
    assert partial == PartialJSON(value=expected, complete=False)


def test_partial_complete_document() -> None:
    assert parse_partial_json('{"a": [1, 2]}') == PartialJSON(value={"a": [1, 2]}, complete=True)


@pytest.mark.parametrize("prefix", ["", "   ", "Sure, here", '{"a": 1 "b"'])
def test_partial_returns_none_when_nothing_usable(prefix: str) -> None:
    assert parse_partial_json(prefix) is None


def test_every_prefix_of_a_document_parses_and_grows_monotonically() -> None:
    document = {
        "title": 'A "quoted" title \u2014 with unicode',
        "count": 1234,
        "ratio": -0.125,
        "flags": [True, False, None],
        "nested": {"items": [{"k": "v"}, {"k": "w", "n": [1, 2, 3]}]},
    }
    text = json.dumps(document, indent=2, ensure_ascii=True)
    previous_keys = 0
    for end in range(1, len(text) + 1):
        partial = parse_partial_json(text[:end])
        if partial is None:
            continue
        assert isinstance(partial.value, dict)
        assert len(partial.value) >= previous_keys
        previous_keys = len(partial.value)
    final = parse_partial_json(text)
    assert final == PartialJSON(value=document, complete=True)
