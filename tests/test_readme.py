"""Keep the README honest: its runnable snippet and its sample outputs are checked here.

Blocks are marked in the README with an HTML comment right above the code fence:

* ``<!-- readme-test: exec -->``: a Python block that is executed;
* ``<!-- readme-test: stdout -->``: must equal what the preceding ``exec`` block printed;
* ``<!-- readme-test: output-of invoice -->``: must equal ``python examples/invoice.py``;
* ``<!-- readme-test: repair-prompt-of invoice -->``: must equal the repair turn the
  invoice example sends on its second attempt.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import invoice

README = Path(__file__).resolve().parents[1] / "README.md"
_MARKED_BLOCK = re.compile(
    r"<!-- readme-test: (?P<directive>[^>]+?) -->\n```[\w-]*\n(?P<body>.*?)\n```", re.DOTALL
)


def _blocks() -> list[tuple[str, str]]:
    return [
        (match["directive"].strip(), match["body"])
        for match in _MARKED_BLOCK.finditer(README.read_text(encoding="utf-8"))
    ]


def _block(directive: str) -> str:
    matches = [body for name, body in _blocks() if name == directive]
    assert len(matches) == 1, f"expected exactly one README block marked {directive!r}"
    return matches[0]


def test_readme_has_the_expected_marked_blocks() -> None:
    assert [name for name, _ in _blocks()] == [
        "exec",
        "stdout",
        "output-of invoice",
        "repair-prompt-of invoice",
    ]


def test_quickstart_runs_and_prints_what_the_readme_says(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = _block("exec")
    exec(compile(code, str(README), "exec"), {"__name__": "readme_quickstart"})
    assert capsys.readouterr().out.rstrip("\n") == _block("stdout")


def test_invoice_output_matches_the_readme(capsys: pytest.CaptureFixture[str]) -> None:
    assert invoice.main([]) == 0
    assert capsys.readouterr().out.rstrip("\n") == _block("output-of invoice")


def test_invoice_repair_prompt_matches_the_readme(capsys: pytest.CaptureFixture[str]) -> None:
    provider = invoice.mock_provider()
    invoice.run(provider)
    capsys.readouterr()
    assert provider.requests[1].messages[-1].content == _block("repair-prompt-of invoice")
