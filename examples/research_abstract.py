"""Extract structured study metadata from a research abstract, optionally streaming.

Run offline (deterministic, no API key, no cost)::

    python examples/research_abstract.py            # one-shot
    python examples/research_abstract.py --stream   # watch the JSON arrive

Run against Claude (needs ANTHROPIC_API_KEY; makes a paid API call)::

    python examples/research_abstract.py --live --stream

The schema captures what a reviewer screening papers needs: design, population,
sample size, and quantitative findings with confidence intervals and p-values.
Validators reject the usual slips - ``"212 adolescents"`` for a sample size,
``"<0.001"`` for a p-value, a study design outside the controlled vocabulary.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from llm_structured_extract import (
    AnthropicProvider,
    ExtractionResult,
    Extractor,
    MockProvider,
    PartialEvent,
    Provider,
    RepairEvent,
    ResultEvent,
)


class StudyDesign(str, Enum):
    RANDOMIZED_CONTROLLED_TRIAL = "randomized_controlled_trial"
    COHORT_STUDY = "cohort_study"
    CASE_CONTROL_STUDY = "case_control_study"
    CROSS_SECTIONAL_STUDY = "cross_sectional_study"
    SYSTEMATIC_REVIEW = "systematic_review"
    META_ANALYSIS = "meta_analysis"
    QUALITATIVE_STUDY = "qualitative_study"
    OTHER = "other"


class ConfidenceInterval(BaseModel):
    lower: float
    upper: float
    level: float = Field(default=0.95, gt=0, lt=1, description="E.g. 0.95 for a 95% CI.")

    @model_validator(mode="after")
    def _ordered(self) -> ConfidenceInterval:
        if self.lower > self.upper:
            raise ValueError(f"lower bound {self.lower} is greater than upper bound {self.upper}")
        return self


class Finding(BaseModel):
    outcome: str = Field(description="What was measured.")
    effect: str = Field(description="Direction and size of the effect, in a short phrase.")
    estimate: float | None = Field(default=None, description="Point estimate, if reported.")
    unit: str | None = Field(default=None, description="Unit of the estimate, e.g. 'minutes'.")
    confidence_interval: ConfidenceInterval | None = None
    p_value: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="A number. For a reported threshold such as 'p < 0.001', use 0.001.",
    )


class ResearchAbstract(BaseModel):
    """Screening metadata for one study."""

    title: str
    authors: list[str] = Field(min_length=1, description="Full names, in the order given.")
    study_design: StudyDesign
    population: str = Field(description="Who was studied, in one phrase.")
    sample_size: int | None = Field(default=None, gt=0, description="Number of participants.")
    objective: str
    key_findings: list[Finding] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(min_length=3, max_length=8, description="Lower-case keywords.")

    @field_validator("keywords")
    @classmethod
    def _lower_case(cls, keywords: list[str]) -> list[str]:
        not_lower = [keyword for keyword in keywords if keyword != keyword.lower()]
        if not_lower:
            raise ValueError(f"keywords must be lower-case: {not_lower}")
        return keywords


SAMPLE_TEXT = """\
Sleep Regularity and Next-Day Working Memory in Adolescents: A Randomized Crossover Trial
Maya R. Okafor, Jonas Lindqvist, Priya Raman, Tomas Echeverria

Background: Irregular sleep schedules are common in adolescence, but their causal effect
on cognition is unclear.
Objective: To test whether a fixed sleep schedule improves next-day working memory compared
with participants' habitual schedule.
Methods: In this randomized crossover trial, 212 adolescents aged 14-17 years completed two
14-day periods (fixed vs habitual schedule) separated by a 7-day washout. Sleep was measured
with wrist actigraphy; working memory was assessed each morning with a 2-back task.
Results: The fixed schedule increased mean nightly sleep by 34 minutes (95% CI, 27 to 41) and
improved 2-back accuracy by 4.2 percentage points (95% CI, 2.9 to 5.5; p < 0.001). Daytime
sleepiness scores fell by 1.8 points (95% CI, 1.1 to 2.5). No serious adverse events occurred.
Conclusions: A fixed sleep schedule produced modest but meaningful gains in working memory.
Limitations include the short intervention period and recruitment from two urban schools.
Keywords: sleep regularity; adolescents; working memory; actigraphy; crossover trial
"""

#: The corrected answer the scripted model gives on its second attempt.
CORRECTED_ANSWER: dict[str, Any] = {
    "title": (
        "Sleep Regularity and Next-Day Working Memory in Adolescents: A Randomized Crossover Trial"
    ),
    "authors": ["Maya R. Okafor", "Jonas Lindqvist", "Priya Raman", "Tomas Echeverria"],
    "study_design": "randomized_controlled_trial",
    "population": "Adolescents aged 14-17 years",
    "sample_size": 212,
    "objective": (
        "To test whether a fixed sleep schedule improves next-day working memory compared "
        "with the habitual schedule."
    ),
    "key_findings": [
        {
            "outcome": "Mean nightly sleep duration",
            "effect": "Increased with the fixed schedule",
            "estimate": 34,
            "unit": "minutes",
            "confidence_interval": {"lower": 27, "upper": 41, "level": 0.95},
            "p_value": None,
        },
        {
            "outcome": "2-back working-memory accuracy",
            "effect": "Improved with the fixed schedule",
            "estimate": 4.2,
            "unit": "percentage points",
            "confidence_interval": {"lower": 2.9, "upper": 5.5, "level": 0.95},
            "p_value": 0.001,
        },
        {
            "outcome": "Daytime sleepiness score",
            "effect": "Decreased with the fixed schedule",
            "estimate": -1.8,
            "unit": "points",
            "confidence_interval": {"lower": -2.5, "upper": -1.1, "level": 0.95},
            "p_value": None,
        },
    ],
    "limitations": [
        "Short intervention period (two 14-day periods)",
        "Recruitment from two urban schools",
    ],
    "keywords": ["sleep regularity", "adolescents", "working memory", "actigraphy"],
}

#: The first attempt: right content, wrong types and vocabulary in three places.
FLAWED_ANSWER: dict[str, Any] = {
    **CORRECTED_ANSWER,
    "study_design": "randomized crossover trial",
    "sample_size": "212 adolescents",
    "key_findings": [
        CORRECTED_ANSWER["key_findings"][0],
        {**CORRECTED_ANSWER["key_findings"][1], "p_value": "<0.001"},
        CORRECTED_ANSWER["key_findings"][2],
    ],
}


def mock_provider(*, chunk_size: int = 48) -> MockProvider:
    """A deterministic stand-in for an LLM: one flawed answer, then the repair."""
    return MockProvider([FLAWED_ANSWER, CORRECTED_ANSWER], chunk_size=chunk_size)


def run(provider: Provider) -> ExtractionResult[ResearchAbstract]:
    return Extractor(provider).extract(SAMPLE_TEXT, ResearchAbstract)


def run_streaming(provider: Provider) -> ExtractionResult[ResearchAbstract]:
    """Stream the extraction, printing each top-level field as it starts to arrive."""
    seen: list[str] = []
    for event in Extractor(provider).stream(SAMPLE_TEXT, ResearchAbstract):
        if isinstance(event, PartialEvent) and isinstance(event.data, dict):
            new_fields = [field for field in event.data if field not in seen]
            if new_fields:
                seen.extend(new_fields)
                print(f"  [attempt {event.attempt}] receiving: {', '.join(new_fields)}")
        elif isinstance(event, RepairEvent):
            print(f"  [attempt {event.attempt.number}] rejected, asking for a repair:")
            for issue in getattr(event.error, "issues", ()):
                print(f"      - {issue.describe()}")
            seen.clear()
        elif isinstance(event, ResultEvent):
            return event.result
    raise AssertionError("stream ended without a result")  # pragma: no cover


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--stream", action="store_true", help="stream partial results")
    parser.add_argument(
        "--live",
        action="store_true",
        help="use Claude via AnthropicProvider (needs ANTHROPIC_API_KEY; paid API call)",
    )
    args = parser.parse_args(argv)
    provider: Provider = AnthropicProvider() if args.live else mock_provider()

    result = run_streaming(provider) if args.stream else run(provider)
    study = result.value
    print()
    print(study.title)
    print(f"  design: {study.study_design.value}, n = {study.sample_size}")
    for finding in study.key_findings:
        ci = finding.confidence_interval
        interval = f" ({ci.level:.0%} CI {ci.lower} to {ci.upper})" if ci else ""
        p_value = f", p <= {finding.p_value}" if finding.p_value is not None else ""
        print(f"  - {finding.outcome}: {finding.estimate} {finding.unit}{interval}{p_value}")
    print(f"attempts: {len(result.attempts)}, repaired: {result.repaired}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
