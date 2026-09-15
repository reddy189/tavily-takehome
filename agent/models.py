"""Schemas passed between pipeline stages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["supported", "partially_supported", "unsupported"]


class PlannedQuery(BaseModel):
    """One focused search the plan stage wants executed."""

    query: str
    topic: Literal["general", "news", "finance"] = "general"
    time_range: Literal["day", "week", "month", "year"] | None = None


class PlanOutput(BaseModel):
    """Structured output of the plan stage. Capped at 3 by the plan prompt;
    the pipeline enforces the cap in code regardless of what the model returns."""

    queries: list[PlannedQuery] = Field(min_length=1)


class EvidenceItem(BaseModel):
    """A single retrieved, deduped source made available to synthesis."""

    id: str
    url: str
    title: str
    content: str
    score: float
    published_date: str | None = None


class Claim(BaseModel):
    """One assertion in the answer, with the evidence ids it claims to draw from."""

    text: str
    cited_source_ids: list[str] = Field(default_factory=list)


class SynthesisOutput(BaseModel):
    claims: list[Claim] = Field(min_length=1)


class ClaimVerdict(BaseModel):
    claim_text: str
    verdict: Verdict
    rationale: str


class JudgeOutput(BaseModel):
    """Structured output of the judge call. `verdicts` must be the same length and
    order as the claims it was given -- the pipeline matches by index, not text."""

    verdicts: list[ClaimVerdict]


class VerificationResult(BaseModel):
    """Result of one verification pass. `judge_ran` distinguishes a full pass
    (existence + judge) from the post-correction pass (existence only).
    `judge_call_made` distinguishes an actual LLM judge call from a full pass that
    had nothing left to judge (e.g. every claim had a nonexistent citation).
    `judge_mismatch` flags that the judge's output couldn't be matched to the claims
    it was given, in which case those claims were conservatively marked unsupported
    rather than silently dropped."""

    nonexistent_citations: list[str] = Field(default_factory=list)
    claim_verdicts: list[ClaimVerdict] = Field(default_factory=list)
    judge_ran: bool = True
    judge_call_made: bool = False
    judge_mismatch: bool = False


class FinalAnswer(BaseModel):
    """What the CLI renders and the eval harness scores."""

    answer_text: str
    sources: list[EvidenceItem]
    was_resynthesized: bool
    initial_verification: VerificationResult
    final_verification: VerificationResult
    metadata: dict = Field(default_factory=dict)
