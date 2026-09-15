"""Tests for the deterministic pipeline logic. No network access or API keys required --
LLM/Tavily calls are mocked throughout."""

from unittest.mock import MagicMock, patch

import pytest

from agent import pipeline
from agent.models import (
    Claim,
    ClaimVerdict,
    EvidenceItem,
    JudgeOutput,
    PlanOutput,
    PlannedQuery,
    SynthesisOutput,
    VerificationResult,
)

# --------------------------------------------------------------------------
# select_evidence: dedupe, score threshold, cap
# --------------------------------------------------------------------------


def _result(url, score, title="T", content="C"):
    return {"url": url, "score": score, "title": title, "content": content}


def test_select_evidence_filters_below_threshold():
    raw = [{"results": [_result("http://low", 0.1), _result("http://high", 0.9)]}]
    evidence = pipeline.select_evidence(raw)
    assert [e.url for e in evidence] == ["http://high"]


def test_select_evidence_dedupes_by_url_keeping_higher_score():
    raw = [
        {"results": [_result("http://a", 0.5)]},
        {"results": [_result("http://a", 0.95)]},
    ]
    evidence = pipeline.select_evidence(raw)
    assert len(evidence) == 1
    assert evidence[0].score == 0.95


def test_select_evidence_caps_and_ranks_by_score():
    raw = [{"results": [_result(f"http://{i}", score=i / 10) for i in range(4, 15)]}]
    evidence = pipeline.select_evidence(raw, max_evidence=3)
    assert len(evidence) == 3
    assert [e.score for e in evidence] == sorted([e.score for e in evidence], reverse=True)
    assert [e.id for e in evidence] == ["s1", "s2", "s3"]


# --------------------------------------------------------------------------
# citation-existence check
# --------------------------------------------------------------------------


def test_retrieve_returns_failed_queries_separately():
    ok = PlannedQuery(query="ok")
    fails = PlannedQuery(query="fails")

    def fake_search(client, planned, max_results=6):
        if planned.query == "fails":
            raise pipeline.TavilySearchError("boom")
        return {"results": [_result("http://a", 0.9)]}

    with patch.object(pipeline, "tavily_search", side_effect=fake_search):
        raw_results, failed = pipeline.retrieve(MagicMock(), [ok, fails])

    assert len(raw_results) == 1
    assert failed == [fails]


def test_check_citation_existence_splits_valid_and_nonexistent():
    evidence = [EvidenceItem(id="s1", url="http://a", title="A", content="a", score=0.9)]
    synthesis = SynthesisOutput(
        claims=[
            Claim(text="grounded claim", cited_source_ids=["s1"]),
            Claim(text="fabricated claim", cited_source_ids=["s9"]),
            Claim(text="uncited claim", cited_source_ids=[]),
        ]
    )
    valid, nonexistent = pipeline.check_citation_existence(synthesis, evidence)
    assert [c.text for c, _ in valid] == ["grounded claim"]
    assert set(nonexistent) == {"fabricated claim", "uncited claim"}


# --------------------------------------------------------------------------
# judge mismatch: fail safe, never silently truncate (fix #1); accurate call
# accounting (fix #4)
# --------------------------------------------------------------------------


def test_judge_claims_raises_on_verdict_count_mismatch():
    claim = Claim(text="c1", cited_source_ids=["s1"])
    evidence_item = EvidenceItem(id="s1", url="http://a", title="A", content="a", score=0.9)

    with patch.object(pipeline, "call_structured", return_value=JudgeOutput(verdicts=[])):
        with pytest.raises(pipeline.JudgeMismatchError):
            pipeline.judge_claims(MagicMock(), [(claim, [evidence_item])])


def test_verify_fails_safe_on_judge_mismatch_marks_claims_unsupported():
    synthesis = SynthesisOutput(claims=[Claim(text="c1", cited_source_ids=["s1"])])
    evidence = [EvidenceItem(id="s1", url="http://a", title="A", content="a", score=0.9)]

    with patch.object(pipeline, "judge_claims", side_effect=pipeline.JudgeMismatchError("boom")):
        result = pipeline.verify(MagicMock(), synthesis, evidence)

    assert result.judge_mismatch is True
    assert result.judge_call_made is True
    assert len(result.claim_verdicts) == 1
    assert result.claim_verdicts[0].verdict == "unsupported"
    assert pipeline.needs_resynthesis(result) is True


def test_verify_skips_judge_call_when_no_valid_citations():
    synthesis = SynthesisOutput(claims=[Claim(text="bad claim", cited_source_ids=["s9"])])
    evidence = [EvidenceItem(id="s1", url="http://a", title="A", content="a", score=0.9)]

    with patch.object(pipeline, "judge_claims") as mock_judge:
        result = pipeline.verify(MagicMock(), synthesis, evidence)

    mock_judge.assert_not_called()
    assert result.judge_call_made is False
    assert result.nonexistent_citations == ["bad claim"]


def test_needs_resynthesis_true_on_nonexistent_citation():
    result = VerificationResult(nonexistent_citations=["x"], claim_verdicts=[], judge_ran=True)
    assert pipeline.needs_resynthesis(result) is True


def test_needs_resynthesis_true_on_unsupported_verdict():
    result = VerificationResult(
        nonexistent_citations=[],
        claim_verdicts=[ClaimVerdict(claim_text="c", verdict="unsupported", rationale="r")],
        judge_ran=True,
    )
    assert pipeline.needs_resynthesis(result) is True


def test_needs_resynthesis_false_when_all_supported():
    result = VerificationResult(
        nonexistent_citations=[],
        claim_verdicts=[ClaimVerdict(claim_text="c", verdict="supported", rationale="r")],
        judge_ran=True,
    )
    assert pipeline.needs_resynthesis(result) is False


# --------------------------------------------------------------------------
# run_pipeline: the bounded resynthesis state machine
# --------------------------------------------------------------------------


def _base_fixtures():
    plan_output = PlanOutput(queries=[PlannedQuery(query="q1")])
    evidence = [EvidenceItem(id="s1", url="http://a", title="A", content="content a", score=0.9)]
    synthesis = SynthesisOutput(claims=[Claim(text="claim one", cited_source_ids=["s1"])])
    return plan_output, evidence, synthesis


def test_run_pipeline_skips_resynthesis_when_fully_supported():
    plan_output, evidence, synthesis = _base_fixtures()
    fully_supported = VerificationResult(
        nonexistent_citations=[],
        claim_verdicts=[ClaimVerdict(claim_text="claim one", verdict="supported", rationale="ok")],
        judge_ran=True,
    )

    with (
        patch.object(pipeline, "get_chat_model", return_value=MagicMock()),
        patch.object(pipeline, "get_tavily_client", return_value=MagicMock()),
        patch.object(pipeline, "plan", return_value=plan_output),
        patch.object(pipeline, "retrieve", return_value=([{"results": []}], [])),
        patch.object(pipeline, "select_evidence", return_value=evidence),
        patch.object(pipeline, "synthesize", return_value=synthesis),
        patch.object(pipeline, "verify", return_value=fully_supported) as mock_verify,
        patch.object(pipeline, "resynthesize") as mock_resynth,
        patch.object(pipeline, "existence_only") as mock_existence_only,
        patch.object(pipeline, "log_run"),
    ):
        result = pipeline.run_pipeline("some question")

    mock_verify.assert_called_once()
    mock_resynth.assert_not_called()
    mock_existence_only.assert_not_called()
    assert result.was_resynthesized is False
    assert result.final_verification == fully_supported


def test_run_pipeline_resynthesizes_exactly_once_and_never_rejudges():
    plan_output, evidence, synthesis = _base_fixtures()
    unsupported = VerificationResult(
        nonexistent_citations=[],
        claim_verdicts=[ClaimVerdict(claim_text="claim one", verdict="unsupported", rationale="no support")],
        judge_ran=True,
    )
    corrected = SynthesisOutput(claims=[Claim(text="softer claim", cited_source_ids=["s1"])])
    existence_pass = VerificationResult(nonexistent_citations=[], claim_verdicts=[], judge_ran=False)

    with (
        patch.object(pipeline, "get_chat_model", return_value=MagicMock()),
        patch.object(pipeline, "get_tavily_client", return_value=MagicMock()),
        patch.object(pipeline, "plan", return_value=plan_output),
        patch.object(pipeline, "retrieve", return_value=([{"results": []}], [])),
        patch.object(pipeline, "select_evidence", return_value=evidence),
        patch.object(pipeline, "synthesize", return_value=synthesis),
        patch.object(pipeline, "verify", return_value=unsupported) as mock_verify,
        patch.object(pipeline, "resynthesize", return_value=corrected) as mock_resynth,
        patch.object(pipeline, "existence_only", return_value=existence_pass) as mock_existence_only,
        patch.object(pipeline, "log_run"),
    ):
        result = pipeline.run_pipeline("some question")

    mock_verify.assert_called_once()  # judge never runs a second time
    mock_resynth.assert_called_once()
    mock_existence_only.assert_called_once()
    assert result.was_resynthesized is True
    assert "softer claim" in result.answer_text
    assert result.final_verification == existence_pass


# --------------------------------------------------------------------------
# render: dangling citation markers must not remain in the answer text (fix #6)
# --------------------------------------------------------------------------


def test_render_strips_dangling_citation_markers_from_answer_text():
    evidence = [EvidenceItem(id="s1", url="http://a", title="A", content="a", score=0.9)]
    synthesis = SynthesisOutput(claims=[Claim(text="claim", cited_source_ids=["s1", "s9"])])

    result = pipeline.render(
        synthesis,
        evidence,
        was_resynthesized=False,
        initial_verification=VerificationResult(judge_ran=True),
        final_verification=VerificationResult(judge_ran=True),
        metadata={},
    )

    assert "[s9]" not in result.answer_text
    assert "[s1]" in result.answer_text
    assert len(result.sources) == 1
    assert result.sources[0].id == "s1"


def test_run_pipeline_llm_call_count_accounts_for_skipped_judge():
    """Reproduces the original bug: when every claim has a nonexistent citation, the
    judge is never called (judge_call_made=False), so the real call count for a
    resynthesized run is 3 (plan, synthesize, resynthesize), not the old hardcoded 4."""
    plan_output, evidence, synthesis = _base_fixtures()
    all_nonexistent = VerificationResult(
        nonexistent_citations=["claim one"], claim_verdicts=[], judge_ran=True, judge_call_made=False
    )
    corrected = SynthesisOutput(claims=[Claim(text="fixed claim", cited_source_ids=["s1"])])
    existence_pass = VerificationResult(nonexistent_citations=[], claim_verdicts=[], judge_ran=False)

    with (
        patch.object(pipeline, "get_chat_model", return_value=MagicMock()),
        patch.object(pipeline, "get_tavily_client", return_value=MagicMock()),
        patch.object(pipeline, "plan", return_value=plan_output),
        patch.object(pipeline, "retrieve", return_value=([{"results": []}], [])),
        patch.object(pipeline, "select_evidence", return_value=evidence),
        patch.object(pipeline, "synthesize", return_value=synthesis),
        patch.object(pipeline, "verify", return_value=all_nonexistent),
        patch.object(pipeline, "resynthesize", return_value=corrected),
        patch.object(pipeline, "existence_only", return_value=existence_pass),
        patch.object(pipeline, "log_run"),
    ):
        result = pipeline.run_pipeline("some question")

    assert result.metadata["llm_call_count"] == 3


def test_run_pipeline_raises_when_no_evidence_survives_selection():
    plan_output, _, _ = _base_fixtures()

    with (
        patch.object(pipeline, "get_chat_model", return_value=MagicMock()),
        patch.object(pipeline, "get_tavily_client", return_value=MagicMock()),
        patch.object(pipeline, "plan", return_value=plan_output),
        patch.object(pipeline, "retrieve", return_value=([{"results": []}], [])),
        patch.object(pipeline, "select_evidence", return_value=[]),
    ):
        with pytest.raises(pipeline.NoEvidenceError):
            pipeline.run_pipeline("some question")
