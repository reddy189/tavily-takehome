"""Tests for CLI rendering logic (fix #5). Pure rendering given a constructed
FinalAnswer -- no pipeline execution, no API keys needed."""

import cli
from agent.models import ClaimVerdict, EvidenceItem, FinalAnswer, VerificationResult


def _answer(**overrides) -> FinalAnswer:
    defaults = dict(
        answer_text="claim [s1]",
        sources=[EvidenceItem(id="s1", url="http://a", title="A", content="a", score=0.9)],
        was_resynthesized=False,
        initial_verification=VerificationResult(judge_ran=True),
        final_verification=VerificationResult(judge_ran=True),
        metadata={"model": "m", "planned_queries": [], "num_evidence": 1, "timings_sec": {}, "total_sec": 0.1},
    )
    defaults.update(overrides)
    return FinalAnswer(**defaults)


def test_debug_panel_labels_verdicts_as_pre_correction_when_resynthesized():
    answer = _answer(
        was_resynthesized=True,
        initial_verification=VerificationResult(
            judge_ran=True,
            judge_call_made=True,
            claim_verdicts=[ClaimVerdict(claim_text="c", verdict="unsupported", rationale="r")],
        ),
    )
    panel_text = cli.render_debug_panel(answer).renderable.plain
    assert "pre-correction" in panel_text
    assert "not re-judged" in panel_text


def test_debug_panel_does_not_label_verdicts_when_not_resynthesized():
    answer = _answer(
        initial_verification=VerificationResult(
            judge_ran=True,
            judge_call_made=True,
            claim_verdicts=[ClaimVerdict(claim_text="c", verdict="supported", rationale="r")],
        )
    )
    panel_text = cli.render_debug_panel(answer).renderable.plain
    assert "pre-correction" not in panel_text
    assert "initial verdicts:" in panel_text


def test_debug_panel_shows_judge_mismatch_warning():
    answer = _answer(
        initial_verification=VerificationResult(judge_ran=True, judge_call_made=True, judge_mismatch=True)
    )
    panel_text = cli.render_debug_panel(answer).renderable.plain
    assert "judge output could not be matched to claims" in panel_text


def test_debug_panel_shows_failed_queries():
    answer = _answer(metadata={"failed_queries": ["a bad query"], "timings_sec": {}, "total_sec": 0.1})
    panel_text = cli.render_debug_panel(answer).renderable.plain
    assert "a bad query" in panel_text
