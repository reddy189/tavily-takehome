"""The six-stage orchestrator: plan -> retrieve -> select -> synthesize -> verify ->
[resynthesize] -> render. No stage loops; resynthesis fires at most once per run.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from langchain_nebius import ChatNebius

from agent.clients import (
    DEFAULT_MODEL,
    TavilySearchError,
    call_structured,
    get_chat_model,
    get_tavily_client,
    tavily_search,
)
from agent.models import (
    Claim,
    ClaimVerdict,
    EvidenceItem,
    FinalAnswer,
    JudgeOutput,
    PlanOutput,
    PlannedQuery,
    SynthesisOutput,
    VerificationResult,
)
from agent.prompts import (
    JUDGE_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    build_judge_user_prompt,
    build_resynthesis_user_prompt,
    build_synthesis_user_prompt,
)

MAX_PLANNED_QUERIES = 3
EVIDENCE_SCORE_THRESHOLD = 0.3
MAX_EVIDENCE = 8


class NoEvidenceError(Exception):
    """All planned searches failed or returned nothing usable."""


class JudgeMismatchError(Exception):
    """The judge returned a verdict count that doesn't match the claims it was given."""


# --------------------------------------------------------------------------
# Optional OpenTelemetry tracing -- a true no-op if ENABLE_TRACING is unset
# or the otel packages aren't installed. Never a hard dependency.
#
# Initialization is lazy (deferred to first use inside run_pipeline) rather than
# done at module import time. Reading ENABLE_TRACING at import time is fragile:
# it can run before a caller's load_dotenv() has populated os.environ, silently
# disabling tracing with no error. Lazy init sidesteps that regardless of any
# entry point's import order.
# --------------------------------------------------------------------------

_tracer = None
_tracer_initialized = False


def _get_tracer():
    global _tracer, _tracer_initialized
    if _tracer_initialized:
        return _tracer
    _tracer_initialized = True
    if os.getenv("ENABLE_TRACING"):
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider()
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            trace.set_tracer_provider(provider)
            _tracer = trace.get_tracer("tavily-research-agent")
        except Exception:
            _tracer = None
    return _tracer


@contextlib.contextmanager
def traced_stage(name: str, **attrs):
    tracer = _get_tracer()
    if tracer is None:
        yield
        return
    with tracer.start_as_current_span(name) as span:
        for key, value in attrs.items():
            span.set_attribute(key, str(value))
        yield


def log_run(record: dict) -> None:
    """Reads RUN_LOG_PATH fresh on every call (rather than caching it at import time)
    so callers -- e.g. eval/run_eval.py -- can redirect it per-run via the env var."""
    path = Path(os.getenv("RUN_LOG_PATH", "runs/runs.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------
# Stage 1: plan
# --------------------------------------------------------------------------


def plan(model: ChatNebius, question: str) -> PlanOutput:
    result = call_structured(model, PLAN_SYSTEM_PROMPT, question, PlanOutput)
    if len(result.queries) > MAX_PLANNED_QUERIES:
        result = PlanOutput(queries=result.queries[:MAX_PLANNED_QUERIES])
    return result


# --------------------------------------------------------------------------
# Stage 2: retrieve
# --------------------------------------------------------------------------


def retrieve(tavily_client, queries: list[PlannedQuery]) -> tuple[list[dict], list[PlannedQuery]]:
    """Runs each planned query once, in parallel. A single query's failure doesn't
    abort the others; failed queries are returned separately so the caller can
    record them (e.g. in run metadata) instead of silently proceeding as if
    nothing happened."""

    raw_results: list[dict] = []
    failed: list[PlannedQuery] = []
    with ThreadPoolExecutor(max_workers=max(len(queries), 1)) as pool:
        futures = {pool.submit(tavily_search, tavily_client, q): q for q in queries}
        for future in as_completed(futures):
            query = futures[future]
            try:
                raw_results.append(future.result())
            except TavilySearchError:
                failed.append(query)
    return raw_results, failed


# --------------------------------------------------------------------------
# Stage 3: select evidence
# --------------------------------------------------------------------------


def select_evidence(
    raw_results: list[dict],
    score_threshold: float = EVIDENCE_SCORE_THRESHOLD,
    max_evidence: int = MAX_EVIDENCE,
) -> list[EvidenceItem]:
    by_url: dict[str, dict] = {}
    for response in raw_results:
        for result in response.get("results", []):
            url = result.get("url")
            if not url:
                continue
            score = result.get("score", 0.0)
            if score < score_threshold:
                continue
            existing = by_url.get(url)
            if existing is None or score > existing.get("score", 0.0):
                by_url[url] = result

    ranked = sorted(by_url.values(), key=lambda r: r.get("score", 0.0), reverse=True)
    top = ranked[:max_evidence]

    return [
        EvidenceItem(
            id=f"s{i}",
            url=r["url"],
            title=r.get("title", "Untitled"),
            content=r.get("content", ""),
            score=r.get("score", 0.0),
            published_date=r.get("published_date"),
        )
        for i, r in enumerate(top, start=1)
    ]


# --------------------------------------------------------------------------
# Stage 4: synthesize
# --------------------------------------------------------------------------


def synthesize(model: ChatNebius, question: str, evidence: list[EvidenceItem]) -> SynthesisOutput:
    user_prompt = build_synthesis_user_prompt(question, evidence)
    return call_structured(model, SYNTHESIS_SYSTEM_PROMPT, user_prompt, SynthesisOutput)


def resynthesize(
    model: ChatNebius,
    question: str,
    evidence: list[EvidenceItem],
    verdicts: list[ClaimVerdict],
    nonexistent_citations: list[str],
) -> SynthesisOutput:
    user_prompt = build_resynthesis_user_prompt(question, evidence, verdicts, nonexistent_citations)
    return call_structured(model, SYNTHESIS_SYSTEM_PROMPT, user_prompt, SynthesisOutput)


# --------------------------------------------------------------------------
# Stage 5: verify
# --------------------------------------------------------------------------


def check_citation_existence(
    synthesis: SynthesisOutput, evidence: list[EvidenceItem]
) -> tuple[list[tuple[Claim, list[EvidenceItem]]], list[str]]:
    """Splits claims into (claim, cited evidence items) pairs whose citations are all
    real, versus the text of claims that cited a nonexistent evidence id."""

    evidence_by_id = {item.id: item for item in evidence}
    valid: list[tuple[Claim, list[EvidenceItem]]] = []
    nonexistent: list[str] = []

    for claim in synthesis.claims:
        if not claim.cited_source_ids:
            nonexistent.append(claim.text)
            continue
        cited_items = []
        all_exist = True
        for source_id in claim.cited_source_ids:
            item = evidence_by_id.get(source_id)
            if item is None:
                all_exist = False
                break
            cited_items.append(item)
        if all_exist:
            valid.append((claim, cited_items))
        else:
            nonexistent.append(claim.text)

    return valid, nonexistent


def judge_claims(
    model: ChatNebius, claims_with_evidence: list[tuple[Claim, list[EvidenceItem]]]
) -> list[ClaimVerdict]:
    """Raises JudgeMismatchError if the judge doesn't return exactly one verdict per
    claim -- callers must not silently truncate/misalign via zip()."""
    if not claims_with_evidence:
        return []

    user_prompt = build_judge_user_prompt(claims_with_evidence)
    result: JudgeOutput = call_structured(model, JUDGE_SYSTEM_PROMPT, user_prompt, JudgeOutput)

    if len(result.verdicts) != len(claims_with_evidence):
        raise JudgeMismatchError(
            f"judge returned {len(result.verdicts)} verdicts for {len(claims_with_evidence)} claims"
        )

    verdicts = list(result.verdicts)
    # Defensive: never trust the model to preserve claim_text verbatim -- overwrite
    # with the actual claim text so downstream rendering/logging is exact.
    for verdict, (claim, _) in zip(verdicts, claims_with_evidence):
        verdict.claim_text = claim.text
    return verdicts


def verify(model: ChatNebius, synthesis: SynthesisOutput, evidence: list[EvidenceItem]) -> VerificationResult:
    valid, nonexistent = check_citation_existence(synthesis, evidence)
    judge_call_made = bool(valid)
    judge_mismatch = False
    verdicts: list[ClaimVerdict] = []

    if valid:
        try:
            verdicts = judge_claims(model, valid)
        except JudgeMismatchError:
            # Fail safe: never let an unjudged claim pass through unflagged. Mark
            # the whole batch unsupported so the existing resynthesis path handles it.
            judge_mismatch = True
            verdicts = [
                ClaimVerdict(
                    claim_text=claim.text,
                    verdict="unsupported",
                    rationale="judge output could not be matched to claims (count mismatch)",
                )
                for claim, _ in valid
            ]

    return VerificationResult(
        nonexistent_citations=nonexistent,
        claim_verdicts=verdicts,
        judge_ran=True,
        judge_call_made=judge_call_made,
        judge_mismatch=judge_mismatch,
    )


def existence_only(synthesis: SynthesisOutput, evidence: list[EvidenceItem]) -> VerificationResult:
    _, nonexistent = check_citation_existence(synthesis, evidence)
    return VerificationResult(
        nonexistent_citations=nonexistent, claim_verdicts=[], judge_ran=False, judge_call_made=False
    )


def needs_resynthesis(verification: VerificationResult) -> bool:
    if verification.nonexistent_citations:
        return True
    return any(v.verdict != "supported" for v in verification.claim_verdicts)


# --------------------------------------------------------------------------
# Stage 6: render
# --------------------------------------------------------------------------


def render(
    synthesis: SynthesisOutput,
    evidence: list[EvidenceItem],
    was_resynthesized: bool,
    initial_verification: VerificationResult,
    final_verification: VerificationResult,
    metadata: dict,
) -> FinalAnswer:
    # Only render markers for ids that actually exist in the evidence set. A citation
    # that's still invalid after resynthesis's existence-only recheck must not show up
    # as a dangling marker in the answer text -- it stays visible in
    # final_verification.nonexistent_citations (surfaced in the debug panel) instead.
    valid_ids = {item.id for item in evidence}
    used_ids: set[str] = set()
    lines = []
    for claim in synthesis.claims:
        real_ids = [sid for sid in claim.cited_source_ids if sid in valid_ids]
        used_ids.update(real_ids)
        markers = "".join(f"[{sid}]" for sid in real_ids)
        lines.append(f"{claim.text} {markers}".rstrip())
    answer_text = "\n".join(lines)

    sources = [item for item in evidence if item.id in used_ids]

    return FinalAnswer(
        answer_text=answer_text,
        sources=sources,
        was_resynthesized=was_resynthesized,
        initial_verification=initial_verification,
        final_verification=final_verification,
        metadata=metadata,
    )


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------


def run_pipeline(question: str, model_name: str = DEFAULT_MODEL) -> FinalAnswer:
    run_start = time.monotonic()
    timings: dict[str, float] = {}

    chat_model = get_chat_model(model_name)
    tavily_client = get_tavily_client()

    with traced_stage("plan"):
        t0 = time.monotonic()
        plan_output = plan(chat_model, question)
        timings["plan"] = time.monotonic() - t0

    with traced_stage("retrieve", num_queries=len(plan_output.queries)):
        t0 = time.monotonic()
        raw_results, failed_queries = retrieve(tavily_client, plan_output.queries)
        timings["retrieve"] = time.monotonic() - t0

    with traced_stage("select"):
        evidence = select_evidence(raw_results)

    if not evidence:
        raise NoEvidenceError(f"No usable evidence found for: {question!r}")

    with traced_stage("synthesize"):
        t0 = time.monotonic()
        synthesis = synthesize(chat_model, question, evidence)
        timings["synthesize"] = time.monotonic() - t0

    with traced_stage("verify"):
        t0 = time.monotonic()
        initial_verification = verify(chat_model, synthesis, evidence)
        timings["verify"] = time.monotonic() - t0

    was_resynthesized = False
    final_verification = initial_verification

    if needs_resynthesis(initial_verification):
        was_resynthesized = True
        with traced_stage("resynthesize"):
            t0 = time.monotonic()
            synthesis = resynthesize(
                chat_model,
                question,
                evidence,
                initial_verification.claim_verdicts,
                initial_verification.nonexistent_citations,
            )
            timings["resynthesize"] = time.monotonic() - t0
        final_verification = existence_only(synthesis, evidence)

    llm_call_count = 2  # plan + synthesize, always made
    if initial_verification.judge_call_made:
        llm_call_count += 1
    if was_resynthesized:
        llm_call_count += 1  # existence-only recheck makes no LLM call

    metadata = {
        "model": model_name,
        "planned_queries": [q.model_dump() for q in plan_output.queries],
        "num_evidence": len(evidence),
        "failed_queries": [q.query for q in failed_queries],
        "timings_sec": timings,
        "total_sec": time.monotonic() - run_start,
        "llm_call_count": llm_call_count,
    }

    final_answer = render(
        synthesis, evidence, was_resynthesized, initial_verification, final_verification, metadata
    )

    log_run(
        {
            "question": question,
            "answer": final_answer.answer_text,
            "was_resynthesized": was_resynthesized,
            "initial_verification": initial_verification.model_dump(),
            "final_verification": final_verification.model_dump(),
            "metadata": metadata,
        }
    )

    return final_answer
