"""Prompt templates and the evidence-formatting helper shared by them.

Resynthesis deliberately reuses SYNTHESIS_SYSTEM_PROMPT (see build_resynthesis_user_prompt)
rather than getting its own template -- it is "synthesize again with extra constraints,"
not a distinct task.
"""

from __future__ import annotations

from agent.models import ClaimVerdict, EvidenceItem

PLAN_SYSTEM_PROMPT = """You turn a user's question into a small set of focused web search queries.

Rules:
- Return at most 3 queries. Most questions need only 1.
- Only split into multiple queries when the question has genuinely distinct facets
  that a single search would under-cover (e.g. "what changed in market X this year"
  covers funding, new entrants, and competitive shifts as separate facets).
- Do not pad with redundant or overlapping queries just to reach 3.
- Set topic="news" only for current-events/politics/sports stories. Set topic="finance"
  only for markets/investing/economic-data questions. Otherwise use "general".
- Set time_range only when the question implies recency ("latest", "this year", "recent").
  Leave it null otherwise.
"""

SYNTHESIS_SYSTEM_PROMPT = """You are a research assistant. Answer the user's question using ONLY
the evidence provided below -- never use outside knowledge.

Rules:
- Break your answer into discrete claims. Each claim is one self-contained assertion.
- Every claim must cite the id(s) of the evidence item(s) it draws from, using the exact
  ids given (e.g. "s1", "s2"). Never invent an id that isn't in the evidence list.
- If a claim can't be supported by any of the evidence, don't make it.
- Do not cite an evidence item for a claim it doesn't actually support -- citing the
  wrong source is treated as a defect, not a formality.
- Keep claims concise and factual.
"""

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checker. For each claim below, you are given
the exact text of the evidence it cites. Decide whether that evidence text actually
supports the claim.

Rules:
- Judge each claim using ONLY the evidence text given for it -- not outside knowledge,
  not other claims' evidence.
- "supported": the evidence text directly states or clearly implies the claim.
- "partially_supported": the evidence text is related but doesn't fully establish the
  claim (e.g. hedges, covers only part of it, or the claim overstates/understates it).
- "unsupported": the evidence text does not support the claim, or contradicts it.
- Give a one-sentence rationale citing the specific words that drove your verdict.
- Return verdicts in the exact same order as the claims are given below. You are matched
  back to claims by position, not by text, so the order and count must match exactly.
"""


def format_evidence_block(evidence: list[EvidenceItem]) -> str:
    lines = []
    for item in evidence:
        lines.append(f"[{item.id}] {item.title}")
        lines.append(f"URL: {item.url}")
        if item.published_date:
            lines.append(f"Published: {item.published_date}")
        lines.append(item.content)
        lines.append("")
    return "\n".join(lines).strip()


def build_synthesis_user_prompt(question: str, evidence: list[EvidenceItem]) -> str:
    return (
        f"Question: {question}\n\n"
        f"Evidence:\n{format_evidence_block(evidence)}"
    )


def build_resynthesis_user_prompt(
    question: str,
    evidence: list[EvidenceItem],
    verdicts: list[ClaimVerdict],
    nonexistent_citations: list[str],
) -> str:
    base = build_synthesis_user_prompt(question, evidence)
    issues = []
    for v in verdicts:
        if v.verdict != "supported":
            issues.append(f'- "{v.claim_text}" was judged {v.verdict}: {v.rationale}')
    for text in nonexistent_citations:
        issues.append(f'- "{text}" cited an evidence id that does not exist')

    feedback = "\n".join(issues) if issues else "- none"
    return (
        f"{base}\n\n"
        "A previous draft of this answer had claims that didn't hold up under review:\n"
        f"{feedback}\n\n"
        "Revise the answer using the same evidence above. For each flagged claim, either "
        "drop it, soften it to what the evidence actually supports, or fix its citation. "
        "Do not introduce new unsupported claims."
    )


def build_judge_user_prompt(claims_with_citations: list) -> str:
    """claims_with_citations: list of (Claim, list[EvidenceItem]) pairs to judge."""
    blocks = []
    for i, (claim, cited) in enumerate(claims_with_citations, start=1):
        cited_text = "\n".join(f"[{e.id}] {e.content}" for e in cited)
        blocks.append(f"Claim {i}: {claim.text}\nCited evidence:\n{cited_text}")
    return "\n\n".join(blocks)
