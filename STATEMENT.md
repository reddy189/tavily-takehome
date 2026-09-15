# Technical Statement

> This reflects the implementation as built and tested with mocked LLM/Tavily clients.
> The pipeline has **not** been run against live Tavily/Nebius APIs yet — no eval
> results, latency figures, or live-validation claims are included here. `eval/run_eval.py`
> is written and ready; results will be added once keys are available.

## The problem

The starting point was a minimal LangChain agent: an unconfigured `TavilySearch` tool
behind a "cite when available" prompt, with no retrieval configuration, no structural
guarantee that a citation was real or that a claim was actually supported by what was
retrieved, no evaluation loop, and no observability. For a search-grounded assistant, an
unverifiable citation is the single most common trust-breaking failure a user or customer
will notice — so that's the problem this project targets directly, rather than adding
unrelated features on top of the same agent loop.

## Core engineering decisions

**A bounded pipeline instead of an LLM-driven agent loop (`create_agent`/ReAct).** The
properties this project optimizes for — retrieval-parameter control, per-stage
debuggability, testability, and comparable evaluation across queries — are exactly what
an open-ended tool-calling loop makes hard, and none of them require an LLM deciding
whether/when/how to search. The trade-off: this loses genuine open-ended multi-hop
adaptivity.

**Direct `tavily-python` SDK call instead of the `langchain_tavily` LangChain tool.**
Checked against the actual wrapper source rather than assumed: it exposes nearly the full
Tavily parameter surface, so the reason to bypass it isn't missing capability. It's that
the wrapper's design — a construction-time/call-time parameter split, and
natural-language, LLM-facing retry-suggestion errors on empty results — is built for
invocation via LLM tool-calling, which this pipeline deliberately doesn't do. Search
parameters are decided in code, so that design doesn't fit and the wrapper's abstraction
is pure overhead. Trade-off: more integration code owned directly (parameter mapping,
typed exceptions) instead of getting it from the wrapper for free.

**A capped planning step (1-3 sub-queries, hard-capped in code) instead of one fixed
query or an open-ended agent loop.** A single search under-covers genuinely compound
questions — the assignment's own example ("what changed in the AI search market this
year") spans distinct facets a single query tends to blur together. The hard cap (not
just a prompt instruction) keeps the mechanism bounded regardless of what the model
returns. Trade-off: no reactive retry if all planned queries under-deliver evidence —
betting on better-targeted queries up front rather than a second round of search.

**Two-layer verification, not three.** A deterministic citation-existence check (the
cited source id must actually be in the retrieved evidence) plus one batched LLM judge
call (does the cited text actually support the claim). A middle lexical/keyword-overlap
layer was deliberately left out: it would share the judge's exact blind spot — a claim
can share every keyword with its source while inverting or misrepresenting what it
actually says — so it would add tuning surface without adding real coverage.

**Exactly one corrective resynthesis, using the same evidence, then stop.** If
verification finds a problem, the model gets one chance to revise using only the evidence
it already has (soften/drop the flagged claim, fix a bad citation) — never a new search,
never a second attempt. This keeps latency and cost bounded and predictable instead of
opening a retry loop. The post-correction pass is existence-only, not re-judged — running
the full batched judge again would double that cost on every corrected answer — so a
corrected answer is trusted-but-not-re-verified for semantic support. That status is
disclosed in run metadata/debug output, not claimed as re-verified and not surfaced as an
inline warning inside the answer text itself.

## Why grounding is designed this way

The goal isn't "verify perfectly" — it's "never silently present an unverified or
fabricated claim as fact." The existence check and the judge catch two genuinely
different failure modes (a citation that doesn't exist at all vs. a real citation that
doesn't actually say what the claim says), the one-shot correction gives the pipeline a
chance to fix what it can with the evidence already in hand, and whatever can't be fixed
is disclosed rather than hidden. A judge whose output can't be reliably matched back to
the claims it was given (wrong verdict count/order) is treated as a verification failure
that conservatively marks those claims unsupported — never silently dropped or truncated
— which routes into the same correction path rather than a separate failure mode. And a
citation that's still invalid after correction can't appear as a working-looking marker
in the rendered answer, even though it stays fully visible in debug metadata.

## Trade-offs accepted

- No agent loop → loses multi-hop adaptivity in exchange for determinism, testability,
  and comparable evaluation.
- Direct SDK integration → more code owned in exchange for parameter control and typed
  error handling instead of LLM-facing retry text.
- Planner capped at 1-3 queries, no reactive retry-if-thin → if all planned searches
  under-deliver, there's no further search to compensate.
- Judge runs unconditionally on every existence-passing claim, never selectively gated →
  a cheap gate would share the judge's exact blind spot, so gating would add complexity
  without solving the actual problem.
- One corrective pass, existence-only recheck, not re-judged → predictable cost/latency
  in exchange for a corrected answer's semantic support being trusted, not re-verified.

## Known limitations

- Recency metadata (`published_date`) is only reliably populated by Tavily for
  `topic="news"` searches, and isn't currently used to influence evidence ranking even
  when present.
- `search_depth="advanced"` is hardcoded for every planned search — a real, untuned
  cost/latency lever (Tavily bills advanced search at a higher credit cost than basic).
- Evidence given to synthesis and the judge is Tavily's snippet only
  (`include_raw_content=False`); grounding is only as good as that snippet, never full
  page content.
- The default model (carried over from the original starter) is unverified against
  Nebius's current model catalog — the first real risk once live calls are made.
- The JSONL run log is a plain file append with no concurrency lock — fine for this
  single-process CLI/eval usage, not safe for concurrent writers.
- `nonexistent_citations` bundles two distinct cases (a fabricated citation id, and a
  claim with no citation at all) under one name — both correctly trigger correction, but
  the naming slightly overclaims for the second case.
- There is no retry/backoff on transient LLM or Tavily call failures — any such error
  propagates straight to a hard pipeline failure.
- Test coverage is intentionally scoped to the deterministic pipeline logic against
  mocked clients — there is no live-API integration test yet, so real retrieval,
  synthesis, and judge quality are unverified until the golden set is actually run.

## What I'd change for production

- Run the golden set (and a larger one) against live APIs to tune
  `EVIDENCE_SCORE_THRESHOLD`, `MAX_EVIDENCE`, and the search-depth policy with real data
  instead of the current placeholder defaults.
- Add live-API integration tests (recorded/replayed fixtures) to catch schema drift from
  Tavily or Nebius without requiring live calls on every test run.
- Add retry/backoff for transient LLM/Tavily failures instead of a hard pipeline failure.
- Move the run log to a concurrency-safe store, or add file locking, if this ever serves
  more than one request at a time.
- Richer trace attributes (token usage, cost, prompt/response previews) instead of just
  stage timing, for real production debugging.
- Consider fetching full page content for higher-stakes queries where a snippet isn't
  enough to judge support confidently — as an explicit, opt-in cost/latency trade-off,
  not a silent default change.
- Validate and pin the model against Nebius's current catalog rather than carrying over
  an unverified default from the original starter.

## Value

For a customer evaluating Tavily for a production research or assistant use case, the
most common trust-breaking failure in a search-grounded system is a citation that's
fabricated or doesn't actually say what the answer claims. This project is a concrete,
inexpensive pattern (bounded LLM call count, no new infrastructure, no vector store or
new tracing platform required) for making that failure mode visible and largely
self-correcting, backed by a repeatable evaluation harness to measure it and
observability to debug it — the kind of pattern worth showing a customer building a
production research assistant on Tavily, not a demo that only happens to work on one
example query.
