# Technical Statement

> Live-validated against real Tavily and Nebius APIs: the full golden set (6 queries) has
> been run end to end. Summary: 35 total claims, 31/35 (89%) judged `supported` on first
> pass; 3 of 6 queries triggered the one-shot resynthesis (each over one
> `partially_supported` claim); zero judge-format mismatches and zero nonexistent
> citations occurred across the run; latency ranged ~4-22s per query. See "Live validation"
> below for what this run actually surfaced, including a real defect it caught.

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

## Live validation

Running against real APIs surfaced one genuine, model-specific defect and confirmed the
grounding mechanism catches real problems, not just constructed test cases.

**The default model had to change, twice, based on evidence, not guesswork.** The
starter's carried-over default, `moonshotai/Kimi-K2.6`, turned out to leak an internal
serving-template token (`<|tool_calls_section_begin|>`) into every structured-output tool
call on this Nebius deployment, breaking JSON parsing for every stage (`plan`,
`synthesize`, `verify`, `resynthesize`) uniformly. Switching `with_structured_output`'s
`method` parameter (`json_schema`/`json_mode`) doesn't help — that path errors inside
`langchain-nebius` itself, independent of model. Querying the account's live model catalog
directly (rather than guessing names) and testing candidates against the *actual*
synthesis prompt (not just a short one) surfaced a second, subtler issue: `openai/gpt-oss-120b`
parsed cleanly on the short `plan` prompt but silently skipped tool-calling — writing valid
JSON as plain message content instead — once the prompt included a full evidence block.
`Qwen/Qwen3-235B-A22B-Instruct-2507` was verified reliable on the real synthesis/judge
workload and is now the default. This is exactly the risk flagged as unverified before any
live testing happened, and it materialized in a more specific way than expected (two
distinct failure modes, not one) — worth noting as a concrete argument for why "unverified
against live behavior" belongs in a limitations section until it's actually been run.

**Resynthesis caught a real overclaim, not a synthetic one.** On the compound query "what
changed in the AI search market this year," the initial synthesis presented two distinct
forecasts' CAGR figures (16.69% and 27.30%) as a single unqualified range, as if they were
one fact rather than two different estimates. The judge marked it `partially_supported`,
flagging that the combined range wasn't cleanly backed by a single source. Resynthesis
fired exactly once, and the corrected answer split it into two properly attributed
statements — the market-size range as one claim, and the two CAGR figures as separate,
source-attributed estimates ("one forecast... while another forecast..."). This happened
without any special-casing for this scenario — it's the designed mechanism working on a
case it wasn't built for in advance.

**One honest correction to the eval design's own assumption:** the golden set's "compound"
category was meant to exercise the planner's multi-query decomposition, but in this run
the planner judged both compound questions well-served by a single, well-targeted query
(with `topic`/`time_range` set appropriately) rather than splitting them. That's a valid
planning judgment call, not a bug — the prompt only asks it to split on genuinely distinct
facets — but it means the golden set doesn't *guarantee* multi-query planning fires; it
only creates the opportunity for it to.

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
- The default model has been live-validated on the golden set (see "Live validation")
  but only against 6 queries in one session — not enough runs to rule out occasional
  tool-calling failures recurring under different prompt shapes or load.
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
- Extend live validation beyond one 6-query session — run the golden set (and a larger
  one) repeatedly over time to catch intermittent tool-calling failures a single run
  wouldn't surface, and add an automated check that fails loudly if `with_structured_output`
  ever returns `None` in production rather than only in this diagnostic session.

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
