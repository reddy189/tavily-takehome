"""The only module that talks to external APIs (Tavily, the LLM provider).

Calling Tavily directly through `tavily-python` rather than the LangChain tool wrapper
is deliberate: this pipeline decides search parameters in code (not via LLM tool-calling),
so the wrapper's construction/call-time param split and LLM-facing retry-suggestion errors
don't fit -- see the technical statement for the full reasoning.
"""

from __future__ import annotations

import os
from typing import TypeVar

from langchain_nebius import ChatNebius
from pydantic import BaseModel
from tavily import TavilyClient

from agent.models import PlannedQuery

DEFAULT_MODEL = "moonshotai/Kimi-K2.6"
DEFAULT_MAX_RESULTS = 6

T = TypeVar("T", bound=BaseModel)


class TavilySearchError(Exception):
    """A Tavily request failed outright (network/auth/etc), as opposed to a query
    that simply returned zero results."""


def get_tavily_client() -> TavilyClient:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    return TavilyClient(api_key=api_key)


def tavily_search(
    client: TavilyClient,
    planned: PlannedQuery,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> dict:
    """Run one search. Returns Tavily's raw response dict (query/results/...).
    Raises TavilySearchError on request failure; a query that legitimately finds
    nothing returns a dict with an empty `results` list, which is not an error."""

    try:
        return client.search(
            query=planned.query,
            search_depth="advanced",
            topic=planned.topic,
            time_range=planned.time_range,
            max_results=max_results,
            include_raw_content=False,
        )
    except Exception as exc:  # tavily-python raises several distinct exception types
        raise TavilySearchError(f"Tavily search failed for '{planned.query}': {exc}") from exc


def get_chat_model(model_name: str = DEFAULT_MODEL) -> ChatNebius:
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise RuntimeError("NEBIUS_API_KEY is not set")
    return ChatNebius(model=model_name)


def call_structured(model: ChatNebius, system_prompt: str, user_content: str, schema: type[T]) -> T:
    """Shared structured-output call used by plan/synthesize/resynthesize/judge."""
    structured_model = model.with_structured_output(schema)
    return structured_model.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    )
