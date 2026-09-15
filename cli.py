# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "langchain>=1.0.0",
#   "langchain-nebius>=0.1.0",
#   "tavily-python>=0.7.0",
#   "pydantic>=2.7",
#   "python-dotenv>=1.0.0",
#   "rich>=13.0.0",
#   "typer>=0.12.0",
# ]
# ///
"""
Grounded research CLI built on Tavily.

Setup:
  1. Create a Tavily API key: https://app.tavily.com
  2. Create a Nebius API key: https://tokenfactory.nebius.com
  3. Export both keys or add them to a .env file:
       TAVILY_API_KEY="tvly-..."
       NEBIUS_API_KEY="..."
  4. Run:
       uv run cli.py "What changed in the AI search market this year?"

Unlike a single streamed agent call, this runs a small bounded pipeline (plan -> search ->
select evidence -> synthesize -> verify citations -> render), so there's one status spinner
for the whole run rather than a token stream. See agent/pipeline.py for the stage detail and
STATEMENT.md for why.
"""

from __future__ import annotations

import os
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agent.clients import DEFAULT_MODEL
from agent.pipeline import NoEvidenceError, run_pipeline

load_dotenv()

app = typer.Typer(add_completion=False)
console = Console()


def require_env(name: str, instructions: str) -> None:
    if os.getenv(name):
        return
    console.print(f"[bold red]Missing {name}[/bold red]")
    console.print(instructions)
    raise typer.Exit(code=1)


def render_sources(sources) -> Table:
    table = Table(title="Sources", border_style="cyan", show_lines=False)
    table.add_column("id", style="dim")
    table.add_column("title")
    table.add_column("url", overflow="fold")
    for item in sources:
        table.add_row(item.id, item.title, item.url)
    return table


def render_debug_panel(answer) -> Panel:
    meta = answer.metadata
    lines = [
        f"model: {meta.get('model')}",
        f"planned queries: {meta.get('planned_queries')}",
        f"evidence used: {meta.get('num_evidence')}",
        f"llm calls: {meta.get('llm_call_count')}",
        f"total time: {meta.get('total_sec', 0):.2f}s "
        f"({', '.join(f'{k}={v:.2f}s' for k, v in meta.get('timings_sec', {}).items())})",
        f"resynthesized: {answer.was_resynthesized}",
    ]
    if meta.get("failed_queries"):
        lines.append(f"failed search queries: {meta['failed_queries']}")

    verdicts = answer.initial_verification.claim_verdicts
    if verdicts:
        if answer.was_resynthesized:
            lines.append("initial verdicts (pre-correction; final answer was existence-checked only, not re-judged):")
        else:
            lines.append("initial verdicts:")
        for v in verdicts:
            lines.append(f"  [{v.verdict}] {v.claim_text} -- {v.rationale}")
    if answer.initial_verification.judge_mismatch:
        lines.append(
            "[!] judge output could not be matched to claims (count mismatch) -- "
            "claims were conservatively marked unsupported"
        )
    if answer.initial_verification.nonexistent_citations:
        lines.append(f"nonexistent citations (initial): {answer.initial_verification.nonexistent_citations}")
    if answer.was_resynthesized and answer.final_verification.nonexistent_citations:
        lines.append(
            f"nonexistent citations (after correction): {answer.final_verification.nonexistent_citations}"
        )
    return Panel(Text("\n".join(lines)), title="Debug / quality metadata", border_style="yellow")


@app.command()
def main(
    question: Annotated[list[str], typer.Argument(help="Question")],
    model: Annotated[str, typer.Option(help="Model name")] = DEFAULT_MODEL,
) -> None:
    """Ask a question and run the grounded research pipeline."""

    require_env(
        "TAVILY_API_KEY",
        "Create one at https://app.tavily.com, then run: export TAVILY_API_KEY='tvly-...'",
    )
    require_env(
        "NEBIUS_API_KEY",
        "Create one at https://tokenfactory.nebius.com, then run: export NEBIUS_API_KEY='...'",
    )

    question_text = " ".join(question)
    console.print(Panel.fit(question_text, title="Question", border_style="cyan"))

    try:
        with console.status("[bold blue]Researching..."):
            answer = run_pipeline(question_text, model_name=model)
    except NoEvidenceError as exc:
        console.print(f"\n[bold red]No usable evidence found:[/bold red] {exc}")
        raise typer.Exit(code=1) from None
    except Exception as exc:
        console.print(f"\n[bold red]Pipeline run failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    console.rule("[bold green]Answer")
    # markup=False: the answer text contains literal "[s1]"-style citation markers,
    # which Rich would otherwise silently swallow as (invalid) markup tags.
    console.print(answer.answer_text, markup=False)
    console.print()
    console.print(render_sources(answer.sources))
    console.print()
    console.print(render_debug_panel(answer))


if __name__ == "__main__":
    app()
