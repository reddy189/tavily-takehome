"""Runs the golden query set through the real pipeline and reports groundedness +
latency. Requires TAVILY_API_KEY and NEBIUS_API_KEY -- this is Phase 2, not something
that runs without keys.

Reuses the pipeline's own JSONL run logger as its data source (redirected to a
dedicated eval log for this run) rather than building a second instrumentation path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

EVAL_LOG_PATH = "runs/eval.jsonl"
os.environ["RUN_LOG_PATH"] = EVAL_LOG_PATH

import json

import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.pipeline import NoEvidenceError, run_pipeline  # noqa: E402

load_dotenv()
console = Console()

GOLDEN_SET_PATH = Path(__file__).parent / "golden_queries.yaml"


def load_golden_set() -> list[dict]:
    with GOLDEN_SET_PATH.open() as f:
        return yaml.safe_load(f)


def run_eval() -> None:
    Path(EVAL_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(EVAL_LOG_PATH).unlink(missing_ok=True)

    queries = load_golden_set()
    table = Table(title="Eval results")
    table.add_column("id")
    table.add_column("category")
    table.add_column("planned queries")
    table.add_column("verdicts (initial)")
    table.add_column("resynthesized")
    table.add_column("latency (s)")

    supported_count = 0
    total_claims = 0
    log_path = Path(EVAL_LOG_PATH)

    for item in queries:
        try:
            run_pipeline(item["question"])
        except NoEvidenceError as exc:
            table.add_row(item["id"], item["category"], "-", f"[red]no evidence: {exc}[/red]", "-", "-")
            continue

        # log_run appended exactly one line for this call; read it back rather than
        # using the in-memory return value, so the report is genuinely sourced from
        # the same JSONL log used for debugging (one mechanism, two consumers).
        record = json.loads(log_path.read_text().strip().splitlines()[-1])

        verdicts = record["initial_verification"]["claim_verdicts"]
        total_claims += len(verdicts)
        supported_count += sum(1 for v in verdicts if v["verdict"] == "supported")

        verdict_summary = ", ".join(v["verdict"] for v in verdicts) or "-"
        table.add_row(
            item["id"],
            item["category"],
            str(len(record["metadata"]["planned_queries"])),
            verdict_summary,
            str(record["was_resynthesized"]),
            f"{record['metadata']['total_sec']:.2f}",
        )

    console.print(table)

    faithfulness_pct = (supported_count / total_claims * 100) if total_claims else 0.0
    console.print(f"\nInitial faithfulness rate: {supported_count}/{total_claims} ({faithfulness_pct:.0f}%)")
    console.print(f"Full run log: {EVAL_LOG_PATH}")


if __name__ == "__main__":
    run_eval()
