"""Tests for lazy tracer initialization (fix #2): ENABLE_TRACING must be read at
first use, not at module import time, so it works regardless of when a caller's
load_dotenv() populates os.environ relative to importing agent.pipeline."""

from agent import pipeline


def _reset_tracer_memoization():
    pipeline._tracer = None
    pipeline._tracer_initialized = False


def test_get_tracer_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("ENABLE_TRACING", raising=False)
    _reset_tracer_memoization()

    assert pipeline._get_tracer() is None


def test_get_tracer_initializes_when_enabled_after_import(monkeypatch):
    # Simulates the exact bug scenario: the env var becomes available only after
    # agent.pipeline has already been imported (e.g. .env loaded later by a caller).
    _reset_tracer_memoization()
    monkeypatch.setenv("ENABLE_TRACING", "1")

    tracer = pipeline._get_tracer()

    assert tracer is not None


def test_get_tracer_is_memoized_after_first_call(monkeypatch):
    monkeypatch.setenv("ENABLE_TRACING", "1")
    _reset_tracer_memoization()

    first = pipeline._get_tracer()
    monkeypatch.delenv("ENABLE_TRACING", raising=False)  # should have no effect now
    second = pipeline._get_tracer()

    assert first is second
