"""Tests for H1: NLI pre-filter stub behavior.

The NLI module is optional. The stub path is what's exercised by default;
the real-model path requires `pip install transformers torch` and
HVE_NLI_ENABLED=1. These tests cover the stub and the short-circuit
plumbing so we know per_ref_verifier doesn't accidentally call the
stub as if it were a real model.
"""
import pytest

import hve_nli
from hve_nli import (
    ENTAILMENT, CONTRADICTION, NEUTRAL,
    predict_stub, maybe_short_circuit,
    is_nli_enabled, get_stats,
)


def test_nli_disabled_by_default(monkeypatch):
    """Without HVE_NLI_ENABLED, the real model path is off."""
    monkeypatch.delenv("HVE_NLI_ENABLED", raising=False)
    assert is_nli_enabled() is False


def test_nli_stub_returns_neutral():
    """The stub always returns NEUTRAL with 0.0 confidence."""
    label, conf = predict_stub("any claim", "any source")
    assert label == NEUTRAL
    assert conf == 0.0


def test_stub_never_short_circuits():
    """The stub must never trigger a short-circuit."""
    result = maybe_short_circuit("claim", "source")
    assert result is None


def test_stub_records_nli_calls():
    before = get_stats()["nli_calls"]
    predict_stub("a", "b")
    after = get_stats()["nli_calls"]
    assert after == before + 1


def test_real_nli_requires_env_and_deps(monkeypatch):
    """Real NLI only on if env + transformers + torch."""
    # Env not set
    monkeypatch.delenv("HVE_NLI_ENABLED", raising=False)
    assert is_nli_enabled() is False

    # Env set but deps missing -> still False
    monkeypatch.setenv("HVE_NLI_ENABLED", "1")
    assert is_nli_enabled() is False  # transformers/torch not installed

    # If we had fake modules installed, the function would proceed to
    # _load_model() which is exercised separately by the real-model
    # integration test (skipped if deps aren't present).
