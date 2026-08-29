"""Smoke tests for the test infrastructure itself.

These tests do not assert any code fixes; they only verify that pytest
fixtures, the OpenAI mock, and module imports work as designed. If these
fail, the test infrastructure is broken and other tests cannot be trusted.
"""
import json
import pytest


def test_mock_openai_is_stable(mock_openai):
    """The same input should produce the same verdict on repeated calls."""
    from verify_hallucinations import llm_call
    out1 = llm_call("system prompt", "user prompt X")
    out2 = llm_call("system prompt", "user prompt X")
    assert out1 == out2
    parsed = json.loads(out1)
    assert parsed["verdict"] in {
        "SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED", "UNVERIFIABLE"
    }


def test_mock_openai_changes_with_input(mock_openai):
    """Different inputs should (in general) produce different verdicts."""
    from verify_hallucinations import llm_call
    a = json.loads(llm_call("s", "alpha alpha alpha"))
    b = json.loads(llm_call("s", "beta beta beta beta beta"))
    # They might collide on the 4-bucket verdict; assert non-trivial diff
    # by checking that across many distinct inputs, we see at least 2 verdicts.
    seen = set()
    for s in ["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh"]:
        seen.add(json.loads(llm_call("s", s))["verdict"])
    assert len(seen) >= 2


def test_mock_openai_override(mock_openai):
    """Test code can pin a specific verdict for the next call."""
    from verify_hallucinations import llm_call
    mock_openai.next_json = json.dumps({
        "verdict": "NOT_SUPPORTED",
        "confidence": 0.95,
        "evidence_quote": "test",
        "evidence_start_phrase": "test",
        "reasoning": "explicit override",
    })
    out = llm_call("s", "anything")
    parsed = json.loads(out)
    assert parsed["verdict"] == "NOT_SUPPORTED"
    assert parsed["confidence"] == 0.95
    # Override should be consumed
    assert mock_openai.next_json is None


def test_short_text_fixture_loads(sample_short_text):
    assert "[1]" in sample_short_text
    assert "[2]" in sample_short_text


def test_multi_ref_claim_fixture_loads(sample_multi_ref_claim):
    assert "[62, 68]" in sample_multi_ref_claim
    assert "[14, 23]" in sample_multi_ref_claim
    assert "[26, 27]" in sample_multi_ref_claim


def test_module_imports():
    """All top-level modules can be imported without running code paths.

    Note: `build_v100_benchmark.py`, `run_v100.py`, and
    `run_competitors_v100.py` are excluded because they have top-level
    file opens that fail outside their expected cwd. This is a separate
    bug (M-tier: top-level side effects) that will be fixed in Phase 2.
    """
    import verify_hallucinations
    import arxiv_extractor
    import benchmark_template
    from statistical_tests import Diff2MeanSig, Diff2MeanConf


@pytest.mark.benchmark
def test_crosscat_dataset_loads(crosscat_input, crosscat_gt):
    """Benchmark dataset is loadable. Marked as a benchmark test (skipped in default CI)."""
    assert len(crosscat_input) > 0
    assert len(crosscat_gt) > 0
    gt_map = {g["question_id"]: g["status"] for g in crosscat_gt}
    for entry in crosscat_input[:5]:
        assert entry["question_id"] in gt_map
