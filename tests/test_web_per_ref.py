"""End-to-end test for the website's per-ref verification path.

This exercises the full web pipeline: claim extraction -> ref resolution ->
per-ref verification -> aggregated response. It writes a sample paper to a
temp file and calls the web app's `_run_verification` directly.
"""
import os
import sys
import json
import pytest


@pytest.fixture
def web_app_path():
    """Ensure web/ is on sys.path so we can import the FastAPI app."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    web_path = os.path.join(repo, "web")
    if web_path not in sys.path:
        sys.path.insert(0, web_path)
    return web_path


def test_web_per_ref_for_fully_resolvable_claim(mock_openai, web_app_path, tmp_path, monkeypatch):
    """A claim whose all refs resolve: response has per_ref_verdicts and verdict."""
    # Build a sample paper text with one cited claim
    paper_text = (
        "Transformers [1] use self-attention.\n\n"
        "References\n\n"
        "[1] Vaswani et al. Attention is all you need. NeurIPS 2017.\n"
    )
    paper_file = tmp_path / "test_paper.txt"
    paper_file.write_text(paper_text)

    # Mock the citation resolver so the arXiv fetch is skipped
    import app as web_app
    import citation_resolver
    monkeypatch.setattr(citation_resolver, "resolve_and_fetch_all", lambda full_text, cited_refs: {
        "1": "Vaswani et al. Attention is all you need. NeurIPS 2017. We propose the Transformer, a new simple network architecture based solely on attention mechanisms."
    })

    # Mock OpenAI to return SUPPORTED
    mock_openai.next_verdict = {
        "verdict": "SUPPORTED",
        "confidence": 0.95,
        "evidence_quote": "based solely on attention mechanisms",
        "evidence_start_phrase": "based solely on attention",
        "reasoning": "paper describes attention-based architecture",
    }

    result = web_app._run_verification(str(paper_file), "test_paper.txt", ".txt", "auto")
    assert result["summary"]["total_claims"] >= 1
    claim = result["claims"][0]
    assert claim["verdict"] == "SUPPORTED"
    assert "per_ref_verdicts" in claim
    assert len(claim["per_ref_verdicts"]) == 1
    assert claim["per_ref_verdicts"][0]["verdict"] == "SUPPORTED"


def test_web_per_ref_for_unresolvable_claim(mock_openai, web_app_path, tmp_path, monkeypatch):
    """A claim with all refs unresolvable: returns UNVERIFIABLE with unresolved_refs."""
    paper_text = (
        "Some claim [1] that we cannot verify.\n\n"
        "References\n\n"
        "[1] Some paper behind a paywall.\n"
    )
    paper_file = tmp_path / "test_paper.txt"
    paper_file.write_text(paper_text)

    # Mock resolver to return empty (paywalled / unresolvable)
    import app as web_app
    import citation_resolver
    monkeypatch.setattr(citation_resolver, "resolve_and_fetch_all", lambda full_text, cited_refs: {})

    result = web_app._run_verification(str(paper_file), "test_paper.txt", ".txt", "auto")
    claim = result["claims"][0]
    assert claim["verdict"] == "UNVERIFIABLE"
    assert claim["citation_exists"] is False
    assert claim["unresolved_refs"] == [1]
    assert claim["per_ref_verdicts"] == []


def test_web_per_ref_unverifiable_does_not_count_in_reliability(mock_openai, web_app_path, tmp_path, monkeypatch):
    """reliability_percent uses verifiable denominator, not total.

    The bug fixed in H5 was that unverifiable claims were folded into
    precision. The fix splits the denominator.
    """
    paper_text = (
        "The first study found that creatine improves muscle strength [1]. "
        "A separate investigation showed that vitamin D affects mood [2].\n\n"
        "References\n\n"
        "[1] First paper on creatine and muscle function.\n"
        "[2] Second paper (paywalled) on vitamin D and mood.\n"
    )
    paper_file = tmp_path / "test_paper.txt"
    paper_file.write_text(paper_text)

    import app as web_app
    import citation_resolver
    monkeypatch.setattr(citation_resolver, "resolve_and_fetch_all", lambda full_text, cited_refs: {
        "1": "Substantial content of the first source paper for keyword overlap."
    })
    mock_openai.next_verdict = {
        "verdict": "SUPPORTED",
        "confidence": 0.95,
        "evidence_quote": "ok",
        "evidence_start_phrase": "ok",
        "reasoning": "supported",
    }

    result = web_app._run_verification(str(paper_file), "test_paper.txt", ".txt", "auto")
    # 2 claims extracted, 1 supported, 1 unverifiable
    assert result["summary"]["total_claims"] == 2
    assert result["summary"]["supported"] == 1
    assert result["summary"]["unverifiable"] == 1
    # verifiable denominator = 2 - 1 = 1; reliability = 1/1 = 100%
    assert result["summary"]["reliability_percent"] == 100.0
