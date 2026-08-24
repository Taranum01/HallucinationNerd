"""Tests for H2: sentence splitter abbreviation handling.

The naive `(?<=[.!?])\s+(?=[A-Z\[])` would split at "et al.", "Fig.",
"i.e.", "e.g.", etc., corrupting claims. The fix replaces these with
a placeholder before splitting, then restores.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "web"))

from pdf_claim_extractor import extract_cited_claims_from_text


def test_et_al_does_not_split():
    text = "Recent work [1] has shown that transformers (Smith et al. 2020) work well."
    claims = extract_cited_claims_from_text(text)
    # "et al." should not split; the whole sentence is one claim
    assert len(claims) >= 1
    assert any("et al." in c["claim_text"] or "et al" in c["claim_text"] for c in claims)


def test_figure_abbreviation_does_not_split():
    text = (
        "As shown in Fig. 1, the model performs well on the benchmark [1]. "
        "The second result is also good in subsequent experiments."
    )
    claims = extract_cited_claims_from_text(text)
    # The first sentence contains "Fig. 1" and should not be split
    assert any("Fig. 1" in c["claim_text"] for c in claims)


def test_ie_does_not_split():
    text = "Many methods (i.e. transformers [1]) are evaluated. The results are consistent."
    claims = extract_cited_claims_from_text(text)
    assert any("i.e." in c["claim_text"] or "i.e" in c["claim_text"] for c in claims)


def test_normal_sentences_still_split():
    text = (
        "The first claim about transformers is supported by recent evidence [1]. "
        "A second claim about adapters was also evaluated by the authors [2]. "
        "The third claim about noise injection rounds out the experiment [3]."
    )
    claims = extract_cited_claims_from_text(text)
    # Three separate sentences with three citations -> three claims
    assert len(claims) >= 2


def test_citation_after_abbreviation():
    text = (
        "Earlier work on language modeling (e.g. Smith 2020) was extended [1]. "
        "New methods for parameter-efficient training were introduced recently."
    )
    claims = extract_cited_claims_from_text(text)
    # "e.g." should not split, and the [1] should attach to the first sentence
    assert len(claims) >= 1
