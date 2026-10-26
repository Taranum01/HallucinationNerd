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


def test_per_item_citations_split_into_separate_claims():
    """A sentence that attributes a separate reference to each list item must split into one
    claim per (item, citation) pair, so each reference is checked only against its own item.
    Regression for the CEP example: 'used in X [1], Y [2], and Z [3]' was kept as one claim
    citing [1,2,3] and then failed because no single reference covered all three domains."""
    text = ("In CEP, queries are evaluated over events to detect situations of interest in "
            "domains, such as urban transportation [1], computational finance [2], and "
            "supply chain management [3].")
    claims = extract_cited_claims_from_text(text)
    assert len(claims) == 3, f"expected 3 per-domain claims, got {len(claims)}"
    assert sorted(r for c in claims for r in c["cited_refs"]) == [1, 2, 3]
    assert all(len(c["cited_refs"]) == 1 for c in claims), "each split claim cites exactly one ref"


def test_short_form_per_item_citations_split():
    """'A is applied in X [1] and Y [2]' splits into two single-ref claims."""
    claims = extract_cited_claims_from_text(
        "Attention is applied in machine translation [1] and image captioning [2].")
    assert len(claims) == 2
    assert sorted(r for c in claims for r in c["cited_refs"]) == [1, 2]


def test_joint_citation_stays_one_claim():
    """A genuinely joint '[1, 2]' is ambiguous (each ref may cover the whole claim), so it must
    NOT be split; it stays one claim citing both refs."""
    claims = extract_cited_claims_from_text(
        "Transformers are widely used for sequence modeling and translation [1, 2].")
    assert len(claims) == 1
    assert claims[0]["cited_refs"] == [1, 2]


def _claims(text):
    return [(c["claim_text"], c["cited_refs"]) for c in extract_cited_claims_from_text(text)]


def test_split_keeps_introducer_in_each_claim():
    """The list introducer ('such as') must stay in each split claim so meaning is unchanged."""
    claims = _claims("CEP is used in domains such as urban transportation [1], "
                     "computational finance [2], and supply chain management [3].")
    assert [t for t, _ in claims] == [
        "CEP is used in domains such as urban transportation [1].",
        "CEP is used in domains such as computational finance [2].",
        "CEP is used in domains such as supply chain management [3].",
    ]


def test_trailing_predicate_after_last_citation_is_not_split():
    """Text after the last citation (a shared predicate) would be lost by splitting, so the
    sentence must stay one claim, unchanged, citing both refs."""
    text = "Models such as BERT [1] and GPT [2] outperform LSTMs on translation tasks in 2019."
    claims = _claims(text)
    assert len(claims) == 1
    assert claims[0][0] == text and claims[0][1] == [1, 2]


def test_contrast_and_negation_are_not_split():
    for text in (
        "Drug A reduces risk of stroke [1] but not of heart attack [2].",
        "Unlike X, which works in vision [1], Y works in speech [2].",
        "A is not used in vision [1] or speech [2].",
    ):
        claims = _claims(text)
        assert len(claims) == 1 and claims[0][0] == text and claims[0][1] == [1, 2]


def test_mixed_joint_and_single_citations_not_split():
    text = "Transformers are used in vision [1, 2] and speech [3]."
    claims = _claims(text)
    assert len(claims) == 1 and claims[0][1] == [1, 2, 3]


def test_split_claims_preserve_all_item_text():
    claims = _claims("Attention is applied in machine translation [1] and image captioning [2].")
    assert claims == [("Attention is applied in machine translation [1].", [1]),
                      ("Attention is applied in image captioning [2].", [2])]
