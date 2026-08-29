"""Tests for C4: citation_resolver cache key collision fix.

The previous key was the first 100 chars of the parsed `raw` text, which
caused collisions when two references had similar opening text (common
for repeated author names or formatting). The new key prefers canonical
identifiers (arxiv_id > doi > pmid) and falls back to a hash of the title.
"""
import sys
import os
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "web"))


def test_same_arxiv_id_same_key():
    from citation_resolver import _make_cache_key
    a = _make_cache_key({"arxiv_id": "2305.14314", "title": "Foo"}, "1")
    b = _make_cache_key({"arxiv_id": "2305.14314", "title": "Bar"}, "2")  # different title
    assert a == b  # arxiv_id is canonical


def test_different_arxiv_id_different_key():
    from citation_resolver import _make_cache_key
    a = _make_cache_key({"arxiv_id": "2305.14314"}, "1")
    b = _make_cache_key({"arxiv_id": "2305.99999"}, "2")
    assert a != b


def test_doi_canonical():
    from citation_resolver import _make_cache_key
    a = _make_cache_key({"doi": "10.1234/abc.123"}, "1")
    b = _make_cache_key({"doi": "10.1234/xyz.999"}, "2")
    assert a != b
    # Case-insensitive
    assert a == _make_cache_key({"doi": "10.1234/ABC.123"}, "3")


def test_pmid_canonical():
    from citation_resolver import _make_cache_key
    a = _make_cache_key({"pmid": "12345"}, "1")
    b = _make_cache_key({"pmid": "99999"}, "2")
    assert a != b


def test_arxiv_takes_precedence_over_doi():
    from citation_resolver import _make_cache_key
    a = _make_cache_key({"arxiv_id": "2305.14314", "doi": "10/x"}, "1")
    b = _make_cache_key({"arxiv_id": "2305.14314", "doi": "10/y"}, "2")
    assert a == b  # arxiv_id wins


def test_title_fallback_hash():
    """When no arxiv/doi/pmid, the key is a hash of the title."""
    from citation_resolver import _make_cache_key
    a = _make_cache_key({"title": "Attention is all you need"}, "1")
    b = _make_cache_key({"title": "Attention is all you need"}, "2")
    assert a == b
    c = _make_cache_key({"title": "A different paper"}, "3")
    assert a != c


def test_keys_no_longer_depend_on_raw_text():
    """The old key depended on the first 100 chars of raw text. Verify that's gone."""
    from citation_resolver import _make_cache_key
    # Two refs with completely different raw text but same arxiv_id should collide
    raw1 = "X. Author. Some title. Venue 2020. arXiv:2305.14314..."
    raw2 = "Y. Other-author. Different title. Other Venue 2021. arXiv:2305.14314..."
    a = _make_cache_key({"arxiv_id": "2305.14314", "raw": raw1}, "1")
    b = _make_cache_key({"arxiv_id": "2305.14314", "raw": raw2}, "2")
    assert a == b


def test_completely_empty_ref_uses_fallback():
    from citation_resolver import _make_cache_key
    # No info at all: title falls back to ref_key, then gets hashed
    k = _make_cache_key({}, "1")
    # Falls back to hashing the ref_key as the title
    assert k.startswith("title:") or k.startswith("raw:")


def test_ref_key_only_uses_ref_key_as_fallback():
    """No canonical ids and no title: hash the ref_key itself."""
    from citation_resolver import _make_cache_key
    a = _make_cache_key({}, "Q42")
    b = _make_cache_key({}, "Q42")  # same ref_key -> same hash
    assert a == b
    c = _make_cache_key({}, "Q99")
    assert a != c
