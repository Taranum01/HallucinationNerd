"""Tests for L12: citation_resolver cache TTL.

The previous cache had no expiry — stale entries (e.g. arXiv papers
with revised versions) could stick around for the server's lifetime.
The new cache is keyed by (timestamp, content) and evicts entries
older than `_CACHE_TTL_SECONDS` (default 1 hour, override via HVE_CACHE_TTL).
"""
import os
import sys
import time
import importlib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "web"))


def test_cache_set_and_get():
    import citation_resolver as cr
    importlib.reload(cr)
    cr._source_cache["k1"] = (time.time(), "content for k1")
    cached = cr._source_cache.get("k1")
    assert cached is not None
    assert cached[1] == "content for k1"


def test_cache_evicts_expired_entries(monkeypatch):
    import citation_resolver as cr
    importlib.reload(cr)
    # Insert a very old entry
    cr._source_cache["old"] = (time.time() - 100000, "stale content")
    cr._source_cache["fresh"] = (time.time(), "fresh content")

    # Call resolve_and_fetch_all with no cited refs to trigger GC
    # (the GC runs at the top of resolve_and_fetch_all)
    cr.resolve_and_fetch_all("dummy text", [])
    # The old entry should be evicted, the fresh one should remain
    assert "old" not in cr._source_cache
    assert "fresh" in cr._source_cache


def test_cache_ttl_override(monkeypatch):
    monkeypatch.setenv("HVE_CACHE_TTL", "60")  # 1 minute
    import citation_resolver as cr
    importlib.reload(cr)
    # Insert an entry 100 seconds old
    cr._source_cache["recent"] = (time.time() - 100, "data")
    cr.resolve_and_fetch_all("dummy", [])
    # 100 seconds > 60 second TTL -> evicted
    assert "recent" not in cr._source_cache
