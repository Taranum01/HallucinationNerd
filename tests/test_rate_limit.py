"""Tests for M19: per-IP rate limiting on /verify.

The rate limiter is a simple in-memory token bucket per IP. Default is
5 requests per 60s, override with HVE_RATE_LIMIT_REQUESTS and
HVE_RATE_LIMIT_WINDOW_SECONDS.
"""
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "web"))


def test_rate_limit_allows_under_limit():
    """Under the limit, all requests are allowed."""
    import importlib
    import web.app as web_app
    importlib.reload(web_app)
    # Set a fresh window for this test
    web_app._rate_buckets.clear()
    for _ in range(5):
        assert web_app._check_rate_limit("192.168.1.1") is True


def test_rate_limit_blocks_over_limit():
    """Over the limit, requests are blocked."""
    import importlib
    import web.app as web_app
    importlib.reload(web_app)
    web_app._rate_buckets.clear()
    for _ in range(5):
        assert web_app._check_rate_limit("10.0.0.1") is True
    # 6th request blocked
    assert web_app._check_rate_limit("10.0.0.1") is False


def test_rate_limit_per_ip():
    """Each IP has its own bucket."""
    import importlib
    import web.app as web_app
    importlib.reload(web_app)
    web_app._rate_buckets.clear()
    for _ in range(5):
        web_app._check_rate_limit("ip-a")
    # ip-b should still be allowed
    assert web_app._check_rate_limit("ip-b") is True
    # ip-a should be blocked
    assert web_app._check_rate_limit("ip-a") is False


def test_rate_limit_window_expiry():
    """After the window expires, requests are allowed again."""
    import importlib
    import web.app as web_app
    importlib.reload(web_app)
    # Set a 1-second window for this test
    web_app._rate_limit_window = 1
    web_app._rate_buckets.clear()
    for _ in range(5):
        assert web_app._check_rate_limit("ip-c") is True
    assert web_app._check_rate_limit("ip-c") is False
    # Wait for the window to expire
    time.sleep(1.1)
    assert web_app._check_rate_limit("ip-c") is True
