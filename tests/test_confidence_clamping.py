"""Tests for C11: confidence clamping helper.

LLM-reported confidence is a 0-1 float but the model can return values
outside the range, NaN, or arbitrary precision. The clamp helper normalizes
these to a safe [0, 1] rounded value.
"""
import math
import pytest

from verify_hallucinations import _clamp_confidence


def test_clamp_keeps_in_range_value():
    assert _clamp_confidence(0.5) == 0.5
    assert _clamp_confidence(0.0) == 0.0
    assert _clamp_confidence(1.0) == 1.0


def test_clamp_rounds_to_4_decimals():
    assert _clamp_confidence(0.123456789) == 0.1235
    assert _clamp_confidence(0.99999) == 1.0  # rounds up to 1.0
    assert _clamp_confidence(0.99994) == 0.9999  # rounds down


def test_clamp_clamps_negative():
    assert _clamp_confidence(-0.5) == 0.0
    assert _clamp_confidence(-100) == 0.0


def test_clamp_clamps_above_one():
    assert _clamp_confidence(1.5) == 1.0
    assert _clamp_confidence(2.0) == 1.0
    assert _clamp_confidence(100) == 1.0


def test_clamp_handles_nan():
    assert _clamp_confidence(float("nan"), default=0.5) == 0.5
    assert _clamp_confidence(float("nan"), default=0.7) == 0.7


def test_clamp_handles_infinity():
    assert _clamp_confidence(float("inf")) == 1.0
    assert _clamp_confidence(float("-inf")) == 0.0


def test_clamp_handles_invalid_input():
    assert _clamp_confidence("not a number", default=0.5) == 0.5
    assert _clamp_confidence(None, default=0.5) == 0.5
    assert _clamp_confidence([1, 2, 3], default=0.5) == 0.5


def test_clamp_default_when_missing():
    """When the value is None or invalid, fall back to default."""
    assert _clamp_confidence(None) == 0.5  # default
    assert _clamp_confidence(None, default=0.3) == 0.3
