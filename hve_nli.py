"""Optional NLI pre-filter for citation verification.

This is the HLD's "Layer 2" NLI gate. When enabled, a fast NLI model
(DeBERTa-v3-large fine-tuned on NLI) classifies each (claim, source)
pair BEFORE the LLM judge. High-confidence SUPPORTED and NOT_SUPPORTED
labels short-circuit the LLM call, saving cost and latency. PARTIALLY
and UNVERIFIABLE always escalate.

The model is OPTIONAL — by default this module exposes a deterministic
stub (returns NEUTRAL with score 0.0, so nothing ever short-circuits).
To enable real NLI:
  1. pip install transformers torch
  2. Set HVE_NLI_ENABLED=1
  3. The module will download ~700MB of model weights on first call

Why optional: the model is large, the install adds a torch dep, and
the benchmark / CLI default-off keeps the existing 89.5% / 89.3% numbers
reproducible without the NLI gate adding new labels.
"""
import os
from typing import Optional, Tuple


# Standard NLI label set
ENTAILMENT = "ENTAILMENT"      # -> SUPPORTED
CONTRADICTION = "CONTRADICTION"  # -> NOT_SUPPORTED
NEUTRAL = "NEUTRAL"            # -> ambiguous, escalate to LLM

# Thresholds for short-circuit. Below these confidence levels, escalate to LLM.
SHORT_CIRCUIT_SUPPORTED_THRESHOLD = 0.85
SHORT_CIRCUIT_NOT_SUPPORTED_THRESHOLD = 0.85


_stub_state = {
    "nli_calls": 0,
    "llm_calls_avoided": 0,
}


def is_nli_enabled() -> bool:
    """Whether real NLI is enabled (env var + deps installed)."""
    if os.getenv("HVE_NLI_ENABLED", "0") != "1":
        return False
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _load_model():
    """Lazy-load the DeBERTa NLI model. Returns (model, tokenizer) or raises."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    # microsoft/deberta-v2-xxlarge-mnli is the canonical NLI model. We
    # use a smaller variant for speed; the model name is configurable via
    # env var for users who want to swap in a custom fine-tune.
    model_name = os.getenv("HVE_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


# Lazy-loaded singleton
_model = None
_tokenizer = None


def _get_model():
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = _load_model()
    return _model, _tokenizer


def predict_stub(claim: str, source: str) -> Tuple[str, float]:
    """Deterministic stub used when NLI is disabled or unavailable.

    Returns NEUTRAL with 0.0 confidence so the per-ref verifier always
    escalates to the LLM. This preserves the existing behavior.
    """
    _stub_state["nli_calls"] += 1
    return NEUTRAL, 0.0


def predict(claim: str, source: str) -> Tuple[str, float]:
    """Run NLI on (claim, source). Returns (label, confidence).

    label is one of ENTAILMENT / CONTRADICTION / NEUTRAL.
    confidence is the model's softmax probability for the returned label.
    """
    if not is_nli_enabled():
        return predict_stub(claim, source)

    model, tokenizer = _get_model()
    import torch
    _stub_state["nli_calls"] += 1

    inputs = tokenizer(
        source, claim,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    # The MoritzaLaurer/DeBERTa-v3-base-mnli-fever-anli model uses:
    # 0=ENTAILMENT, 1=NEUTRAL, 2=CONTRADICTION (verify in model.config.id2label)
    id2label = model.config.id2label
    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    label = id2label[best_idx]
    confidence = probs[best_idx]

    # Normalize to our 3-label set
    label_upper = label.upper()
    if "ENTAIL" in label_upper:
        return ENTAILMENT, confidence
    if "CONTRADICT" in label_upper:
        return CONTRADICTION, confidence
    return NEUTRAL, confidence


def maybe_short_circuit(claim: str, source: str) -> Optional[Tuple[str, float, str]]:
    """If NLI is confident, return (mapped_verdict, confidence, evidence_quote)
    without calling the LLM. Otherwise return None to signal escalation.

    Mapped verdicts use the engine's vocabulary: SUPPORTED, NOT_SUPPORTED.
    NEUTRAL maps to None (escalate).
    """
    label, confidence = predict(claim, source)
    if label == ENTAILMENT and confidence >= SHORT_CIRCUIT_SUPPORTED_THRESHOLD:
        _stub_state["llm_calls_avoided"] += 1
        return ("SUPPORTED", confidence, "")  # no quote — would need a span-finder
    if label == CONTRADICTION and confidence >= SHORT_CIRCUIT_NOT_SUPPORTED_THRESHOLD:
        _stub_state["llm_calls_avoided"] += 1
        return ("NOT_SUPPORTED", confidence, "")
    return None


def get_stats() -> dict:
    """Stats on NLI usage: total calls, LLM calls avoided."""
    return dict(_stub_state)
