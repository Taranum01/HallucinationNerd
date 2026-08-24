"""Shared pytest fixtures for the HVE test suite.

Mock OpenAI client, sample inputs, dataset fixtures. The mock is deterministic
per call (a stable hash of the input) so tests can be reproducible.
"""
import json
import os
import sys
import hashlib
import pytest
from pathlib import Path

# Make the repo root importable so tests can `from verify_hallucinations import ...`
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ─── OpenAI mock ────────────────────────────────────────────────────────────

class _FakeChoice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """Stable, deterministic mock of openai.OpenAI().chat.completions.create.

    The verdict depends on a stable hash of the system+user prompts so
    tests can assert against it. Override via the `mock_verdict` side-effect
    on the fixture.
    """

    def __init__(self):
        self.next_verdict = None  # optional override
        self.next_json = None  # optional raw JSON
        self.call_count = 0
        self.calls = []  # for assertion

    def create(self, model, messages, temperature=None, response_format=None, **kwargs):
        from verify_hallucinations import CONFIG
        self.call_count += 1
        sys_prompt = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_prompt = next((m["content"] for m in messages if m["role"] == "user"), "")
        self.calls.append({"model": model, "sys": sys_prompt, "user": user_prompt, "kwargs": kwargs})

        if self.next_json is not None:
            content = self.next_json
            self.next_json = None  # consumed
        elif self.next_verdict is not None:
            content = json.dumps(self.next_verdict)
            self.next_verdict = None  # consumed
        else:
            # Deterministic by content hash
            content = self._default_response(sys_prompt, user_prompt)

        # rate_limit_delay is a real sleep, so swap it out for tests
        import verify_hallucinations as vh
        original = vh.CONFIG.get("rate_limit_delay", 1.0)
        vh.CONFIG["rate_limit_delay"] = 0.0
        try:
            return _FakeResponse(content)
        finally:
            vh.CONFIG["rate_limit_delay"] = original

    def _default_response(self, sys_prompt, user_prompt):
        # Hash the user prompt; map the first byte to a verdict.
        h = hashlib.sha256(user_prompt.encode("utf-8", errors="ignore")).digest()
        bucket = h[0] % 4
        verdict = ["SUPPORTED", "PARTIALLY_SUPPORTED", "NOT_SUPPORTED", "UNVERIFIABLE"][bucket]
        return json.dumps({
            "verdict": verdict,
            "confidence": 0.5 + (h[1] / 510.0),  # 0.5..0.999 deterministic
            "evidence_quote": "",
            "evidence_start_phrase": "",
            "reasoning": "mock response",
        })


@pytest.fixture
def mock_openai(monkeypatch):
    """Patch verify_hallucinations.get_client so tests never hit OpenAI."""
    fake = _FakeCompletions()
    import verify_hallucinations as vh

    class _Client:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": fake})()

    monkeypatch.setattr(vh, "get_client", lambda: _Client())
    # Reset the module-level singleton so the patch takes effect
    monkeypatch.setattr(vh, "client", None, raising=False)
    return fake


# ─── Sample inputs ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_short_text():
    """A 2-paragraph text with numbered citations [1] and [2]."""
    return (
        "Recent work has shown that transformers [1] achieve strong results on "
        "language modeling. The original architecture [2] was proposed in 2017.\n\n"
        "Subsequent work extended these ideas to many domains."
    )


@pytest.fixture
def sample_multi_ref_claim():
    """A claim citing 5 refs in one sentence — the core regression case."""
    return (
        "Transformers [62, 68] use self-attention, while adapters [14, 23] and "
        "noise injection [26, 27] are common parameter-efficient techniques."
    )


@pytest.fixture
def sample_paywalled_claim():
    """A claim with a citation that should fail to resolve (fake DOI)."""
    return "We propose a method that improves accuracy by 30% [1]."


@pytest.fixture
def sample_pdf_extracted_text():
    """A paper body with references section, used by pdf_claim_extractor."""
    return (
        "Transformers have revolutionized NLP. The original paper [1] introduced "
        "the architecture, and subsequent work has scaled it [2, 3].\n\n"
        "More recent work explores efficient variants [4].\n\n"
        "References\n\n"
        "[1] Vaswani et al. Attention is all you need. NeurIPS 2017.\n"
        "[2] Devlin et al. BERT. NAACL 2019.\n"
        "[3] Brown et al. GPT-3. 2020.\n"
        "[4] Hu et al. LoRA. 2021.\n"
    )


# ─── Datasets ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def crosscat_input():
    p = REPO_ROOT / "datasets" / "crosscat_input.json"
    if not p.exists():
        pytest.skip(f"dataset not found: {p}")
    return json.loads(p.read_text())


@pytest.fixture(scope="session")
def crosscat_gt():
    p = REPO_ROOT / "datasets" / "crosscat_gt.json"
    if not p.exists():
        pytest.skip(f"dataset not found: {p}")
    return json.loads(p.read_text())


@pytest.fixture(scope="session")
def samefield_input():
    p = REPO_ROOT / "datasets" / "samefield_input.json"
    if not p.exists():
        pytest.skip(f"dataset not found: {p}")
    return json.loads(p.read_text())


@pytest.fixture(scope="session")
def samefield_gt():
    p = REPO_ROOT / "datasets" / "samefield_gt.json"
    if not p.exists():
        pytest.skip(f"dataset not found: {p}")
    return json.loads(p.read_text())


# ─── Test helpers ───────────────────────────────────────────────────────────

@pytest.fixture
def tmp_result_dir(tmp_path):
    """A per-test output directory for citation_verification runs."""
    d = tmp_path / "results"
    d.mkdir()
    return d
