"""Re-export shim for backward compatibility.

The canonical engine is the root `verify_hallucinations.py`. The web app
imports from this module so existing deployment configs that reference
`web/verify_hallucinations` keep working, but all behavior is the root
engine's.

Phase 3 of the code-review plan: the previous fork at this path diverged
(dropped --strictness, simplified metrics). The canonical version now lives
in the root; this file is a thin re-export.
"""
from verify_hallucinations import *  # noqa: F401,F403

# `import *` skips underscore-prefixed names, but the web app imports these
# private helpers explicitly, so re-export them so uploads don't fail.
from verify_hallucinations import _parse_pdf, _find_relevant_spans  # noqa: F401
