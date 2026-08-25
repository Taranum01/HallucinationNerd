"""
HallucinationNerd Web — Citation Hallucination Verification Service
FastAPI + Uvicorn + Jinja2
"""

import os
import sys
import json
import asyncio
import tempfile
import hashlib
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

load_dotenv()

# The verification engine files (verify_hallucinations.py, arxiv_extractor.py)
# are co-located in this directory for deployment.

app = FastAPI(title="HallucinationNerd", description="Citation Hallucination Verification")
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/verify")
async def verify_document(
    request: Request,
    file: UploadFile = File(...),
    source_type: str = Form(default="auto"),
    databases: str = Form(default=""),
    custom_database: str = Form(default=""),
):
    """
    Accept a document upload, extract claims + citations, verify each one.
    For claims with no inline citation, optionally run a backup search over the
    user-selected databases (comma-separated keys in `databases`, plus an optional
    `custom_database` search-URL template containing '{query}').
    Returns JSON with per-claim results.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Server misconfigured: no API key")

    # Read uploaded file
    content_bytes = await file.read()
    filename = file.filename or "uploaded_file"
    suffix = Path(filename).suffix.lower()

    # Save to temp file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content_bytes)
        tmp_path = tmp.name

    db_list = [d.strip() for d in databases.split(",") if d.strip()]
    try:
        results = await asyncio.to_thread(
            _run_verification, tmp_path, filename, suffix, source_type, db_list, custom_database.strip()
        )
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        os.unlink(tmp_path)


def _extract_clause_for_ref(claim_text: str, ref_key: str) -> Optional[str]:
    """
    For a claim citing many refs like "methods use attention [62,68], adapters [14,23], noise [26,27]",
    extract just the clause relevant to the ref we're verifying against.
    E.g., if ref_key="14", return "specialized adapters" context.
    """
    import re
    # Find where this ref number appears in the claim
    # Look for [X, ref_key, Y] or [ref_key] patterns
    patterns = [
        rf'\[([^\]]*\b{ref_key}\b[^\]]*)\]',  # [14, 23] containing our ref
    ]
    for pattern in patterns:
        match = re.search(pattern, claim_text)
        if match:
            # Get text before this citation bracket (the clause it's attached to)
            pos = match.start()
            # Walk backwards to find clause start (comma, semicolon, or sentence start)
            clause_start = max(0, pos - 150)
            for i in range(pos - 1, max(0, pos - 150), -1):
                if claim_text[i] in ',;.':
                    clause_start = i + 1
                    break
            clause = claim_text[clause_start:match.end()].strip()
            if len(clause) > 20:
                return clause
    return None


def _extract_citation_rich_sections(text: str, max_chars: int = 12000) -> str:
    """
    For full papers, extract the sections that contain citations to other papers
    (typically Related Work, Introduction, Background). These are the sections
    where claims cite external sources that we can verify.
    """
    import re

    if len(text) <= max_chars:
        return text

    # Only count real bracketed citations like [1] or [1, 2, 3]
    citation_pattern = re.compile(r'\[\d+(?:,\s*\d+)*\]')
    if not citation_pattern.findall(text):
        return text[:max_chars]

    # Try to find specific sections by their headers
    # Common section headers in academic papers
    section_patterns = [
        r'(?:^|\n)\s*\d*\.?\s*(?:Related\s+Work|RELATED\s+WORK)',
        r'(?:^|\n)\s*\d*\.?\s*(?:Background|BACKGROUND)',
        r'(?:^|\n)\s*\d*\.?\s*(?:Introduction|INTRODUCTION)',
        r'(?:^|\n)\s*\d*\.?\s*(?:Literature\s+Review|LITERATURE\s+REVIEW)',
        r'(?:^|\n)\s*\d*\.?\s*(?:Previous\s+Work|PREVIOUS\s+WORK)',
    ]

    extracted_sections = []
    for pattern in section_patterns:
        match = re.search(pattern, text)
        if match:
            start = match.start()
            # Find the next section header (a line starting with a number or all-caps word)
            rest = text[start + len(match.group()):]
            next_header = re.search(r'\n\s*\d+\.?\s+[A-Z]|\n\s*[A-Z]{2,}[a-z]', rest)
            end = start + len(match.group()) + (next_header.start() if next_header else min(8000, len(rest)))
            section = text[start:end]
            if len(citation_pattern.findall(section)) >= 3:  # Must have at least 3 citations
                extracted_sections.append(section)

    if extracted_sections:
        combined = '\n\n'.join(extracted_sections)
        return combined[:max_chars]

    # Fallback: just take chunks of text that contain citations
    lines = text.split('\n')
    citation_lines = []
    for i, line in enumerate(lines):
        if citation_pattern.search(line) and len(line) > 60:
            # Include some context (2 lines before, the line itself)
            start_idx = max(0, i - 2)
            context = '\n'.join(lines[start_idx:i+1])
            citation_lines.append(context)

    if citation_lines:
        return '\n\n'.join(citation_lines)[:max_chars]

    return text[:max_chars]


def _verify_one(claim_text: str, source_content: str) -> dict:
    """Verify a single claim against a block of source content (used by the backup
    search for uncited claims). Returns a dict with verdict/confidence/evidence."""
    import re as _re
    import hashlib as _hl
    from verify_hallucinations import verify_citations_for_question, _find_relevant_spans

    clean_text = _re.sub(r'\[\d+(?:,\s*\d+)*\]', '', claim_text).strip()
    chunked = _find_relevant_spans(clean_text, source_content)
    if not chunked or len(chunked) < 50:
        chunked = source_content[:15000]

    out = {"verdict": "UNVERIFIABLE", "confidence": 0.0, "evidence_quote": "", "reasoning": ""}
    verification = verify_citations_for_question(
        entry={
            "question_id": _hl.md5((clean_text + "backup").encode()).hexdigest()[:12],
            "synopsis": f"{clean_text} [1]",
            "retrieved_articles": [{"id": "1", "content": chunked}],
        },
        single_claim=True,
    )
    if verification:
        v = verification[0] if isinstance(verification, list) else verification
        if hasattr(v, "verdict"):
            out.update({"verdict": v.verdict, "confidence": v.confidence,
                        "evidence_quote": v.evidence_quote, "reasoning": v.reasoning})
        else:
            out.update({"verdict": v.get("verdict", "UNVERIFIABLE"),
                        "confidence": v.get("confidence", 0.0),
                        "evidence_quote": v.get("evidence_quote", ""),
                        "reasoning": v.get("reasoning", "")})
    return out


def _run_verification(file_path: str, filename: str, suffix: str, source_type: str,
                      databases: list = None, custom_database: str = "") -> dict:
    """
    Run the HallucinationNerd pipeline on the uploaded file.
    This runs in a thread to not block the event loop.
    `databases` is the list of user-selected backup-search databases (used for
    claims with no inline citation); `custom_database` is an optional user-supplied
    search-URL template.
    """
    databases = databases or []
    from verify_hallucinations import (
        _parse_pdf,
        decompose_claims,
        verify_citations_for_question,
        run_citation_verification,
    )

    # Step 1: Extract text based on file type
    if suffix == ".pdf":
        text = _parse_pdf(file_path)
    elif suffix in (".html", ".htm"):
        from bs4 import BeautifulSoup
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
    elif suffix == ".json":
        with open(file_path, "r") as f:
            data = json.load(f)
        # If JSON matches our input schema, use it directly
        if isinstance(data, list) and data and "synopsis" in data[0]:
            return _verify_structured_input(data)
        elif isinstance(data, dict) and "synopsis" in data:
            return _verify_structured_input([data])
        else:
            text = json.dumps(data, indent=2)
    else:
        # Plain text
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    if not text or len(text.strip()) < 50:
        return {"error": "Could not extract sufficient text from the uploaded file."}

    # Step 2: Extract claims with citations using deterministic regex extraction
    # (more reliable than LLM-based decompose_claims for preserving citation markers)
    from pdf_claim_extractor import extract_cited_claims_from_text, split_multi_ref_claims

    claims = extract_cited_claims_from_text(text)

    # If regex extraction found nothing, fall back to LLM-based decomposition
    if not claims:
        text_for_decomposition = _extract_citation_rich_sections(text)
        claims = decompose_claims(text_for_decomposition)

    if not claims:
        return {"error": "No verifiable claims with citations found in the document."}

    # Step 2.5: Split multi-citation sentences into atomic per-ref claims
    # This matches what the benchmark does (one claim, one source)
    claims = split_multi_ref_claims(claims)

    # Step 2.5: Resolve citations to actual source content
    from citation_resolver import resolve_and_fetch_all

    # Collect all unique cited refs across all claims
    all_cited = set()
    for claim_data in claims:
        for ref in claim_data.get("cited_refs", []):
            all_cited.add(str(ref))

    # Resolve all citations at once (fetches from arXiv, DOI, PubMed, URLs)
    # Use FULL text here (not trimmed) since References section is at the end
    resolved_sources = {}
    if all_cited:
        resolved_sources = resolve_and_fetch_all(text, list(all_cited))

    # Step 3: Verify each claim against its resolved source(s)
    results = []
    for claim_data in claims:
        claim_text = claim_data.get("claim_text", claim_data.get("claim", ""))
        cited_refs = claim_data.get("cited_refs", [])

        result = {
            "claim": claim_text,
            "cited_refs": cited_refs,
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "evidence_quote": "",
            "reasoning": "",
            "citation_exists": None,
        }

        if not cited_refs:
            # No inline citation. If the user enabled backup databases, search them
            # for supporting evidence; otherwise report it as uncited.
            if databases or custom_database:
                from citation_resolver import backup_search
                bcontent, bsource = backup_search(claim_text, databases, custom_database)
                if bcontent:
                    bv = _verify_one(claim_text, bcontent)
                    if bv["verdict"] in ("SUPPORTED", "PARTIALLY_SUPPORTED"):
                        result["verdict"] = "BACKUP_FOUND"
                        result["backup_source"] = bsource
                        result["confidence"] = bv["confidence"]
                        result["evidence_quote"] = bv["evidence_quote"]
                        result["reasoning"] = (
                            f"No inline citation. Supporting evidence found via {bsource} "
                            f"(verdict against that source: {bv['verdict']})."
                        )
                    else:
                        result["verdict"] = "NO_BACKUP_FOUND"
                        result["backup_source"] = bsource
                        result["reasoning"] = (
                            f"No inline citation. Searched {bsource}, but the top candidate "
                            f"did not support the claim ({bv['verdict']})."
                        )
                else:
                    result["verdict"] = "NO_BACKUP_FOUND"
                    searched = ", ".join(databases) if databases else "custom database"
                    result["reasoning"] = (
                        f"No inline citation. No supporting source found in the selected "
                        f"database(s): {searched}."
                    )
            else:
                result["reasoning"] = "No citation provided for this claim."
            results.append(result)
            continue

        # Try ALL cited refs until we find one we can access (not just the first)
        source_content = None
        ref_key = None
        for ref in cited_refs:
            rk = str(ref)
            content = resolved_sources.get(rk)
            if content and len(content) > 100:
                source_content = content
                ref_key = rk
                break

        if not source_content:
            result["reasoning"] = f"Could not access any source for references [{', '.join(str(r) for r in cited_refs)}]. The cited sources may be behind a paywall, unavailable, or could not be resolved."
            result["citation_exists"] = False
            results.append(result)
            continue

        # We have source content — verify the claim against it
        result["citation_exists"] = True

        # For claims with many refs, extract just the clause relevant to the ref we're checking
        # This makes verification more precise (full sentence might cover multiple papers)
        verification_text = claim_text
        if len(cited_refs) > 3:
            # Try to extract the specific clause near this ref's mention
            clause = _extract_clause_for_ref(claim_text, ref_key)
            if clause:
                verification_text = clause

        # The verification engine uses positional indexing (ref [1] = articles[0]).
        # Strip original citation markers and map to [1] since we provide exactly one article.
        # Use keyword-overlap chunking to find the relevant passage (same as benchmark)
        import re as _re
        from verify_hallucinations import _find_relevant_spans
        clean_text = _re.sub(r'\[\d+(?:,\s*\d+)*\]', '', verification_text).strip()
        
        # Chunk the source to the most relevant passage (same technique as benchmark)
        chunked_content = _find_relevant_spans(clean_text, source_content)
        if not chunked_content or len(chunked_content) < 50:
            chunked_content = source_content[:15000]
        
        synopsis_with_ref = f"{clean_text} [1]"
        verification = verify_citations_for_question(
            entry={
                "question_id": hashlib.md5(claim_text.encode()).hexdigest()[:12],
                "synopsis": synopsis_with_ref,
                "retrieved_articles": [{"id": "1", "content": chunked_content}],
            },
            single_claim=True,
        )
        if verification:
            v = verification[0] if isinstance(verification, list) else verification
            if hasattr(v, 'verdict'):
                result.update({
                    "verdict": v.verdict,
                    "confidence": v.confidence,
                    "evidence_quote": v.evidence_quote,
                    "reasoning": v.reasoning,
                })
            else:
                result.update({
                    "verdict": v.get("verdict", "UNVERIFIABLE"),
                    "confidence": v.get("confidence", 0.0),
                    "evidence_quote": v.get("evidence_quote", ""),
                    "reasoning": v.get("reasoning", ""),
                })

        results.append(result)

    # Step 4: Compute summary
    total = len(results)
    supported = sum(1 for r in results if r["verdict"] == "SUPPORTED")
    partial = sum(1 for r in results if r["verdict"] == "PARTIALLY_SUPPORTED")
    not_supported = sum(1 for r in results if r["verdict"] == "NOT_SUPPORTED")
    contradicted = sum(1 for r in results if r["verdict"] == "CONTRADICTED")
    unverifiable = sum(1 for r in results if r["verdict"] == "UNVERIFIABLE")
    backup_found = sum(1 for r in results if r["verdict"] == "BACKUP_FOUND")
    no_backup_found = sum(1 for r in results if r["verdict"] == "NO_BACKUP_FOUND")

    # Verifiable = everything we could actually assess (backup outcomes count).
    verifiable = supported + partial + not_supported + contradicted + backup_found + no_backup_found
    reliability_pct = (supported + partial + backup_found) / verifiable * 100 if verifiable > 0 else 0

    return {
        "filename": filename,
        "summary": {
            "total_claims": total,
            "supported": supported,
            "partially_supported": partial,
            "not_supported": not_supported,
            "contradicted": contradicted,
            "unverifiable": unverifiable,
            "backup_found": backup_found,
            "no_backup_found": no_backup_found,
            "reliability_percent": round(reliability_pct, 1),
        },
        "claims": results,
    }


def _verify_structured_input(data: list) -> dict:
    """Handle JSON input that matches our structured schema."""
    from verify_hallucinations import verify_citations_for_question

    all_results = []
    for entry in data:
        try:
            results = verify_citations_for_question(entry, single_claim=True)
            if results:
                for r in results:
                    # Handle both dataclass objects and dicts
                    if hasattr(r, 'claim_text'):
                        all_results.append({
                            "claim": r.claim_text,
                            "cited_refs": r.cited_refs,
                            "verdict": r.verdict,
                            "confidence": r.confidence,
                            "evidence_quote": r.evidence_quote,
                            "reasoning": r.reasoning,
                            "citation_exists": True,
                        })
                    else:
                        all_results.append({
                            "claim": r.get('claim_text', ''),
                            "cited_refs": r.get('cited_refs', []),
                            "verdict": r.get('verdict', 'UNVERIFIABLE'),
                            "confidence": r.get('confidence', 0.0),
                            "evidence_quote": r.get('evidence_quote', ''),
                            "reasoning": r.get('reasoning', ''),
                            "citation_exists": True,
                        })
        except Exception as e:
            all_results.append({
                "claim": entry.get("synopsis", ""),
                "cited_refs": [],
                "verdict": "UNVERIFIABLE",
                "confidence": 0.0,
                "evidence_quote": "",
                "reasoning": f"Processing error: {str(e)}",
            })

    total = len(all_results)
    supported = sum(1 for r in all_results if r["verdict"] == "SUPPORTED")
    partial = sum(1 for r in all_results if r["verdict"] == "PARTIALLY_SUPPORTED")
    not_supported = sum(1 for r in all_results if r["verdict"] == "NOT_SUPPORTED")
    contradicted = sum(1 for r in all_results if r["verdict"] == "CONTRADICTED")
    unverifiable = sum(1 for r in all_results if r["verdict"] == "UNVERIFIABLE")

    verifiable = total - unverifiable
    reliability_pct = (supported + partial) / verifiable * 100 if verifiable > 0 else 0

    return {
        "filename": "structured_input.json",
        "summary": {
            "total_claims": total,
            "supported": supported,
            "partially_supported": partial,
            "not_supported": not_supported,
            "contradicted": contradicted,
            "unverifiable": unverifiable,
            "reliability_percent": round(reliability_pct, 1),
        },
        "claims": all_results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
