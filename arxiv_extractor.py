"""
Generic ArXiv Citation Extractor
=================================
Extracts (claim, cited paper) pairs from any arXiv paper's Related Work section.

Supports ALL common academic citation formats:
1. (Author et al., 2020)     — parenthetical author-year (ACL/NeurIPS)
2. [N] or [N, M]             — numbered brackets (IEEE/CVPR)
3. [Author et al., 2020]     — bracketed author-year
4. Author et al. (2020)      — narrative author-year
5. (N) or (N, M)             — parenthetical numbers

Usage:
    from arxiv_extractor import extract_citations_from_pdf
    claims = extract_citations_from_pdf("paper.pdf", section="related work")
    # Returns: [{"clause": "...", "arxiv_id": "2305.14314", ...}, ...]
"""

import re
import fitz  # PyMuPDF


def detect_citation_format(text):
    """Detect citation format(s) used in text. Returns sorted by count (most common first)."""
    formats = []
    
    # Format 1: (Author et al., Year)
    f1 = re.findall(r'\(([A-Z][\w\'\-]+(?:\s*et al\.)?),?\s*\d{4}[a-z]?\)', text)
    if f1:
        formats.append(("author_year_paren", len(f1)))
    
    # Format 3: [Author et al., Year]
    f3 = re.findall(r'\[([A-Z][\w\'\-]+(?:\s*et al\.)?),?\s*\d{4}[a-z]?\]', text)
    if f3:
        formats.append(("author_year_bracket", len(f3)))
    
    # Format 4: Author et al. (Year) — narrative
    f4 = re.findall(r'([A-Z][\w\'\-]+(?:\s*et al\.)?)\s*\(\d{4}[a-z]?\)', text)
    if f4:
        formats.append(("author_year_narrative", len(f4)))
    
    # Format 2: [N] or [N, M] — only count if no author-year brackets
    if not f3:
        f2 = re.findall(r'\[(\d+(?:,\s*\d+)*)\]', text)
        if f2:
            formats.append(("numbered_bracket", len(f2)))
    
    # Format 5: (N) — only if no author-year parens and numbers are small
    if not f1 and not f4:
        f5 = re.findall(r'\((\d+(?:,\s*\d+)*)\)', text)
        if f5:
            formats.append(("numbered_paren", len(f5)))
    
    formats.sort(key=lambda x: x[1], reverse=True)
    return formats


def extract_citations_from_pdf(pdf_path, section="related work"):
    """
    Extract (claim, arxiv_id) pairs from a paper's specified section.
    
    Args:
        pdf_path: Path to PDF file
        section: Section to extract from (default: "related work")
    
    Returns:
        List of dicts with keys: clause, arxiv_id, and format-specific keys
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    
    # Find section
    section_start = text.lower().find(section)
    if section_start == -1 and section == "related work":
        section_start = text.lower().find("background")
    if section_start == -1:
        return []
    
    # Find end of section
    text_after = text[section_start + 50:]
    next_section = re.search(r'\n\d+\.?\s+[A-Z]', text_after)
    section_end = section_start + 50 + (next_section.start() if next_section else 5000)
    section_text = text[section_start:section_end].replace('\n', ' ')
    
    # Get bibliography
    ref_start = text.lower().rfind("\nreferences")
    bibliography = text[ref_start:] if ref_start != -1 else ""
    
    # Detect format
    formats = detect_citation_format(section_text)
    if not formats:
        return []
    
    primary = formats[0][0]
    
    # ---- NUMBERED FORMATS ----
    if primary in ("numbered_bracket", "numbered_paren"):
        return _extract_numbered(section_text, bibliography, primary)
    
    # ---- AUTHOR-YEAR FORMATS ----
    return _extract_author_year(section_text, bibliography, primary)


def _extract_numbered(section_text, bibliography, fmt):
    """Extract from numbered citation formats [N] or (N)."""
    if fmt == "numbered_bracket":
        num_pattern = r'\[([\d,\s]+)\]'
        cite_pattern = r'\[[\d,\s]+\]'
    else:
        num_pattern = r'\(([\d,\s]+)\)'
        cite_pattern = r'\([\d,\s]+\)'
    
    # Parse numbered bibliography
    bib_entries = {}
    for m in re.finditer(r'\[(\d+)\]\s*(.+?)(?=\[\d+\]|\Z)', bibliography, re.DOTALL):
        bib_entries[int(m.group(1))] = m.group(2).strip().replace('\n', ' ')
    
    # Fallback: "1. Entry" format
    if not bib_entries:
        for m in re.finditer(r'(\d+)\.\s+(.+?)(?=\d+\.\s|\Z)', bibliography, re.DOTALL):
            bib_entries[int(m.group(1))] = m.group(2).strip().replace('\n', ' ')
    
    results = []
    seen_nums = set()
    
    for m in re.finditer(num_pattern, section_text):
        nums = [int(x.strip()) for x in m.group(1).split(',')]
        clause = _get_clause_before(section_text, m.start(), cite_pattern)
        
        if len(clause) < 20:
            continue
        
        for num in nums[:1]:
            if num in seen_nums or num not in bib_entries:
                continue
            seen_nums.add(num)
            
            arxiv_id = re.search(r'(\d{4}\.\d{4,5})', bib_entries[num])
            if arxiv_id:
                results.append({
                    "clause": clause,
                    "arxiv_id": arxiv_id.group(1),
                    "cite_num": num,
                    "bib_entry": bib_entries[num][:100],
                    "format": "numbered",
                })
    
    return results


def _extract_author_year(section_text, bibliography, fmt):
    """Extract from author-year citation formats."""
    if fmt == "author_year_paren":
        pattern = r'\(([A-Z][\w\'\-]+(?:\s*et al\.)?),?\s*(\d{4})[a-z]?\)'
        cite_pattern = r'\([A-Z][\w\'\-]+.*?\d{4}[a-z]?\)'
    elif fmt == "author_year_bracket":
        pattern = r'\[([A-Z][\w\'\-]+(?:\s*et al\.)?),?\s*(\d{4})[a-z]?\]'
        cite_pattern = r'\[[A-Z][\w\'\-]+.*?\d{4}[a-z]?\]'
    elif fmt == "author_year_narrative":
        pattern = r'([A-Z][\w\'\-]+(?:\s*et al\.)?)\s*\((\d{4})[a-z]?\)'
        cite_pattern = r'[A-Z][\w\'\-]+(?:\s*et al\.)?\s*\(\d{4}[a-z]?\)'
    
    bib_entries = re.split(r'\n(?=[A-Z][\w\'\-]+)', bibliography[20:])
    
    results = []
    seen = set()
    
    for m in re.finditer(pattern, section_text):
        author = m.group(1).replace(' et al.', '').strip()
        year = m.group(2)
        
        if (author, year) in seen:
            continue
        seen.add((author, year))
        
        # Get clause
        if fmt == "author_year_narrative":
            # For narrative, the claim comes AFTER the citation
            start_pos = m.end()
            next_cite = re.search(cite_pattern, section_text[start_pos:])
            sent_end = section_text.find('.', start_pos)
            end_pos = min(
                start_pos + (next_cite.start() if next_cite else 300),
                sent_end + 1 if sent_end != -1 else start_pos + 300
            )
            clause = f"{m.group(0)} {section_text[start_pos:end_pos].strip()}"
        else:
            clause = _get_clause_before(section_text, m.start(), cite_pattern)
        
        if len(clause) < 20:
            continue
        
        # Match to bibliography
        arxiv_id = _find_in_bibliography(author, year, bib_entries, claim_text=clause)
        if arxiv_id:
            results.append({
                "clause": clause,
                "arxiv_id": arxiv_id,
                "author": author,
                "year": year,
                "format": fmt,
            })
    
    return results


def _get_clause_before(text, end_pos, cite_pattern):
    """Get the clause immediately before a citation position."""
    prev_cite_end = 0
    for prev_m in re.finditer(cite_pattern, text[:end_pos]):
        prev_cite_end = prev_m.end()
    sent_start = text.rfind('.', max(0, end_pos - 300), end_pos)
    clause_start = max(prev_cite_end, sent_start + 1) if sent_start > prev_cite_end else prev_cite_end
    return text[clause_start:end_pos].strip().strip(',;').strip()


def _find_in_bibliography(author, year, bib_entries, claim_text=""):
    """Find arXiv ID for an author+year in bibliography entries.
    
    Improved matching:
    1. Author name must START the entry (not just appear somewhere)
    2. Year must appear within 200 chars of start
    3. If multiple matches, use claim keywords to disambiguate
    """
    author_lower = author.lower().strip()
    if not author_lower:
        return None
    
    candidates = []
    
    for entry in bib_entries:
        entry_lower = entry.lower().strip()
        
        # Author must START the entry (first word match)
        entry_first_word = entry_lower.split(',')[0].split('.')[0].strip()
        
        # Check if author last name matches the start of the entry
        if not (entry_first_word.startswith(author_lower) or 
                entry_lower.startswith(author_lower)):
            continue
        
        # Year must appear within first 200 chars (near the author)
        if year not in entry[:200]:
            continue
        
        # Has arXiv ID?
        match = re.search(r'(\d{4}\.\d{4,5})', entry)
        if not match:
            continue
        
        candidates.append({
            "arxiv_id": match.group(1),
            "entry": entry,
        })
    
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]["arxiv_id"]
    
    # Multiple matches — disambiguate using claim keywords
    if claim_text:
        claim_words = set(w.lower() for w in claim_text.split() if len(w) > 4)
        best_score = -1
        best_id = candidates[0]["arxiv_id"]
        
        for cand in candidates:
            entry_words = set(w.lower() for w in cand["entry"].split() if len(w) > 4)
            overlap = len(claim_words & entry_words)
            if overlap > best_score:
                best_score = overlap
                best_id = cand["arxiv_id"]
        
        return best_id
    
    # Default to first match
    return candidates[0]["arxiv_id"]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python arxiv_extractor.py <paper.pdf> [section]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    section = sys.argv[2] if len(sys.argv) > 2 else "related work"
    
    results = extract_citations_from_pdf(pdf_path, section)
    print(f"Extracted {len(results)} citation-claim pairs:")
    for r in results:
        print(f"  [{r.get('cite_num', r.get('author',''))}] \"{r['clause'][:60]}\"")
        print(f"    → arXiv:{r['arxiv_id']}")
