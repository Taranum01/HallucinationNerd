"""
Rebuild the random benchmark with stronger claim-quality filtering.
Reuses already-downloaded PDFs in arxiv_test/random200/.

Quality gates (address the 3 false-positive causes found in error analysis):
  1. FULL SENTENCE extraction (not clause fragments)
  2. DROP nominal/definitional citations (just a method name + [N], no assertion)
  3. STRONGER cited-paper validation (claim<->paper keyword overlap >= 4, catches
     wrong-paper matches like the "AviationGPT" mismatch)
"""
import json, os, re, copy, random
import fitz

random.seed(42)
PDF_DIR = "arxiv_test/random200"

STOP = {'the','a','an','is','are','was','were','been','be','have','has','had','do','does',
        'did','will','would','could','should','may','that','this','these','those','it','its',
        'and','or','but','not','for','on','with','at','by','from','as','into','through','of',
        'in','to','no','we','our','their','they','which','using','used','use','based','can'}


def pdf_text(path):
    try:
        doc = fitz.open(path); t = "".join(p.get_text() for p in doc); doc.close(); return t
    except Exception:
        return ""


def full_sentence_around(text, anchor_phrase):
    """Locate anchor_phrase in text and return the full sentence containing it.
    Normalizes whitespace so PDF line-breaks don't defeat the match."""
    norm = re.sub(r'\s+', ' ', text)
    anchor = re.sub(r'\s+', ' ', anchor_phrase).strip()
    # Try progressively shorter anchor prefixes to survive OCR/ligature noise
    pos = -1
    for L in (60, 40, 25):
        if len(anchor) >= L:
            pos = norm.find(anchor[:L])
            if pos >= 0:
                break
    if pos < 0:
        # last resort: match on a distinctive mid-anchor chunk
        words = anchor.split()
        if len(words) >= 4:
            probe = " ".join(words[1:5])
            pos = norm.find(probe)
    if pos < 0:
        return None
    ws = max(0, pos - 400)
    sent_start = ws
    for m in re.finditer(r'[.!?]\s+', norm[ws:pos]):
        sent_start = ws + m.end()
    after = norm[pos:pos + 500]
    em = re.search(r'\]\s*[.!?]', after) or re.search(r'[.!?]\s', after)
    sent_end = pos + (em.end() if em else 200)
    return norm[sent_start:sent_end].strip()


def is_substantive_claim(claim):
    """Reject fragments and nominal/definitional citations."""
    c = re.sub(r'\s*\[\d+\]\s*\.?\s*$', '', claim.strip()).strip()
    words = c.split()
    if len(words) < 8:
        return False
    if c[0].islower() or c[0] in '.,;:':
        return False
    if re.match(r'^(and|or|but|which|that|where|while|with|including|such as)\b', c, re.I):
        return False
    # Nominal pattern: "Something (ACRONYM) [N]" with few other words → just a name-drop
    # Heuristic: must contain a verb-like token (very rough: presence of common verbs)
    verbs = re.findall(r'\b(is|are|was|were|has|have|show|shows|showed|propose|proposes|proposed|'
                       r'introduce|introduces|introduced|demonstrate|demonstrates|use|uses|used|'
                       r'provide|provides|provided|achieve|achieves|enable|enables|improve|improves|'
                       r'address|addresses|require|requires|can|allow|allows|present|presents|'
                       r'reduce|reduces|leverage|leverages|apply|applies|rely|relies|report|reports)\b',
                       c, re.I)
    if not verbs:
        return False
    return True


def kw_overlap(claim, paper_text, n=4):
    cw = set(w.lower().strip('.,;:()[]') for w in claim.split()
             if len(w) > 3 and w.lower() not in STOP)
    pw = set(w.lower().strip('.,;:()[]') for w in paper_text[:2000].split()
             if len(w) > 3 and w.lower() not in STOP)
    return len(cw & pw) >= n


def main():
    detail = json.load(open(f"{PDF_DIR}/correct_entries_detail.json"))
    print(f"Reprocessing {len(detail)} candidate entries with quality filters...\n")

    kept = []
    dropped = {"no_sentence": 0, "not_substantive": 0, "low_overlap": 0}
    category_pool = {}

    for d in detail:
        src_id = d["source_paper"]; cited_id = d["cited_paper"]; src_cat = d["source_category"]
        old_clause = d["synopsis"]
        src_path = f"{PDF_DIR}/src_{src_id.replace('.','_')}.pdf"
        cited_path = f"{PDF_DIR}/cited_{cited_id.replace('.','_')}.pdf"
        if not (os.path.exists(src_path) and os.path.exists(cited_path)):
            continue

        src_text = pdf_text(src_path)
        cited_text = pdf_text(cited_path)
        if len(cited_text) < 500:
            continue

        # 1. full sentence
        anchor = re.sub(r'\s*\[\d+\]\s*\.?\s*$', '', old_clause).strip()
        sentence = full_sentence_around(src_text, anchor)
        if not sentence:
            dropped["no_sentence"] += 1; continue

        claim = re.sub(r'\[\d+\]', '', sentence).strip()  # strip citation markers
        claim = re.sub(r'\s+', ' ', claim).strip()

        # 2. substantive
        if not is_substantive_claim(claim + " [1]"):
            dropped["not_substantive"] += 1; continue

        # 3. strong overlap with cited paper
        if not kw_overlap(claim, cited_text, n=4):
            dropped["low_overlap"] += 1; continue

        kept.append({
            "question_id": f"RND2-{src_id}-{cited_id}",
            "source_category": src_cat,
            "cited_paper": cited_id,
            "synopsis": f"{claim} [1].",
            "retrieved_articles": [{"id": cited_id, "content": cited_text[:15000]}],
        })
        category_pool.setdefault(src_cat, []).append((cited_id, cited_text[:15000]))

    print(f"KEPT (clean correct claims): {len(kept)}")
    print(f"Dropped — no full sentence found: {dropped['no_sentence']}")
    print(f"Dropped — not substantive (fragment/nominal): {dropped['not_substantive']}")
    print(f"Dropped — low overlap (wrong-paper match): {dropped['low_overlap']}")

    # Build correct + cross-category swaps
    all_entries, all_gt = [], []
    for e in kept:
        all_entries.append({"question_id": e["question_id"], "synopsis": e["synopsis"],
                            "retrieved_articles": e["retrieved_articles"]})
        all_gt.append({"question_id": e["question_id"], "status": "CORRECT"})

    for e in kept:
        others = [c for c in category_pool if c != e["source_category"] and category_pool[c]]
        if not others:
            continue
        sc = random.choice(others)
        sid, stext = random.choice(category_pool[sc])
        if sid == e["cited_paper"]:
            continue
        all_entries.append({"question_id": f"{e['question_id']}-SWAPPED", "synopsis": e["synopsis"],
                            "retrieved_articles": [{"id": sid, "content": stext}]})
        all_gt.append({"question_id": f"{e['question_id']}-SWAPPED", "status": "SWAPPED"})

    nc = sum(1 for g in all_gt if g["status"]=="CORRECT")
    ns = sum(1 for g in all_gt if g["status"]=="SWAPPED")
    with open(f"{PDF_DIR}/clean_benchmark_input.json", "w") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)
    with open(f"{PDF_DIR}/clean_benchmark_gt.json", "w") as f:
        json.dump(all_gt, f, indent=2, ensure_ascii=False)
    print(f"\n=== CLEAN BENCHMARK ===\nTotal: {len(all_entries)} ({nc} correct + {ns} swapped)")
    print(f"\u2713 Saved clean_benchmark_input.json / clean_benchmark_gt.json")


if __name__ == "__main__":
    main()
