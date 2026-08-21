"""
Build the SAME-FIELD (hard-negative) variant of the random benchmark.

Design (addresses professor's redline: "Do the same procedure as above but choosing
articles from this smaller set" -- i.e. swap with a paper from the SAME arXiv
category instead of a different one):

  * POSITIVES: identical to the cross-category clean benchmark (same 95 correct
    (claim, cited-paper) pairs, same quality filters, same seed). This isolates the
    single variable we are changing -- negative difficulty.

  * HARD NEGATIVES: for each correct claim, replace the cited paper with a DIFFERENT
    paper drawn randomly from the SAME arXiv primary category. Same-field papers
    share vocabulary and topic, so this is a strictly harder negative than the
    cross-category swap.

Reuses already-downloaded PDFs in arxiv_test/random200/. No network needed.
"""
import json, os, re, random
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
    norm = re.sub(r'\s+', ' ', text)
    anchor = re.sub(r'\s+', ' ', anchor_phrase).strip()
    pos = -1
    for L in (60, 40, 25):
        if len(anchor) >= L:
            pos = norm.find(anchor[:L])
            if pos >= 0:
                break
    if pos < 0:
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
    c = re.sub(r'\s*\[\d+\]\s*\.?\s*$', '', claim.strip()).strip()
    words = c.split()
    if len(words) < 8:
        return False
    if c[0].islower() or c[0] in '.,;:':
        return False
    if re.match(r'^(and|or|but|which|that|where|while|with|including|such as)\b', c, re.I):
        return False
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
    print(f"Reprocessing {len(detail)} candidate entries (same filters as clean benchmark)...\n")

    kept = []
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

        anchor = re.sub(r'\s*\[\d+\]\s*\.?\s*$', '', old_clause).strip()
        sentence = full_sentence_around(src_text, anchor)
        if not sentence:
            continue

        claim = re.sub(r'\[\d+\]', '', sentence).strip()
        claim = re.sub(r'\s+', ' ', claim).strip()

        if not is_substantive_claim(claim + " [1]"):
            continue
        if not kw_overlap(claim, cited_text, n=4):
            continue

        kept.append({
            "question_id": f"HARDNEG-{src_id}-{cited_id}",
            "source_category": src_cat,
            "cited_paper": cited_id,
            "synopsis": f"{claim} [1].",
            "retrieved_articles": [{"id": cited_id, "content": cited_text[:15000]}],
        })
        category_pool.setdefault(src_cat, []).append((cited_id, cited_text[:15000]))

    print(f"KEPT (clean correct claims): {len(kept)}")
    print("Category distribution of correct claims:")
    for c in sorted(category_pool):
        print(f"  {c}: {len(category_pool[c])}")

    # Positives
    all_entries, all_gt = [], []
    for e in kept:
        all_entries.append({"question_id": e["question_id"], "synopsis": e["synopsis"],
                            "retrieved_articles": e["retrieved_articles"]})
        all_gt.append({"question_id": e["question_id"], "status": "CORRECT"})

    # SAME-FIELD hard negatives: swap with a different paper from the SAME category
    made, skipped_singleton = 0, 0
    for e in kept:
        same_cat = [p for p in category_pool[e["source_category"]] if p[0] != e["cited_paper"]]
        if not same_cat:
            skipped_singleton += 1
            continue
        sid, stext = random.choice(same_cat)
        all_entries.append({"question_id": f"{e['question_id']}-SWAPPED", "synopsis": e["synopsis"],
                            "retrieved_articles": [{"id": sid, "content": stext}]})
        all_gt.append({"question_id": f"{e['question_id']}-SWAPPED", "status": "SWAPPED"})
        made += 1

    nc = sum(1 for g in all_gt if g["status"] == "CORRECT")
    ns = sum(1 for g in all_gt if g["status"] == "SWAPPED")
    with open(f"{PDF_DIR}/hardneg_benchmark_input.json", "w") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)
    with open(f"{PDF_DIR}/hardneg_benchmark_gt.json", "w") as f:
        json.dump(all_gt, f, indent=2, ensure_ascii=False)
    print(f"\n=== HARD-NEGATIVE (SAME-FIELD) BENCHMARK ===")
    print(f"Total: {len(all_entries)} ({nc} correct + {ns} same-field swapped)")
    print(f"Hard negatives made: {made}; skipped (only one paper in category): {skipped_singleton}")
    print("\u2713 Saved hardneg_benchmark_input.json / hardneg_benchmark_gt.json")


if __name__ == "__main__":
    main()
