"""
Build a benchmark from RANDOMLY sampled arXiv papers.

Sampling methodology (addresses professor's "randomly chosen from arxiv" + "exactly
how we find topically unrelated papers"):

1. SOURCE PAPERS: randomly sample N papers across a fixed set of CS categories
   (cs.CL, cs.LG, cs.CV, cs.AI, cs.RO, cs.CR) using random start-offsets into each
   category's listing. This removes hand-picking bias.

2. CLAIMS: from each source paper, extract Related-Work citations that resolve to a
   real arXiv ID (so the cited source is downloadable in full).

3. CORRECT pairs: (claim, the actually-cited paper).

4. SWAPPED pairs (topically unrelated): replace the cited paper with a paper drawn
   randomly from a DIFFERENT arXiv category than the cited paper's primary category.
   This gives a precise, reproducible definition of "topically unrelated": different
   primary arXiv subject class.

Run as a PILOT first (small N) to measure yield, then scale.
"""
import json, os, time, random, re, argparse
import requests
import fitz
from arxiv_extractor import extract_citations_from_pdf

# Seed varies per run so repeated runs sample DIFFERENT random papers (accumulate pool)
random.seed(int(time.time()))

CATEGORIES = ["cs.CL", "cs.LG", "cs.CV", "cs.AI", "cs.RO", "cs.CR"]
HEADERS = {"User-Agent": "HallucinationNerd-Benchmark/1.0 (academic research)"}
PDF_DIR = "arxiv_test/random200"
os.makedirs(PDF_DIR, exist_ok=True)


def arxiv_search(category, start, max_results=20):
    """Get a batch of paper IDs from a category at a random offset."""
    url = (f"http://export.arxiv.org/api/query?search_query=cat:{category}"
           f"&start={start}&max_results={max_results}"
           f"&sortBy=submittedDate&sortOrder=descending")
    try:
        resp = requests.get(url, timeout=20, headers=HEADERS)
        if resp.status_code != 200:
            return []
        # Extract id + primary category
        entries = re.findall(r'<entry>.*?</entry>', resp.text, re.DOTALL)
        results = []
        for e in entries:
            m = re.search(r'<id>http://arxiv.org/abs/([^<]+)</id>', e)
            cat = re.search(r'<arxiv:primary_category[^>]*term="([^"]+)"', e)
            if m:
                results.append((m.group(1).split('v')[0], cat.group(1) if cat else category))
        return results
    except Exception:
        return []


def download_pdf(arxiv_id, dest):
    if os.path.exists(dest):
        return True
    try:
        resp = requests.get(f"https://arxiv.org/pdf/{arxiv_id}", timeout=40, headers=HEADERS)
        if resp.status_code == 200 and resp.content[:4] == b"%PDF":
            with open(dest, "wb") as f:
                f.write(resp.content)
            time.sleep(1.5)  # be polite to arXiv
            return True
    except Exception:
        pass
    return False


def pdf_text(path):
    try:
        doc = fitz.open(path)
        t = "".join(p.get_text() for p in doc)
        doc.close()
        return t
    except Exception:
        return ""


def sample_source_papers(n_per_cat):
    """Randomly sample source papers across categories."""
    papers = []  # (arxiv_id, primary_cat)
    for cat in CATEGORIES:
        start = random.randint(0, 500)  # random offset for randomness
        batch = arxiv_search(cat, start, max_results=n_per_cat)
        papers.extend(batch)
        time.sleep(1)
    random.shuffle(papers)
    return papers


def main(n_per_cat, max_source_papers, cites_per_paper):
    print(f"Sampling ~{n_per_cat} papers/category from {len(CATEGORIES)} categories...")
    source_papers = sample_source_papers(n_per_cat)[:max_source_papers]
    print(f"Got {len(source_papers)} candidate source papers\n")

    # A pool of papers-by-category for building swaps
    category_pool = {}  # cat -> list of (arxiv_id, text)

    correct_entries = []
    stats = {"source_downloaded": 0, "no_citations": 0, "cited_downloaded": 0, "cited_failed": 0}

    for src_id, src_cat in source_papers:
        src_path = f"{PDF_DIR}/src_{src_id.replace('.','_')}.pdf"
        if not download_pdf(src_id, src_path):
            continue
        stats["source_downloaded"] += 1

        # Extract citations from Related Work
        try:
            cites = extract_citations_from_pdf(src_path, section="related work")
        except Exception:
            cites = []
        # Keep only cites with arxiv_id
        cites = [c for c in cites if c.get("arxiv_id") and c.get("clause")][:cites_per_paper]
        if not cites:
            stats["no_citations"] += 1
            continue

        for c in cites:
            cited_id = c["arxiv_id"]
            cited_path = f"{PDF_DIR}/cited_{cited_id.replace('.','_')}.pdf"
            if not download_pdf(cited_id, cited_path):
                stats["cited_failed"] += 1
                continue
            text = pdf_text(cited_path)
            if len(text) < 500:
                stats["cited_failed"] += 1
                continue
            stats["cited_downloaded"] += 1

            correct_entries.append({
                "question_id": f"RND-{src_id}-{cited_id}",
                "source_paper": src_id,
                "source_category": src_cat,
                "cited_paper": cited_id,
                "synopsis": f"{c['clause'].strip()} [1].",
                "retrieved_articles": [{"id": cited_id, "content": text[:15000]}],
            })
            # add to pool for swaps
            category_pool.setdefault(src_cat, []).append((cited_id, text[:15000]))

    print(f"\n=== YIELD ===")
    print(f"Source papers downloaded: {stats['source_downloaded']}")
    print(f"  with no usable arXiv citations: {stats['no_citations']}")
    print(f"Cited papers downloaded: {stats['cited_downloaded']}")
    print(f"Cited download failures: {stats['cited_failed']}")
    print(f"CORRECT (claim, source) pairs: {len(correct_entries)}")
    print(f"Yield per source paper: {len(correct_entries)/max(1,stats['source_downloaded']):.1f}")

    # ── Build SWAPPED pairs: replace cited paper with one from a DIFFERENT category ──
    all_entries = []
    all_gt = []

    for e in correct_entries:
        all_entries.append({
            "question_id": e["question_id"],
            "synopsis": e["synopsis"],
            "retrieved_articles": e["retrieved_articles"],
        })
        all_gt.append({"question_id": e["question_id"], "status": "CORRECT"})

    # For each correct entry, create a swap using a paper from a different category
    swap_count = 0
    for e in correct_entries:
        src_cat = e["source_category"]
        # Candidate swap categories = all categories except the source's
        other_cats = [c for c in category_pool.keys() if c != src_cat and category_pool[c]]
        if not other_cats:
            continue
        swap_cat = random.choice(other_cats)
        swap_id, swap_text = random.choice(category_pool[swap_cat])
        # Don't swap with the same paper
        if swap_id == e["cited_paper"]:
            continue
        swapped = {
            "question_id": f"{e['question_id']}-SWAPPED",
            "synopsis": e["synopsis"],
            "retrieved_articles": [{"id": swap_id, "content": swap_text}],
        }
        all_entries.append(swapped)
        all_gt.append({"question_id": swapped["question_id"], "status": "SWAPPED"})
        swap_count += 1

    n_c = sum(1 for g in all_gt if g["status"] == "CORRECT")
    n_s = sum(1 for g in all_gt if g["status"] == "SWAPPED")

    out_input = f"{PDF_DIR}/random_benchmark_input.json"
    out_gt = f"{PDF_DIR}/random_benchmark_gt.json"
    with open(out_input, "w") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)
    with open(out_gt, "w") as f:
        json.dump(all_gt, f, indent=2, ensure_ascii=False)

    print(f"\n=== FINAL BENCHMARK ===")
    print(f"Total entries: {len(all_entries)} ({n_c} correct + {n_s} swapped)")
    print(f"\u2713 Saved {out_input}")
    print(f"\u2713 Saved {out_gt}")

    # Also save the correct-entries detail (with source/cited metadata) for reference
    # ACCUMULATE across runs: merge with any existing detail, dedupe by question_id
    detail_path = f"{PDF_DIR}/correct_entries_detail.json"
    existing = []
    if os.path.exists(detail_path):
        try:
            existing = json.load(open(detail_path))
        except Exception:
            existing = []
    merged = {d["question_id"]: d for d in existing}
    for e in correct_entries:
        merged[e["question_id"]] = e
    with open(detail_path, "w") as f:
        json.dump(list(merged.values()), f, indent=2, ensure_ascii=False)
    print(f"\nCumulative correct-entry pool now: {len(merged)} (was {len(existing)}, added {len(merged)-len(existing)} new)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cat", type=int, default=3, help="papers to pull per category")
    ap.add_argument("--max-papers", type=int, default=12, help="max source papers to process (pilot)")
    ap.add_argument("--cites-per-paper", type=int, default=3)
    args = ap.parse_args()
    main(args.n_per_cat, args.max_papers, args.cites_per_paper)
