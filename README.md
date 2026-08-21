# HallucinationNerd

**A citation-level hallucination verification engine for synopsis-generating systems.**

HallucinationNerd checks whether cited sources actually support the claims attributed to them. Given any text with inline citations — a RAG synopsis, a research paper's Related Work, or an agentic system's report — it verifies each (claim, source) pair and returns per-claim verdicts with evidence pointers.

![Architecture](figures/hallucinationnerd-architecture.png)

## Key Results

Evaluated on two benchmarks built from **randomly sampled arXiv papers**, where a subset of genuine citations are swapped with unrelated papers to create objective, annotation-free ground truth.

| Benchmark | Accuracy | Precision | Recall |
|-----------|----------|-----------|--------|
| Random arXiv — cross-category swaps (200 pairs) | **92.1%** | 86.1% | 100% |
| Random arXiv — same-field hard negatives (187 pairs) | **90.2%** | 83.9% | 97.5% |

Against the systems we ran on identical data — **MiniCheck** and **RAGAS** — HallucinationNerd is far ahead: both competitors sit near chance (52–56% accuracy) because they flag roughly half of the *genuine* citations, while HallucinationNerd keeps false positives low. Paired sign-flip permutation tests confirm the gap is significant (p < 0.0001 on both benchmarks). Additional NLI baselines (HHEM 2.1, AlignScore, SummaCConv) are evaluated in the paper.

| System | Cross-category Acc | Same-field Acc |
|---|---|---|
| **HallucinationNerd** | **92.1%** | **90.2%** |
| RAGAS | 54.5% | 55.6% |
| MiniCheck | 52.5% | 51.9% |

## How It Works

1. **Claim Extraction** — Decomposes text into atomic, decontextualized claims with citation markers
2. **Reference Matching** — Retrieves source content (PDF, web, text) and finds relevant passages via keyword-overlap chunking
3. **Document Interrogation** — Asks the source document itself whether it supports the claim using a structured LLM judge
4. **Aggregation** — Produces per-claim verdicts, evidence quotes, and overall reliability scores

For uncited claims, the system searches PubMed for backup references (`--search-backup` mode).

## Installation

```bash
pip install -r requirements.txt
```

Optional, for PDF and web-page source support:
```bash
pip install pymupdf beautifulsoup4 requests
```

Requires an OpenAI API key in `.env`:
```
OPENAI_API_KEY=your-key-here
```

## Usage

```bash
# Verify citations in a document
python verify_hallucinations.py --input data.json --output-dir results/ --task citation

# Also search PubMed for uncited claims
python verify_hallucinations.py --input data.json --output-dir results/ --task citation --search-backup

# Adjust strictness: 'lenient' (default, precision-oriented) or 'strict' (recall-oriented)
python verify_hallucinations.py --input data.json --output-dir results/ --task citation --strictness strict
```

## Input Format

```json
[
  {
    "question_id": "Q1",
    "question": "What was asked",
    "synopsis": "The answer with [1] citations to verify",
    "retrieved_articles": [
      {
        "id": "1",
        "title": "Source paper title",
        "content": "Full text of source...",
        "url": "https://...",
        "path": "/path/to/file.pdf"
      }
    ]
  }
]
```

Sources can be provided as pre-extracted text (`content`), URLs to fetch (`url`), or PDF/text file paths (`path`).

## Citation Formats Supported

The `arxiv_extractor.py` module auto-detects and extracts citations in:

- `(Author et al., 2020)` — parenthetical author-year
- `[1]`, `[2, 3]` — numbered brackets
- `[Author et al., 2020]` — bracketed author-year
- `Author et al. (2020)` — narrative author-year
- `(1)`, `(2, 3)` — parenthetical numbers

## Verdict Taxonomy

| Verdict | Meaning |
|---------|---------|
| `SUPPORTED` | Source directly states or logically implies the claim |
| `PARTIALLY_SUPPORTED` | Source discusses the topic; gist correct but specifics differ |
| `NOT_SUPPORTED` | Source does not contain information relevant to this claim |
| `CONTRADICTED` | Source explicitly states the opposite |
| `UNVERIFIABLE` | Source content unavailable |
| `BACKUP_FOUND` | No citation provided; PubMed search found supporting evidence |
| `NO_BACKUP_FOUND` | No citation provided; no supporting evidence found in PubMed |

## Benchmarks & Reproduction

The `datasets/` folder contains the two benchmarks used above, each as a self-contained
`*_input.json` (claim + embedded source text) and `*_gt.json` (CORRECT / SWAPPED labels):

- `clean_benchmark_*` — cross-category, 200 pairs (100 genuine + 100 swaps)
- `hardneg_benchmark_*` — same-field hard negatives, 187 pairs (95 genuine + 92 swaps)

```bash
# Score any NLI/LLM system against a benchmark (edit the tool call inside)
python benchmark_template.py

# Reproduce HallucinationNerd verdicts
python run_clean.py       # cross-category
python run_hardneg.py     # same-field

# Reproduce competitor baselines (needs: pip install minicheck ragas datasets)
python run_competitors_clean.py both
python run_competitors_hardneg.py both
```

The benchmark builders (`build_random_benchmark.py`, `rebuild_clean_benchmark.py`,
`rebuild_hardneg_benchmark.py`, `topup_clean_to_200.py`) document exactly how the
benchmarks were sampled and quality-filtered from arXiv. They download source PDFs at
run time; the raw PDFs are not shipped here since their text is already embedded in the
dataset JSONs.

## Statistical Validation

```bash
python statistical_tests/Diff2MeanSig.py    # permutation test (p-value)
python statistical_tests/Diff2MeanConf.py   # bootstrap confidence interval
```

## Citation

If you use this work, please cite:

```
@article{wasu2026hallucinationnerd,
  title={HallucinationNerd: A Framework and Tool for Detecting Citation Hallucinations in Synopsis-Generating Systems},
  author={Wasu, Taranumpreet Kaur and Kashyap, Harsh and Chennur, Vishnu and Shasha, Dennis},
  year={2026}
}
```

## Authors

- **Taranumpreet Kaur Wasu** — Thapar Institute of Engineering and Technology, Patiala, India
- **Harsh Kashyap** — Thapar Institute of Engineering and Technology, Patiala, India
- **Vishnu Chennur** — Downingtown STEM Academy, Downingtown, PA, USA
- **Dennis Shasha** — Department of Computer Science, New York University, New York, USA
