"""
Benchmark Template — How to Evaluate a New Tool
================================================

This script shows how we evaluated MiniCheck and RAGAS against our dataset.
Use this as a template to evaluate other tools (e.g. HHEM, AlignScore, SummaCConv).

INPUT:
  - claims: list of text claims
  - docs: list of source documents
  - ground_truth: is_swapped (True = hallucination, False = correct)

YOUR JOB:
  1. Install the new tool
  2. For each (claim, document) pair, get the tool's verdict
  3. Compare against ground truth
  4. Report accuracy, precision, recall, F1

Run: python benchmark_template.py
"""

import json

BENCHMARK = "cross_topic"  # or "hard_negatives"

FILES = {
    "cross_topic": {
        "input": "datasets/arxiv_50_permutation_input.json",
        "gt": "datasets/arxiv_50_permutation_gt.json",
    },
    "hard_negatives": {
        "input": "datasets/arxiv_hard_negatives_input.json",
        "gt": "datasets/arxiv_hard_negatives_gt.json",
    },
}[BENCHMARK]

# --- STEP 1: Load the test data ---
# Each entry is already a single (claim, document) pair -- one synopsis claim
# plus its single cited article, with ground truth status CORRECT/SWAPPED.
with open(FILES["input"]) as f:
    entries = json.load(f)
with open(FILES["gt"]) as f:
    gt_map = {g["question_id"]: g["status"] for g in json.load(f)}

# --- STEP 2: Prepare (claim, document) pairs ---
pairs = []
for e in entries:
    qid = e["question_id"]
    status = gt_map.get(qid)
    if status is None:
        continue
    pairs.append({
        "question_id": qid,
        "claim": e["synopsis"],
        "document": e["retrieved_articles"][0]["content"][:3000],  # truncate for most tools
        "is_swapped": (status == "SWAPPED"),  # True = should be flagged
    })

print(f"Loaded {len(pairs)} test pairs from '{BENCHMARK}'")
print(f"  Swapped (should detect): {sum(1 for p in pairs if p['is_swapped'])}")
print(f"  Correct (should not flag): {sum(1 for p in pairs if not p['is_swapped'])}")

# --- STEP 3: Run YOUR tool here ---
# Replace this section with the new tool's code.
# The tool should return True (hallucination) or False (supported) for each pair.

tool_predictions = []
for pair in pairs:
    claim = pair["claim"]
    document = pair["document"]

    # ======================================
    # YOUR CODE HERE
    # Example for MiniCheck:
    #   from minicheck.minicheck import MiniCheck
    #   scorer = MiniCheck(model_name='flan-t5-large')
    #   pred_labels, _, _, _ = scorer.score(docs=[document], claims=[claim])
    #   is_hallucination = (pred_labels[0] == 0)
    #
    # Example for a HuggingFace NLI model (AlignScore/SummaCConv-style):
    #   score = model.predict(document, claim)
    #   is_hallucination = (score < 0.5)
    # ======================================

    is_hallucination = False  # REPLACE THIS with actual tool output
    tool_predictions.append(is_hallucination)

# --- STEP 4: Calculate metrics ---
tp = tn = fp = fn = 0
for i, pair in enumerate(pairs):
    is_swapped = pair["is_swapped"]
    tool_says_hallucination = tool_predictions[i]

    if is_swapped and tool_says_hallucination: tp += 1
    elif is_swapped and not tool_says_hallucination: fn += 1
    elif not is_swapped and not tool_says_hallucination: tn += 1
    elif not is_swapped and tool_says_hallucination: fp += 1

total = tp + tn + fp + fn
accuracy = (tp + tn) / total if total else 0
precision = tp / (tp + fp) if (tp + fp) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print(f"\nRESULTS:")
print(f"  Accuracy:  {accuracy:.1%}")
print(f"  Precision: {precision:.1%}")
print(f"  Recall:    {recall:.1%}")
print(f"  F1:        {f1:.1%}")
print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")
