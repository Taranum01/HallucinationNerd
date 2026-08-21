"""
Run MiniCheck and RAGAS on the CLEAN random benchmark, keyed by question_id,
producing per-claim correctness for comparison + paired tests against HallucinationNerd.
"""
import json, os, sys

INPUT = "datasets/hardneg_benchmark_input.json"
GT = "datasets/hardneg_benchmark_gt.json"
os.makedirs("results/HARDNEG", exist_ok=True)

entries = json.load(open(INPUT))
gt_map = {g["question_id"]: g["status"] for g in json.load(open(GT))}

qids = [e["question_id"] for e in entries]
claims = [e["synopsis"] for e in entries]
docs = [e["retrieved_articles"][0]["content"][:2000] for e in entries]  # match prior competitor doc handling


def score_metrics(correct_map, name):
    tp = tn = fp = fn = 0
    for qid, correct in correct_map.items():
        gt = gt_map.get(qid)
        if gt is None:
            continue
        is_swapped = (gt == "SWAPPED")
        # correct==1 means the tool's flag matched ground truth
        # reconstruct flag: flagged iff (is_swapped and correct) or (not is_swapped and not correct)
        flagged = (is_swapped and correct) or ((not is_swapped) and (not correct))
        if is_swapped and flagged: tp += 1
        elif is_swapped and not flagged: fn += 1
        elif not is_swapped and not flagged: tn += 1
        else: fp += 1
    t = tp + tn + fp + fn
    acc = (tp + tn) / t * 100 if t else 0
    prec = tp / (tp + fp) * 100 if tp + fp else 0
    rec = tp / (tp + fn) * 100 if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"{name}: n={t} Acc={acc:.1f}% Prec={prec:.1f}% Rec={rec:.1f}% F1={f1:.1f}% (TP={tp} FP={fp} FN={fn})")
    return dict(n=t, acc=acc, prec=prec, rec=rec, f1=f1, tp=tp, tn=tn, fp=fp, fn=fn)


def run_minicheck():
    from minicheck.minicheck import MiniCheck
    print("Running MiniCheck...", flush=True)
    scorer = MiniCheck(model_name="flan-t5-large", enable_prefix_caching=False)
    pred_labels, _, _, _ = scorer.score(docs=docs, claims=claims)
    correct = {}
    for qid, pred in zip(qids, pred_labels):
        gt = gt_map.get(qid)
        if gt is None: continue
        is_swapped = (gt == "SWAPPED")
        flagged = (pred == 0)  # 0 = not supported = flagged as hallucination
        correct[qid] = 1 if (is_swapped == flagged) else 0
    json.dump(correct, open("results/HARDNEG/minicheck_correct.json", "w"), indent=2)
    return correct


def run_ragas():
    import os as _os
    key = [l for l in open(".env").readlines() if l.startswith("OPENAI_API_KEY")][0].split("=", 1)[1].strip().strip('"')
    _os.environ["OPENAI_API_KEY"] = key
    from ragas.metrics import faithfulness
    from ragas import evaluate
    from datasets import Dataset
    print("Running RAGAS...", flush=True)
    ds = Dataset.from_dict({
        "question": ["Verify this claim"] * len(claims),
        "answer": claims,
        "contexts": [[d] for d in docs],
    })
    result = evaluate(ds, metrics=[faithfulness])
    scores = result.to_pandas()["faithfulness"].tolist()
    correct = {}
    for qid, score in zip(qids, scores):
        gt = gt_map.get(qid)
        if gt is None: continue
        is_swapped = (gt == "SWAPPED")
        flagged = (score < 0.5) if score == score else False  # NaN-safe
        correct[qid] = 1 if (is_swapped == flagged) else 0
    json.dump(correct, open("results/HARDNEG/ragas_correct.json", "w"), indent=2)
    return correct


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("minicheck", "both"):
        mc = run_minicheck()
        score_metrics(mc, "MiniCheck")
    if which in ("ragas", "both"):
        rg = run_ragas()
        score_metrics(rg, "RAGAS")
    print("DONE", flush=True)
