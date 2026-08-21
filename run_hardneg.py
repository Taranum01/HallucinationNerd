import json, os
from dataclasses import asdict
from verify_hallucinations import verify_citations_for_question

INPUT = "datasets/hardneg_benchmark_input.json"
CLEAN = "results/CLEAN200/citation_verification.jsonl"
OUT = "results/HARDNEG/citation_verification.jsonl"
os.makedirs("results/HARDNEG", exist_ok=True)

entries = json.load(open(INPUT))

# Map reusable positive verdicts from the clean benchmark: RND2-{src}-{cited} -> record
clean = {}
for l in open(CLEAN):
    r = json.loads(l)
    clean[r["question_id"]] = r

done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["question_id"] for l in open(OUT)}

reused, ran, errs = 0, 0, 0
with open(OUT, "a") as out:
    for i, e in enumerate(entries, 1):
        qid = e["question_id"]
        if qid in done:
            continue
        # Positive? reuse the identical clean-benchmark verdict (same claim + cited content)
        if not qid.endswith("SWAPPED"):
            rnd_id = qid.replace("HARDNEG-", "RND2-", 1)
            src = clean.get(rnd_id)
            if src is not None:
                rec = dict(src)
                rec["question_id"] = qid
                rec["claim_id"] = qid + "-c0"
                out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
                reused += 1
                print(f"[{i}/{len(entries)}] {qid}: REUSED {rec['verdict']}", flush=True)
                continue
        # Hard negative (or unmapped positive): run fresh
        try:
            r = verify_citations_for_question(e, single_claim=True, n_votes=5)
            for x in r:
                out.write(json.dumps(asdict(x), ensure_ascii=False) + "\n")
            out.flush()
            ran += 1
            print(f"[{i}/{len(entries)}] {qid}: {r[0].verdict if r else 'NONE'}", flush=True)
        except Exception as ex:
            errs += 1
            print(f"[{i}] ERR {ex}", flush=True)

print(f"DONE reused={reused} ran={ran} errs={errs}", flush=True)
