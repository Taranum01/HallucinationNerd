import json, os
from dataclasses import asdict
from verify_hallucinations import verify_citations_for_question
INPUT="arxiv_test/random200/v100_all_input.json"
OUT="results/V100/citation_verification.jsonl"
os.makedirs("results/V100", exist_ok=True)
entries=json.load(open(INPUT))
done=set()
if os.path.exists(OUT):
    done={json.loads(l)["question_id"] for l in open(OUT)}
with open(OUT,"a") as out:
    for i,e in enumerate(entries,1):
        if e["question_id"] in done: continue
        try:
            r=verify_citations_for_question(e,single_claim=True,n_votes=5)
            for x in r: out.write(json.dumps(asdict(x),ensure_ascii=False)+"\n")
            out.flush()
            print(f"[{i}/{len(entries)}] {e['question_id']}: {r[0].verdict if r else 'NONE'}",flush=True)
        except Exception as ex:
            print(f"[{i}] ERR {ex}",flush=True)
print("DONE",flush=True)
