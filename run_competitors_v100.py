import json, os, sys
BENCH = sys.argv[1] if len(sys.argv)>1 else "v100_crosscat"
INPUT=f"arxiv_test/random200/{BENCH}_input.json"
GT=f"arxiv_test/random200/{BENCH}_gt.json"
OUTDIR=f"results/{BENCH.upper()}"; os.makedirs(OUTDIR,exist_ok=True)
entries=json.load(open(INPUT)); gt_map={g["question_id"]:g["status"] for g in json.load(open(GT))}
qids=[e["question_id"] for e in entries]; claims=[e["synopsis"] for e in entries]
docs=[e["retrieved_articles"][0]["content"][:2000] for e in entries]
def run_minicheck():
    from minicheck.minicheck import MiniCheck
    sc=MiniCheck(model_name="flan-t5-large",enable_prefix_caching=False)
    pred,_,_,_=sc.score(docs=docs,claims=claims); correct={}
    for qid,p in zip(qids,pred):
        g=gt_map.get(qid);
        if g is None: continue
        inc=(g=="INCORRECT"); flag=(p==0); correct[qid]=1 if (inc==flag) else 0
    json.dump(correct,open(f"{OUTDIR}/minicheck_correct.json","w"),indent=2); return correct
def run_ragas():
    key=[l for l in open(".env") if l.startswith("OPENAI_API_KEY")][0].split("=",1)[1].strip().strip('"')
    os.environ["OPENAI_API_KEY"]=key
    from ragas.metrics import faithfulness; from ragas import evaluate; from datasets import Dataset
    ds=Dataset.from_dict({"question":["Verify"]*len(claims),"answer":claims,"contexts":[[d] for d in docs]})
    scores=evaluate(ds,metrics=[faithfulness]).to_pandas()["faithfulness"].tolist(); correct={}
    for qid,s in zip(qids,scores):
        g=gt_map.get(qid)
        if g is None: continue
        inc=(g=="INCORRECT"); flag=(s<0.5) if s==s else False; correct[qid]=1 if (inc==flag) else 0
    json.dump(correct,open(f"{OUTDIR}/ragas_correct.json","w"),indent=2); return correct
run_minicheck(); run_ragas(); print(f"{BENCH} DONE")
