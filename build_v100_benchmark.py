"""
Build the 100-PAPER benchmark (professor's target: ~100 distinct source papers).

- Select exactly ONE clean genuine claim from each of 100 distinct source papers
  (deterministic: seed=42, sorted sources, first passing claim per source).
- Build TWO parallel benchmarks over the SAME 100 genuine claims:
    * cross-category (out-of-field): replacement paper from a DIFFERENT arXiv category
    * same-field   (in-field):       replacement paper from the SAME arXiv category
  Terminology: "replacement" (not "swap"); labels CORRECT / INCORRECT.
Outputs:
  arxiv_test/random200/v100_crosscat_input.json / _gt.json
  arxiv_test/random200/v100_samefield_input.json / _gt.json
"""
import json, re, os, random, fitz
random.seed(42)
PDF_DIR = "arxiv_test/random200"
TARGET = 100
STOP = set('the a an is are was were been be have has had do does did will would could should may that this these those it its and or but not for on with at by from as into through of in to no we our their they which using used use based can'.split())

def pt(p):
    try:
        d=fitz.open(p); t="".join(pg.get_text() for pg in d); d.close(); return t
    except: return ""
def full_sentence(text,a):
    n=re.sub(r'\s+',' ',text); a=re.sub(r'\s+',' ',a).strip(); pos=-1
    for L in (60,40,25):
        if len(a)>=L:
            pos=n.find(a[:L])
            if pos>=0: break
    if pos<0: return None
    ws=max(0,pos-400); ss=ws
    for m in re.finditer(r'[.!?]\s+',n[ws:pos]): ss=ws+m.end()
    af=n[pos:pos+500]; em=re.search(r'\]\s*[.!?]',af) or re.search(r'[.!?]\s',af)
    return n[ss:pos+(em.end() if em else 200)].strip()
def substantive(c):
    c=re.sub(r'\s*\[\d+\]\s*\.?\s*$','',c.strip()).strip()
    if len(c.split())<8 or c[0].islower(): return False
    return bool(re.findall(r'\b(is|are|was|were|has|have|show|shows|showed|propose|proposed|introduce|introduced|demonstrate|use|uses|used|provide|provides|achieve|enable|improve|improves|address|require|can|allow|present|presents|reduce|leverage|apply|rely|report|reports)\b',c,re.I))
def overlap(cl,t,n=4):
    cw={w.lower().strip('.,;:()[]') for w in cl.split() if len(w)>3 and w.lower() not in STOP}
    pw={w.lower().strip('.,;:()[]') for w in t[:2000].split() if len(w)>3 and w.lower() not in STOP}
    return len(cw&pw)>=n

detail=json.load(open(f"{PDF_DIR}/correct_entries_detail.json"))
# gather clean claims grouped by source (sorted for determinism)
per_source={}
cat_of={}
for d in sorted(detail,key=lambda x:(x["source_paper"],x["cited_paper"])):
    src=d["source_paper"]; cid=d["cited_paper"]; cat=d["source_category"]
    sp=f"{PDF_DIR}/src_{src.replace('.','_')}.pdf"; cp=f"{PDF_DIR}/cited_{cid.replace('.','_')}.pdf"
    if not(os.path.exists(sp) and os.path.exists(cp)): continue
    ct=pt(cp)
    if len(ct)<500: continue
    s=full_sentence(pt(sp),re.sub(r'\s*\[\d+\]\s*\.?\s*$','',d["synopsis"]).strip())
    if not s: continue
    claim=re.sub(r'\s+',' ',re.sub(r'\[\d+\]','',s)).strip()
    if not substantive(claim+" [1]") or not overlap(claim,ct): continue
    if src in per_source: continue  # ONE claim per source
    per_source[src]={"cited":cid,"claim":claim,"content":ct[:15000],"cat":cat}
    cat_of[cid]=(cat,ct[:15000])

sources=sorted(per_source)[:TARGET]
print(f"distinct clean sources available: {len(per_source)}; using {len(sources)}")

# category pool for replacements (from the chosen genuine cited docs)
cat_pool={}
for s in sources:
    e=per_source[s]; cat_pool.setdefault(e["cat"],[]).append((e["cited"],e["content"]))

def build(mode):
    inp=[]; gt=[]
    for s in sources:
        e=per_source[s]
        qid=f"V100-{s}-{e['cited']}"
        inp.append({"question_id":qid,"synopsis":f"{e['claim']} [1].","retrieved_articles":[{"id":e["cited"],"content":e["content"]}]})
        gt.append({"question_id":qid,"status":"CORRECT"})
        # replacement
        if mode=="crosscat":
            cands=[c for c in cat_pool if c!=e["cat"] and cat_pool[c]]
        else:
            cands=[e["cat"]] if len([p for p in cat_pool.get(e["cat"],[]) if p[0]!=e["cited"]])>0 else []
        if not cands: continue
        c=random.choice(cands)
        pool=[p for p in cat_pool[c] if p[0]!=e["cited"]]
        if not pool: continue
        rid,rtext=random.choice(pool)
        suf="XREPL" if mode=="crosscat" else "SREPL"; rq=f"{qid}-{suf}"
        inp.append({"question_id":rq,"synopsis":f"{e['claim']} [1].","retrieved_articles":[{"id":rid,"content":rtext}]})
        gt.append({"question_id":rq,"status":"INCORRECT"})
    return inp,gt

for mode,tag in [("crosscat","v100_crosscat"),("samefield","v100_samefield")]:
    inp,gt=build(mode)
    json.dump(inp,open(f"{PDF_DIR}/{tag}_input.json","w"),indent=2,ensure_ascii=False)
    json.dump(gt,open(f"{PDF_DIR}/{tag}_gt.json","w"),indent=2,ensure_ascii=False)
    nc=sum(1 for g in gt if g["status"]=="CORRECT"); ni=sum(1 for g in gt if g["status"]=="INCORRECT")
    print(f"{tag}: {len(inp)} pairs ({nc} correct + {ni} replaced)")
