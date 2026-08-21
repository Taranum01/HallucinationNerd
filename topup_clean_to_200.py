"""
Top up the CROSS-CATEGORY (clean) benchmark from 190 -> 200 pairs, WITHOUT disturbing
the existing 190. We keep the existing 95 correct + 95 swaps exactly (so their HN /
competitor verdicts stay valid) and APPEND 5 new clean positives + 5 new cross-category
swaps drawn from newly sampled papers.
"""
import json, os, re, random
import fitz

random.seed(4242)  # separate seed; existing entries are never reshuffled
PDF_DIR = "arxiv_test/random200"
N_ADD = 5

STOP = {'the','a','an','is','are','was','were','been','be','have','has','had','do','does',
        'did','will','would','could','should','may','that','this','these','those','it','its',
        'and','or','but','not','for','on','with','at','by','from','as','into','through','of',
        'in','to','no','we','our','their','they','which','using','used','use','based','can'}

def pdf_text(path):
    try:
        doc=fitz.open(path); t="".join(p.get_text() for p in doc); doc.close(); return t
    except Exception: return ""

def full_sentence_around(text, anchor_phrase):
    norm=re.sub(r'\s+',' ',text); anchor=re.sub(r'\s+',' ',anchor_phrase).strip()
    pos=-1
    for L in (60,40,25):
        if len(anchor)>=L:
            pos=norm.find(anchor[:L])
            if pos>=0: break
    if pos<0:
        w=anchor.split()
        if len(w)>=4: pos=norm.find(" ".join(w[1:5]))
    if pos<0: return None
    ws=max(0,pos-400); sent_start=ws
    for m in re.finditer(r'[.!?]\s+',norm[ws:pos]): sent_start=ws+m.end()
    after=norm[pos:pos+500]
    em=re.search(r'\]\s*[.!?]',after) or re.search(r'[.!?]\s',after)
    sent_end=pos+(em.end() if em else 200)
    return norm[sent_start:sent_end].strip()

def is_substantive_claim(claim):
    c=re.sub(r'\s*\[\d+\]\s*\.?\s*$','',claim.strip()).strip()
    if len(c.split())<8: return False
    if c[0].islower() or c[0] in '.,;:': return False
    if re.match(r'^(and|or|but|which|that|where|while|with|including|such as)\b',c,re.I): return False
    verbs=re.findall(r'\b(is|are|was|were|has|have|show|shows|showed|propose|proposes|proposed|'
                     r'introduce|introduces|introduced|demonstrate|demonstrates|use|uses|used|'
                     r'provide|provides|provided|achieve|achieves|enable|enables|improve|improves|'
                     r'address|addresses|require|requires|can|allow|allows|present|presents|'
                     r'reduce|reduces|leverage|leverages|apply|applies|rely|relies|report|reports)\b',c,re.I)
    return bool(verbs)

def kw_overlap(claim, paper_text, n=4):
    cw=set(w.lower().strip('.,;:()[]') for w in claim.split() if len(w)>3 and w.lower() not in STOP)
    pw=set(w.lower().strip('.,;:()[]') for w in paper_text[:2000].split() if len(w)>3 and w.lower() not in STOP)
    return len(cw & pw)>=n

# --- existing benchmark (keep intact) ---
inp=json.load(open(f"{PDF_DIR}/clean_benchmark_input.json"))
gt=json.load(open(f"{PDF_DIR}/clean_benchmark_gt.json"))
existing_ids={e["question_id"] for e in inp}
existing_correct_srccited={tuple(qid.split("RND2-")[1].split("-")[:2]) for qid in existing_ids if not qid.endswith("SWAPPED") and qid.startswith("RND2-")}
print(f"existing: {len(inp)} pairs, {sum(1 for g in gt if g['status']=='CORRECT')} correct")

# --- process full pool, collect clean positives keyed by (src,cited) ---
detail=json.load(open(f"{PDF_DIR}/correct_entries_detail.json"))
category_pool={}   # cat -> [(cited_id, text)]  (for building swaps)
clean_all=[]       # all clean positives
for d in detail:
    src=d["source_paper"]; cited=d["cited_paper"]; cat=d["source_category"]
    sp=f"{PDF_DIR}/src_{src.replace('.','_')}.pdf"; cp=f"{PDF_DIR}/cited_{cited.replace('.','_')}.pdf"
    if not (os.path.exists(sp) and os.path.exists(cp)): continue
    st=pdf_text(sp); ct=pdf_text(cp)
    if len(ct)<500: continue
    anchor=re.sub(r'\s*\[\d+\]\s*\.?\s*$','',d["synopsis"]).strip()
    sent=full_sentence_around(st,anchor)
    if not sent: continue
    claim=re.sub(r'\s+',' ',re.sub(r'\[\d+\]','',sent)).strip()
    if not is_substantive_claim(claim+" [1]"): continue
    if not kw_overlap(claim,ct,n=4): continue
    entry={"question_id":f"RND2-{src}-{cited}","source_category":cat,"cited_paper":cited,
           "synopsis":f"{claim} [1].","content":ct[:15000]}
    clean_all.append(entry)
    category_pool.setdefault(cat,[]).append((cited,ct[:15000]))

# NEW positives = clean ones whose (src,cited) not already in benchmark
new_pos=[e for e in clean_all if tuple(e["question_id"].split("RND2-")[1].split("-")[:2]) not in existing_correct_srccited]
# dedupe by question_id, stable
seen=set(); uniq=[]
for e in new_pos:
    if e["question_id"] in seen or e["question_id"] in existing_ids: continue
    seen.add(e["question_id"]); uniq.append(e)
print(f"candidate NEW clean positives available: {len(uniq)}")
random.shuffle(uniq)
add=uniq[:N_ADD]
assert len(add)>=N_ADD, f"only {len(add)} new clean positives; sample more papers"

# --- append 5 correct + 5 cross-category swaps ---
added_correct=added_swap=0
for e in add:
    inp.append({"question_id":e["question_id"],"synopsis":e["synopsis"],
                "retrieved_articles":[{"id":e["cited_paper"],"content":e["content"]}]})
    gt.append({"question_id":e["question_id"],"status":"CORRECT"}); added_correct+=1
    # cross-category swap
    others=[c for c in category_pool if c!=e["source_category"] and category_pool[c]]
    sc=random.choice(others); sid,stext=random.choice(category_pool[sc])
    while sid==e["cited_paper"]:
        sid,stext=random.choice(category_pool[sc])
    sq=f"{e['question_id']}-SWAPPED"
    inp.append({"question_id":sq,"synopsis":e["synopsis"],"retrieved_articles":[{"id":sid,"content":stext}]})
    gt.append({"question_id":sq,"status":"SWAPPED"}); added_swap+=1

json.dump(inp,open(f"{PDF_DIR}/clean_benchmark_input.json","w"),indent=2,ensure_ascii=False)
json.dump(gt,open(f"{PDF_DIR}/clean_benchmark_gt.json","w"),indent=2,ensure_ascii=False)
nc=sum(1 for g in gt if g["status"]=="CORRECT"); ns=sum(1 for g in gt if g["status"]=="SWAPPED")
print(f"added {added_correct} correct + {added_swap} swaps")
print(f"NEW TOTAL: {len(inp)} pairs ({nc} correct + {ns} swapped)")
print("new positive ids:")
for e in add: print("  ", e["question_id"])
