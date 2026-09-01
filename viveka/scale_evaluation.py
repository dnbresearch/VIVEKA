#!/usr/bin/env python3
"""
VIVEKA Full-Scale Evaluation for ICDM 2026
=============================================
Scales up all experiments to paper-ready sample sizes + 2 new experiments.

Phase 1: Large-scale mining (100+ repos, all extractors)         ~60 min
Phase 2: OCP cross-linking (all OCP repos)                       ~30 min
Phase 3: LLM quality scoring (30 repos)                          ~15 min
Phase 4: LLM hybrid verification (20 repos)                      ~15 min
Phase 5: Domain analysis (breakdown by venue/field)               instant
Phase 6: Novel insight detection (insights not in paper abstract) ~15 min
Phase 7: Cost rollup                                              instant

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python3 viveka_scale_evaluation.py --max-repos 500

  # Without LLM (Phase 1,2,5,7 only)
  python3 viveka_scale_evaluation.py --max-repos 500 --skip-llm
"""

import argparse,json,os,re,shutil,subprocess,sys,time,io,contextlib
from collections import defaultdict
from dataclasses import dataclass,field
from pathlib import Path
from typing import Optional
from difflib import SequenceMatcher
import yaml

WORK_DIR = Path("./insightminer_work")
RESULTS_DIR = Path("./validation_results/scale")
WORK_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True,exist_ok=True)
CLONE_TIMEOUT = 60  # increased from 45 for larger repos

SKIP_REPOS = {
    "https://github.com/yandex-research/tabular-dl-tabr",
    "https://github.com/yandex-research/tabular-dl-num-embeddings",
    "https://github.com/yandex-research/rtdl-revisiting-models",
    "https://github.com/yandex-research/rtdl",
}

# =====================================================================
# Core (all fixes, compact)
# =====================================================================
IDENTITY_RE=[re.compile(p,re.IGNORECASE) for p in [
    r".*path$",r".*dir$",r".*file$",r".*root$",r".*_dir$",r".*_path$",
    r".*_file$",r".*_root$",r".*\.dir$",r".*\.path$",r".*\.file$",
    r".*\.root$",r".*meta_file.*",r".*image_dir.*",r".*data_dir.*",
    r".*checkpoints.*",r".*log_dir.*",r".*vis_dir.*",r".*save_dir.*",
    r".*output_dir.*",r".*pretrain.*",r".*pertrain.*",r".*workspace.*",
    r".*exp_name.*",r".*run_name.*"]]
NOISE_RE=[re.compile(p,re.IGNORECASE) for p in [
    r".*print_freq.*",r".*log_freq.*",r".*save_freq.*",r".*display.*",
    r".*verbose.*",r".*workers$",r".*num_workers.*"]]
KNOWN_DS=["mvtec","btad","mpdd","visa","cifar","imagenet","coco","voc","cityscapes",
    "ade20k","kinetics","ucf","hmdb","mnist","svhn","celeba","scannet","shapenet",
    "modelnet","squad","glue","superglue","wmt","ntu","ptb","wiki"]
HPARAM_COL_RE=re.compile(r"(epoch|step|iter|batch|lr|learning.rate|seed|layer|dim|head|"
    r"param|flop|gflop|memory|gpu|time|hour|min|sec|training|warmup|decay|size)",re.IGNORECASE)

def clsp(k):
    for p in IDENTITY_RE:
        if p.match(k): return "identity"
    for p in NOISE_RE:
        if p.match(k): return "noise"
    return "semantic"

def sdiffs(p1,p2):
    sh=set(p1)&set(p2)
    if len(sh)<3: return None
    d={"semantic":[],"identity":[],"noise":[]}
    for k in sorted(sh):
        if p1[k]!=p2[k]: d[clsp(k)].append((k,p1[k],p2[k]))
    return d if sum(len(v) for v in d.values())>0 else None

def eds(src,params=None):
    for ds in KNOWN_DS:
        if ds in src.lower(): return ds
    if params:
        for k,v in params.items():
            if clsp(k)=="identity" and isinstance(v,str):
                for ds in KNOWN_DS:
                    if ds in v.lower(): return ds
    parts=Path(src).parts
    return parts[-2] if len(parts)>=2 else None

def fld(d,pfx="",sep="."):
    it={}
    if not isinstance(d,dict): return it
    for k,v in d.items():
        nk=f"{pfx}{sep}{k}" if pfx else k
        if isinstance(v,dict): it.update(fld(v,nk,sep))
        elif isinstance(v,list) and len(v)<=10 and all(isinstance(x,(int,float,str,bool)) for x in v):
            it[nk]=v
        elif isinstance(v,(int,float,str,bool)) or v is None: it[nk]=v
    return it

def tpv(s):
    if not isinstance(s,str): return s
    s=s.strip()
    if s.lower()=="true": return True
    if s.lower()=="false": return False
    if s.lower() in ("none","null"): return None
    try: return int(s)
    except: pass
    try: return float(s)
    except: pass
    return s

def ilm(col,vals):
    if HPARAM_COL_RE.search(col.lower()): return False
    nums=[v for v in vals if isinstance(v,(int,float))]
    if not nums: return False
    if all(isinstance(v,int) and v>100 for v in nums): return False
    if len(nums)>=2:
        mn,mx=min(nums),max(nums)
        if mn>0 and mx/mn>1000: return False
    return True

def sclone(url,dest):
    u=url if url.endswith(".git") else url+".git"
    try:
        p=subprocess.Popen(["git","clone","--depth","1","--single-branch",u,str(dest)],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,preexec_fn=os.setsid)
        try:
            p.wait(timeout=CLONE_TIMEOUT); return p.returncode==0
        except subprocess.TimeoutExpired:
            try: os.killpg(os.getpgid(p.pid),9)
            except: pass
            p.wait(); return False
    except: return False

@dataclass
class N:
    id:int; src:str; p:dict; lbl:str=""; dsid:str=""; etype:str=""
@dataclass
class E:
    s:int; t:int; sd:list; idd:list; nd:list; nsd:int; et:str
@dataclass
class I:
    tp:str; desc:str; ev:dict; sig:float; par:list

# --- Extractors ---
def ext_yaml(rp):
    cfgs=[]
    ex={".git","node_modules","__pycache__","wandb","build","dist",".eggs","outputs",".tox","docs","test","tests"}
    kw={"learning_rate","lr","batch_size","epochs","num_layers","hidden_size","dropout",
        "weight_decay","optimizer","model","dataset","train","seed","max_epochs","warmup","scheduler"}
    for pat in ["**/*.yaml","**/*.yml","**/*.json"]:
        for f in rp.glob(pat):
            if any(e in f.parts for e in ex): continue
            if f.stat().st_size>200000 or f.stat().st_size<20: continue
            if f.name.startswith(".") or f.name in {"package.json","package-lock.json","tsconfig.json",
                ".pre-commit-config.yaml","mkdocs.yml","docker-compose.yml","codecov.yml"}: continue
            try:
                c=f.read_text(errors="ignore")
                if sum(1 for k in kw if k in c.lower())<2: continue
                p=yaml.safe_load(c) if f.suffix in (".yaml",".yml") else json.loads(c)
                if not isinstance(p,dict): continue
                fl=fld(p)
                if len(fl)<3: continue
                src=str(f.relative_to(rp))
                cfgs.append({"source_file":src,"params":fl,"n_params":len(fl),
                    "dataset_id":eds(src,fl),"experiment_type":Path(src).stem.lower(),
                    "method":"yaml"})
            except: continue
    return cfgs

def ext_argparse(rp):
    res=[]
    ex={".git","__pycache__","build","wandb","test","tests"}
    pat=re.compile(r"""add_argument\s*\(\s*['"](--[\w-]+)['"].*?(?:default\s*=\s*([^,\)]+))?""",re.DOTALL)
    for f in rp.glob("**/*.py"):
        if any(e in f.parts for e in ex): continue
        if f.stat().st_size>300000: continue
        try:
            c=f.read_text(errors="ignore")
            if "add_argument" not in c: continue
            params={}
            for m in pat.finditer(c):
                n=m.group(1).lstrip("-").replace("-","_")
                d=m.group(2)
                if d: params[n]=tpv(d.strip().strip("'\""))
            if len(params)>=3:
                src=str(f.relative_to(rp))
                res.append({"source_file":src,"params":params,"n_params":len(params),
                    "dataset_id":eds(src,params),"experiment_type":Path(src).stem.lower(),
                    "method":"argparse"})
        except: continue
    return res

def ext_shell(rp):
    cfgs=[]
    ex={".git","node_modules","__pycache__","wandb","build"}
    flag_re=re.compile(r'--([\w-]+)[=\s]+([^\s\\]+)')
    ml=["train","python","torch","cuda","gpu","epoch","model","lr"]
    for f in rp.glob("**/*.sh"):
        if any(e in f.parts for e in ex): continue
        if f.stat().st_size>100000 or f.stat().st_size<30: continue
        try:
            c=f.read_text(errors="ignore")
            if sum(1 for h in ml if h in c.lower())<2: continue
            params={}
            for m in flag_re.finditer(c):
                nm=m.group(1).replace("-","_")
                params[nm]=tpv(m.group(2).strip("'\""))
            if len(params)>=3:
                src=str(f.relative_to(rp))
                cfgs.append({"source_file":src,"params":params,"n_params":len(params),
                    "dataset_id":eds(src,params),"experiment_type":Path(src).stem.lower(),
                    "method":"shell"})
        except: continue
    return cfgs

def ext_readme(rp):
    tables=[]
    for name in ["README.md","readme.md","Readme.md"]:
        f=rp/name
        if not f.exists(): continue
        try:
            cur=[]
            for line in f.read_text(errors="ignore").split("\n"):
                s=line.strip()
                if "|" in s and s.startswith("|"): cur.append(s)
                else:
                    if len(cur)>=3:
                        t=_pt(cur)
                        if t: tables.append(t)
                    cur=[]
            if len(cur)>=3:
                t=_pt(cur)
                if t: tables.append(t)
        except: continue
    return tables

def _pt(lines):
    rows=[]
    for l in lines:
        cells=[c.strip() for c in l.split("|") if c.strip()]
        if cells: rows.append(cells)
    if len(rows)<3: return None
    hdr=rows[0]
    data=[r for r in rows[1:] if not all(set(c.strip())<=set("-: ") for c in r)]
    if not data: return None
    recs=[{hdr[i]:tpv(r[i]) for i in range(min(len(hdr),len(r)))} for r in data]
    if sum(1 for r in recs for v in r.values() if isinstance(v,(int,float)))<2: return None
    return {"headers":hdr,"records":recs,"n_rows":len(recs)}

# --- Graph + Mining ---
def build_g(cfgs):
    nodes=[N(i,c["source_file"],c["params"],Path(c["source_file"]).stem,
        c.get("dataset_id") or "",c.get("experiment_type") or "") for i,c in enumerate(cfgs)]
    edges=[]
    max_nodes = min(len(nodes), 200)
    for i in range(max_nodes):
        for j in range(i+1,max_nodes):
            d=sdiffs(nodes[i].p,nodes[j].p)
            if not d: continue
            ns,ni=len(d["semantic"]),len(d["identity"])
            if ns==0 and ni>0: et="dataset_switch"
            elif ns==1 and ni==0: et="single_ablation"
            elif ns==1 and ni>0: et="cross_dataset_ablation"
            elif ns<=3: et="multi_ablation"
            else: continue
            edges.append(E(i,j,d["semantic"],d["identity"],d["noise"],ns,et))
    return nodes,edges

def mine(cfgs,nodes,edges,tables):
    ins=[]
    for etype,itype in [("single_ablation","clean_ablation"),("cross_dataset_ablation","cross_dataset_ablation")]:
        bp=defaultdict(list)
        for e in edges:
            if e.et==etype: bp[e.sd[0][0]].append(e)
        for p,pe in bp.items():
            vs=set()
            for e in pe: vs.add(str(e.sd[0][1])); vs.add(str(e.sd[0][2]))
            ins.append(I(itype,f"'{p}' ablated: {sorted(vs)} across {len(pe)} pairs",
                {"param":p,"values":sorted(vs),"n_pairs":len(pe)},
                min(len(vs)/4,1) if etype=="single_ablation" else min(len(vs)/5,0.8),[p]))
    dsw=[e for e in edges if e.et=="dataset_switch"]
    if dsw:
        ds=set()
        for e in dsw: ds.add(nodes[e.s].dsid); ds.add(nodes[e.t].dsid)
        ds.discard("")
        if len(ds)>=2:
            ins.append(I("dataset_comparison",f"Same config across {len(ds)} datasets: {sorted(ds)}",
                {"datasets":sorted(ds)},min(len(ds)/4,1),list(ds)))
    pv=defaultdict(list)
    for c in cfgs:
        for k,v in c["params"].items():
            if isinstance(v,(int,float)) and not isinstance(v,bool) and clsp(k)=="semantic":
                pv[k].append(v)
    for p,vals in pv.items():
        dist=sorted(set(vals))
        if len(dist)>=3:
            ins.append(I("parameter_range",f"'{p}' takes {len(dist)} values: {dist[:8]}",
                {"param":p,"values":dist[:20]},min(len(dist)/8,1),[p]))
    seen=set(); rins=[]
    for t in tables:
        recs,hdrs=t["records"],t["headers"]
        ncols=[h for h in hdrs if sum(1 for r in recs if isinstance(r.get(h),(int,float)))>=2 and ilm(h,[r.get(h) for r in recs])]
        if not ncols: continue
        lcol=None
        for h in hdrs:
            if sum(1 for r in recs if isinstance(r.get(h),str) and len(str(r.get(h,"")))>1)>=2:
                lcol=h; break
        for col in ncols:
            vals=[(r.get(col),r) for r in recs if isinstance(r.get(col),(int,float))]
            if len(vals)<2: continue
            vs=sorted(vals,key=lambda x:x[0],reverse=True)
            bv,br=vs[0]; wv,wr=vs[-1]
            bl=str(br.get(lcol,"top")) if lcol else "top"
            wl=str(wr.get(lcol,"bottom")) if lcol else "bottom"
            dk=(col,bl,wl,bv,wv)
            if dk in seen: continue
            seen.add(dk)
            g=bv-wv; rg=g/max(abs(wv),1e-6)
            if rg>0.01:
                ar=[{"label":str(r.get(lcol,"e") if lcol else "e"),"value":v} for v,r in vs]
                i=I("result_comparison",f"On '{col}': '{bl}'={bv} vs '{wl}'={wv} ({rg*100:.1f}% gap)",
                    {"metric":col,"best":{"label":bl,"value":bv},"worst":{"label":wl,"value":wv},
                     "all_results":ar,"n_entries":len(vals)},min(rg,1),[col])
                ins.append(i); rins.append(i)
    if len(cfgs)>=3:
        ak=set()
        for c in cfgs: ak.update(c["params"].keys())
        const,vary={},{}
        for k in ak:
            if clsp(k)=="noise": continue
            vs=set(str(c["params"].get(k,"")) for c in cfgs if k in c["params"])
            if len(vs)==1 and clsp(k)=="semantic": const[k]=list(vs)[0]
            elif len(vs)>1: vary[k]=len(vs)
        if const and vary:
            sv={k:v for k,v in vary.items() if clsp(k)=="semantic"}
            ins.append(I("experiment_family",f"{len(const)} fixed, {len(sv)} varying semantic params",
                {"n_configs":len(cfgs),"varying":dict(sorted(sv.items(),key=lambda x:-x[1])[:8])},
                1.0 if sv else 0.5,list(sv.keys())[:10]))
    rbl={}
    for i in rins:
        for row in i.ev.get("all_results",[]):
            lb=str(row["label"]).lower().strip(); m=i.ev["metric"]
            if lb not in rbl: rbl[lb]={}
            rbl[lb][m]=row["value"]
    if rbl:
        n2r={}
        for node in nodes:
            bm,bs=None,0
            did=node.dsid.lower() if node.dsid else ""
            sp=node.src.lower().replace("/"," ").replace("_"," ").replace("-"," ")
            for lb in rbl:
                lc=lb.replace("-","").replace("_","").replace(" ","")
                sc=0
                if did and did in lc: sc+=3
                if did and lc in did: sc+=3
                for pt in sp.split():
                    if len(pt)>2 and pt in lc: sc+=2
                sc+=SequenceMatcher(None,did,lc).ratio()*2
                if sc>bs: bs=sc; bm=lb
            if bm and bs>=2: n2r[node.id]={"label":bm,"metrics":rbl[bm],"score":bs}
        for edge in edges:
            sr,tr=n2r.get(edge.s),n2r.get(edge.t)
            if not sr or not tr: continue
            if sr["label"]==tr["label"]: continue
            if nodes[edge.s].etype!=nodes[edge.t].etype: continue
            for m in set(sr["metrics"])&set(tr["metrics"]):
                vs_m,vt_m=sr["metrics"][m],tr["metrics"][m]
                if not isinstance(vs_m,(int,float)) or not isinstance(vt_m,(int,float)): continue
                d=vt_m-vs_m
                if abs(d)<0.01: continue
                cc=[f"{k}:{v1}→{v2}" for k,v1,v2 in edge.sd]
                sds=nodes[edge.s].dsid or sr["label"]; tds=nodes[edge.t].dsid or tr["label"]
                dp=[]
                if sds!=tds: dp.append(f"Dataset:{sds}→{tds}")
                if cc: dp.append(f"Config:{','.join(cc)}")
                dp.append(f"{m}:{vs_m}→{vt_m}({'↑' if d>0 else '↓'}{abs(d):.2f})")
                ins.append(I("config_result_link"," | ".join(dp),
                    {"metric":m,"value_from":vs_m,"value_to":vt_m,"delta":round(d,4),
                     "config_changes":{k:{"from":v1,"to":v2} for k,v1,v2 in edge.sd}},
                    min(abs(d)/max(abs(vs_m),1),1),[dd[0] for dd in edge.sd]+[m]))
    ins.sort(key=lambda x:x.sig,reverse=True)
    return ins

def analyze(rp):
    yc=ext_yaml(rp); ac=ext_argparse(rp); sc=ext_shell(rp); rt=ext_readme(rp)
    all_c=yc+ac+sc
    stats={"yaml":len(yc),"argparse":len(ac),"shell":len(sc)}
    if not all_c:
        return {"status":"no_configs","n_configs":0,"n_insights":0,"insight_types":{},
                "insights":[],"extraction":stats,"n_result_tables":len(rt),"n_edges":0}
    ap=set()
    for c in all_c: ap.update(c["params"].keys())
    pc={"semantic":sum(1 for p in ap if clsp(p)=="semantic"),
        "identity":sum(1 for p in ap if clsp(p)=="identity"),
        "noise":sum(1 for p in ap if clsp(p)=="noise")}
    nodes,edges=build_g(all_c)
    et=defaultdict(int)
    for e in edges: et[e.et]+=1
    ins=mine(all_c,nodes,edges,rt)
    tc=defaultdict(int)
    for i in ins: tc[i.tp]+=1
    ct="\n".join([f"--- {c['source_file']} ---\n"+"\n".join(f"  {k}: {v}" for k,v in list(c["params"].items())[:25])
        for c in all_c[:5]])
    tt="\n".join([f"--- Table ---\nHeaders:{t['headers']}\n"+"\n".join(f"  {r}" for r in t["records"][:4]) for t in rt[:3]])
    return {"status":"ok","n_configs":len(all_c),"n_edges":len(edges),"n_insights":len(ins),
        "n_result_tables":len(rt),"param_class":pc,"edge_types":dict(et),"insight_types":dict(tc),
        "extraction":stats,"configs_text":ct,"tables_text":tt,
        "insights":[{"type":i.tp,"description":i.desc,"significance":i.sig,
            "params":i.par,"evidence":i.ev} for i in ins]}

def run_repo(entry):
    url=entry.get("repo_url","").rstrip("/")
    if url in SKIP_REPOS:
        return {"status":"skipped","n_configs":0,"n_insights":0,"insight_types":{},
                "insights":[],"extraction":{},"n_result_tables":0,"n_edges":0},0.0
    rd=WORK_DIR/"current_repo"
    if rd.exists(): shutil.rmtree(rd,ignore_errors=True)
    t0=time.time()
    if not sclone(entry["repo_url"],rd):
        return {"status":"clone_failed","n_configs":0,"n_insights":0,"insight_types":{},
                "insights":[],"extraction":{},"n_result_tables":0,"n_edges":0},time.time()-t0
    try:
        f=io.StringIO()
        with contextlib.redirect_stdout(f):
            r=analyze(rd)
    except Exception as e:
        r={"status":"error","error":str(e)[:200],"n_configs":0,"n_insights":0,
           "insight_types":{},"insights":[],"extraction":{},"n_result_tables":0,"n_edges":0}
    if rd.exists(): shutil.rmtree(rd,ignore_errors=True)
    return r,time.time()-t0

def call_llm(prompt,api_key,max_tokens=1500):
    import requests
    try:
        resp=requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key":api_key,"content-type":"application/json","anthropic-version":"2023-06-01"},
            json={"model":"claude-sonnet-4-20250514","max_tokens":max_tokens,
                  "messages":[{"role":"user","content":prompt}]},timeout=60)
        if resp.status_code==200:
            t=resp.json()["content"][0]["text"]
            return {"text":t,"tokens":len(prompt)//4+len(t)//4,"ok":True}
        return {"text":f"Error {resp.status_code}","tokens":0,"ok":False}
    except Exception as e:
        return {"text":str(e),"tokens":0,"ok":False}

def select(dataset,n,with_res=False):
    seen=set(); u=[]
    for e in dataset:
        url=e.get("repo_url","").rstrip("/").lower()
        if url and url not in seen: seen.add(url); u.append(e)
    if with_res:
        return [e for e in u if e.get("has_structured_results")][:n]
    # Group by venue, cap per venue to ensure diversity
    bv=defaultdict(list)
    for e in u:
        vid=e.get("venue_id",e.get("venue_label","unknown"))
        bv[vid].append(e)
    n_venues=max(len(bv),1)
    per_venue=max(n//n_venues,5)  # at least 5 per venue
    sel=[]
    # First pass: take up to per_venue from each venue
    for v,ents in sorted(bv.items()):
        sel.extend(ents[:per_venue])
    # Second pass: if still under n, fill from largest venues
    if len(sel)<n:
        for v,ents in sorted(bv.items(),key=lambda x:-len(x[1])):
            for e in ents[per_venue:]:
                if len(sel)>=n: break
                sel.append(e)
    return sel[:n]

# =====================================================================
# PHASE 1: Large-scale mining
# =====================================================================
def phase1(dataset,max_repos,resume=True):
    print("\n"+"="*70)
    print(f"PHASE 1: LARGE-SCALE MINING ({max_repos} repos)")
    print("="*70)
    sel=select(dataset,max_repos)
    print(f"Selected {len(sel)} repos\n")
    
    # Resume: load previously processed results
    results=[]
    done_urls=set()
    incremental_path=RESULTS_DIR/"phase1_incremental.json"
    if resume and incremental_path.exists():
        try:
            with open(incremental_path) as f:
                prev=json.load(f)
            results=prev
            done_urls={r.get("url","").rstrip("/").lower() for r in prev if r.get("url")}
            print(f"  RESUMING: loaded {len(prev)} already-processed repos, skipping them\n")
        except:
            print("  Could not load incremental file, starting fresh\n")
    
    total_time=0
    skipped=0
    for i,entry in enumerate(sel):
        # Skip already processed
        entry_url=entry.get("repo_url","").rstrip("/").lower()
        if entry_url in done_urls:
            skipped+=1
            continue
        
        title=entry.get("title","")[:50]
        venue=entry.get("venue_id",entry.get("venue","OCP"))[:20]
        print(f"[{i+1}/{len(sel)}] {title}...",end=" ",flush=True)
        a,t=run_repo(entry)
        total_time+=t
        r={"idx":i,"title":title,"url":entry.get("repo_url"),"venue":venue,"time":round(t,1)}
        r.update({k:a.get(k,0) for k in ["status","n_configs","n_edges","n_insights","n_result_tables"]})
        r["insight_types"]=a.get("insight_types",{})
        r["extraction"]=a.get("extraction",{})
        r["top3"]=[{"type":ins["type"],"desc":ins["description"][:100],"sig":ins["significance"]}
            for ins in a.get("insights",[])[:3]]
        # IMPORTANT: Keep full insights for community hypothesis pipeline
        r["_insights"]=a.get("insights",[])
        r["_configs_text"]=a.get("configs_text","")
        r["_tables_text"]=a.get("tables_text","")
        if a.get("status")=="skipped": print("SKIPPED")
        else: print(f"cfg={r['n_configs']} ins={r['n_insights']} ({t:.0f}s)")
        results.append(r)

        # Save incrementally every 25 NEW repos
        new_count=len(results)-len(done_urls)
        if new_count > 0 and new_count % 25 == 0:
            _save_phase1_incremental(results)

    if skipped:
        print(f"\n  Skipped {skipped} already-processed repos")
    print(f"  New repos processed: {len(results)-len(done_urls)}")
    print(f"  Total in results: {len(results)}")
    print(f"  New processing time: {total_time:.0f}s ({total_time/60:.1f}min)")
    
    # Final save
    _save_phase1_incremental(results)
    return results

def _save_phase1_incremental(results):
    """Save Phase 1 preserving _insights, dropping bulky text fields."""
    save_data = [{k:v for k,v in r.items() if k not in ("_configs_text","_tables_text")}
                 for r in results]
    with open(RESULTS_DIR/"phase1_incremental.json","w") as f:
        json.dump(save_data,f,indent=1,default=str)

# =====================================================================
# PHASE 2-7: Same as original (abbreviated comments)
# =====================================================================
def phase2(dataset,p1_results,max_repos):
    print("\n"+"="*70)
    print("PHASE 2: OCP CROSS-LINKING")
    print("="*70)
    ocp=[e for e in dataset if e.get("has_structured_results") and e.get("ocp_benchmarks")]
    sel=ocp[:max_repos]
    print(f"OCP repos with benchmarks: {len(sel)}\n")
    results=[]
    for i,entry in enumerate(sel):
        title=entry.get("title","")[:50]
        ocp_data=entry.get("ocp_benchmarks",[])
        ocp_metrics={}
        for bm in ocp_data:
            mn=bm.get("model_name",bm.get("paper_title","unknown"))
            met=bm.get("metrics",{})
            if met: ocp_metrics[str(mn).lower().strip()]={k:v for k,v in met.items() if isinstance(v,(int,float))}
        url=entry.get("repo_url","").rstrip("/")
        p1_match=None
        for r in p1_results:
            if r.get("url","").rstrip("/")==url: p1_match=r; break
        if p1_match and p1_match.get("status")=="ok":
            n_configs=p1_match["n_configs"]; insights=p1_match.get("_insights",[])
        else:
            print(f"[{i+1}/{len(sel)}] {title}...",end=" ",flush=True)
            a,t=run_repo(entry)
            n_configs=a.get("n_configs",0); insights=a.get("insights",[])
            print(f"cfg={n_configs}")
        links=[]
        for ok,om in ocp_metrics.items():
            for ins in insights:
                if ins["type"] in ("clean_ablation","cross_dataset_ablation","experiment_family","dataset_comparison"):
                    links.append({"ocp_model":ok,"ocp_metrics":om,"viveka_insight":ins["description"][:100],"link_type":"enrichment"})
                    break
        results.append({"title":title,"url":url,"n_configs":n_configs,"n_ocp":len(ocp_metrics),"cross_links":len(links),"details":links[:3]})
    with_links=sum(1 for r in results if r["cross_links"]>0)
    analyzed=sum(1 for r in results if r["n_configs"]>0)
    print(f"\n  Analyzed: {analyzed}, With cross-links: {with_links} ({100*with_links//max(analyzed,1)}%)")
    return results

def phase3(p1_results,api_key,max_repos=30):
    print("\n"+"="*70)
    print(f"PHASE 3: LLM QUALITY SCORING ({max_repos} repos)")
    print("="*70)
    if not api_key: print("  Skipped"); return {"skipped":True}
    candidates=[r for r in p1_results if r.get("status")=="ok" and r.get("n_insights",0)>0]
    sel=candidates[:max_repos]
    print(f"Scoring insights from {len(sel)} repos\n")
    all_scores=[]; results=[]
    for i,r in enumerate(sel):
        title=r["title"][:50]; insights=r.get("_insights",[])[:15]
        if not insights: continue
        print(f"[{i+1}/{len(sel)}] {title}...",end=" ",flush=True)
        itxt="\n".join([f"{j+1}. [{ins['type']}] {ins['description']}" for j,ins in enumerate(insights)])
        prompt=f"""Rate each insight 1-5:\n1=Trivial 2=Low 3=Moderate 4=Valuable 5=Highly valuable\nINSIGHTS:\n{itxt}\nRespond ONLY JSON array: [{{"n":1,"s":3,"r":"reason"}},...]. No markdown."""
        llm=call_llm(prompt,api_key,1500); scores=_ps(llm["text"]); all_scores.extend(scores)
        avg=sum(scores)/len(scores) if scores else 0
        type_scores=defaultdict(list)
        for j,s in enumerate(scores):
            if j<len(insights): type_scores[insights[j]["type"]].append(s)
        print(f"{len(scores)} rated, avg={avg:.1f}")
        results.append({"title":title,"n_rated":len(scores),"avg":round(avg,2),
            "type_scores":{t:round(sum(s)/len(s),2) for t,s in type_scores.items()}})
    if all_scores:
        print(f"\n  Overall: {len(all_scores)} insights rated, mean={sum(all_scores)/len(all_scores):.2f}")
    return {"results":results,"all_scores":all_scores,"mean":round(sum(all_scores)/len(all_scores),2) if all_scores else 0}

def _ps(text):
    scores=[]
    try:
        t=text.strip()
        if t.startswith("```"): t=t.split("```")[1].strip()
        if t.startswith("json"): t=t[4:].strip()
        for item in json.loads(t):
            s=item.get("s",item.get("score",0))
            if isinstance(s,(int,float)) and 1<=s<=5: scores.append(int(s))
    except: pass
    return scores

def phase4(p1_results,api_key,max_repos=20):
    print("\n"+"="*70)
    print(f"PHASE 4: HYBRID VIVEKA+LLM ({max_repos} repos)")
    print("="*70)
    if not api_key: print("  Skipped"); return {"skipped":True}
    candidates=[r for r in p1_results if r.get("status")=="ok" and r.get("n_insights",0)>0]
    sel=candidates[:max_repos]; print(f"Testing {len(sel)} repos\n"); results=[]
    for i,r in enumerate(sel):
        title=r["title"][:50]; insights=r.get("_insights",[])[:15]
        ct=r.get("_configs_text","")[:2000]; tt=r.get("_tables_text","")[:1500]
        if not insights: continue
        print(f"[{i+1}/{len(sel)}] {title}...",end=" ",flush=True)
        itxt="\n".join([f"{j+1}. [{ins['type']}] {ins['description']}" for j,ins in enumerate(insights)])
        prompt=f"""VIVEKA found these insights:\n{itxt}\nCONFIGS:\n{ct}\nRESULTS:\n{tt}\nFor each: verdict, enhancement.\nJSON array: [{{"n":1,"v":"correct","e":"enhancement or null"}},...]. Only JSON."""
        llm=call_llm(prompt,api_key,2000)
        sp=f"""Analyze these configs and results. List every insight.\nCONFIGS:\n{ct}\nRESULTS:\n{tt}\nNumbered list."""
        llm2=call_llm(sp,api_key,1500)
        standalone_n=len([l for l in llm2["text"].split("\n") if l.strip() and l.strip()[0].isdigit()])
        verified=0; enhanced=0
        try:
            t=llm["text"].strip()
            if t.startswith("```"): t=t.split("```")[1].strip()
            if t.startswith("json"): t=t[4:].strip()
            for item in json.loads(t):
                v=item.get("v","")
                if v in ("correct","partially_correct"): verified+=1
                if item.get("e") and item["e"]!="null": enhanced+=1
        except: verified=llm["text"].lower().count("correct")
        viveka_n=r["n_insights"]; hybrid_n=viveka_n+enhanced
        print(f"V={viveka_n} LLM={standalone_n} Hybrid={hybrid_n} ver={verified}")
        results.append({"title":title,"viveka_n":viveka_n,"standalone_n":standalone_n,
            "hybrid_n":hybrid_n,"verified":verified,"enhanced":enhanced,
            "tokens":llm.get("tokens",0)+llm2.get("tokens",0)})
    return results

def phase5(p1_results):
    print("\n"+"="*70)
    print("PHASE 5: DOMAIN ANALYSIS")
    print("="*70)
    by_venue=defaultdict(list)
    for r in p1_results: by_venue[r.get("venue","unknown")[:30]].append(r)
    print(f"\n  {'Venue':<30} {'Repos':>5} {'Analyzed':>8} {'w/Insights':>10} {'Avg Ins':>8}")
    print(f"  {'-'*65}")
    domain_stats={}
    for venue,repos in sorted(by_venue.items(),key=lambda x:-len(x[1])):
        ok=[r for r in repos if r.get("status")=="ok"]
        wi=[r for r in ok if r["n_insights"]>0]
        avg_ins=sum(r["n_insights"] for r in ok)/len(ok) if ok else 0
        print(f"  {venue:<30} {len(repos):>5} {len(ok):>8} {len(wi):>10} {avg_ins:>7.1f}")
        domain_stats[venue]={"total":len(repos),"analyzed":len(ok),"with_insights":len(wi),"avg_insights":round(avg_ins,1)}
    return domain_stats

def phase6(p1_results,dataset,api_key,max_repos=20):
    print("\n"+"="*70)
    print("PHASE 6: NOVEL INSIGHT DETECTION")
    print("="*70)
    if not api_key: print("  Skipped"); return {"skipped":True}
    candidates=[]
    for r in p1_results:
        if r.get("status")!="ok" or r.get("n_insights",0)==0: continue
        url=r.get("url","").rstrip("/")
        for e in dataset:
            if e.get("repo_url","").rstrip("/")==url:
                abstract=e.get("abstract","")
                if abstract: candidates.append({"repo_result":r,"abstract":abstract,"title":r["title"]}); break
    sel=candidates[:max_repos]; print(f"Repos with abstracts: {len(sel)}\n"); results=[]
    for i,c in enumerate(sel):
        title=c["title"][:50]; insights=c["repo_result"].get("_insights",[])[:10]
        if not insights: continue
        print(f"[{i+1}/{len(sel)}] {title}...",end=" ",flush=True)
        itxt="\n".join([f"{j+1}. [{ins['type']}] {ins['description']}" for j,ins in enumerate(insights)])
        prompt=f"""PAPER ABSTRACT:\n{c['abstract'][:1500]}\n\nAUTOMATED TOOL FINDINGS:\n{itxt}\n\nClassify each: "reported"/"implicit"/"novel".\nJSON array: [{{"n":1,"class":"reported/implicit/novel","reason":"brief"}},...]. Only JSON."""
        llm=call_llm(prompt,api_key,1500)
        novel=0; implicit=0; reported=0; details=[]
        try:
            t=llm["text"].strip()
            if t.startswith("```"): t=t.split("```")[1].strip()
            if t.startswith("json"): t=t[4:].strip()
            for item in json.loads(t):
                cl=item.get("class","")
                if cl=="novel": novel+=1
                elif cl=="implicit": implicit+=1
                elif cl=="reported": reported+=1
                details.append({"insight_n":item.get("n",0),"class":cl,"reason":item.get("reason","")[:100]})
        except: pass
        print(f"reported={reported} implicit={implicit} novel={novel}")
        results.append({"title":title,"reported":reported,"implicit":implicit,"novel":novel,"total":len(insights),"details":details[:5]})
    total_novel=sum(r["novel"] for r in results); total_all=sum(r["total"] for r in results)
    print(f"\n  Total: {total_all}, Novel: {total_novel} ({100*total_novel//max(total_all,1)}%)")
    return results

def phase7(p1,p4):
    print("\n"+"="*70)
    print("PHASE 7: COST ANALYSIS")
    print("="*70)
    ok=[r for r in p1 if r.get("status")=="ok"]; mean_t=0
    if ok:
        times=[r["time"] for r in ok]; mean_t=sum(times)/len(times)
        print(f"\n  VIVEKA: {mean_t:.1f}s/repo, $0")
        print(f"  At 5000 repos: ~{mean_t*5000/3600:.0f} hours, $0")
    total_tokens=0
    if isinstance(p4,list):
        total_tokens=sum(r.get("tokens",0) for r in p4)
        per_repo=total_tokens/max(len(p4),1)
        print(f"\n  LLM: {total_tokens:,} tokens, ~${total_tokens*5/1e6:.2f}")
    return {"viveka_mean_time":round(mean_t,1),"llm_tokens":total_tokens,"llm_cost":round(total_tokens*5/1e6,2)}

def report(p1,p2,p3,p4,p5,p6,p7r):
    L=["="*70,"VIVEKA FULL-SCALE EVALUATION — ICDM 2026","="*70]
    ok=[r for r in p1 if r.get("status")=="ok"]
    L.append(f"\n--- PHASE 1: LARGE-SCALE MINING ---")
    L.append(f"Repos: {len(p1)} attempted, {len(ok)} analyzed")
    if ok:
        wi=sum(1 for r in ok if r["n_insights"]>0)
        L.append(f"With insights: {wi}/{len(ok)} ({100*wi//len(ok)}%)")
        L.append(f"Total insights: {sum(r['n_insights'] for r in ok)}")
        L.append(f"Mean configs: {sum(r['n_configs'] for r in ok)/len(ok):.1f}")
        L.append(f"Mean time: {sum(r['time'] for r in ok)/len(ok):.1f}s")
        ty=sum(r.get("extraction",{}).get("yaml",0) for r in ok)
        ta=sum(r.get("extraction",{}).get("argparse",0) for r in ok)
        ts=sum(r.get("extraction",{}).get("shell",0) for r in ok)
        L.append(f"Extraction: yaml={ty} argparse={ta} shell={ts}")
        it=defaultdict(int)
        for r in ok:
            for t,c in r.get("insight_types",{}).items(): it[t]+=c
        for t,c in sorted(it.items(),key=lambda x:-x[1]): L.append(f"  {t}: {c}")
    L.append(f"\n--- PHASE 5: DOMAIN ANALYSIS ---")
    if isinstance(p5,dict):
        for v,s in sorted(p5.items(),key=lambda x:-x[1]["total"]):
            L.append(f"  {v[:25]:<26} repos={s['total']} analyzed={s['analyzed']} insights={s['with_insights']} avg={s['avg_insights']}")
    L.append(f"\n--- PHASE 7: COST ---")
    L.append(f"VIVEKA: {p7r.get('viveka_mean_time',0)}s/repo, $0")
    L.append(f"LLM: ${p7r.get('llm_cost',0):.2f}")
    return "\n".join(L)

# =====================================================================
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--max-repos",type=int,default=500)
    parser.add_argument("--dataset",default="./expmine_output/merged_dataset.json")
    parser.add_argument("--skip-llm",action="store_true")
    parser.add_argument("--no-resume",action="store_true",help="Start fresh, ignore previous incremental results")
    args=parser.parse_args()

    with open(args.dataset) as f:
        dataset=json.load(f)
    print(f"Loaded {len(dataset)} entries\n")
    api_key="" if args.skip_llm else os.environ.get("ANTHROPIC_API_KEY","")

    p1=phase1(dataset,args.max_repos,resume=not args.no_resume)
    p2=phase2(dataset,p1,max_repos=50)
    p3=phase3(p1,api_key,max_repos=30)
    p4=phase4(p1,api_key,max_repos=20)
    p5=phase5(p1)
    p6=phase6(p1,dataset,api_key,max_repos=20)
    p7r=phase7(p1,p4 if isinstance(p4,list) else [])

    rpt=report(p1,p2,p3,p4,p5,p6,p7r)
    print("\n\n"+rpt)

    # Save all — FIXED: preserve _insights in phase1
    for name,data in [("phase1",p1),("phase2",p2),("phase3",p3),("phase4",p4),
                      ("phase5",p5),("phase6",p6),("phase7",p7r)]:
        with open(RESULTS_DIR/f"{name}.json","w") as f:
            if name=="phase1":
                # FIXED: Keep _insights for community hypothesis pipeline
                # Only drop _configs_text and _tables_text (bulky, not needed)
                save_data=[{k:v for k,v in r.items()
                           if k not in ("_configs_text","_tables_text")}
                          for r in data]
                json.dump(save_data,f,indent=1,default=str)
            else:
                json.dump(data,f,indent=2,default=str)
    with open(RESULTS_DIR/"full_report.txt","w") as f:
        f.write(rpt)
    print(f"\nAll saved to {RESULTS_DIR}/")

if __name__=="__main__":
    main()
