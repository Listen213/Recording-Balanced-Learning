from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

def load_yaml(path):
    with open(path,'r',encoding='utf-8') as f: return yaml.safe_load(f)

def save_json(obj,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False)

def canonical_mmsi(v):
    if pd.isna(v): return ''
    s=str(v).strip()
    if s.endswith('.0'):
        try: s=str(int(float(s)))
        except Exception: pass
    return s

def gini(a):
    a=np.sort(np.asarray(a,dtype=float))
    if len(a)==0 or a.sum()<=0: return 0.0
    n=len(a); return float((2*np.arange(1,n+1)-n-1).dot(a)/(n*a.sum()))

def cv(a):
    a=np.asarray(a,dtype=float)
    return float(a.std(ddof=0)/a.mean()) if len(a) and a.mean()>0 else 0.0

def sha256_file(path,block=1<<20):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()
