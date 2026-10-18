from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import csv, re
import numpy as np
import torch

@dataclass
class Recording:
    recording_id: str
    label: int
    embeddings: np.ndarray
    source: str | None = None
    start_sec: np.ndarray | None = None
    component_id: str | None = None

class RecordingCollection:
    def __init__(self, recordings, label_to_idx=None, manifest_hashes=None):
        self.recordings=list(recordings)
        self.label_to_idx=label_to_idx or {}
        self.manifest_hashes=manifest_hashes or []
        self.by_id={r.recording_id:r for r in self.recordings}
    def __len__(self): return len(self.recordings)
    @property
    def labels(self): return np.asarray([r.label for r in self.recordings],dtype=np.int64)
    def class_counts(self): return dict(Counter(int(r.label) for r in self.recordings))
    def component_counts(self):
        seen=set(); out=Counter()
        for r in self.recordings:
            k=r.component_id or r.recording_id
            key=(int(r.label),str(k))
            if key not in seen:
                seen.add(key); out[int(r.label)]+=1
        return dict(out)

    @classmethod
    def from_cache(cls,path,source_map=None):
        obj=torch.load(path,map_location="cpu",weights_only=False)
        emb=_np(obj.get("embedding",obj.get("embeddings")))
        labels=_np(obj.get("labels",obj.get("label"))).astype(np.int64)
        rids_raw=obj.get("recording_ids",obj.get("recording_id"))
        if emb is None or labels is None or rids_raw is None:
            raise KeyError(f"Cache {path} must contain embedding(s), labels, recording_ids")
        rids=[str(x) for x in list(rids_raw)]
        start_raw=obj.get("start_sec")
        starts=_np(start_raw).astype(float) if start_raw is not None else None
        comp_raw=obj.get("component_ids")
        comps=[str(x) for x in list(comp_raw)] if comp_raw is not None else None
        source_raw=obj.get("source_types")
        sources=[str(x) for x in list(source_raw)] if source_raw is not None else None

        groups={}; order=[]
        for i,rid in enumerate(rids):
            if rid not in groups:
                groups[rid]=[]; order.append(rid)
            groups[rid].append(i)

        recs=[]
        for rid in order:
            idx=np.asarray(groups[rid],dtype=np.int64)
            if starts is not None:
                idx=idx[np.argsort(starts[idx],kind="stable")]
            ys=labels[idx]
            vals,cnts=np.unique(ys,return_counts=True)
            y=int(vals[np.argmax(cnts)])
            src=(sources[idx[0]] if sources is not None else match_source(rid,source_map or {}))
            comp=(comps[idx[0]] if comps is not None else rid)
            recs.append(Recording(
                rid,y,emb[idx].astype(np.float32),src,
                starts[idx].copy() if starts is not None else None,
                comp
            ))
        mh=obj.get("manifest_sha256",[])
        if isinstance(mh,str): mh=[mh]
        return cls(recs,obj.get("label_to_idx",{}),mh)

def _np(x):
    if x is None: return None
    if isinstance(x,np.ndarray): return x
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    return np.asarray(x)

def _norm(s):
    s=Path(str(s)).stem.lower().strip()
    return re.sub(r"[^a-z0-9]+","",s)

def _rid_candidates(rid):
    s=Path(str(rid)).stem
    out=[s]
    if "__" in s:
        parts=s.split("__")
        if parts[0].isdigit():
            out.append("__".join(parts[1:]))
        if len(parts)>=3 and parts[0].isdigit() and len(parts[1])<=2:
            out.append("__".join(parts[2:]))
    return list(dict.fromkeys(out))

def match_source(rid,source_map):
    for c in _rid_candidates(rid):
        if c in source_map: return source_map[c]
        n=_norm(c)
        if n in source_map: return source_map[n]
    return None

def discover_source_map(meta_csv=None):
    out={}
    if not meta_csv or not Path(meta_csv).exists():
        return out
    with open(meta_csv,"r",encoding="utf-8-sig",errors="ignore",newline="") as f:
        for row in csv.DictReader(f):
            fn=row.get("filename")
            src=row.get("type")
            if not fn or not src: continue
            stem=Path(str(fn)).stem
            out[stem]=str(src).strip()
            out[_norm(stem)]=str(src).strip()
    return out
