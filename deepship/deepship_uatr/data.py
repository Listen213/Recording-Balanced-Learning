from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
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
    def labels(self):
        return np.asarray([r.label for r in self.recordings],dtype=np.int64)

    def class_counts(self):
        return dict(Counter(int(r.label) for r in self.recordings))

    def component_counts(self):
        seen=set(); out=Counter()
        for r in self.recordings:
            key=(int(r.label),str(r.component_id or r.recording_id))
            if key not in seen:
                seen.add(key)
                out[int(r.label)]+=1
        return dict(out)

    @classmethod
    def from_cache(cls,path):
        obj=torch.load(path,map_location="cpu",weights_only=False)
        emb=_np(obj.get("embedding",obj.get("embeddings")))
        labels=_np(obj.get("labels",obj.get("label"))).astype(np.int64)
        rids_raw=obj.get("recording_ids",obj.get("recording_id"))
        if emb is None or labels is None or rids_raw is None:
            raise KeyError(f"Cache {path} must contain embedding(s), labels, recording_ids")

        rids=[str(x) for x in list(rids_raw)]
        starts=_np(obj.get("start_sec"))
        starts=starts.astype(float) if starts is not None else None

        comps_raw=obj.get("component_ids")
        comps=[str(x) for x in list(comps_raw)] if comps_raw is not None else None

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
            recs.append(Recording(
                recording_id=rid,
                label=y,
                embeddings=emb[idx].astype(np.float32),
                source=(sources[idx[0]] if sources is not None else None),
                start_sec=(starts[idx].copy() if starts is not None else None),
                component_id=(comps[idx[0]] if comps is not None else rid),
            ))

        mh=obj.get("manifest_sha256",[])
        if isinstance(mh,str): mh=[mh]
        return cls(recs,obj.get("label_to_idx",{}),mh)

def _np(x):
    if x is None: return None
    if isinstance(x,np.ndarray): return x
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    return np.asarray(x)
