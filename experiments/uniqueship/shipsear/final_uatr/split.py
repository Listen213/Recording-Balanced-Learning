from __future__ import annotations
from collections import Counter,defaultdict
from pathlib import Path
import csv, hashlib, math, random, re
import numpy as np
import torch, torchaudio

TAXONOMY={
    "ocean liner":"A_Large_Merchant_Ship",
    "roro":"A_Large_Merchant_Ship",
    "dredger":"A_Large_Merchant_Ship",
    "motorboat":"B_Small_Speedboat",
    "tugboat":"B_Small_Speedboat",
    "pilot ship":"B_Small_Speedboat",
    "sailboat":"B_Small_Speedboat",
    "yacht":"B_Small_Speedboat",
    "high speed motorboat":"B_Small_Speedboat",
    "zodiac":"B_Small_Speedboat",
    "fishboat":"C_Fishing_Boat",
    "trawler":"C_Fishing_Boat",
    "mussel boat":"C_Fishing_Boat",
    "passengers":"D_Ferry",
    "natural ambient noise":"E_Ambient_Noise",
}
LABELS=[
    "A_Large_Merchant_Ship",
    "B_Small_Speedboat",
    "C_Fishing_Boat",
    "D_Ferry",
    "E_Ambient_Noise",
]
LABEL_TO_IDX={x:i for i,x in enumerate(LABELS)}

def norm_type(s):
    return re.sub(r"\s+"," ",str(s).strip().lower())

def sha256_file(path, chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def build_recordings(audio_root,csv_path):
    root=Path(audio_root)
    wavs={p.name.lower():p for p in root.rglob("*.wav")}
    rows=list(csv.DictReader(open(csv_path,"r",encoding="utf-8-sig",newline="")))
    if not rows:
        raise RuntimeError("empty metadata CSV")
    if not {"filename","type"}.issubset(rows[0].keys()):
        raise ValueError("CSV must contain columns filename,type")

    out=[]
    for row in rows:
        fn=str(row["filename"]).strip()
        src=str(row["type"]).strip()
        key=norm_type(src)
        if key not in TAXONOMY:
            raise RuntimeError(f"Unknown ShipsEar type: {src!r}")
        p=wavs.get(fn.lower()) or wavs.get((fn+".wav").lower())
        if p is None:
            raise FileNotFoundError(f"WAV not found for CSV filename={fn!r}")
        info=torchaudio.info(str(p))
        lab=TAXONOMY[key]
        out.append({
            "path":str(p.resolve()),
            "filename":p.name,
            "recording_id":Path(fn).stem,
            "source_type":src,
            "source_type_norm":key,
            "label_name":lab,
            "label":LABEL_TO_IDX[lab],
            "sample_rate":int(info.sample_rate),
            "num_frames":int(info.num_frames),
            "duration_sec":float(info.num_frames/max(1,info.sample_rate)),
            "channels":int(info.num_channels),
            "sha256":sha256_file(p),
        })
    return out

def _window_spectrum(w, sr, target_sr=8000):
    w=w.float()
    if sr!=target_sr:
        w=torchaudio.functional.resample(w,sr,target_sr)
        sr=target_sr
    if w.numel()<1024:
        w=torch.nn.functional.pad(w,(0,1024-w.numel()))
    sp=torch.stft(
        w,n_fft=1024,hop_length=512,win_length=1024,
        window=torch.hann_window(1024),return_complex=True
    ).abs().pow(2).mean(-1)
    x=torch.log1p(sp)
    x=torch.nn.functional.interpolate(
        x[None,None,:],size=128,mode="linear",align_corners=False
    )[0,0]
    x=(x-x.mean())/(x.std(unbiased=False)+1e-6)
    return x.numpy().astype(np.float32)

def acoustic_fingerprint(path, target_sr=8000, window_sec=20.0, n_windows=3):
    """
    Multi-window spectral fingerprint: start / middle / end when possible.
    It is intentionally simple and deterministic. Near-duplicate grouping is a
    leakage-control device, not a learned feature used by the classifier.
    """
    w,sr=torchaudio.load(path)
    w=w.mean(0)
    win=max(1024,int(round(float(window_sec)*sr)))
    n=w.numel()
    if n<=win:
        starts=[0]
    else:
        if n_windows<=1:
            starts=[0]
        else:
            starts=np.linspace(0,n-win,num=int(n_windows)).round().astype(int).tolist()
    feats=[]
    for st in starts:
        feats.append(_window_spectrum(w[st:st+win],sr,target_sr))
    # Repeat last view if the file is short, so fingerprint dimension is fixed.
    while len(feats)<int(n_windows):
        feats.append(feats[-1].copy())
    x=np.concatenate(feats[:int(n_windows)],axis=0)
    x=(x-x.mean())/(x.std()+1e-6)
    return x.astype(np.float32)

def cosine(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    return float(np.dot(a,b)/((np.linalg.norm(a)+1e-12)*(np.linalg.norm(b)+1e-12)))

class DSU:
    def __init__(self,n):
        self.p=list(range(n)); self.sz=[1]*n
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]
            x=self.p[x]
        return x
    def union(self,a,b):
        a=self.find(a); b=self.find(b)
        if a==b: return
        if self.sz[a]<self.sz[b]: a,b=b,a
        self.p[b]=a; self.sz[a]+=self.sz[b]

def build_components(records, fingerprints, threshold=0.997):
    n=len(records)
    dsu=DSU(n)
    by_sha=defaultdict(list)
    for i,r in enumerate(records):
        by_sha[r["sha256"]].append(i)
    for ids in by_sha.values():
        for j in ids[1:]:
            dsu.union(ids[0],j)

    pairs=[]
    for i in range(n):
        for j in range(i+1,n):
            sim=cosine(fingerprints[i],fingerprints[j])
            if sim>=float(threshold):
                dsu.union(i,j)
                pairs.append({
                    "recording_a":records[i]["recording_id"],
                    "recording_b":records[j]["recording_id"],
                    "label_a":records[i]["label_name"],
                    "label_b":records[j]["label_name"],
                    "cosine":sim,
                })

    groups=defaultdict(list)
    for i,r in enumerate(records):
        groups[dsu.find(i)].append(i)

    comps=[]
    conflicts=[]
    for k,ids in groups.items():
        labs={records[i]["label_name"] for i in ids}
        if len(labs)>1:
            conflicts.append({
                "recordings":[records[i]["recording_id"] for i in ids],
                "labels":sorted(labs),
            })
            continue
        rid_sorted=sorted(records[i]["recording_id"] for i in ids)
        cid="cmp_"+hashlib.sha1("|".join(rid_sorted).encode()).hexdigest()[:12]
        comps.append({
            "component_id":cid,
            "label_name":next(iter(labs)),
            "label":records[ids[0]]["label"],
            "indices":ids,
            "recording_ids":rid_sorted,
            "size":len(ids),
            "source_types":sorted({records[i]["source_type_norm"] for i in ids}),
        })
    return comps,sorted(pairs,key=lambda x:-x["cosine"]),conflicts

def _valid_remove(test_set, cand, comps, records, min_support_components):
    new=set(test_set)|{cand}
    lab=comps[cand]["label_name"]
    class_ids=[i for i,c in enumerate(comps) if c["label_name"]==lab]
    support=[i for i in class_ids if i not in new]
    if len(support)<int(min_support_components):
        return False

    # Keep at least one support component for every raw subtype observed in class.
    all_sub=Counter()
    sup_sub=Counter()
    for i in class_ids:
        for s in comps[i]["source_types"]:
            all_sub[s]+=1
    for i in support:
        for s in comps[i]["source_types"]:
            sup_sub[s]+=1
    for s in all_sub:
        if sup_sub[s]<1:
            return False
    return True

def split_components(records, comps, seed=20261014, test_ratio=0.30, min_support_components=6):
    """
    Atomic near-duplicate components are assigned wholly to support_pool or test.
    The support pool must retain >= min_support_components per coarse class and
    at least one component for every raw subtype represented in that class.
    """
    rng=random.Random(int(seed))
    test_comp=set()

    for lab in LABELS:
        ids=[i for i,c in enumerate(comps) if c["label_name"]==lab]
        rng.shuffle(ids)
        if len(ids)<int(min_support_components)+1:
            raise RuntimeError(
                f"{lab} has only {len(ids)} independent components; "
                f"need >= {int(min_support_components)+1}"
            )
        total_rec=sum(comps[i]["size"] for i in ids)
        target=max(1,int(round(total_rec*float(test_ratio))))

        # Try small/randomized components first to approach target smoothly.
        ids=sorted(ids,key=lambda i:(comps[i]["size"],rng.random()))
        count=0
        improved=True
        while improved:
            improved=False
            best=None; best_gap=abs(target-count)
            for i in ids:
                if i in test_comp: continue
                if not _valid_remove(test_comp,i,comps,records,min_support_components):
                    continue
                nc=count+comps[i]["size"]
                gap=abs(target-nc)
                # Prefer moves toward target; require at least one test component.
                if gap<best_gap or (count==0 and best is None):
                    best=i; best_gap=gap
            if best is not None and (count==0 or best_gap<abs(target-count)):
                test_comp.add(best)
                count+=comps[best]["size"]
                improved=True

        if not any(i in test_comp for i in ids):
            valid=[i for i in ids if _valid_remove(test_comp,i,comps,records,min_support_components)]
            if not valid:
                raise RuntimeError(f"Cannot allocate a test component for {lab}")
            best=min(valid,key=lambda i:abs(target-comps[i]["size"]))
            test_comp.add(best)

    support=[]; test=[]
    for i,c in enumerate(comps):
        target=test if i in test_comp else support
        for ridx in c["indices"]:
            rr=dict(records[ridx])
            rr["component_id"]=c["component_id"]
            target.append(rr)

    return support,test,test_comp

def audit_split(support,test,near_pairs):
    sr={r["recording_id"] for r in support}
    tr={r["recording_id"] for r in test}
    sh={r["sha256"] for r in support}
    th={r["sha256"] for r in test}
    cross_near=[
        p for p in near_pairs
        if ((p["recording_a"] in sr and p["recording_b"] in tr) or
            (p["recording_b"] in sr and p["recording_a"] in tr))
    ]
    return {
        "recording_id_overlap":len(sr&tr),
        "sha256_overlap":len(sh&th),
        "cross_split_near_duplicate_edges":cross_near,
        "cross_split_near_duplicate_count":len(cross_near),
    }

def class_recording_counts(rows):
    return dict(Counter(r["label_name"] for r in rows))

def class_component_counts(rows):
    by=defaultdict(set)
    for r in rows:
        by[r["label_name"]].add(r["component_id"])
    return {k:len(v) for k,v in by.items()}
