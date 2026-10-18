from __future__ import annotations
from pathlib import Path
from collections import defaultdict, Counter
import hashlib, json, math, os, re
import numpy as np
import torch, torchaudio, wave

CLASS_NAMES=["Cargo","Passengership","Tanker","Tug"]
LABEL_TO_IDX={c:i for i,c in enumerate(CLASS_NAMES)}

# Published DeepShip recording counts are used only as an inventory sanity check,
# never as labels or model information.
REFERENCE_RECORDING_COUNTS={
    "Cargo":110,
    "Passengership":193,
    "Tanker":240,
    "Tug":70,
}

DATE_RE=re.compile(r"^(20\d{6})")

def sha256_file(path,chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def session_date_from_name(name):
    m=DATE_RE.match(str(name))
    return m.group(1) if m else None

def _basic_wav_info(path):
    """Robust WAV metadata: stdlib wave first, torchaudio full-load fallback."""
    try:
        with wave.open(str(path),"rb") as f:
            return int(f.getframerate()), int(f.getnframes()), int(f.getnchannels())
    except Exception:
        w,sr=torchaudio.load(str(path))
        return int(sr), int(w.shape[-1]), int(w.shape[0])

def _audio_info(path):
    sr,n,ch=_basic_wav_info(path)
    return {
        "path":str(Path(path).resolve()),
        "filename":Path(path).name,
        "sample_rate":sr,
        "num_frames":n,
        "duration_sec":float(n/max(1,sr)),
        "channels":ch,
        "sha256":sha256_file(path),
    }

def discover_recording_units(root):
    """
    DeepShip-aware recording unit discovery.

    For each class:
      * each immediate child directory containing WAVs recursively = one recording/session unit;
      * each WAV directly under the class directory = one recording unit.

    This supports both the original session-directory layout and flattened subsets.
    """
    root=Path(root)
    units=[]
    empty_dirs=[]
    ignored=[]

    for cname in CLASS_NAMES:
        cdir=root/cname
        if not cdir.exists():
            raise FileNotFoundError(f"Missing class directory: {cdir}")

        # Immediate session directories.
        for child in sorted([p for p in cdir.iterdir() if p.is_dir()]):
            wavs=sorted(child.rglob("*.wav"))+sorted(child.rglob("*.WAV"))
            wavs=list(dict.fromkeys(wavs))
            if not wavs:
                empty_dirs.append(str(child))
                continue
            audio=[_audio_info(p) for p in wavs]
            rid=f"{cname}/{child.name}"
            file_hashes=sorted(a["sha256"] for a in audio)
            unit_sig=hashlib.sha256("|".join(file_hashes).encode()).hexdigest()
            units.append({
                "recording_id":rid,
                "unit_name":child.name,
                "unit_kind":"session_dir",
                "class_name":cname,
                "label_name":cname,
                "label":LABEL_TO_IDX[cname],
                "session_date":session_date_from_name(child.name),
                "audio_files":audio,
                "n_wav_files":len(audio),
                "duration_sec":float(sum(a["duration_sec"] for a in audio)),
                "unit_sha256":unit_sig,
            })

        # WAVs directly under class directory.
        root_wavs=sorted(cdir.glob("*.wav"))+sorted(cdir.glob("*.WAV"))
        root_wavs=list(dict.fromkeys(root_wavs))
        for p in root_wavs:
            audio=[_audio_info(p)]
            rid=f"{cname}/{p.stem}"
            units.append({
                "recording_id":rid,
                "unit_name":p.stem,
                "unit_kind":"root_wav",
                "class_name":cname,
                "label_name":cname,
                "label":LABEL_TO_IDX[cname],
                "session_date":session_date_from_name(p.stem),
                "audio_files":audio,
                "n_wav_files":1,
                "duration_sec":float(audio[0]["duration_sec"]),
                "unit_sha256":audio[0]["sha256"],
            })

        for p in sorted(cdir.iterdir()):
            if p.is_file() and p.suffix.lower()!=".wav":
                ignored.append(str(p))

    # Stable order.
    units=sorted(units,key=lambda x:(x["label"],x["recording_id"]))
    return units,empty_dirs,ignored

def inventory_summary(units):
    by=defaultdict(list)
    for u in units: by[u["class_name"]].append(u)

    out={"classes":{}}
    total_units=0; total_wavs=0; total_sec=0.0
    severe=[]
    for cname in CLASS_NAMES:
        arr=by[cname]
        units_n=len(arr)
        wavs_n=sum(u["n_wav_files"] for u in arr)
        sec=sum(u["duration_sec"] for u in arr)
        ref=REFERENCE_RECORDING_COUNTS[cname]
        ratio=max(units_n,wavs_n)/float(ref) if ref else 1.0
        cls={
            "recording_units":units_n,
            "wav_files":wavs_n,
            "duration_hours":sec/3600.0,
            "reference_recordings":ref,
            "inventory_ratio_vs_reference":ratio,
            "session_dir_units":sum(u["unit_kind"]=="session_dir" for u in arr),
            "root_wav_units":sum(u["unit_kind"]=="root_wav" for u in arr),
            "dated_units":sum(bool(u["session_date"]) for u in arr),
        }
        out["classes"][cname]=cls
        if max(units_n,wavs_n)<0.60*ref:
            severe.append(cname)
        total_units+=units_n; total_wavs+=wavs_n; total_sec+=sec

    out.update({
        "total_recording_units":total_units,
        "total_wav_files":total_wavs,
        "total_duration_hours":total_sec/3600.0,
        "potentially_incomplete_classes":severe,
        "looks_like_full_or_near_full_dataset":len(severe)==0,
    })
    return out

def _read_window(path, frac=0.5, seconds=20.0, target_sr=8000):
    sr,n,_=_basic_wav_info(path)
    want=max(1024,int(round(float(seconds)*sr)))
    if n<=want:
        off=0; num=n
    else:
        center=int(round(float(frac)*(n-1)))
        off=max(0,min(n-want,center-want//2))
        num=want
    w,_=torchaudio.load(str(path),frame_offset=off,num_frames=num)
    w=w.mean(0).float()
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

def recording_fingerprint(unit,target_sr=8000,window_sec=20.0,n_views=3):
    files=[Path(a["path"]) for a in unit["audio_files"]]
    views=[]
    if len(files)==1:
        fracs=np.linspace(0.1,0.9,int(n_views))
        for f in fracs:
            views.append(_read_window(files[0],float(f),window_sec,target_sr))
    else:
        idx=np.linspace(0,len(files)-1,int(n_views)).round().astype(int)
        for j in idx:
            views.append(_read_window(files[int(j)],0.5,window_sec,target_sr))
    while len(views)<int(n_views):
        views.append(views[-1].copy())
    x=np.concatenate(views[:int(n_views)])
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

def build_components(units,fingerprints,threshold=0.997,group_same_day=False):
    """
    Components combine:
      * exact duplicate unit signatures / individual WAV hashes;
      * optional same-class same-day session groups (legacy option only);
      * high acoustic-fingerprint similarity.

    DeepShip V2 calls this with group_same_day=False. Calendar date alone is not
    treated as proof that two official recording/session units are the same sample.
    """
    n=len(units); dsu=DSU(n)
    exact_edges=[]; date_edges=[]; near_edges=[]

    # Exact unit signatures.
    by_unit_sha=defaultdict(list)
    for i,u in enumerate(units): by_unit_sha[u["unit_sha256"]].append(i)
    for ids in by_unit_sha.values():
        for j in ids[1:]:
            dsu.union(ids[0],j)
            exact_edges.append((ids[0],j,"unit_sha256"))

    # Any exact WAV duplicated across units.
    by_file_sha=defaultdict(list)
    for i,u in enumerate(units):
        for af in u["audio_files"]:
            by_file_sha[af["sha256"]].append(i)
    for ids in by_file_sha.values():
        ids=sorted(set(ids))
        for j in ids[1:]:
            dsu.union(ids[0],j)
            exact_edges.append((ids[0],j,"wav_sha256"))

    # Same-day within same class.
    if group_same_day:
        by_day=defaultdict(list)
        for i,u in enumerate(units):
            if u.get("session_date"):
                by_day[(u["class_name"],u["session_date"])].append(i)
        for ids in by_day.values():
            for j in ids[1:]:
                dsu.union(ids[0],j)
                date_edges.append((ids[0],j))

    # Acoustic similarity only within the same class; cross-class high similarities
    # are reported as diagnostics but never merged across labels.
    cross_class_high=[]
    for i in range(n):
        for j in range(i+1,n):
            sim=cosine(fingerprints[i],fingerprints[j])
            if sim>=float(threshold):
                rec={
                    "recording_a":units[i]["recording_id"],
                    "recording_b":units[j]["recording_id"],
                    "class_a":units[i]["class_name"],
                    "class_b":units[j]["class_name"],
                    "cosine":sim,
                }
                if units[i]["label"]==units[j]["label"]:
                    dsu.union(i,j)
                    near_edges.append(rec)
                else:
                    cross_class_high.append(rec)

    groups=defaultdict(list)
    for i in range(n):
        groups[dsu.find(i)].append(i)

    comps=[]
    conflicts=[]
    for _,ids in groups.items():
        labs={units[i]["class_name"] for i in ids}
        if len(labs)>1:
            conflicts.append({
                "recording_ids":[units[i]["recording_id"] for i in ids],
                "classes":sorted(labs),
            })
            continue
        rids=sorted(units[i]["recording_id"] for i in ids)
        cid="cmp_"+hashlib.sha1("|".join(rids).encode()).hexdigest()[:12]
        comps.append({
            "component_id":cid,
            "class_name":next(iter(labs)),
            "label":units[ids[0]]["label"],
            "indices":ids,
            "recording_ids":rids,
            "size":len(ids),
            "dates":sorted({units[i]["session_date"] for i in ids if units[i]["session_date"]}),
        })

    diagnostics={
        "exact_edges_count":len(exact_edges),
        "same_day_edges_count":len(date_edges),
        "near_duplicate_edges":sorted(near_edges,key=lambda x:-x["cosine"]),
        "cross_class_high_similarity":sorted(cross_class_high,key=lambda x:-x["cosine"]),
    }
    return comps,diagnostics,conflicts

def split_components(units,comps,seed=20261015,test_ratio=0.30,min_support_components=6):
    rng=np.random.default_rng(int(seed))
    test_ids=set()

    by=defaultdict(list)
    for i,c in enumerate(comps): by[c["class_name"]].append(i)

    for cname in CLASS_NAMES:
        ids=by[cname]
        if len(ids)<int(min_support_components)+1:
            raise RuntimeError(
                f"{cname}: only {len(ids)} independent components; "
                f"need >= {int(min_support_components)+1}"
            )
        ids=list(ids); rng.shuffle(ids)
        n_test=max(1,int(round(len(ids)*float(test_ratio))))
        n_test=min(n_test,len(ids)-int(min_support_components))
        chosen=ids[:n_test]
        test_ids.update(chosen)

    support=[]; test=[]
    for i,c in enumerate(comps):
        target=test if i in test_ids else support
        for ui in c["indices"]:
            u=dict(units[ui])
            u["component_id"]=c["component_id"]
            target.append(u)
    return support,test,test_ids

def audit_split(support,test,diagnostics):
    sr={u["recording_id"] for u in support}
    tr={u["recording_id"] for u in test}
    sf={af["sha256"] for u in support for af in u["audio_files"]}
    tf={af["sha256"] for u in test for af in u["audio_files"]}
    sc={u["component_id"] for u in support}
    tc={u["component_id"] for u in test}
    cross_near=[]
    for e in diagnostics["near_duplicate_edges"]:
        a,b=e["recording_a"],e["recording_b"]
        if (a in sr and b in tr) or (a in tr and b in sr):
            cross_near.append(e)
    return {
        "recording_id_overlap":len(sr&tr),
        "wav_sha256_overlap":len(sf&tf),
        "component_overlap":len(sc&tc),
        "cross_split_near_duplicate_count":len(cross_near),
        "cross_split_near_duplicate_edges":cross_near,
    }

def class_counts(rows,key="recording"):
    if key=="recording":
        return dict(Counter(u["class_name"] for u in rows))
    if key=="component":
        d=defaultdict(set)
        for u in rows: d[u["class_name"]].add(u["component_id"])
        return {k:len(v) for k,v in d.items()}
    raise ValueError(key)
