from __future__ import annotations
from pathlib import Path
import hashlib, json, random
import numpy as np

def ensure_dir(p):
    p=Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj, path):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))

def sha256_file(path, chunk=1024*1024):
    h=hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b=f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def sha256_text(s):
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()

def soft_clip_probs(p, eps=1e-8):
    p=np.asarray(p, dtype=np.float64)
    p=np.clip(p, eps, None)
    return p/np.clip(p.sum(axis=-1, keepdims=True), eps, None)

def geometric_fusion(p, q, alpha=0.5, eps=1e-8):
    p=np.clip(np.asarray(p,float),eps,1.0)
    q=np.clip(np.asarray(q,float),eps,1.0)
    z=float(alpha)*np.log(p)+(1.0-float(alpha))*np.log(q)
    z=z-z.max(axis=-1,keepdims=True)
    e=np.exp(z)
    return e/np.clip(e.sum(axis=-1,keepdims=True),eps,None)
