from __future__ import annotations
from pathlib import Path
import json, os, time
import numpy as np
from .common import save_json, load_json


def atomic_json(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)


def atomic_npz(path, **arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with open(tmp,'wb') as f:
        np.savez_compressed(f,**arrays)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def draw_paths(root,setting,seed):
    stem=f"{setting.replace('-','_')}_seed{int(seed)}"
    root=Path(root)
    return root/f"{stem}.npz", root/f"{stem}.json"


def save_draw_checkpoint(root,setting,seed,lock_sha,support,diag,met,methods):
    npz_path,json_path=draw_paths(root,setting,seed)
    resource={}
    for m in methods:
        resource[m]={
            'fit_seconds':float(met[m].get('fit_seconds',0.0)),
            'predict_seconds':float(met[m].get('predict_seconds',0.0)),
            'pipeline_seconds':float(met[m].get('pipeline_seconds',0.0)),
            'trainable_params':int(met[m].get('trainable_params',0)),
        }
    arrays={
        'y':np.asarray(diag['y'],dtype=np.int64),
        'rids':np.asarray(diag['rids'],dtype=object),
        'components':np.asarray(diag['components'],dtype=object),
    }
    for m in methods:
        arrays[f'{m}_probs']=np.asarray(diag['probs'][m],dtype=np.float32)
    atomic_npz(npz_path,**arrays)
    meta={
        'lock_sha256':lock_sha,
        'setting':setting,
        'seed':int(seed),
        'created_at_unix':time.time(),
        'support':[
            {
                'recording_id':r.recording_id,
                'component_id':r.component_id,
                'label':int(r.label),
                'source':r.source,
                'n_segments':int(len(r.embeddings)),
            }
            for r in support
        ],
        'resource':resource,
        'pt_info':diag.get('pt_info',{}),
        'npz_file':npz_path.name,
    }
    atomic_json(meta,json_path)
    return npz_path,json_path


def save_full_checkpoint(root,seed,lock_sha,support_count,diag,met,methods):
    setting='full-support-reference'
    npz_path,json_path=draw_paths(root,setting,seed)
    resource={}
    for m in methods:
        resource[m]={
            'fit_seconds':float(met[m].get('fit_seconds',0.0)),
            'predict_seconds':float(met[m].get('predict_seconds',0.0)),
            'pipeline_seconds':float(met[m].get('pipeline_seconds',0.0)),
            'trainable_params':int(met[m].get('trainable_params',0)),
        }
    arrays={
        'y':np.asarray(diag['y'],dtype=np.int64),
        'rids':np.asarray(diag['rids'],dtype=object),
        'components':np.asarray(diag['components'],dtype=object),
    }
    for m in methods:
        arrays[f'{m}_probs']=np.asarray(diag['probs'][m],dtype=np.float32)
    atomic_npz(npz_path,**arrays)
    meta={
        'lock_sha256':lock_sha,
        'setting':setting,
        'seed':int(seed),
        'created_at_unix':time.time(),
        'support_recordings':int(support_count),
        'resource':resource,
        'pt_info':diag.get('pt_info',{}),
        'npz_file':npz_path.name,
    }
    atomic_json(meta,json_path)
    return npz_path,json_path


def checkpoint_valid(root,setting,seed,lock_sha,methods):
    npz_path,json_path=draw_paths(root,setting,seed)
    if not npz_path.exists() or not json_path.exists(): return False
    try:
        meta=json.loads(json_path.read_text(encoding='utf-8'))
        if meta.get('lock_sha256')!=lock_sha: return False
        with np.load(npz_path,allow_pickle=True) as z:
            if not {'y','rids','components'}.issubset(z.files): return False
            n=len(z['y'])
            for m in methods:
                k=f'{m}_probs'
                if k not in z.files or z[k].shape[0]!=n: return False
        return True
    except Exception:
        return False


def load_draw_checkpoint(root,setting,seed,lock_sha,methods):
    if not checkpoint_valid(root,setting,seed,lock_sha,methods):
        raise RuntimeError(f'invalid checkpoint {setting} seed={seed}')
    npz_path,json_path=draw_paths(root,setting,seed)
    meta=json.loads(json_path.read_text(encoding='utf-8'))
    with np.load(npz_path,allow_pickle=True) as z:
        arrays={k:z[k].copy() for k in z.files}
    return meta,arrays
