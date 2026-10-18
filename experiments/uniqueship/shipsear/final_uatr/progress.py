from __future__ import annotations
from pathlib import Path
import json, os, time


def fmt_seconds(x):
    if x is None: return '--'
    x=max(0,int(round(float(x))))
    h=x//3600; m=(x%3600)//60; s=x%60
    if h: return f'{h:02d}:{m:02d}:{s:02d}'
    return f'{m:02d}:{s:02d}'


def eta_from(start,done,total):
    if done<=0 or total<=done: return 0.0 if total<=done else None
    elapsed=time.time()-float(start)
    return elapsed/done*(total-done)


def write_progress(path,**kwargs):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=dict(kwargs)
    data['updated_at_unix']=time.time()
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)
    return data
