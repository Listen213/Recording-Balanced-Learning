from __future__ import annotations
import json,math
from dataclasses import dataclass

@dataclass(frozen=True)
class SegmentRef:
    recording_id:str
    path:str
    label:int
    label_name:str
    source_type:str
    component_id:str
    start_sec:float
    duration_sec:float
    sample_rate:int

def expand_segments(manifest_path):
    with open(manifest_path,"r",encoding="utf-8") as f:
        m=json.load(f)
    seg=float(m.get("segment_sec",5.0))
    hop=float(m.get("hop_sec",seg))
    out=[]
    for r in m["recordings"]:
        d=float(r["duration_sec"])
        starts=[0.0] if d<=seg else [
            i*hop for i in range(int(math.floor((d-seg)/hop))+1)
        ]
        for s in starts:
            out.append(SegmentRef(
                r["recording_id"],r["path"],int(r["label"]),r["label_name"],
                r.get("source_type",""),str(r.get("component_id",r["recording_id"])),
                float(s),seg,int(r["sample_rate"])
            ))
    return m,out
