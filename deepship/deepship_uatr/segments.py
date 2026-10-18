from __future__ import annotations
import json,math
from dataclasses import dataclass

@dataclass(frozen=True)
class SegmentRef:
    recording_id:str
    audio_path:str
    file_index:int
    label:int
    label_name:str
    source_type:str
    component_id:str
    local_start_sec:float
    global_start_sec:float
    duration_sec:float
    sample_rate:int

def expand_segments(manifest_path):
    with open(manifest_path,"r",encoding="utf-8") as f:
        m=json.load(f)
    seg=float(m.get("segment_sec",5.0))
    hop=float(m.get("hop_sec",seg))
    out=[]

    for r in m["recordings"]:
        global_offset=0.0
        files=r["audio_files"]
        for fi,af in enumerate(files):
            d=float(af["duration_sec"])
            if d<=seg:
                starts=[0.0]
            else:
                starts=[i*hop for i in range(int(math.floor((d-seg)/hop))+1)]
            for s in starts:
                out.append(SegmentRef(
                    recording_id=r["recording_id"],
                    audio_path=af["path"],
                    file_index=int(fi),
                    label=int(r["label"]),
                    label_name=r["label_name"],
                    source_type=r.get("session_date",""),
                    component_id=str(r.get("component_id",r["recording_id"])),
                    local_start_sec=float(s),
                    global_start_sec=float(global_offset+s),
                    duration_sec=seg,
                    sample_rate=int(af["sample_rate"]),
                ))
            global_offset += d
    return m,out
