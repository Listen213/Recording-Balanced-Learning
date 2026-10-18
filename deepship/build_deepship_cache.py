from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import torch,torchaudio
from tqdm import tqdm
from deepship_uatr.beats import OfficialBEATs
from deepship_uatr.segments import expand_segments
from deepship_uatr.common import sha256_file

def load_full(path,target_sr):
    wav,sr=torchaudio.load(path)
    wav=wav.mean(0)
    if sr!=target_sr:
        wav=torchaudio.functional.resample(wav,sr,target_sr)
    wav=wav.float()
    wav=wav-wav.mean()
    return wav.clamp(-1,1)

def process(model,chunks,ids,embs,device):
    w=torch.stack(chunks)
    f=model.preprocess(w).cpu()
    with torch.no_grad(),torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type=="cuda"
    ):
        e=model.forward_fbank(f.to(device)).float().cpu()
    for j,i in enumerate(ids):
        embs[i]=e[j]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--beats-source-dir",default="third_party/beats")
    p.add_argument("--beats-ckpt",default="weights/BEATs_iter3_plus_AS2M.pt")
    p.add_argument("--batch-size",type=int,default=24)
    a=p.parse_args()

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta,segs=expand_segments(a.manifest)
    sr=int(meta.get("target_sr",16000))
    want=int(round(float(meta.get("segment_sec",5))*sr))
    model=OfficialBEATs(a.beats_source_dir,a.beats_ckpt).to(device).eval()

    by=defaultdict(list)
    for i,s in enumerate(segs):
        by[s.audio_path].append((i,s))

    embs=[None]*len(segs)
    with torch.no_grad():
        for path,items in tqdm(by.items(),desc=f"cache {Path(a.manifest).stem}"):
            full=load_full(path,sr)
            chunks=[]; ids=[]
            for idx,s in items:
                st=int(round(s.local_start_sec*sr))
                x=full[st:st+want]
                if x.numel()<want:
                    x=torch.nn.functional.pad(x,(0,want-x.numel()))
                chunks.append(x[:want]); ids.append(idx)
                if len(chunks)>=a.batch_size:
                    process(model,chunks,ids,embs,device)
                    chunks=[]; ids=[]
            if chunks:
                process(model,chunks,ids,embs,device)

    ridnames=[]; rid2={}
    for s in segs:
        if s.recording_id not in rid2:
            rid2[s.recording_id]=len(ridnames)
            ridnames.append(s.recording_id)

    obj={
        "dataset":"DeepShip",
        "manifest":str(Path(a.manifest).resolve()),
        "manifest_sha256":sha256_file(a.manifest),
        "label_to_idx":meta["label_to_idx"],
        "labels":torch.tensor([s.label for s in segs]),
        "recording_ids":[s.recording_id for s in segs],
        "component_ids":[s.component_id for s in segs],
        "source_types":[s.source_type for s in segs],
        "recording_index":torch.tensor([rid2[s.recording_id] for s in segs]),
        "recording_names":ridnames,
        "start_sec":torch.tensor([s.global_start_sec for s in segs]),
        "embedding":torch.stack(embs).half(),
        "device_used":str(device),
    }
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    torch.save(obj,a.out)
    print(json.dumps({
        "dataset":"DeepShip",
        "out":a.out,
        "segments":len(segs),
        "recordings":len(ridnames),
        "embedding_shape":list(obj["embedding"].shape),
        "device":str(device),
        "manifest_sha256":obj["manifest_sha256"],
    },indent=2))

if __name__=="__main__":
    main()
