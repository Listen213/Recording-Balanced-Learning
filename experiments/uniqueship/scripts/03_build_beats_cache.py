#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torchaudio
from tqdm import tqdm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import load_yaml, save_json
from vendor.beats_wrapper import OfficialBEATs

def load_clip(path,target_sr,want):
    wav,sr=torchaudio.load(path); wav=wav.mean(0).float()
    if sr!=target_sr: wav=torchaudio.functional.resample(wav,sr,target_sr)
    wav=wav-wav.mean(); wav=wav.clamp(-1,1)
    if wav.numel()<want: wav=torch.nn.functional.pad(wav,(0,want-wav.numel()))
    return wav[:want]

def embed_batch(model,wavs,device):
    w=torch.stack(wavs); f=model.preprocess(w).cpu()
    with torch.no_grad(), torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        e=model.forward_fbank(f.to(device)).float().cpu()
    return e.numpy().astype(np.float16)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/uniqueship_subset.yaml'); ap.add_argument('--cleanup-chunks',action='store_true'); a=ap.parse_args()
    cfg=load_yaml(a.config); proj=Path(__file__).resolve().parents[1]
    mf=proj/cfg['project']['selected_manifest']; out=proj/cfg['project']['cache_file']
    df=pd.read_csv(mf); target_sr=int(cfg['features']['target_sr']); want=int(round(float(cfg['features']['clip_sec'])*target_sr)); bs=int(cfg['features']['batch_size']); cs=int(cfg['features']['chunk_size'])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=OfficialBEATs(proj/'vendor'/'beats_src',proj/'vendor'/'BEATs_iter3_plus_AS2M.pt').to(device).eval()
    chunk_dir=proj/'cache'/'chunks'; chunk_dir.mkdir(parents=True,exist_ok=True)
    chunks=[]
    for st in range(0,len(df),cs):
        en=min(st+cs,len(df)); cp=chunk_dir/f'chunk_{st:06d}_{en:06d}.npz'; chunks.append(cp)
        if cp.exists():
            z=np.load(cp,allow_pickle=True)
            if len(z['X'])==en-st: print('SKIP',cp); continue
        sub=df.iloc[st:en]; all_e=[]; wavs=[]
        for _,row in tqdm(sub.iterrows(),total=len(sub),desc=f'BEATs {st}:{en}'):
            wavs.append(load_clip(row['abs_path'],target_sr,want))
            if len(wavs)>=bs:
                all_e.append(embed_batch(model,wavs,device)); wavs=[]
        if wavs: all_e.append(embed_batch(model,wavs,device))
        X=np.concatenate(all_e,axis=0)
        np.savez_compressed(cp,X=X,canonical_path=sub.canonical_path.astype(str).to_numpy(),label=sub.label.astype(str).to_numpy(),source_id=sub.source_id.astype(str).to_numpy())
        print('WROTE',cp,X.shape)
    # merge
    X=[]; paths=[]; labels=[]; sources=[]
    for cp in chunks:
        z=np.load(cp,allow_pickle=True); X.append(z['X']); paths.append(z['canonical_path']); labels.append(z['label']); sources.append(z['source_id'])
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,X=np.concatenate(X),canonical_path=np.concatenate(paths),label=np.concatenate(labels),source_id=np.concatenate(sources))
    save_json({'cache':str(out),'rows':len(df),'shape':[len(df),int(X[0].shape[1])],'device':str(device)},proj/'outputs'/'cache_summary.json')
    print('FINAL',out)
    if a.cleanup_chunks:
        for cp in chunks: cp.unlink()
if __name__=='__main__': main()
