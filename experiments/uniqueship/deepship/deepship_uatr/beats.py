from __future__ import annotations
import importlib,sys
from pathlib import Path
import torch
import torch.nn as nn

class OfficialBEATs(nn.Module):
    def __init__(self,source_dir,checkpoint_path):
        super().__init__()
        src=Path(source_dir); ck=Path(checkpoint_path)
        req=[src/"BEATs.py",src/"backbone.py",src/"modules.py"]
        miss=[str(x) for x in req if not x.exists()]
        if miss: raise FileNotFoundError("Missing BEATs source files: "+", ".join(miss))
        if not ck.exists(): raise FileNotFoundError(f"Missing BEATs checkpoint: {ck}")
        s=str(src.resolve()); sys.path.insert(0,s)
        try:
            mod=importlib.import_module("BEATs")
            blob=torch.load(str(ck),map_location="cpu",weights_only=False)
            cfg=mod.BEATsConfig(blob["cfg"])
            m=mod.BEATs(cfg)
            m.load_state_dict(blob["model"],strict=True)
            if getattr(m,"predictor",None) is not None:
                m.predictor=None
            self.model=m
            self.out_dim=int(cfg.encoder_embed_dim)
        finally:
            if sys.path and sys.path[0]==s:
                sys.path.pop(0)
        for p in self.model.parameters():
            p.requires_grad=False

    def preprocess(self,wav):
        return self.model.preprocess(wav.float())

    def forward_fbank(self,fbank):
        x=fbank.unsqueeze(1)
        x=self.model.patch_embedding(x)
        x=x.reshape(x.shape[0],x.shape[1],-1).transpose(1,2)
        x=self.model.layer_norm(x)
        if self.model.post_extract_proj is not None:
            x=self.model.post_extract_proj(x)
        x=self.model.dropout_input(x)
        x,_=self.model.encoder(x,padding_mask=None)
        return x.mean(1)
