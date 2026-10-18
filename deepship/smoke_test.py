from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import torch

from deepship_uatr.data import RecordingCollection
from deepship_uatr.sampling import sample_support
from deepship_uatr.eval import evaluate_draw,METHODS
from deepship_uatr.metrics import hierarchical_paired_bootstrap_fast_all_metrics

def make_cache(path,recs,dim=24,segs=8,seed=0):
    rng=np.random.default_rng(seed)
    E=[]; y=[]; ids=[]; starts=[]; comps=[]; sources=[]
    for rid,c,comp,src in recs:
        center=np.zeros(dim,np.float32)
        center[c%dim]=2.0
        center[(c+4)%dim]=0.7
        x=center+rng.normal(0,0.9,(segs,dim)).astype(np.float32)
        E.append(x)
        y.extend([c]*segs)
        ids.extend([rid]*segs)
        starts.extend(np.arange(segs)*5.0)
        comps.extend([comp]*segs)
        sources.extend([src]*segs)
    torch.save({
        "embedding":torch.tensor(np.concatenate(E)),
        "labels":torch.tensor(y),
        "recording_ids":ids,
        "component_ids":comps,
        "source_types":sources,
        "start_sec":torch.tensor(starts,dtype=torch.float32),
        "label_to_idx":{"Cargo":0,"Passengership":1,"Tanker":2,"Tug":3},
        "manifest_sha256":"synthetic",
    },path)

def main():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        sup=[]; tst=[]
        for c in range(4):
            for i in range(8):
                sup.append((f"sup_c{c}_{i}",c,f"sc{c}_{i}",f"d{i}"))
            for i in range(4):
                tst.append((f"test_c{c}_{i}",c,f"tc{c}_{i}",f"d{i}"))
        make_cache(root/"support.pt",sup,seed=1)
        make_cache(root/"test.pt",tst,seed=2)

        support_pool=RecordingCollection.from_cache(root/"support.pt")
        test=RecordingCollection.from_cache(root/"test.pt")
        pt_cfg={"rank":2,"steps":24,"lr":0.05,"l2":0.3,"temperature":0.25}

        A=[];B=[]; y=None; comps=None
        for seed in [101,102,103]:
            draw=sample_support(support_pool,2,seed,True)
            assert len(draw)==8
            assert len({r.component_id for r in draw})==8
            # Synthetic sources are unique per recording; source-aware mode should
            # therefore also avoid duplicate source/date values within each class.
            for c in range(4):
                rows=[r for r in draw if r.label==c]
                assert len({r.source for r in rows})==2
            met,diag=evaluate_draw(draw,test.recordings,4,seed,pt_cfg)
            assert sorted(met)==sorted(METHODS)
            A.append(diag["probs"]["S1"])
            B.append(diag["probs"]["S0"])
            y=diag["y"]; comps=diag["components"]

        res,meta,_=hierarchical_paired_bootstrap_fast_all_metrics(
            y,comps,np.stack(A),np.stack(B),iters=50,seed=9
        )
        assert set(res)=={"macro_f1","balanced_accuracy","accuracy"}
        print("SMOKE TEST PASSED",METHODS)
        print("component-aware + date-aware support sampling PASS")
        print("fast hierarchical bootstrap PASS")

if __name__=="__main__":
    main()
