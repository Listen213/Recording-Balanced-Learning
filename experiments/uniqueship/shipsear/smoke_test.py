from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import torch

from final_uatr.data import RecordingCollection
from final_uatr.sampling import sample_support
from final_uatr.eval import evaluate_draw,METHODS
from final_uatr.metrics import (
    hierarchical_paired_bootstrap_fast_all_metrics,
    hierarchical_paired_bootstrap_slow,
)
from final_uatr.checkpoint import save_draw_checkpoint,checkpoint_valid,load_draw_checkpoint
from final_uatr.split import build_components,split_components,audit_split,LABELS


def make_cache(path,recs,dim=24,segs=8,seed=0):
    rng=np.random.default_rng(seed)
    E=[]; y=[]; ids=[]; starts=[]; comps=[]; sources=[]
    for rid,c,comp,src in recs:
        center=np.zeros(dim,np.float32)
        center[c%dim]=2.2; center[(c+5)%dim]=0.8
        x=center+rng.normal(0,0.9,(segs,dim)).astype(np.float32)
        E.append(x); y.extend([c]*segs); ids.extend([rid]*segs)
        starts.extend(np.arange(segs)*5.0); comps.extend([comp]*segs); sources.extend([src]*segs)
    torch.save({
        'embedding':torch.tensor(np.concatenate(E)),
        'labels':torch.tensor(y),
        'recording_ids':ids,
        'component_ids':comps,
        'source_types':sources,
        'start_sec':torch.tensor(starts,dtype=torch.float32),
        'label_to_idx':{LABELS[i]:i for i in range(5)},
        'manifest_sha256':'synthetic',
    },path)


def test_split_logic():
    records=[]; fps=[]; rng=np.random.default_rng(0)
    for c,lab in enumerate(LABELS):
        for i in range(10):
            rid=f'c{c}_{i}'
            records.append({
                'recording_id':rid,'label_name':lab,'label':c,
                'sha256':f'{c}_{i}','source_type_norm':f'src{c}_{i%2}',
                'source_type':f'src{c}_{i%2}','path':rid,
            })
            v=rng.normal(size=16); v=v/np.linalg.norm(v); fps.append(v)
    comps,pairs,conflicts=build_components(records,np.asarray(fps),threshold=0.999999)
    assert not conflicts
    support,test,_=split_components(records,comps,seed=123,test_ratio=0.3,min_support_components=6)
    audit=audit_split(support,test,pairs)
    assert audit['recording_id_overlap']==0
    assert audit['sha256_overlap']==0
    assert audit['cross_split_near_duplicate_count']==0


def main():
    test_split_logic()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        support=[]; test=[]
        for c in range(5):
            for i in range(8): support.append((f'sup_c{c}_{i}',c,f'sc{c}_{i}',f'type{c}_{i%2}'))
            for i in range(3): test.append((f'test_c{c}_{i}',c,f'tc{c}_{i}',f'type{c}_{i%2}'))
        make_cache(root/'support.pt',support,seed=1)
        make_cache(root/'test.pt',test,seed=2)
        sup=RecordingCollection.from_cache(root/'support.pt')
        tst=RecordingCollection.from_cache(root/'test.pt')
        pt_cfg={'rank':2,'steps':24,'lr':0.05,'l2':0.3,'temperature':0.25}

        pF0=[]; pS1=[]; y=None; comps=None
        ck=root/'ckpt'; lock_sha='synthetic_lock'
        for seed in [101,102,103]:
            draw=sample_support(sup,2,seed,True)
            met,diag=evaluate_draw(draw,tst.recordings,5,seed,pt_cfg)
            assert sorted(met)==sorted(METHODS)
            save_draw_checkpoint(ck,'2-rec',seed,lock_sha,draw,diag,met,METHODS)
            assert checkpoint_valid(ck,'2-rec',seed,lock_sha,METHODS)
            meta,arr=load_draw_checkpoint(ck,'2-rec',seed,lock_sha,METHODS)
            pF0.append(arr['F0_probs']); pS1.append(arr['S1_probs'])
            y=arr['y']; comps=arr['components']

        A=np.stack(pF0); B=np.stack(pS1)
        fast,meta,_=hierarchical_paired_bootstrap_fast_all_metrics(y,comps,A,B,iters=120,seed=77)
        for metric in ['macro_f1','balanced_accuracy','accuracy']:
            slow=hierarchical_paired_bootstrap_slow(y,comps,A,B,iters=120,seed=77,metric=metric)
            for k in ['mean_delta','low95','high95','p_delta_le_0']:
                if not np.isclose(fast[metric][k],slow[k],rtol=0,atol=1e-12):
                    raise AssertionError((metric,k,fast[metric][k],slow[k]))
        print('SMOKE TEST PASSED',METHODS)
        print('checkpoint save/load/resume path PASS')
        print('FAST bootstrap == SLOW reference EXACT (tested all 3 metrics)')
        print('fast bootstrap elapsed:',round(meta['elapsed_seconds'],4),'s')

if __name__=='__main__': main()
