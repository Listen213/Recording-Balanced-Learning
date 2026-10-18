#!/usr/bin/env python3
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

METHODS = ["s0", "classbal", "s1"]
PAIRS = [("s0","classbal"),("classbal","s1"),("s0","s1")]

def macro_f1_from_confusion(cm):
    tp=np.diagonal(cm,axis1=1,axis2=2).astype(float)
    true_n=cm.sum(axis=2,dtype=float); pred_n=cm.sum(axis=1,dtype=float)
    den=true_n+pred_n
    f1=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0)
    return f1.mean(axis=1)

def prepare_cube(pred,k):
    z=pred[(pred["k"].astype(str)==str(k)) & (pred["head"]=="lr") & (pred["scaler"]=="segment") & pred["method"].isin(METHODS)].copy()
    draws=np.array(sorted(z.seed.astype(int).unique()),dtype=int)
    srcs=np.array(sorted(z.source_id.astype(str).unique()),dtype=object)
    labels=np.array(sorted(z.true.astype(str).unique()),dtype=object)
    dmap={v:i for i,v in enumerate(draws)}; smap={v:i for i,v in enumerate(srcs)}; lmap={v:i for i,v in enumerate(labels)}
    D,T,C=len(draws),len(srcs),len(labels)
    truth=np.full(T,-1,dtype=np.int16)
    gt=z[["source_id","true"]].drop_duplicates()
    if gt.groupby("source_id").true.nunique().max()!=1: raise RuntimeError("inconsistent truth by source")
    for sid,t in gt.itertuples(index=False,name=None): truth[smap[str(sid)]]=lmap[str(t)]
    cubes={}
    for method in METHODS:
        a=z[z.method==method]
        if len(a)!=D*T: raise RuntimeError(f"{method}: expected {D*T} prediction rows for k={k}, got {len(a)}")
        mat=np.full((D,T),-1,dtype=np.int16)
        for seed,sid,p in a[["seed","source_id","pred"]].itertuples(index=False,name=None):
            mat[dmap[int(seed)],smap[str(sid)]]=lmap[str(p)]
        if (mat<0).any(): raise RuntimeError(f"incomplete matrix for {method}, k={k}")
        cubes[method]=mat
    return draws,srcs,labels,truth,cubes

def hierarchical_bootstrap(pred,k,iters,seed,batch):
    draws,srcs,labels,truth,cubes=prepare_cube(pred,k)
    D,T,C=len(draws),len(srcs),len(labels)
    rng=np.random.default_rng(seed)
    vals={f"{b}-{a}":np.empty(iters,float) for a,b in PAIRS}
    done=0; t0=time.time()
    while done<iters:
        B=min(batch,iters-done)
        didx=rng.integers(0,D,size=(B,D))
        sidx=rng.integers(0,T,size=(B,D,T))
        yt=truth[sidx]; groups=B*D
        offsets=(np.arange(groups,dtype=np.int64)*(C*C))[:,None]
        yt2=yt.reshape(groups,T).astype(np.int64,copy=False)
        scores={}
        for method in METHODS:
            sel=cubes[method][didx]
            yp=np.take_along_axis(sel,sidx,axis=2).reshape(groups,T).astype(np.int64,copy=False)
            codes=yt2*C+yp
            bc=np.bincount((offsets+codes).ravel(),minlength=groups*C*C)
            cm=bc.reshape(groups,C,C)
            scores[method]=100*macro_f1_from_confusion(cm).reshape(B,D).mean(axis=1)
        for a,b in PAIRS: vals[f"{b}-{a}"][done:done+B]=scores[b]-scores[a]
        done+=B
        if done==iters or done%max(100,batch*10)==0:
            rate=done/max(time.time()-t0,1e-9); eta=(iters-done)/max(rate,1e-9)
            print(f"k={k}: {done}/{iters}, eta={eta:.1f}s",flush=True)
    rows=[]
    for name,v in vals.items():
        rows.append({"k":k,"comparison":name,"mean_effect":float(v.mean()),"ci_low":float(np.quantile(v,.025)),"ci_high":float(np.quantile(v,.975)),"iterations":iters,"n_draws":D,"n_test_sources":T,"n_classes":C})
    return rows

def draw_paired(metrics,iters,seed):
    few=metrics[metrics.k.astype(str)!="full"].copy(); out=[]; rng=np.random.default_rng(seed)
    piv=few[few.method.isin(METHODS)].pivot_table(index=["k","seed"],columns="method",values="macro_f1",aggfunc="first").dropna()
    for k,g in piv.groupby(level=0):
        for a,b in PAIRS:
            d=(g[b]-g[a]).to_numpy(float)
            boot=np.array([rng.choice(d,size=len(d),replace=True).mean() for _ in range(iters)])
            out.append({"k":str(k),"comparison":f"{b}-{a}","mean_delta":float(d.mean()),"sd_delta":float(d.std(ddof=1)),"min_delta":float(d.min()),"max_delta":float(d.max()),"ci_low":float(np.quantile(boot,.025)),"ci_high":float(np.quantile(boot,.975)),"n":len(d),"draw_bootstrap_iters":iters})
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--primary-metrics',required=True); ap.add_argument('--primary-predictions',required=True)
    ap.add_argument('--classbal-metrics',required=True); ap.add_argument('--classbal-predictions',required=True)
    ap.add_argument('--out-dir',default='outputs/final_summary'); ap.add_argument('--hier-iters',type=int,default=5000); ap.add_argument('--draw-iters',type=int,default=10000); ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--seed',type=int,default=20261016)
    a=ap.parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    pm=pd.read_csv(a.primary_metrics,dtype={'k':str}); cm=pd.read_csv(a.classbal_metrics,dtype={'k':str}); metrics=pd.concat([pm,cm],ignore_index=True)
    pp=pd.read_csv(a.primary_predictions,dtype={'k':str,'source_id':str,'true':str,'pred':str}); cp=pd.read_csv(a.classbal_predictions,dtype={'k':str,'source_id':str,'true':str,'pred':str}); pred=pd.concat([pp,cp],ignore_index=True)
    few=metrics[metrics.k!='full']
    summary=few.groupby(['k','method']).agg(macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),balanced_accuracy_mean=('balanced_accuracy','mean'),balanced_accuracy_sd=('balanced_accuracy','std'),n=('macro_f1','size')).reset_index()
    summary.to_csv(out/'method_summary.csv',index=False)
    full=pm[pm.k=='full'][['method','accuracy','macro_f1','balanced_accuracy']]
    full.to_csv(out/'full_support.csv',index=False)
    draw_paired(metrics,a.draw_iters,a.seed).to_csv(out/'paired_draw_bootstrap.csv',index=False)
    rows=[]
    for k in (2,3,4): rows.extend(hierarchical_bootstrap(pred,k,a.hier_iters,a.seed+k,a.batch_size))
    pd.DataFrame(rows).to_csv(out/'hierarchical_bootstrap_all_pairs.csv',index=False)
    print(f'WROTE {out}')

if __name__=='__main__': main()
