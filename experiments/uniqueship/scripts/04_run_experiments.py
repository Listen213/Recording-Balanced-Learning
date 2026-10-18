#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import load_yaml, save_json, cv, gini

def load_cache(path):
    z=np.load(path,allow_pickle=True)
    return {k:z[k] for k in z.files}

def fold_data(cache,protocol_csv):
    p=pd.read_csv(protocol_csv)
    idx={str(x):i for i,x in enumerate(cache['canonical_path'].astype(str))}
    missing=[x for x in p.canonical_path.astype(str) if x not in idx]
    if missing: raise RuntimeError(f'{len(missing)} protocol clips absent from cache; first={missing[:3]}')
    rows=np.array([idx[x] for x in p.canonical_path.astype(str)],dtype=int)
    return {'X':cache['X'][rows].astype(np.float32),'label':p.label.astype(str).to_numpy(),'source_id':p.source_id.astype(str).to_numpy(),'split':p.split.astype(str).to_numpy(),'canonical_path':p.canonical_path.astype(str).to_numpy()}

def sources_by_class(d,split='support'):
    m=d['split']==split; df=pd.DataFrame({'label':d['label'][m],'source':d['source_id'][m]})
    labn=df.groupby('source').label.nunique();
    if (labn>1).any(): raise RuntimeError('source has multiple labels')
    out=defaultdict(list)
    for lab,sid in df.drop_duplicates('source').itertuples(index=False,name=None): out[lab].append(sid)
    return {k:sorted(v) for k,v in out.items()}

def draw_idx(d,k,seed):
    rng=np.random.default_rng(seed); chosen=[]; by=sources_by_class(d)
    for lab in sorted(by):
        if len(by[lab])<k: raise RuntimeError(f'{lab}: only {len(by[lab])} support sources < {k}')
        chosen += rng.choice(np.array(by[lab],dtype=object),size=k,replace=False).tolist()
    return np.flatnonzero((d['split']=='support') & np.isin(d['source_id'],chosen)),chosen

def weights(labels,sources,method,scale='mean1'):
    if method=='s0': w=np.ones(len(labels),float)
    else:
        df=pd.DataFrame({'label':labels,'source':sources}); seg_class=df.label.value_counts().to_dict(); seg_src=df.source.value_counts().to_dict(); srcs=df.drop_duplicates('source').groupby('label').source.nunique().to_dict()
        if method=='classbal': w=np.array([1/seg_class[c] for c in labels],float)
        elif method=='s1': w=np.array([1/(srcs[c]*seg_src[s]) for c,s in zip(labels,sources)],float)
        else: raise ValueError(method)
    if scale=='mean1': w=w/max(w.mean(),1e-12)
    return w

def scaler_fit(X,labels,sources,mode):
    if mode=='none': return None
    sc=StandardScaler()
    if mode=='segment': sc.fit(X); return sc
    if mode=='source': sc.fit(X,sample_weight=weights(labels,sources,'s1','mean1')); return sc
    raise ValueError(mode)

def train(X,y,w,head,C,max_iter):
    if head=='lr': m=LogisticRegression(C=C,solver='lbfgs',max_iter=max_iter); m.fit(X,y,sample_weight=w)
    elif head=='linear_svc': m=LinearSVC(C=C,max_iter=max(10000,max_iter)); m.fit(X,y,sample_weight=w)
    elif head=='sgd_log':
        m=SGDClassifier(loss='log_loss',alpha=1/max(C*len(y),1e-9),max_iter=max(5000,max_iter),tol=1e-5,random_state=0); m.fit(X,y,sample_weight=w)
    else: raise ValueError(head)
    return m

def scores(model,X):
    if hasattr(model,'predict_proba'): return model.predict_proba(X),np.asarray(model.classes_)
    s=model.decision_function(X); s=np.asarray(s)
    if s.ndim==1: s=np.stack([-s,s],axis=1)
    return s,np.asarray(model.classes_)

def eval_sources(model,sc,d,le):
    m=d['split']=='test'; X=d['X'][m]; X=sc.transform(X) if sc is not None else X
    scs,classes=scores(model,X); src=d['source_id'][m]; lab=d['label'][m]; rows=[]
    for sid in sorted(set(src)):
        ix=np.flatnonzero(src==sid); mean=scs[ix].mean(0); pred_enc=classes[int(np.argmax(mean))]; pred=le.inverse_transform([int(pred_enc)])[0]
        rows.append({'source_id':sid,'true':lab[ix][0],'pred':pred})
    yt=[r['true'] for r in rows]; yp=[r['pred'] for r in rows]
    met={'accuracy':100*accuracy_score(yt,yp),'balanced_accuracy':100*balanced_accuracy_score(yt,yp),'macro_f1':100*f1_score(yt,yp,average='macro',zero_division=0),'n_test_sources':len(rows)}
    return met,rows

def equal_source_idx(d,base_idx,strategy,seed):
    rng=np.random.default_rng(seed); src=d['source_id'][base_idx]; cnt=pd.Series(src).value_counts(); m=int(cnt.min()) if strategy=='min' else max(1,int(round(cnt.median())))
    out=[]
    for sid in sorted(cnt.index):
        ix=base_idx[src==sid]; out += rng.choice(ix,size=m,replace=len(ix)<m).tolist()
    return np.asarray(out,dtype=int),m

def imb(labels,sources):
    a=pd.Series(sources).value_counts().to_numpy(float); b=pd.Series(labels).value_counts().to_numpy(float)
    return {'source_clip_cv':cv(a),'source_clip_gini':gini(a),'source_clip_maxmin':float(a.max()/max(a.min(),1)),'class_clip_cv':cv(b)}

def one_run(d,k,seed,method='s0',scale='mean1',scaler_mode='segment',head='lr',C=.1,max_iter=2000,sampler=None,inner_seed=0,full=False):
    if full: idx=np.flatnonzero(d['split']=='support'); chosen=sorted(set(d['source_id'][idx]))
    else: idx,chosen=draw_idx(d,k,seed)
    sampler_m=None
    if sampler:
        idx,sampler_m=equal_source_idx(d,idx,sampler,seed*1000+inner_seed); method_train='s0'
    else: method_train=method
    labels=d['label'][idx]; src=d['source_id'][idx]; le=LabelEncoder().fit(d['label']); y=le.transform(labels)
    w=weights(labels,src,method_train,scale); sc=scaler_fit(d['X'][idx],labels,src,scaler_mode); X=sc.transform(d['X'][idx]) if sc is not None else d['X'][idx]
    model=train(X,y,w,head,C,max_iter); met,preds=eval_sources(model,sc,d,le); meta={'k':'full' if full else k,'seed':seed,'method':method if not sampler else 'equal_source_'+sampler,'weight_scale':scale,'scaler':scaler_mode,'head':head,'C':C,'inner_seed':inner_seed if sampler else None,'sampler_m':sampler_m}
    meta.update(imb(labels,src)); met.update(meta)
    for r in preds: r.update({x:meta[x] for x in ['k','seed','method','head','scaler','C']})
    return met,preds

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/uniqueship_subset.yaml'); ap.add_argument('--draws',type=int,default=None); ap.add_argument('--mode',choices=['primary','classbal','sampler','heads','all'],default='primary'); ap.add_argument('--out-dir',default='outputs/experiments'); a=ap.parse_args()
    cfg=load_yaml(a.config); proj=Path(__file__).resolve().parents[1]; cache=load_cache(proj/cfg['project']['cache_file']); folds=[int(cfg['project']['official_fold'])]; draws=a.draws or int(cfg['few_vessel']['draws']); seeds=list(range(int(cfg['few_vessel']['seed_start']),int(cfg['few_vessel']['seed_start'])+draws)); C=float(cfg['classifier']['C']); mi=int(cfg['classifier']['max_iter'])
    all_metrics=[]; all_preds=[]
    for fold in folds:
        d=fold_data(cache,proj/cfg['project']['protocol_file'])
        tasks=[]
        if a.mode in ['primary','all']: tasks += [('s0','segment','lr',None),('s1','segment','lr',None)]
        if a.mode in ['classbal','all']: tasks += [('classbal','segment','lr',None)]
        if a.mode in ['heads','all']:
            for h in cfg['controls']['heads']:
                if h!='lr': tasks += [('s0','segment',h,None),('s1','segment',h,None)]
            tasks += [('s0','source','lr',None),('s1','source','lr',None)]
        for k in cfg['few_vessel']['k_values']:
            for seed in seeds:
                for method,sm,head,sampler in tasks:
                    met,p=one_run(d,int(k),seed,method,scaler_mode=sm,head=head,C=C,max_iter=mi,sampler=sampler); met['fold']=fold; all_metrics.append(met)
                    for r in p: r['fold']=fold
                    all_preds.extend(p)
                if a.mode in ['sampler','all']:
                    for st in cfg['controls']['sampler_strategies']:
                        for inner in range(int(cfg['controls']['sampler_inner_seeds'])):
                            met,p=one_run(d,int(k),seed,'s0',sampler=st,inner_seed=inner,C=C,max_iter=mi); met['fold']=fold; all_metrics.append(met)
                            for r in p: r['fold']=fold
                            all_preds.extend(p)
        # full support primary once
        if a.mode in ['primary','all']:
            for method in ['s0','s1']:
                met,p=one_run(d,0,9991,method,C=C,max_iter=mi,full=True); met['fold']=fold; all_metrics.append(met)
                for r in p: r['fold']=fold
                all_preds.extend(p)
    out=proj/a.out_dir; out.mkdir(parents=True,exist_ok=True); tag=f"subset_fold{folds[0]}_{a.mode}_{draws}draws"
    pd.DataFrame(all_metrics).to_csv(out/f'{tag}_metrics.csv',index=False); pd.DataFrame(all_preds).to_csv(out/f'{tag}_predictions.csv',index=False)
    print('WROTE',out/f'{tag}_metrics.csv',len(all_metrics),'rows')
if __name__=='__main__': main()
