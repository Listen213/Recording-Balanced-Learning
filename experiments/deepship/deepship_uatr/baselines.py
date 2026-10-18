from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .common import soft_clip_probs


def rec_stats(x):
    x=np.asarray(x,dtype=np.float64)
    return np.concatenate([x.mean(0),x.std(0)],axis=0)


@dataclass
class SegmentEvidenceModel:
    scaler: StandardScaler
    clf: LogisticRegression
    classes_: np.ndarray
    balance_mode: str

    @classmethod
    def fit(cls, recordings, seed=42, balance_mode='recording', C=0.1):
        if balance_mode not in {'recording','uniform'}:
            raise ValueError(balance_mode)
        X=[]; y=[]; w=[]
        nc=Counter(int(r.label) for r in recordings)
        n_classes=max(len(nc),1)
        total_segments=sum(len(r.embeddings) for r in recordings)
        if total_segments<=0: raise ValueError('no support segments')
        for r in recordings:
            a=np.asarray(r.embeddings,dtype=np.float32); n=len(a)
            if n<=0: continue
            X.append(a); y.extend([int(r.label)]*n)
            if balance_mode=='recording':
                ww=1.0/(nc[int(r.label)]*n)
            else:
                ww=n_classes/total_segments
            w.extend([ww]*n)
        X=np.concatenate(X,axis=0); y=np.asarray(y,np.int64); w=np.asarray(w,float)
        sc=StandardScaler().fit(X)
        clf=LogisticRegression(C=float(C),max_iter=2000,solver='lbfgs',random_state=int(seed))
        clf.fit(sc.transform(X),y,sample_weight=w)
        return cls(sc,clf,clf.classes_.copy(),balance_mode)

    def transform(self,x):
        return self.scaler.transform(np.asarray(x,dtype=np.float32)).astype(np.float32)

    def segment_logits_full(self,x,n_classes):
        z=self.clf.decision_function(self.scaler.transform(np.asarray(x,dtype=np.float32)))
        if z.ndim==1:
            z=np.stack([-z,z],axis=1)
        out=np.full((len(z),n_classes),-30.0,dtype=np.float64)
        out[:,self.classes_.astype(int)]=z
        return out

    def segment_probs_full(self,x,n_classes):
        q=self.clf.predict_proba(self.scaler.transform(np.asarray(x,dtype=np.float32)))
        out=np.full((len(q),n_classes),1e-8,dtype=np.float64)
        out[:,self.classes_.astype(int)]=q
        return soft_clip_probs(out)

    def predict_recording_mean(self,x,n_classes):
        return soft_clip_probs(self.segment_probs_full(x,n_classes).mean(axis=0))


@dataclass
class RecordingStatsModel:
    scaler: StandardScaler
    clf: LogisticRegression
    classes_: np.ndarray
    @classmethod
    def fit(cls,recordings,seed=42):
        X=np.stack([rec_stats(r.embeddings) for r in recordings]); y=np.asarray([r.label for r in recordings],np.int64)
        nc=Counter(y.tolist()); w=np.asarray([1.0/nc[int(v)] for v in y],float)
        sc=StandardScaler().fit(X)
        clf=LogisticRegression(C=0.05,max_iter=2000,solver='lbfgs',random_state=int(seed))
        clf.fit(sc.transform(X),y,sample_weight=w)
        return cls(sc,clf,clf.classes_.copy())
    def predict_recording(self,x,n_classes):
        q=self.clf.predict_proba(self.scaler.transform(rec_stats(x)[None]))[0]
        out=np.full(n_classes,1e-8,float); out[self.classes_.astype(int)]=q
        return soft_clip_probs(out)
