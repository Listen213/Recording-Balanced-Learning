from __future__ import annotations
from collections import defaultdict
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix


def classification_metrics(y_true, probs, component_ids=None):
    y=np.asarray(y_true,dtype=np.int64)
    p=np.asarray(probs,dtype=float)
    pred=p.argmax(1)
    out={
        "accuracy":float(accuracy_score(y,pred)),
        "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "confusion_matrix":confusion_matrix(y,pred).tolist(),
        "n_recordings":int(len(y)),
    }
    if component_ids is not None:
        yc,pc=aggregate_components(y,p,component_ids)
        cp=pc.argmax(1)
        out.update({
            "component_accuracy":float(accuracy_score(yc,cp)),
            "component_macro_f1":float(f1_score(yc,cp,average="macro",zero_division=0)),
            "component_balanced_accuracy":float(balanced_accuracy_score(yc,cp)),
            "component_confusion_matrix":confusion_matrix(yc,cp).tolist(),
            "n_components":int(len(yc)),
        })
    return out


def aggregate_components(y,probs,component_ids):
    y=np.asarray(y,dtype=np.int64)
    p=np.asarray(probs,dtype=float)
    cids=np.asarray(component_ids,dtype=object)
    order=[]; groups=defaultdict(list)
    for i,c in enumerate(cids.tolist()):
        if c not in groups: order.append(c)
        groups[c].append(i)
    ys=[]; ps=[]
    for c in order:
        idx=np.asarray(groups[c],dtype=np.int64)
        vals=np.unique(y[idx])
        if len(vals)!=1:
            raise RuntimeError(f"component {c} has mixed labels {vals.tolist()}")
        ys.append(int(vals[0]))
        q=p[idx].mean(axis=0)
        q=q/np.clip(q.sum(),1e-12,None)
        ps.append(q)
    return np.asarray(ys,dtype=np.int64),np.stack(ps)


def per_class_recall(y,probs,n_classes):
    y=np.asarray(y,dtype=np.int64)
    pred=np.asarray(probs).argmax(1)
    out=[]
    for c in range(int(n_classes)):
        idx=(y==c)
        out.append(float(np.mean(pred[idx]==c)) if np.any(idx) else None)
    return out


def aggregate_rows(rows,setting,method):
    sub=[r for r in rows if r["setting"]==setting and r["method"]==method]
    out={"setting":setting,"method":method,"n_support_draws":len(sub)}
    for k in [
        "accuracy","macro_f1","balanced_accuracy",
        "component_accuracy","component_macro_f1","component_balanced_accuracy",
        "fit_seconds","predict_seconds","pipeline_seconds","trainable_params"
    ]:
        vals=[r[k] for r in sub if r.get(k) is not None]
        if vals:
            a=np.asarray(vals,dtype=float)
            out[k+"_mean"]=float(a.mean())
            out[k+"_std"]=float(a.std(ddof=0))
    if sub and sub[0].get("per_class_recall") is not None:
        A=np.asarray([r["per_class_recall"] for r in sub],dtype=float)
        out["per_class_recall_mean"]=np.nanmean(A,axis=0).tolist()
        out["per_class_recall_std"]=np.nanstd(A,axis=0).tolist()
    return out


def _metric(y,p,name):
    pred=p.argmax(1)
    if name=="macro_f1":
        return f1_score(y,pred,average="macro",zero_division=0)
    if name=="balanced_accuracy":
        return balanced_accuracy_score(y,pred)
    if name=="accuracy":
        return accuracy_score(y,pred)
    raise ValueError(name)


def hierarchical_paired_bootstrap_slow(
    y, component_ids, probs_a_by_draw, probs_b_by_draw,
    iters=5000, seed=42, metric="macro_f1"
):
    """Reference implementation retained for validation only."""
    y=np.asarray(y,dtype=np.int64)
    cids=np.asarray(component_ids,dtype=object)
    A=np.asarray(probs_a_by_draw,dtype=float)
    B=np.asarray(probs_b_by_draw,dtype=float)
    if A.shape!=B.shape or A.ndim!=3:
        raise ValueError("probs arrays must be [draw, recording, class] and equal shape")
    if A.shape[1]!=len(y):
        raise ValueError("query length mismatch")

    comp_order=[]; comp_idx={}
    for i,c in enumerate(cids.tolist()):
        if c not in comp_idx:
            comp_order.append(c); comp_idx[c]=[]
        comp_idx[c].append(i)

    comp_label={}
    for c in comp_order:
        vals=np.unique(y[np.asarray(comp_idx[c],dtype=int)])
        if len(vals)!=1:
            raise RuntimeError(f"mixed-label component {c}")
        comp_label[c]=int(vals[0])

    by_class=defaultdict(list)
    for c in comp_order:
        by_class[comp_label[c]].append(c)

    rng=np.random.default_rng(int(seed))
    deltas=[]
    D=A.shape[0]
    for _ in range(int(iters)):
        dsel=rng.choice(D,D,replace=True)
        sampled_components=[]
        sampled_labels=[]
        for lab in sorted(by_class):
            pool=by_class[lab]
            pick=rng.choice(len(pool),len(pool),replace=True)
            for j in pick:
                sampled_components.append(pool[int(j)])
                sampled_labels.append(lab)
        yy=np.asarray(sampled_labels,dtype=np.int64)

        draw_deltas=[]
        for d in dsel:
            pa=[]; pb=[]
            for c in sampled_components:
                idx=np.asarray(comp_idx[c],dtype=int)
                qa=A[int(d),idx].mean(axis=0); qa=qa/np.clip(qa.sum(),1e-12,None)
                qb=B[int(d),idx].mean(axis=0); qb=qb/np.clip(qb.sum(),1e-12,None)
                pa.append(qa); pb.append(qb)
            pa=np.stack(pa); pb=np.stack(pb)
            draw_deltas.append(float(_metric(yy,pa,metric)-_metric(yy,pb,metric)))
        deltas.append(float(np.mean(draw_deltas)))

    d=np.asarray(deltas,dtype=float)
    return {
        "metric":metric,
        "mean_delta":float(d.mean()),
        "low95":float(np.quantile(d,0.025)),
        "high95":float(np.quantile(d,0.975)),
        "p_delta_le_0":float(np.mean(d<=0)),
        "iters":int(iters),
    }


def _component_predictions(y, component_ids, probs_by_draw):
    """Aggregate recording probabilities to fixed test components once."""
    y=np.asarray(y,dtype=np.int64)
    cids=np.asarray(component_ids,dtype=object)
    P=np.asarray(probs_by_draw,dtype=np.float64)
    if P.ndim!=3 or P.shape[1]!=len(y):
        raise ValueError("probs must be [draw, recording, class]")

    comp_order=[]; groups={}
    for i,c in enumerate(cids.tolist()):
        if c not in groups:
            groups[c]=[]; comp_order.append(c)
        groups[c].append(i)

    comp_y=[]
    comp_probs=[]
    for c in comp_order:
        idx=np.asarray(groups[c],dtype=np.int64)
        vals=np.unique(y[idx])
        if len(vals)!=1:
            raise RuntimeError(f"mixed-label component {c}")
        comp_y.append(int(vals[0]))
        q=P[:,idx,:].mean(axis=1)
        q=q/np.clip(q.sum(axis=1,keepdims=True),1e-12,None)
        comp_probs.append(q)

    # [component, draw, class] -> [draw, component, class]
    comp_probs=np.stack(comp_probs,axis=0).transpose(1,0,2)
    return np.asarray(comp_y,dtype=np.int64), np.asarray(comp_order,dtype=object), comp_probs


def _all_metrics_from_pred(pred, y_boot, classes):
    """
    pred:   [R, D, N]
    y_boot: [R, N]
    Returns each metric as [R, D].
    """
    pred=np.asarray(pred,dtype=np.int16)
    yb=np.asarray(y_boot,dtype=np.int16)

    acc=(pred==yb[:,None,:]).mean(axis=2,dtype=np.float64)
    recalls=[]
    f1s=[]
    for c in classes:
        truth=(yb==int(c))[:,None,:]
        tp=((pred==int(c)) & truth).sum(axis=2,dtype=np.int32)
        true_n=truth.sum(axis=2,dtype=np.int32)  # [R,1]
        pred_n=(pred==int(c)).sum(axis=2,dtype=np.int32)
        rec=np.divide(tp,true_n,out=np.zeros_like(tp,dtype=np.float64),where=true_n>0)
        den=true_n+pred_n
        f1=np.divide(2.0*tp,den,out=np.zeros_like(tp,dtype=np.float64),where=den>0)
        recalls.append(rec)
        f1s.append(f1)
    bal=np.mean(np.stack(recalls,axis=0),axis=0)
    mf1=np.mean(np.stack(f1s,axis=0),axis=0)
    return {"accuracy":acc,"balanced_accuracy":bal,"macro_f1":mf1}


def hierarchical_paired_bootstrap_fast_all_metrics(
    y, component_ids, probs_a_by_draw, probs_b_by_draw,
    iters=5000, seed=42
):
    """
    Vectorized equivalent of the original hierarchical paired bootstrap.

    Scientific definition is unchanged:
      - support draws are resampled with replacement;
      - test near-duplicate components are resampled WITHIN class;
      - paired A/B metrics use exactly the same resample;
      - per-replicate delta is averaged over the resampled support draws.

    All three metrics are computed from the SAME bootstrap resamples, matching
    the original runner which called each metric with the same seed.
    """
    t0=time.perf_counter()
    y=np.asarray(y,dtype=np.int64)
    A=np.asarray(probs_a_by_draw,dtype=np.float64)
    B=np.asarray(probs_b_by_draw,dtype=np.float64)
    if A.shape!=B.shape or A.ndim!=3:
        raise ValueError("probs arrays must be [draw, recording, class] and equal shape")
    if A.shape[1]!=len(y):
        raise ValueError("query length mismatch")

    comp_y, comp_order, Acomp=_component_predictions(y,component_ids,A)
    comp_y_b, comp_order_b, Bcomp=_component_predictions(y,component_ids,B)
    if not np.array_equal(comp_y,comp_y_b) or not np.array_equal(comp_order,comp_order_b):
        raise RuntimeError("component structure mismatch between paired branches")

    classes=sorted(np.unique(comp_y).tolist())
    by_class={c:np.where(comp_y==c)[0] for c in classes}
    D=A.shape[0]
    R=int(iters)
    N=len(comp_y)

    # Preserve the original RNG call order exactly: draw sample first, then
    # each class-specific component sample, replicate by replicate.
    rng=np.random.default_rng(int(seed))
    draw_samples=np.empty((R,D),dtype=np.int16 if D<32767 else np.int32)
    comp_samples=np.empty((R,N),dtype=np.int16 if N<32767 else np.int32)
    for r in range(R):
        draw_samples[r]=rng.choice(D,D,replace=True)
        off=0
        for c in classes:
            pool=by_class[c]
            pick=rng.choice(len(pool),len(pool),replace=True)
            n=len(pool)
            comp_samples[r,off:off+n]=pool[pick]
            off+=n

    # Component-level argmax can be precomputed because normalization after
    # component averaging does not alter argmax.
    Apred=Acomp.argmax(axis=2).astype(np.int16)  # [D,M]
    Bpred=Bcomp.argmax(axis=2).astype(np.int16)

    # Advanced indexing broadcasts [R,D,1] with [R,1,N] -> [R,D,N].
    pa=Apred[draw_samples[:,:,None],comp_samples[:,None,:]]
    pb=Bpred[draw_samples[:,:,None],comp_samples[:,None,:]]
    yb=comp_y[comp_samples]

    ma=_all_metrics_from_pred(pa,yb,classes)
    mb=_all_metrics_from_pred(pb,yb,classes)

    results={}
    raw={}
    for metric in ["macro_f1","balanced_accuracy","accuracy"]:
        delta=(ma[metric]-mb[metric]).mean(axis=1)
        raw[metric]=delta
        results[metric]={
            "metric":metric,
            "mean_delta":float(delta.mean()),
            "low95":float(np.quantile(delta,0.025)),
            "high95":float(np.quantile(delta,0.975)),
            "p_delta_le_0":float(np.mean(delta<=0)),
            "iters":R,
        }
    elapsed=time.perf_counter()-t0
    meta={
        "elapsed_seconds":float(elapsed),
        "n_draws":int(D),
        "n_components":int(N),
        "n_classes":int(len(classes)),
        "vectorized":True,
        "rng_order_matches_reference":True,
    }
    return results,meta,raw


# Backward-compatible name. The fast final runner calls the all-metric version
# directly so it does not repeat the same resampling three times.
def hierarchical_paired_bootstrap(
    y, component_ids, probs_a_by_draw, probs_b_by_draw,
    iters=5000, seed=42, metric="macro_f1"
):
    results,_,_=hierarchical_paired_bootstrap_fast_all_metrics(
        y,component_ids,probs_a_by_draw,probs_b_by_draw,iters=iters,seed=seed
    )
    return results[metric]
