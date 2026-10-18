from __future__ import annotations
import time
import numpy as np
from .baselines import SegmentEvidenceModel
from .prototypes import PrototypeModel
from .common import geometric_fusion
from .metrics import classification_metrics, per_class_recall

METHODS=["S0","S1","PB","PT","F0"]

def _seg_params(m):
    return int(m.clf.coef_.size + m.clf.intercept_.size)

def _predict_segment(model,recs,n_classes):
    return np.stack([model.predict_recording_mean(r.embeddings,n_classes) for r in recs])

def _predict_proto(model,recs):
    return np.stack([model.predict_recording(r) for r in recs])

def evaluate_draw(support,query,n_classes,seed,pt_cfg):
    y=np.asarray([r.label for r in query],dtype=np.int64)
    rids=np.asarray([r.recording_id for r in query],dtype=object)
    comps=np.asarray([r.component_id or r.recording_id for r in query],dtype=object)

    timing={}; params={}

    t=time.perf_counter()
    s0=SegmentEvidenceModel.fit(support,seed=seed,balance_mode="uniform")
    fit_s0=time.perf_counter()-t
    t=time.perf_counter(); p_s0=_predict_segment(s0,query,n_classes)
    pred_s0=time.perf_counter()-t
    timing["S0"]=(fit_s0,pred_s0,fit_s0+pred_s0)
    params["S0"]=_seg_params(s0)

    t=time.perf_counter()
    s1=SegmentEvidenceModel.fit(support,seed=seed+17,balance_mode="recording")
    fit_s1=time.perf_counter()-t
    t=time.perf_counter(); p_s1=_predict_segment(s1,query,n_classes)
    pred_s1=time.perf_counter()-t
    timing["S1"]=(fit_s1,pred_s1,fit_s1+pred_s1)
    params["S1"]=_seg_params(s1)

    # PB uses the S1-standardized recording representation. Report total
    # downstream pipeline time including the S1 preprocessing/model fit.
    t=time.perf_counter()
    pb=PrototypeModel.fit_base(
        s1,support,n_classes,temperature=float(pt_cfg["temperature"])
    )
    fit_pb_extra=time.perf_counter()-t
    t=time.perf_counter(); p_pb=_predict_proto(pb,query)
    pred_pb_extra=time.perf_counter()-t
    timing["PB"]=(fit_s1+fit_pb_extra,pred_pb_extra,fit_s1+fit_pb_extra+pred_pb_extra)
    params["PB"]=_seg_params(s1)

    t=time.perf_counter()
    pt=PrototypeModel.fit_tuned(
        s1,support,n_classes,
        rank=int(pt_cfg["rank"]),
        steps=int(pt_cfg["steps"]),
        lr=float(pt_cfg["lr"]),
        l2=float(pt_cfg["l2"]),
        temperature=float(pt_cfg["temperature"]),
        seed=seed+503,
    )
    fit_pt_extra=time.perf_counter()-t
    t=time.perf_counter(); p_pt=_predict_proto(pt,query)
    pred_pt_extra=time.perf_counter()-t
    timing["PT"]=(fit_s1+fit_pt_extra,pred_pt_extra,fit_s1+fit_pt_extra+pred_pt_extra)
    params["PT"]=_seg_params(s1)+int(pt.info.get("trainable_params",n_classes*int(pt_cfg["rank"])))

    t=time.perf_counter()
    p_f0=geometric_fusion(p_s1,p_pt,0.5)
    pred_f0=time.perf_counter()-t
    timing["F0"]=(
        fit_s1+fit_pt_extra,
        pred_s1+pred_pt_extra+pred_f0,
        fit_s1+fit_pt_extra+pred_s1+pred_pt_extra+pred_f0
    )
    params["F0"]=params["PT"]

    probs={"S0":p_s0,"S1":p_s1,"PB":p_pb,"PT":p_pt,"F0":p_f0}
    metrics={}
    for m in METHODS:
        d=classification_metrics(y,probs[m],comps)
        d["per_class_recall"]=per_class_recall(y,probs[m],n_classes)
        d["fit_seconds"]=float(timing[m][0])
        d["predict_seconds"]=float(timing[m][1])
        d["pipeline_seconds"]=float(timing[m][2])
        d["trainable_params"]=int(params[m])
        metrics[m]=d

    return metrics,{
        "y":y,"rids":rids,"components":comps,"probs":probs,
        "pt_info":pt.info,
    }
