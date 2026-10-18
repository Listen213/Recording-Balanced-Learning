from __future__ import annotations
import argparse,csv,json,os,time,traceback
from pathlib import Path
import numpy as np

from deepship_uatr.common import sha256_file
from deepship_uatr.data import RecordingCollection
from deepship_uatr.sampling import sample_support
from deepship_uatr.eval import evaluate_draw,METHODS
from deepship_uatr.metrics import (
    classification_metrics,per_class_recall,aggregate_rows,
    hierarchical_paired_bootstrap_fast_all_metrics,
)
from deepship_uatr.checkpoint import (
    save_draw_checkpoint,save_full_checkpoint,checkpoint_valid,load_draw_checkpoint,
    atomic_json,atomic_npz,
)
from deepship_uatr.progress import fmt_seconds,write_progress
from verify_deepship_lock import verify

COMPARISONS=[
    ("S1","S0"),   # core recording-balancing hypothesis
    ("PT","PB"),   # low-rank prototype-tuning hypothesis
    ("F0","S0"),
    ("F0","S1"),
    ("F0","PB"),
    ("F0","PT"),
]

def load_bootstrap_checkpoint(path,lock_sha):
    path=Path(path)
    if not path.exists():
        return {"lock_sha256":lock_sha,"blocks":{}}
    obj=json.loads(path.read_text(encoding="utf-8"))
    if obj.get("lock_sha256")!=lock_sha:
        raise RuntimeError("bootstrap checkpoint belongs to a different DEEPSHIP_LOCK")
    obj.setdefault("blocks",{})
    return obj

def block_key(setting,a,b):
    return f"{setting}|{a}_minus_{b}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lock",default="deepship_lock/DEEPSHIP_LOCK.json")
    ap.add_argument("--out-dir",default="runs/deepship_final")
    ap.add_argument("--resume",action="store_true")
    a=ap.parse_args()

    lock=verify(a.lock)
    lock_sha=sha256_file(a.lock)
    cfg=lock["config"]
    out=Path(a.out_dir)
    checkpoints=out/"checkpoints"/"draws"
    boot_ckpt_path=out/"checkpoints"/"bootstrap_results.json"
    complete=out/"FINAL_COMPLETE.json"
    running=out/"RUNNING.json"
    progress=out/"PROGRESS.json"

    if complete.exists():
        raise RuntimeError(f"DeepShip external validation already completed: {complete}")
    if running.exists() and not a.resume:
        raise RuntimeError(
            f"Found {running}. If the previous process was killed, use --resume. "
            "Completed checkpoints will be reused."
        )

    p=lock["paths"]
    support_pool=RecordingCollection.from_cache(p["support_cache"])
    test=RecordingCollection.from_cache(p["test_cache"])
    n_classes=max(max(support_pool.labels),max(test.labels))+1

    shots=list(cfg["few_recording_protocol"]["shots"])
    seeds=list(cfg["few_recording_protocol"]["support_seeds"])
    source_aware=bool(cfg["few_recording_protocol"].get("source_aware",False))
    include_full=bool(cfg["few_recording_protocol"]["include_full_support_reference"])
    pt_cfg=cfg["prototype_tuning"]
    boot_iters=int(cfg["statistics"]["hierarchical_bootstrap_iters"])
    boot_seed=int(cfg["statistics"]["bootstrap_seed"])

    tasks=[]
    for shot in shots:
        for seed in seeds:
            tasks.append((f"{shot}-rec",int(seed),int(shot)))
    if include_full:
        tasks.append(("full-support-reference",9991,None))

    total=len(tasks)
    total_boot=len(shots)*len(COMPARISONS)
    checkpoints.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)

    done_before=sum(
        checkpoint_valid(checkpoints,s,seed,lock_sha,METHODS)
        for s,seed,_ in tasks
    )
    atomic_json({
        "lock_sha256":lock_sha,
        "pid":os.getpid(),
        "started_at_unix":time.time(),
        "resume":bool(a.resume),
        "completed_draws_at_start":int(done_before),
        "policy":"Progress/checkpoint status is visible; DeepShip test metrics remain hidden until all statistics finish.",
    },running)

    overall_start=time.time()

    try:
        print("DeepShip external validation started.",flush=True)
        print("Methods/hyperparameters are ShipsEar-locked; DeepShip is NOT used for retuning.",flush=True)
        print(f"Evaluation tasks: {total}; bootstrap blocks: {total_boot}.",flush=True)
        print("Metrics remain hidden until FINAL_COMPLETE.json is written.",flush=True)

        done=0
        draw_times=[]
        for setting,seed,shot in tasks:
            if checkpoint_valid(checkpoints,setting,seed,lock_sha,METHODS):
                done+=1
                print(f"[EVAL {done}/{total}] checkpoint OK: {setting} seed={seed} (skip)",flush=True)
                continue

            t0=time.perf_counter()
            support=(support_pool.recordings if shot is None
                     else sample_support(support_pool,shot,seed,source_aware))
            met,diag=evaluate_draw(support,test.recordings,n_classes,seed,pt_cfg)

            if shot is None:
                save_full_checkpoint(checkpoints,seed,lock_sha,len(support),diag,met,METHODS)
            else:
                save_draw_checkpoint(checkpoints,setting,seed,lock_sha,support,diag,met,METHODS)

            dt=time.perf_counter()-t0
            draw_times.append(dt)
            done+=1
            mean_dt=float(np.mean(draw_times))
            eta=mean_dt*(total-done)
            write_progress(
                progress,stage="evaluation",
                eval_done=done,eval_total=total,
                bootstrap_done=0,bootstrap_total=total_boot,
                current=f"{setting} seed={seed}",
                elapsed_seconds=time.time()-overall_start,
                eta_seconds=eta,lock_sha256=lock_sha,
            )
            print(
                f"[EVAL {done}/{total}] {setting} seed={seed} | "
                f"draw={fmt_seconds(dt)} | elapsed={fmt_seconds(time.time()-overall_start)} | "
                f"ETA~{fmt_seconds(eta)} | checkpoint SAVED",
                flush=True
            )

        atomic_json({
            "lock_sha256":lock_sha,
            "completed_at_unix":time.time(),
            "draw_checkpoints":total,
            "note":"All DeepShip predictions checkpointed; no test metric released yet."
        },out/"EVALUATION_COMPLETE.json")
        print(f"[stage] Evaluation complete: {total}/{total}. Loading sealed checkpoints...",flush=True)

        rows=[]; draws=[]; pred_store={}
        draw_probs={f"{k}-rec":{m:[] for m in METHODS} for k in shots}
        test_y=None; test_comp=None

        for setting,seed,shot in tasks:
            meta,arr=load_draw_checkpoint(checkpoints,setting,seed,lock_sha,METHODS)
            y=np.asarray(arr["y"],dtype=np.int64)
            rids=np.asarray(arr["rids"],dtype=object)
            comps=np.asarray(arr["components"],dtype=object)
            test_y=y; test_comp=comps

            key=(f"{setting.replace('-','_')}_seed{seed}"
                 if shot is not None else "full_support_reference")
            pred_store[key+"_y"]=y
            pred_store[key+"_rids"]=rids
            pred_store[key+"_components"]=comps

            if shot is None:
                draws.append({
                    "setting":setting,"seed":seed,
                    "support_recordings":int(meta.get("support_recordings",len(support_pool.recordings))),
                    "note":"Reference only; not part of few-recording primary claims."
                })
            else:
                draws.append({"setting":setting,"seed":seed,"support":meta["support"]})

            for m in METHODS:
                probs=np.asarray(arr[f"{m}_probs"],dtype=np.float64)
                pred_store[key+f"_{m}_probs"]=probs.astype(np.float32)
                d=classification_metrics(y,probs,comps)
                d["per_class_recall"]=per_class_recall(y,probs,n_classes)
                d.update(meta["resource"][m])
                d.update({"setting":setting,"seed":int(seed),"method":m})
                rows.append(d)
                if shot is not None:
                    draw_probs[setting][m].append(probs)

        settings=[f"{k}-rec" for k in shots]
        if include_full:
            settings.append("full-support-reference")
        summary=[aggregate_rows(rows,s,m) for s in settings for m in METHODS]

        boot_state=load_bootstrap_checkpoint(boot_ckpt_path,lock_sha)
        completed_blocks=len(boot_state["blocks"])
        print(
            f"[stage] Fast hierarchical bootstrap: {completed_blocks}/{total_boot} blocks checkpointed. "
            f"Each block computes Macro-F1 / Balanced Accuracy / Accuracy over {boot_iters} resamples.",
            flush=True
        )
        boot_times=[]

        for si,shot in enumerate(shots):
            setting=f"{shot}-rec"
            for ai,(ma,mb) in enumerate(COMPARISONS):
                k=block_key(setting,ma,mb)
                if k in boot_state["blocks"]:
                    print(f"[BOOT {completed_blocks}/{total_boot}] checkpoint OK: {setting} {ma} vs {mb}",flush=True)
                    continue

                A=np.stack(draw_probs[setting][ma],axis=0)
                B=np.stack(draw_probs[setting][mb],axis=0)
                seed=boot_seed+1000*si+ai*37

                t0=time.perf_counter()
                results,meta,_=hierarchical_paired_bootstrap_fast_all_metrics(
                    test_y,test_comp,A,B,iters=boot_iters,seed=seed
                )
                dt=time.perf_counter()-t0
                boot_times.append(dt)
                payload={
                    "setting":setting,
                    "comparison":f"{ma}_minus_{mb}",
                    "seed":int(seed),
                    "meta":meta,
                    "results":results,
                }
                boot_state["blocks"][k]=payload
                atomic_json(boot_state,boot_ckpt_path)
                completed_blocks+=1

                mean_dt=float(np.mean(boot_times))
                eta=mean_dt*(total_boot-completed_blocks)
                write_progress(
                    progress,stage="bootstrap",
                    eval_done=total,eval_total=total,
                    bootstrap_done=completed_blocks,bootstrap_total=total_boot,
                    current=f"{setting} {ma} vs {mb}",
                    elapsed_seconds=time.time()-overall_start,
                    eta_seconds=eta,last_block_seconds=dt,
                    lock_sha256=lock_sha,
                )
                print(
                    f"[BOOT {completed_blocks}/{total_boot}] {setting} {ma} vs {mb} | "
                    f"{boot_iters} resamples × 3 metrics | block={dt:.3f}s | "
                    f"ETA~{fmt_seconds(eta)} | checkpoint SAVED",
                    flush=True
                )

        paired=[]
        for shot in shots:
            setting=f"{shot}-rec"
            for ma,mb in COMPARISONS:
                payload=boot_state["blocks"][block_key(setting,ma,mb)]
                for metric in ["macro_f1","balanced_accuracy","accuracy"]:
                    r=dict(payload["results"][metric])
                    r.update({"setting":setting,"comparison":f"{ma}_minus_{mb}"})
                    paired.append(r)

        print("[stage] Finalizing DeepShip summary / predictions / paper statistics...",flush=True)
        atomic_json(summary,out/"summary.json")
        atomic_json(rows,out/"detail.json")
        atomic_json(draws,out/"support_draws.json")
        atomic_json(paired,out/"hierarchical_paired_bootstrap.json")
        atomic_npz(out/"predictions.npz",**pred_store)

        tmp=out/"summary.csv.tmp"
        with open(tmp,"w",newline="",encoding="utf-8") as f:
            keys=sorted({kk for r in summary for kk in r})
            w=csv.DictWriter(f,fieldnames=keys)
            w.writeheader(); w.writerows(summary)
        os.replace(tmp,out/"summary.csv")

        completion={
            "dataset":"DeepShip",
            "experiment_role":"external_validation",
            "method_locked_on":"ShipsEar",
            "lock_sha256":lock_sha,
            "completed_at_unix":time.time(),
            "elapsed_seconds":float(time.time()-overall_start),
            "n_test_recordings":int(len(test.recordings)),
            "n_test_components":int(len(set(r.component_id or r.recording_id for r in test.recordings))),
            "support_draws_per_setting":len(seeds),
            "methods":METHODS,
            "bootstrap_engine":"vectorized exact-definition v2",
            "bootstrap_blocks":total_boot,
            "bootstrap_iters_per_block":boot_iters,
            "policy":"DeepShip external validation complete. Do not retune ShipsEar-locked method on these outcomes."
        }
        atomic_json(completion,complete)
        write_progress(
            progress,stage="complete",
            eval_done=total,eval_total=total,
            bootstrap_done=total_boot,bootstrap_total=total_boot,
            current="complete",elapsed_seconds=time.time()-overall_start,
            eta_seconds=0,lock_sha256=lock_sha,
        )
        if running.exists(): running.unlink()

        print("\n===== DEEPSHIP FINAL SUMMARY =====")
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        print("\n===== HIERARCHICAL PAIRED BOOTSTRAP =====")
        print(json.dumps(paired,ensure_ascii=False,indent=2))
        print(f"\nDEEPSHIP EXTERNAL VALIDATION COMPLETE in {fmt_seconds(time.time()-overall_start)}.")
        print("Do not retune the method on these outcomes.")

    except Exception:
        traceback.print_exc()
        try:
            write_progress(
                progress,stage="crashed",current="see log / traceback",
                elapsed_seconds=time.time()-overall_start,
                lock_sha256=lock_sha,
            )
        except Exception:
            pass
        raise

if __name__=="__main__":
    main()
