from __future__ import annotations
import argparse,csv,json,os,time,traceback
from pathlib import Path
import numpy as np

from final_uatr.common import save_json,sha256_file,set_seed
from final_uatr.data import RecordingCollection
from final_uatr.sampling import sample_support
from final_uatr.eval import evaluate_draw,METHODS
from final_uatr.metrics import (
    classification_metrics,per_class_recall,aggregate_rows,
    hierarchical_paired_bootstrap_fast_all_metrics,
)
from final_uatr.checkpoint import (
    save_draw_checkpoint,save_full_checkpoint,checkpoint_valid,load_draw_checkpoint,
    atomic_json,atomic_npz,
)
from final_uatr.progress import fmt_seconds,eta_from,write_progress
from verify_final_lock import verify


def load_bootstrap_checkpoint(path,lock_sha):
    path=Path(path)
    if not path.exists(): return {"lock_sha256":lock_sha,"blocks":{}}
    obj=json.loads(path.read_text(encoding='utf-8'))
    if obj.get('lock_sha256')!=lock_sha:
        raise RuntimeError('bootstrap checkpoint belongs to a different FINAL_LOCK')
    obj.setdefault('blocks',{})
    return obj


def block_key(setting,base):
    return f'{setting}|F0_minus_{base}'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--lock',default='final_lock/FINAL_LOCK.json')
    ap.add_argument('--out-dir',default='runs/final_fast')
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args()

    lock=verify(a.lock)
    lock_sha=sha256_file(a.lock)
    cfg=lock['config']
    out=Path(a.out_dir)
    checkpoints=out/'checkpoints'/'draws'
    boot_ckpt_path=out/'checkpoints'/'bootstrap_results.json'
    complete=out/'FINAL_COMPLETE.json'
    running=out/'RUNNING.json'
    progress=out/'PROGRESS.json'

    if complete.exists():
        raise RuntimeError(f'Final evaluation already completed: {complete}. Do not rerun.')
    if running.exists() and not a.resume:
        raise RuntimeError(
            f'Found {running}. If the previous process was killed, use --resume. '
            'Checkpoints will be reused; no completed draw will be recomputed.'
        )

    out.mkdir(parents=True,exist_ok=True)
    checkpoints.mkdir(parents=True,exist_ok=True)

    if running.exists() and a.resume:
        run_state=json.loads(running.read_text(encoding='utf-8'))
        if run_state.get('lock_sha256')!=lock_sha:
            raise RuntimeError('RUNNING marker lock mismatch')
        overall_start=float(run_state.get('started_at_unix',time.time()))
        run_state['resumed_at_unix']=time.time()
        run_state['pid']=os.getpid()
        atomic_json(run_state,running)
    else:
        overall_start=time.time()
        atomic_json({
            'lock_sha256':lock_sha,
            'started_at_unix':overall_start,
            'pid':os.getpid(),
            'runner':'run_final_fast_suite.py',
            'policy':'Progress/ETA visible; final test metrics remain hidden until all bootstrap blocks complete.',
        },running)

    try:
        p=lock['paths']
        support_pool=RecordingCollection.from_cache(p['support_cache'])
        test=RecordingCollection.from_cache(p['test_cache'])
        n_classes=max(max(support_pool.labels),max(test.labels))+1

        shots=list(cfg['few_recording_protocol']['shots'])
        seeds=list(cfg['few_recording_protocol']['support_seeds'])
        source_aware=bool(cfg['few_recording_protocol']['source_aware'])
        include_full=bool(cfg['few_recording_protocol']['include_full_support_reference'])
        pt_cfg=cfg['prototype_tuning']
        boot_iters=int(cfg['statistics']['hierarchical_bootstrap_iters'])
        boot_seed=int(cfg['statistics']['bootstrap_seed'])

        tasks=[]
        for shot in shots:
            for seed in seeds:
                tasks.append((f'{shot}-rec',int(seed),int(shot)))
        if include_full:
            tasks.append(('full-support-reference',9991,None))
        total=len(tasks)

        done_before=0
        for setting,seed,_ in tasks:
            if checkpoint_valid(checkpoints,setting,seed,lock_sha,METHODS):
                done_before+=1
        print(f'[startup] FINAL_LOCK verified. Existing valid draw checkpoints: {done_before}/{total}',flush=True)
        print('[startup] Test metrics remain hidden until FINAL_COMPLETE.json is written.',flush=True)

        eval_start=time.time()
        done=done_before
        draw_times=[]
        for setting,seed,shot in tasks:
            if checkpoint_valid(checkpoints,setting,seed,lock_sha,METHODS):
                print(f'[EVAL {done}/{total}] checkpoint OK: {setting} seed={seed} (skip recompute)',flush=True)
                continue

            t0=time.perf_counter()
            if shot is None:
                support=support_pool.recordings
            else:
                support=sample_support(support_pool,shot,seed,source_aware)
            met,diag=evaluate_draw(support,test.recordings,n_classes,seed,pt_cfg)
            if shot is None:
                save_full_checkpoint(checkpoints,seed,lock_sha,len(support),diag,met,METHODS)
            else:
                save_draw_checkpoint(checkpoints,setting,seed,lock_sha,support,diag,met,METHODS)
            dt=time.perf_counter()-t0
            draw_times.append(dt)
            done+=1
            # ETA uses observed work in this invocation plus already completed count only for fraction.
            mean_dt=float(np.mean(draw_times)) if draw_times else 0.0
            eta=mean_dt*(total-done) if draw_times else None
            write_progress(
                progress,
                stage='evaluation',eval_done=done,eval_total=total,
                bootstrap_done=0,bootstrap_total=12,
                current=f'{setting} seed={seed}',
                elapsed_seconds=time.time()-overall_start,
                eta_seconds=eta,
                lock_sha256=lock_sha,
            )
            print(
                f'[EVAL {done}/{total}] {setting} seed={seed} | '
                f'draw={fmt_seconds(dt)} | elapsed={fmt_seconds(time.time()-overall_start)} | '
                f'ETA~{fmt_seconds(eta)} | checkpoint SAVED',flush=True
            )

        # Explicit evaluation-stage marker. Contains no test metric.
        atomic_json({
            'lock_sha256':lock_sha,'completed_at_unix':time.time(),
            'draw_checkpoints':total,
            'note':'All predictions checkpointed. No test metric released yet.'
        },out/'EVALUATION_COMPLETE.json')
        print(f'[stage] Evaluation complete: {total}/{total}. Loading sealed checkpoints for statistics...',flush=True)

        # Reconstruct final rows and predictions from sealed draw checkpoints.
        rows=[]; draws=[]; pred_store={}
        draw_probs={f'{k}-rec':{m:[] for m in METHODS} for k in shots}
        test_y=None; test_comp=None; test_rids=None

        for setting,seed,shot in tasks:
            meta,arr=load_draw_checkpoint(checkpoints,setting,seed,lock_sha,METHODS)
            y=np.asarray(arr['y'],dtype=np.int64)
            rids=np.asarray(arr['rids'],dtype=object)
            comps=np.asarray(arr['components'],dtype=object)
            test_y=y; test_comp=comps; test_rids=rids
            key=f"{setting.replace('-','_')}_seed{seed}" if shot is not None else 'full_support_reference'
            pred_store[key+'_y']=y
            pred_store[key+'_rids']=rids
            pred_store[key+'_components']=comps

            if shot is None:
                draws.append({
                    'setting':setting,'seed':seed,
                    'support_recordings':int(meta.get('support_recordings',len(support_pool.recordings))),
                    'note':'Reference only; not part of few-recording primary claims.',
                })
            else:
                draws.append({'setting':setting,'seed':seed,'support':meta['support']})

            for m in METHODS:
                probs=np.asarray(arr[f'{m}_probs'],dtype=np.float64)
                pred_store[key+f'_{m}_probs']=probs.astype(np.float32)
                d=classification_metrics(y,probs,comps)
                d['per_class_recall']=per_class_recall(y,probs,n_classes)
                resource=meta['resource'][m]
                d.update(resource)
                d.update({'setting':setting,'seed':int(seed),'method':m})
                rows.append(d)
                if shot is not None:
                    draw_probs[setting][m].append(probs)

        settings=[f'{k}-rec' for k in shots]
        if include_full: settings.append('full-support-reference')
        summary=[aggregate_rows(rows,s,m) for s in settings for m in METHODS]

        # Fast bootstrap: 12 paired comparison blocks, 3 metrics per block.
        comparisons=['S0','S1','PB','PT']
        total_blocks=len(shots)*len(comparisons)
        boot_state=load_bootstrap_checkpoint(boot_ckpt_path,lock_sha)
        completed_blocks=len(boot_state['blocks'])
        print(
            f'[stage] Fast hierarchical bootstrap: {completed_blocks}/{total_blocks} blocks already checkpointed. '
            f'Each block computes all 3 metrics over {boot_iters} identical resamples.',flush=True
        )
        boot_times=[]
        boot_start=time.time()

        for si,shot in enumerate(shots):
            setting=f'{shot}-rec'
            A=np.stack(draw_probs[setting]['F0'],axis=0)
            for base in comparisons:
                k=block_key(setting,base)
                if k in boot_state['blocks']:
                    print(f'[BOOT {completed_blocks}/{total_blocks}] checkpoint OK: {setting} F0 vs {base}',flush=True)
                    continue
                B=np.stack(draw_probs[setting][base],axis=0)
                seed=boot_seed+1000*si+METHODS.index(base)*17
                t0=time.perf_counter()
                results,meta,_=hierarchical_paired_bootstrap_fast_all_metrics(
                    test_y,test_comp,A,B,iters=boot_iters,seed=seed
                )
                dt=time.perf_counter()-t0
                boot_times.append(dt)
                payload={
                    'setting':setting,'comparison':f'F0_minus_{base}',
                    'seed':int(seed),'meta':meta,'results':results,
                }
                boot_state['blocks'][k]=payload
                atomic_json(boot_state,boot_ckpt_path)
                completed_blocks+=1
                mean_dt=float(np.mean(boot_times)) if boot_times else 0.0
                eta=mean_dt*(total_blocks-completed_blocks)
                write_progress(
                    progress,
                    stage='bootstrap',eval_done=total,eval_total=total,
                    bootstrap_done=completed_blocks,bootstrap_total=total_blocks,
                    current=f'{setting} F0 vs {base}',
                    elapsed_seconds=time.time()-overall_start,
                    eta_seconds=eta,
                    last_block_seconds=dt,
                    lock_sha256=lock_sha,
                )
                print(
                    f'[BOOT {completed_blocks}/{total_blocks}] {setting} F0 vs {base} | '
                    f'{boot_iters} resamples × 3 metrics | block={dt:.3f}s | '
                    f'ETA~{fmt_seconds(eta)} | checkpoint SAVED',flush=True
                )

        paired=[]
        for shot in shots:
            setting=f'{shot}-rec'
            for base in comparisons:
                payload=boot_state['blocks'][block_key(setting,base)]
                for metric in ['macro_f1','balanced_accuracy','accuracy']:
                    r=dict(payload['results'][metric])
                    r.update({'setting':setting,'comparison':f'F0_minus_{base}'})
                    paired.append(r)

        # Atomically materialize final artifacts only after all 12 bootstrap blocks.
        print('[stage] Finalizing summary / predictions / paper statistics...',flush=True)
        atomic_json(summary,out/'summary.json')
        atomic_json(rows,out/'detail.json')
        atomic_json(draws,out/'support_draws.json')
        atomic_json(paired,out/'hierarchical_paired_bootstrap.json')
        atomic_npz(out/'predictions.npz',**pred_store)

        with open(out/'summary.csv.tmp','w',newline='',encoding='utf-8') as f:
            keys=sorted({kk for r in summary for kk in r})
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(summary)
        os.replace(out/'summary.csv.tmp',out/'summary.csv')

        completion={
            'lock_sha256':lock_sha,
            'completed_at_unix':time.time(),
            'elapsed_seconds':float(time.time()-overall_start),
            'n_test_recordings':int(len(test.recordings)),
            'n_test_components':int(len(set(r.component_id or r.recording_id for r in test.recordings))),
            'support_draws_per_setting':len(seeds),
            'methods':METHODS,
            'bootstrap_engine':'vectorized exact-definition v2',
            'bootstrap_blocks':total_blocks,
            'bootstrap_iters_per_block':boot_iters,
            'policy':'Final test complete. Do not retune methods on these outcomes.',
        }
        atomic_json(completion,complete)
        write_progress(
            progress,stage='complete',eval_done=total,eval_total=total,
            bootstrap_done=total_blocks,bootstrap_total=total_blocks,
            current='complete',elapsed_seconds=time.time()-overall_start,
            eta_seconds=0,lock_sha256=lock_sha,
        )
        if running.exists(): running.unlink()

        print('\n===== FINAL SUMMARY =====')
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        print('\n===== HIERARCHICAL PAIRED BOOTSTRAP =====')
        print(json.dumps(paired,ensure_ascii=False,indent=2))
        print(f"\nFINAL TEST COMPLETE in {fmt_seconds(time.time()-overall_start)}.")
        print('Do not retune methods on these outcomes.')

    except Exception:
        traceback.print_exc()
        # Keep RUNNING + all completed checkpoints for a true resume.
        try:
            write_progress(
                progress,stage='crashed',current='see log / traceback',
                elapsed_seconds=time.time()-overall_start,
                lock_sha256=lock_sha,
            )
        except Exception:
            pass
        raise

if __name__=='__main__':
    main()
