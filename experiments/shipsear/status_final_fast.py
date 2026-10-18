from __future__ import annotations
from pathlib import Path
import argparse,json,os,time
from final_uatr.progress import fmt_seconds


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',default='runs/final_fast')
    a=ap.parse_args()
    rd=Path(a.run_dir)
    pf=rd/'PROGRESS.json'; cf=rd/'FINAL_COMPLETE.json'; rf=rd/'RUNNING.json'
    if cf.exists():
        c=json.loads(cf.read_text())
        print('STATUS: COMPLETE')
        print('elapsed:',fmt_seconds(c.get('elapsed_seconds')))
        print('results:',rd/'summary.json')
        return
    if pf.exists():
        p=json.loads(pf.read_text())
        print('STATUS:',str(p.get('stage','unknown')).upper())
        if 'eval_done' in p:
            print(f"evaluation: {p.get('eval_done')}/{p.get('eval_total')}")
        if 'bootstrap_done' in p:
            print(f"bootstrap: {p.get('bootstrap_done')}/{p.get('bootstrap_total')}")
        print('current:',p.get('current','--'))
        print('elapsed:',fmt_seconds(p.get('elapsed_seconds')))
        print('ETA:',fmt_seconds(p.get('eta_seconds')))
        print('last update:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(p.get('updated_at_unix',0))))
    else:
        print('STATUS: NOT STARTED (no PROGRESS.json)')
    if rf.exists():
        r=json.loads(rf.read_text())
        pid=r.get('pid')
        alive=False
        try:
            os.kill(int(pid),0); alive=True
        except Exception: pass
        print('pid:',pid,'alive' if alive else 'not alive / stale marker')

if __name__=='__main__': main()
