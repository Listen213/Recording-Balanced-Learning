from __future__ import annotations
from collections import defaultdict
import numpy as np

def sample_support(col, shots, seed, source_aware=True):
    """
    Draw K raw recordings per class.
    No two selected recordings may come from the same near-duplicate component.
    """
    rng=np.random.default_rng(int(seed))
    by=defaultdict(list)
    for r in col.recordings:
        by[int(r.label)].append(r)

    selected=[]
    for y in sorted(by):
        cand=list(by[y])
        by_comp=defaultdict(list)
        for r in cand:
            by_comp[str(r.component_id or r.recording_id)].append(r)
        comps=list(by_comp)
        if len(comps)<int(shots):
            raise ValueError(
                f"class {y} has only {len(comps)} independent near-duplicate components < shots={shots}"
            )

        # First choose components, optionally maximizing source subtype diversity.
        if source_aware and any(r.source for r in cand):
            comp_source={}
            for cid,rs in by_comp.items():
                ss=[r.source for r in rs if r.source]
                comp_source[cid]=ss[0] if ss else f"__unknown__{cid}"
            groups=defaultdict(list)
            for cid in comps:
                groups[comp_source[cid]].append(cid)
            keys=list(groups); rng.shuffle(keys)
            chosen_comps=[]
            for k in keys:
                if len(chosen_comps)>=shots: break
                g=groups[k]
                chosen_comps.append(g[int(rng.integers(len(g)))])
            if len(chosen_comps)<shots:
                rem=[c for c in comps if c not in set(chosen_comps)]
                idx=rng.choice(len(rem),shots-len(chosen_comps),replace=False)
                chosen_comps.extend(rem[int(i)] for i in idx)
        else:
            idx=rng.choice(len(comps),shots,replace=False)
            chosen_comps=[comps[int(i)] for i in idx]

        # If one component contains multiple near-duplicate files, select exactly one
        # raw recording as the shot representative.
        for cid in chosen_comps:
            rs=by_comp[cid]
            selected.append(rs[int(rng.integers(len(rs)))])
    return selected
