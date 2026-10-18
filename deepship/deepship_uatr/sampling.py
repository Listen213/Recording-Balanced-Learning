from __future__ import annotations
from collections import defaultdict
import numpy as np

def sample_support(col, shots, seed, source_aware=False):
    """
    K recording units per class.

    Hard independence:
      - no two selected recordings may come from the same exact/near-duplicate component.

    Optional soft correlation control (source_aware=True):
      - when non-empty source/date metadata provides >=K distinct values, do not select
        two recordings from the same source/date in one few-recording draw.
      - blank source metadata (e.g. flattened Cargo filenames) falls back to normal
        component-aware sampling.

    Same-day recordings remain distinct recording units; date is never used as a
    hard identity merge.
    """
    rng = np.random.default_rng(int(seed))
    by = defaultdict(list)
    for r in col.recordings:
        by[int(r.label)].append(r)

    selected = []
    for y in sorted(by):
        by_comp = defaultdict(list)
        for r in by[y]:
            by_comp[str(r.component_id or r.recording_id)].append(r)

        comps = list(by_comp)
        if len(comps) < int(shots):
            raise ValueError(
                f"class {y} has only {len(comps)} exact/near-duplicate components < shots={shots}"
            )

        # Choose one representative per component for draw construction.
        reps = []
        for comp in comps:
            rs = by_comp[comp]
            reps.append((comp, rs[int(rng.integers(len(rs)))]))

        if source_aware:
            nonempty_sources = {str(r.source) for _, r in reps if r.source not in (None, "", "None")}
        else:
            nonempty_sources = set()

        if source_aware and len(nonempty_sources) >= int(shots):
            # Random recording order + greedy unique-date/source selection.
            order = np.arange(len(reps))
            rng.shuffle(order)
            picked = []
            used_sources = set()
            for ii in order:
                comp, r = reps[int(ii)]
                src = str(r.source) if r.source not in (None, "", "None") else ""
                if src and src in used_sources:
                    continue
                picked.append((comp, r))
                if src:
                    used_sources.add(src)
                if len(picked) == int(shots):
                    break

            # Should normally be enough; deterministic fallback keeps component uniqueness.
            if len(picked) < int(shots):
                used_comp = {c for c, _ in picked}
                for ii in order:
                    comp, r = reps[int(ii)]
                    if comp in used_comp:
                        continue
                    picked.append((comp, r))
                    used_comp.add(comp)
                    if len(picked) == int(shots):
                        break
        else:
            idx = rng.choice(len(reps), int(shots), replace=False)
            picked = [reps[int(i)] for i in idx]

        selected.extend(r for _, r in picked)

    return selected
