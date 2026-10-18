from __future__ import annotations
import argparse,json
from pathlib import Path

DISPLAY={
    "S0":"Segment ERM",
    "S1":"Recording-balanced",
    "PB":"Equal-recording Proto",
    "PT":"Low-rank ProtoTune",
    "F0":"DGE Fusion (Proposed)",
}

def fmt(mean,std,scale=100.0):
    if mean is None: return "--"
    return f"{scale*mean:.2f}±{scale*(std or 0):.2f}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",default="runs/final_test")
    ap.add_argument("--out-dir",default="paper_outputs")
    a=ap.parse_args()
    rd=Path(a.run_dir); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((rd/"summary.json").read_text())
    paired=json.loads((rd/"hierarchical_paired_bootstrap.json").read_text())

    settings=[]
    for r in summary:
        if r["setting"] not in settings: settings.append(r["setting"])
    methods=["S0","S1","PB","PT","F0"]

    lookup={(r["setting"],r["method"]):r for r in summary}
    lines=["# Final ShipsEar Results","","## Macro-F1 (%)",""]
    header="| Method | "+" | ".join(settings)+" |"
    sep="|---|"+"|".join(["---:"]*len(settings))+"|"
    lines += [header,sep]
    for m in methods:
        vals=[]
        for s in settings:
            r=lookup[(s,m)]
            vals.append(fmt(r.get("macro_f1_mean"),r.get("macro_f1_std")))
        lines.append("| "+DISPLAY[m]+" | "+" | ".join(vals)+" |")

    lines += ["","## Balanced Accuracy (%)","",header,sep]
    for m in methods:
        vals=[]
        for s in settings:
            r=lookup[(s,m)]
            vals.append(fmt(r.get("balanced_accuracy_mean"),r.get("balanced_accuracy_std")))
        lines.append("| "+DISPLAY[m]+" | "+" | ".join(vals)+" |")

    lines += ["","## Proposed-vs-baseline hierarchical bootstrap","",
              "| Setting | Comparison | Metric | Δ mean (pp) | 95% CI (pp) | P(Δ≤0) |",
              "|---|---|---|---:|---:|---:|"]
    for r in paired:
        lines.append(
            f"| {r['setting']} | {r['comparison']} | {r['metric']} | "
            f"{100*r['mean_delta']:.2f} | "
            f"[{100*r['low95']:.2f}, {100*r['high95']:.2f}] | "
            f"{r['p_delta_le_0']:.4f} |"
        )
    (out/"FINAL_RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    # LaTeX table, focused on the 2/3/4-rec main protocol.
    main_settings=[s for s in settings if s.endswith("-rec")]
    tex=[
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{ShipsEar few-recording results (mean$\pm$std over support draws, \%).}",
        r"\begin{tabular}{l" + "c"*len(main_settings) + "}",
        r"\hline",
        "Method & " + " & ".join(s.replace("-rec","-rec/class") for s in main_settings) + r" \\",
        r"\hline",
    ]
    for m in methods:
        vals=[]
        for s in main_settings:
            r=lookup[(s,m)]
            vals.append(fmt(r.get("macro_f1_mean"),r.get("macro_f1_std")).replace("±",r"$\pm$"))
        name=DISPLAY[m].replace("_",r"\_")
        tex.append(name+" & "+" & ".join(vals)+r" \\")
    tex += [r"\hline",r"\end{tabular}",r"\label{tab:shipsear_final}",r"\end{table}"]
    (out/"FINAL_RESULTS.tex").write_text("\n".join(tex)+"\n",encoding="utf-8")

    print(out/"FINAL_RESULTS.md")
    print(out/"FINAL_RESULTS.tex")

if __name__=="__main__":
    main()
