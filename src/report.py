"""Collect results/*.json into a Markdown summary and a LaTeX table for the report.

    python -m src.report     -> prints Markdown, writes results/table.tex
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLS = ["clean", "lowerbody", "furniture", "furniture_noisy", "self_occ", "one_arm", "temporal_block", "random_30"]
SHORT = {"clean": "Clean", "lowerbody": "Lower", "furniture": "Furn.", "furniture_noisy": "Furn.+N",
         "self_occ": "Self", "one_arm": "Arm$^\\dagger$", "temporal_block": "Temp.$^\\dagger$",
         "random_30": "Rand.$^\\ddagger$"}
ORDER = [("bilstm", "none"), ("bilstm", "geometric"), ("cnn1d", "none"), ("cnn1d", "geometric"),
         ("stgcn", "none"), ("stgcn", "random"), ("stgcn", "lowerbody"), ("stgcn", "geometric")]
NAME = {"bilstm": "Bi-LSTM", "cnn1d": "1D-CNN", "stgcn": "ST-GCN"}
AUG = {"none": "--", "random": "rand. joint", "lowerbody": "lower-body", "geometric": "geometric (ours)"}


def main():
    res = {}
    for f in (ROOT / "results").glob("*.json"):
        r = json.loads(f.read_text())
        if "tests" not in r or r.get("dataset", "ntu") != "ntu":
            continue                              # Toyota / NTU-13 results have their own table
        res[(r["model"], r["aug"])] = r
    if not res:
        print("no results yet")
        return
    rows = [k for k in ORDER if k in res] + [k for k in res if k not in ORDER]

    print("| Model | Aug | " + " | ".join(COLS) + " | params | ms |")
    print("|" + "---|" * (len(COLS) + 4))
    for k in rows:
        r = res[k]
        print(f"| {NAME.get(k[0], k[0])} | {k[1]} | "
              + " | ".join(f"{100 * r['tests'][c]['acc']:.1f}" for c in COLS)
              + f" | {r['params'] / 1e6:.2f}M | {r['latency_ms_bs1']:.1f} |")
    any_r = next(iter(res.values()))
    print("\nHidden joints per protocol: " + ", ".join(f"{c} {100 * any_r['hidden_frac'][c]:.0f}%" for c in COLS))

    # best per column (bold)
    best = {c: max(res[k]["tests"][c]["acc"] for k in rows) for c in COLS}
    lines = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}ll" + "c" * len(COLS) + r"@{}}", r"\toprule",
             "Model & Train aug. & " + " & ".join(SHORT[c] for c in COLS) + r" \\", r"\midrule"]
    for k in rows:
        r = res[k]
        cells = []
        for c in COLS:
            v = r["tests"][c]["acc"]
            s = f"{100 * v:.1f}"
            cells.append(r"\textbf{" + s + "}" if abs(v - best[c]) < 1e-9 else s)
        lines.append(f"{NAME.get(k[0], k[0])} & {AUG.get(k[1], k[1])} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Top-1 accuracy (\%) on the 22-class NTU-60 domestic subset (cross-subject). "
              r"$^\dagger$Never seen in training by any variant. $^\ddagger$Seen only by the random-joint variant.}",
              r"\label{tab:results}", r"\end{table*}"]
    (ROOT / "results/table.tex").write_text("\n".join(lines) + "\n")
    print("\nwrote results/table.tex")


if __name__ == "__main__":
    main()
