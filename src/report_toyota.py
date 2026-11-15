"""Tables for the Toyota benchmark (results/toyota_stgcn_*.json) and NTU->Toyota transfer.

    python -m src.report_toyota      -> prints Markdown, writes results/toyota_table.tex
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUGS = ["none", "random", "lowerbody", "geometric"]
AUG = {"none": "--", "random": "rand. joint", "lowerbody": "lower-body (prop.)", "geometric": "geometric (ours)"}
COLS = ["clean", "furniture", "self_occ", "one_arm", "temporal_block", "random_30"]
SHORT = {"clean": "Real test", "furniture": "+Furn.", "self_occ": "+Self", "one_arm": "+Arm",
         "temporal_block": "+Temp.", "random_30": "+Rand."}


def main():
    rows = []
    for aug in AUGS:
        f = ROOT / f"results/toyota_stgcn_{aug}.json"
        if f.exists():
            rows.append((aug, json.loads(f.read_text())))
    if rows:
        print("Toyota Smarthome cross-subject, ST-GCN (top-1 / mean-class %)")
        print("| Train aug | " + " | ".join(COLS) + " |")
        print("|---|" + "---|" * len(COLS))
        for aug, r in rows:
            print(f"| {aug} | " + " | ".join(f"{100 * r['tests'][c]['acc']:.1f} / {100 * r['tests'][c]['mean_class_acc']:.1f}"
                                         for c in COLS) + " |")
        best = {c: max(r["tests"][c]["mean_class_acc"] for _, r in rows) for c in COLS}
        L = [r"\begin{tabular}{@{}lcc" + "c" * (len(COLS) - 1) + r"@{}}", r"\toprule",
             r"Train aug. & \multicolumn{2}{c}{Real test (no sim.)} & " + " & ".join(SHORT[c] for c in COLS[1:]) + r" \\",
             r" & Top-1 & Mean cls. & \multicolumn{" + str(len(COLS) - 1) + r"}{c}{Mean class acc.} \\", r"\midrule"]
        for aug, r in rows:
            cells = [f"{100 * r['tests']['clean']['acc']:.1f}"]
            for c in COLS:
                v = r["tests"][c]["mean_class_acc"]
                cells.append((r"\textbf{%.1f}" if abs(v - best[c]) < 1e-9 else "%.1f") % (100 * v))
            L.append(f"{AUG[aug]} & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (ROOT / "results/toyota_table.tex").write_text("\n".join(L) + "\n")
        print("wrote results/toyota_table.tex")
    t = ROOT / "results/transfer.json"
    if t.exists():
        r = json.loads(t.read_text())
        print(f"\nNTU(13 joints) -> Toyota zero-shot, {r['n']} clips, classes: {', '.join(r['classes'])}")
        for aug, m in r["models"].items():
            print(f"  {aug:10s} top-1 {100 * m['acc']:5.1f}  mean-class {100 * m['mean_class_acc']:5.1f}")


if __name__ == "__main__":
    main()
