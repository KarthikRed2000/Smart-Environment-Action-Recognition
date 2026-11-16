"""Tables for the Toyota benchmark (results/toyota_stgcn_<aug>[_sK].json) and NTU->Toyota transfer.

    python -m src.report_toyota      -> prints Markdown, writes results/toyota_table.tex
"""
import json
from pathlib import Path

from . import seeds as S

ROOT = Path(__file__).resolve().parents[1]
AUGS = ["none", "random", "lowerbody", "geometric", "combined"]
AUG = {"none": "--", "random": "rand. joint", "lowerbody": "lower-body (prop.)",
       "geometric": "geometric (ours)", "combined": "combined (ours)"}
COLS = ["clean", "furniture", "self_occ", "one_arm", "temporal_block", "random_30"]
SHORT = {"clean": "Real test", "furniture": "+Furn.", "self_occ": "+Self", "one_arm": "+Arm$^\\dagger$",
         "temporal_block": "+Temp.$^\\dagger$", "random_30": "+Rand."}


def main():
    rows = [(aug, S.load(f"toyota_stgcn_{aug}")) for aug in AUGS]
    rows = [(a, rs) for a, rs in rows if rs]
    get = lambda rs, c, k: [r["tests"][c][k] for r in rs]
    plain = lambda v: S.fmt(v).replace(S.PM, "±")
    if rows:
        print("Toyota Smarthome cross-subject, ST-GCN (top-1 / mean-class %, mean over seeds)")
        print("| Train aug | n | " + " | ".join(COLS) + " |")
        print("|---|---|" + "---|" * len(COLS))
        for aug, rs in rows:
            print(f"| {aug} | {len(rs)} | " + " | ".join(
                f"{S.fmt(get(rs, c, 'acc'), std=False)} / {plain(get(rs, c, 'mean_class_acc'))}"
                for c in COLS) + " |")
        best = {c: max(S.ms(get(rs, c, "mean_class_acc"))[0] for _, rs in rows) for c in COLS}
        best_top1 = max(S.ms(get(rs, "clean", "acc"))[0] for _, rs in rows)
        b = lambda s, hit: r"\textbf{" + s + "}" if hit else s
        L = [r"\begin{tabular}{@{}l" + "c" * (len(COLS) + 1) + r"@{}}", r"\toprule",
             r"Train aug. & \multicolumn{2}{c}{Real test (no sim.)} & " + " & ".join(SHORT[c] for c in COLS[1:]) + r" \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-" + str(len(COLS) + 2) + "}",
             r" & Top-1 & Mean cls. & \multicolumn{" + str(len(COLS) - 1) + r"}{c}{Mean class acc.} \\", r"\midrule"]
        for aug, rs in rows:
            t1 = get(rs, "clean", "acc")
            cells = [b(S.fmt(t1), abs(S.ms(t1)[0] - best_top1) < 1e-9)]
            for c in COLS:
                v = get(rs, c, "mean_class_acc")
                cells.append(b(S.fmt(v), abs(S.ms(v)[0] - best[c]) < 1e-9))
            L.append(f"{AUG[aug]} & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (ROOT / "results/toyota_table.tex").write_text("\n".join(L) + "\n")
        print("wrote results/toyota_table.tex")
    t = ROOT / "results/transfer.json"
    if t.exists():
        r = json.loads(t.read_text())
        if "n" in r:
            print(f"\nNTU(13 joints) -> Toyota zero-shot, {r['n']} clips, classes: {', '.join(r['classes'])}")
            for proto in ("raw", "harmonized"):
                for aug, m in r.get(proto, {}).items():
                    print(f"  [{proto:10s}] {aug:10s} ({len(m['runs'])} seeds) top-1 {100 * m['acc']:5.1f}±{100 * m['acc_std']:.1f}"
                          f"  mean-class {100 * m['mean_class_acc']:5.1f}±{100 * m['mean_class_acc_std']:.1f}")


if __name__ == "__main__":
    main()
