"""
Re-simulate an optimized schedule at full fidelity, plot it and dump the connection matrices.

Usage:
    .venv/bin/python report.py results/stage2.json --ncol 40 --cycles 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import smb4
from optimize import decode
from smb4 import COMP_NAMES, HENRY, LIN_KA, LIN_KD, OUT, SWITCH_TIME, connections_for

COLORS = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"]   # R, T1, T2, T3
UNIT_NAMES = {0: "FEED", 1: "ELUENT", 2: "OUT_R", 3: "OUT_T1", 4: "OUT_T2", 5: "OUT_T3",
              6: "COL_6", 7: "COL_7", 8: "COL_8", 9: "COL_9"}


def connection_text(sch):
    lines = ["# Connection matrices for one cycle (4 switches x %d sub-steps)." % len(sch.substeps),
             "# rows: [from, to, port_from, port_to, comp_from, comp_to, flow m^3/s]",
             "# units: 0 FEED, 1 ELUENT, 2 OUT_R, 3 OUT_T1, 4 OUT_T2, 5 OUT_T3, 6-9 columns",
             "# The pattern repeats every cycle (4 * %.0f s)." % SWITCH_TIME, ""]
    t = 0.0
    for sw in range(4):
        for k, s in enumerate(sch.substeps):
            dur = s.frac * SWITCH_TIME
            lines.append(f"## switch {sw}, sub-step {k}:  t = {t:7.2f} .. {t + dur:7.2f} s")
            rows = np.array(connections_for(s, sw)).reshape(-1, 7)
            for r in rows:
                lines.append(f"    {int(r[0])}, {int(r[1])}, -1, -1, -1, -1, {r[6]:.4e},"
                             f"   # {UNIT_NAMES[int(r[0])]} -> {UNIT_NAMES[int(r[1])]}")
            t += dur
    return "\n".join(lines)


def substep_table(sch):
    hdr = (f"{'step':>4} {'dur[s]':>7} {'feed':>9} {'eluent':>9} {'->T3':>9} {'->T2':>9} "
           f"{'->T1':>9} {'->R':>9} | column flow P0..P3 [m^3/s]")
    out = [hdr]
    for k, s in enumerate(sch.substeps):
        qs, _ = s.column_flows()
        o = s.outlet_flows()
        qtxt = " ".join(f"{q:9.2e}" for q in qs)
        out.append(f"{k:>4} {s.frac * SWITCH_TIME:7.1f} {s.F:9.2e} {s.E:9.2e} {o['T3']:9.2e} "
                   f"{o['T2']:9.2e} {o['T1']:9.2e} {o['R']:9.2e} | {qtxt}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--ncol", type=int, default=40)
    ap.add_argument("--cycles", type=int, default=12)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    best = json.loads(Path(a.result).read_text())
    sch = decode(best["z"], best.get("layout", "series"), best.get("free_feed", False))
    res = smb4.simulate(sch, n_cycles=a.cycles, n_col=a.ncol, keep=True)
    sim, meta = res["sim"], res["meta"]
    outdir = Path(a.outdir)
    stem = Path(a.result).stem

    M = res["masses"]
    comp = M / M.sum(axis=1, keepdims=True)
    summary = {
        "isotherm": {"lin_ka": LIN_KA, "lin_kd": LIN_KD, "henry": HENRY},
        "ncol": a.ncol, "cycles": a.cycles,
        "purity": dict(zip(["OUT_R", "OUT_T1", "OUT_T2", "OUT_T3"], res["purity"].round(4).tolist())),
        "recovery": dict(zip(COMP_NAMES, res["recovery"].round(4).tolist())),
        "mass_balance_closure": res["closure"].round(4).tolist(),
        "css_delta_last_two_cycles": res["css_delta"],
        "outlet_composition": comp.round(4).tolist(),
    }
    print(json.dumps(summary, indent=2))
    print(substep_table(sch))
    (outdir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    (outdir / f"{stem}_connections.txt").write_text(
        substep_table(sch) + "\n\n" + connection_text(sch) + "\n")

    # ---- plots ---------------------------------------------------------
    t = np.asarray(sim.root.output.solution.solution_times)
    t_end = meta["section_times"][-1]
    win = t >= t_end - 2 * 4 * SWITCH_TIME
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, (name, unit) in zip(axes.ravel(), OUT.items()):
        c = np.asarray(sim.root.output.solution[f"unit_{unit:03d}"].solution_outlet)
        for i in range(4):
            ax.plot(t[win] / 60, c[win, i], color=COLORS[i], lw=1.3, label=COMP_NAMES[i])
        j = list(OUT).index(name)
        ax.set_title(f"OUT_{name}  purity {res['purity'][j] * 100:.1f} %")
        ax.set_ylabel("Concentration (mM)")
        for sb in meta["section_times"]:
            if t_end - 2 * 4 * SWITCH_TIME <= sb <= t_end:
                ax.axvline(sb / 60, color="k", lw=0.3, alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("Time (min)")
    axes[0, 0].legend(ncol=4, fontsize=8)
    fig.suptitle("Outlet concentrations, last two cycles at cyclic steady state "
                 "(grey lines = sub-step / switch boundaries)")
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_outlets.png", dpi=130)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    left = np.zeros(4)
    for i in range(4):
        ax.barh(["OUT_R", "OUT_T1", "OUT_T2", "OUT_T3"], comp[:, i] * 100, left=left,
                color=COLORS[i], label=COMP_NAMES[i])
        left += comp[:, i] * 100
    ax.axvline(95, color="k", ls="--", lw=1)
    ax.set_xlabel("Composition of outlet stream (% of solute mass, one cycle)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_composition.png", dpi=130)
    Path(res["file"]).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
