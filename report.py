"""
Re-simulate an optimized cascade at full fidelity, plot it and dump the connection matrices.

Usage:
    .venv/bin/python report.py results/cascade_opt.json --ncol 40 --cycles 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import smb_cascade as sc
from optimize import eluent_ratio

COLORS = ["#7f7f7f", "#1f77b4", "#ff7f0e"]   # R, T1, T2
OUT_NAMES = [f"OUT_{n}" for n in sc.OUT]
UNIT_NAMES = {sc.FEED: "FEED", sc.ELUENT: "ELUENT", **{u: f"OUT_{n}" for n, u in sc.OUT.items()},
              **{c0 + k: f"U{i + 1}_COL_{c0 + k}" for i, c0 in enumerate(sc.COL0) for k in range(sc.N_POS)}}


def design_table(d: sc.Design):
    geo = d.geometry()
    lines = [f"switch time {d.switch_time:.0f} s (both units)", ""]
    for unit in (1, 2):
        g = geo[f"unit{unit}"]
        m = d.m1 if unit == 1 else d.m2
        lines += [f"unit {unit}: column L = {g['length'] * 100:.2f} cm, D = {g['diameter'] * 100:.2f} cm, "
                  f"V = {g['volume'] * 1e6:.1f} mL",
                  "  m (zone I..IV):  " + "  ".join(f"{x:9.3f}" for x in m),
                  "  Q zone [m^3/s]:  " + "  ".join(f"{x:9.3e}" for x in d.zone_flows(unit)),
                  "  ports [m^3/s]:   " + "  ".join(f"{k} {v:9.3e}" for k, v in d.port_flows(unit).items()),
                  ""]
    return "\n".join(lines)


def connection_text(d: sc.Design):
    lines = ["# Connection matrices for one cycle (4 switches); the pattern repeats every cycle.",
             "# rows: [from, to, port_from, port_to, comp_from, comp_to, flow m^3/s]", ""]
    for sw in range(sc.N_POS):
        lines.append(f"## switch {sw}:  t = {sw * d.switch_time:.0f} .. {(sw + 1) * d.switch_time:.0f} s")
        for r in np.array(sc.connections(d, sw)).reshape(-1, 7):
            lines.append(f"    {int(r[0])}, {int(r[1])}, -1, -1, -1, -1, {r[6]:.4e},"
                         f"   # {UNIT_NAMES[int(r[0])]} -> {UNIT_NAMES[int(r[1])]}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--ncol", type=int, default=40)
    ap.add_argument("--cycles", type=int, default=30)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()

    best = json.loads(Path(a.result).read_text())
    d = sc.Design(m1=np.array(best["m1"]), m2=np.array(best["m2"]))
    res = sc.simulate(d, n_cycles=a.cycles, n_col=a.ncol, keep=True)
    sim, meta = res["sim"], res["meta"]
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(a.result).stem

    M = res["masses"]
    comp = M / M.sum(axis=1, keepdims=True)
    summary = {
        "isotherm": {"lin_ka": sc.LIN_KA, "lin_kd": sc.LIN_KD, "henry": sc.HENRY},
        "ncol": a.ncol, "cycles": a.cycles,
        "purity": dict(zip(OUT_NAMES, res["purity"].round(4).tolist())),
        "recovery": dict(zip(sc.COMP_NAMES, res["recovery"].round(4).tolist())),
        "mass_balance_closure": res["closure"].round(4).tolist(),
        "css_delta_last_two_cycles": res["css_delta"],
        "feed_flow_m3_s": d.feed_flow,
        "eluent_to_feed_ratio": eluent_ratio(d),
        "m_unit1": d.m1.tolist(), "m_unit2": d.m2.tolist(),
        "geometry": d.geometry(),
        "outlet_composition": comp.round(4).tolist(),
    }
    print(json.dumps(summary, indent=2))
    print(design_table(d))
    (outdir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2))
    (outdir / f"{stem}_connections.txt").write_text(design_table(d) + "\n" + connection_text(d) + "\n")

    # ---- plots ---------------------------------------------------------
    t = np.asarray(sim.root.output.solution.solution_times)
    t_end = meta["section_times"][-1]
    t_win = t_end - 2 * sc.N_POS * d.switch_time
    win = t >= t_win
    sources = {"R": "unit 1 raffinate", "T1": "unit 2 raffinate", "T2": "unit 2 extract"}
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for j, (ax, (name, unit)) in enumerate(zip(axes, sc.OUT.items())):
        c = np.asarray(sim.root.output.solution[f"unit_{unit:03d}"].solution_outlet)
        for i in range(sc.N_COMP):
            ax.plot(t[win] / 60, c[win, i], color=COLORS[i], lw=1.3, label=sc.COMP_NAMES[i])
        ax.set_title(f"OUT_{name} ({sources[name]})  purity {res['purity'][j] * 100:.1f} %")
        ax.set_ylabel("Concentration (mM)")
        for sb in meta["section_times"]:
            if sb >= t_win:
                ax.axvline(sb / 60, color="k", lw=0.3, alpha=0.25)
    axes[-1].set_xlabel("Time (min)")
    axes[0].legend(ncol=3, fontsize=8)
    fig.suptitle("Outlet concentrations, last two cycles at cyclic steady state (grey lines = switches)")
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_outlets.png", dpi=130)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    left = np.zeros(len(sc.OUT))
    for i in range(sc.N_COMP):
        ax.barh(OUT_NAMES, comp[:, i] * 100, left=left, color=COLORS[i], label=sc.COMP_NAMES[i])
        left += comp[:, i] * 100
    ax.axvline(95, color="k", ls="--", lw=1)
    ax.set_xlabel("Composition of outlet stream (% of solute mass, one cycle)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(outdir / f"{stem}_composition.png", dpi=130)


if __name__ == "__main__":
    main()
