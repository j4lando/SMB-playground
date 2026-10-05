"""
Optimize the two-unit SMB cascade (smb_cascade.py) with CMA-ES.

Decision vector z in [0, 1]^8 -> zone flow-rate ratios of both units:
    unit 1: m_I, m_II, m_III - m_II, m_IV        unit 2: m_I, m_II, m_III - m_II, m_IV
Unit-1 column dimensions follow from matching unit-1 extract flow to unit-2 feed flow at the
shared switch time; unit-2 columns, isotherm, switch time and feed concentration are fixed.

Objective: maximise fresh-feed throughput subject to every outlet purity >= PURITY_TARGET,
with a small penalty on eluent consumption (eluent / feed).

Usage:
    .venv/bin/python optimize.py --out results/cascade_opt.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import cma
import numpy as np

import smb_cascade as sc

PURITY_TARGET = 0.95
MARGIN = 0.003
F_REF = 1e-8              # throughput scale [m^3/s]
W_ELUENT = 0.02           # weight on eluent / feed ratio
# bounds of each decoded variable
BOUNDS = np.array([
    [6.0, 15.0],    # unit 1 m_I
    [0.0, 7.0],     # unit 1 m_II
    [0.0, 7.0],     # unit 1 m_III - m_II
    [-2.0, 3.0],    # unit 1 m_IV
    [6.0, 15.0],    # unit 2 m_I
    [4.0, 8.0],     # unit 2 m_II
    [0.0, 3.0],     # unit 2 m_III - m_II
    [2.0, 8.0],     # unit 2 m_IV
])


def decode(z):
    x = BOUNDS[:, 0] + np.clip(np.asarray(z, float), 0, 1) * (BOUNDS[:, 1] - BOUNDS[:, 0])
    m1 = np.array([x[0], x[1], x[1] + x[2], x[3]])
    m2 = np.array([x[4], x[5], x[5] + x[6], x[7]])
    return sc.Design(m1=m1, m2=m2)


def encode(m1, m2):
    x = np.array([m1[0], m1[1], m1[2] - m1[1], m1[3], m2[0], m2[1], m2[2] - m2[1], m2[3]])
    return (x - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])


def eluent_ratio(d: sc.Design):
    return (d.port_flows(1)["eluent"] + d.port_flows(2)["eluent"]) / d.feed_flow


def objective(res, d: sc.Design):
    short = np.maximum(0.0, PURITY_TARGET + MARGIN - res["purity"])
    j = -d.feed_flow / F_REF + W_ELUENT * eluent_ratio(d) + 1000.0 * float(np.sum(short))
    if res["css_delta"] > 2e-3:   # not at cyclic steady state yet
        j += res["css_delta"]
    return j


def evaluate(args):
    z, ncol, cycles = args
    t = time.time()
    d = decode(z)
    base = dict(z=list(map(float, z)), m1=d.m1.tolist(), m2=d.m2.tolist())
    if not d.is_feasible():   # some port or zone flow <= 0: skip the simulation
        return dict(**base, J=100.0, error="infeasible flows", dt=0.0)
    try:
        res = sc.simulate(d, n_cycles=cycles, n_col=ncol)
    except Exception as e:  # solver failure -> bad score
        return dict(**base, J=100.0, error=str(e)[:200], dt=time.time() - t)
    return dict(**base, J=objective(res, d), purity=res["purity"].tolist(),
                recovery=res["recovery"].tolist(), feed=d.feed_flow, eluent_ratio=eluent_ratio(d),
                v1_over_v2=d.v1 / sc.V2, css=res["css_delta"], dt=time.time() - t)


def triangle_start():
    """Conservative design from linear triangle theory (ideal, no dispersion)."""
    return encode([9.5, 1.5, 4.5, 0.15], [9.5, 6.3, 7.3, 5.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncol", type=int, default=40)
    ap.add_argument("--cycles", type=int, default=12)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--popsize", type=int, default=9)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".log.jsonl")
    meta = {"ncol": a.ncol, "cycles": a.cycles}

    z0 = triangle_start()
    es = cma.CMAEvolutionStrategy(z0, a.sigma, {"bounds": [0, 1], "popsize": a.popsize,
                                               "seed": a.seed, "verbose": -9})
    with mp.Pool(a.workers) as pool:
        best = evaluate((z0, a.ncol, a.cycles))
        print(f"start J={best['J']:.4f} purity={np.round(best['purity'], 4)} "
              f"feed={best['feed']:.3e}", flush=True)
        for it in range(a.iters):
            X = es.ask()
            results = pool.map(evaluate, [(x, a.ncol, a.cycles) for x in X])
            es.tell(X, [r["J"] for r in results])
            with log.open("a") as fh:
                for r in results:
                    fh.write(json.dumps({"iter": it, **r}) + "\n")
            gen_best = min(results, key=lambda r: r["J"])
            if gen_best["J"] < best["J"]:
                best = gen_best
                out.write_text(json.dumps({**best, **meta}, indent=2))
            print(f"it {it:3d}  gen J={gen_best['J']:.4f}  best J={best['J']:.4f}  "
                  f"purity={np.round(best['purity'], 4)}  feed={best['feed']:.3e}  "
                  f"E/F={best['eluent_ratio']:.1f}  V1/V2={best['v1_over_v2']:.3f}  "
                  f"sigma={es.sigma:.3f}", flush=True)
            if es.stop():
                print("cma stop:", es.stop())
                break
    out.write_text(json.dumps({**best, **meta}, indent=2))


if __name__ == "__main__":
    main()
