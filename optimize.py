"""
Optimize the per-switch connection matrices of the 4-outlet SMB (smb4.py) with CMA-ES.

Decision vector z in [0, 1]^n; per sub-step k (K sub-steps per switch period):
    frac   duration share of the switch period
    feed   feed share (time-averaged feed flow is F_AVG, or the last entry of z with --free-feed)
    e0..e3 eluent flow into each position
    w0..w3 fraction of each position's outflow sent to its product outlet (rest -> next column)
Columns, isotherm, switch time and feed concentration are fixed.

Goals:  purity      maximise the worst outlet purity
        throughput  maximise feed flow subject to every purity >= PURITY_TARGET

Usage:
    .venv/bin/python optimize.py --goal throughput --free-feed --init results/hand_design.json \
        --sigma 0.02 --iters 80 --out results/B_throughput_hand_seed.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import cma
import numpy as np

import smb4
from smb4 import Schedule, SubStep

K = 2                    # sub-steps per switch period
F_AVG = 1.0e-7           # default time-averaged feed flow [m^3/s] (tutorial: 0.98e-7)
F_RANGE = (0.5e-7, 4e-7) # feed range when it is a decision variable
E_MAX = 1.5e-6           # max eluent flow into one position [m^3/s]
E_MIN = 2e-8             # keeps every column flowing
PURITY_TARGET = 0.95
VARS = ["frac", "feed", "e0", "e1", "e2", "e3", "w0", "w1", "w2", "w3"]


def deadzone(z, dz=0.1):
    """[0,1] -> [0,1] with a flat zero below dz, so a stream can be switched fully off."""
    return np.clip((z - dz) / (1 - dz), 0.0, 1.0)


def twosided(z, dz=0.1):
    """[0,1] -> [0,1] with flat 0 below dz and flat 1 above 1-dz (fully closed / fully open)."""
    return np.clip((z - dz) / (1 - 2 * dz), 0.0, 1.0)


def decode(z, free_feed=False):
    z = np.clip(np.asarray(z, float), 0.0, 1.0)
    f_avg = F_AVG
    if free_feed:
        f_avg = F_RANGE[0] + (F_RANGE[1] - F_RANGE[0]) * z[-1]
        z = z[:-1]
    z = z.reshape(K, len(VARS))
    frac = 0.05 + 0.95 * z[:, 0]
    frac = frac / frac.sum()
    share = deadzone(z[:, 1])
    if np.sum(frac * share) <= 1e-12:
        share = np.ones(K)
    feed = f_avg * share / np.sum(frac * share)
    return Schedule([SubStep(frac=float(frac[k]), e=E_MIN + (E_MAX - E_MIN) * z[k, 2:6],
                             w=twosided(z[k, 6:10]), F=float(feed[k])) for k in range(K)])


def objective(res, f_avg, goal):
    p = res["purity"]
    if goal == "purity":
        j = -float(np.min(p)) - 0.25 * float(np.mean(p))
    else:   # small safety margin above the target
        j = -f_avg / 1e-7 + 200.0 * float(np.sum(np.maximum(0.0, PURITY_TARGET + 0.003 - p)))
    if res["css_delta"] > 2e-3:   # not at cyclic steady state yet
        j += res["css_delta"]
    return j


def evaluate(args):
    z, ncol, cycles, goal, free_feed = args
    t = time.time()
    try:
        sch = decode(z, free_feed)
        res = smb4.simulate(sch, n_cycles=cycles, n_col=ncol)
        f_avg = sum(s.F * s.frac for s in sch.substeps)
        return dict(z=list(map(float, z)), J=objective(res, f_avg, goal), purity=res["purity"].tolist(),
                    feed=f_avg, recovery=res["recovery"].tolist(), css=res["css_delta"],
                    dt=time.time() - t)
    except Exception as e:  # solver failure -> bad score
        return dict(z=list(map(float, z)), J=1.0, error=str(e)[:200], dt=time.time() - t)


def naive_start(free_feed=False):
    """Half of every column's outflow withdrawn, feed in both sub-steps, equal eluent everywhere."""
    z = np.tile([0.5, 1.0, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5], K)
    return np.append(z, (F_AVG - F_RANGE[0]) / (F_RANGE[1] - F_RANGE[0])) if free_feed else z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncol", type=int, default=40)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--popsize", type=int, default=18)
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--init", type=str, default=None, help="result json to start from (default: naive)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--goal", choices=["purity", "throughput"], default="purity")
    ap.add_argument("--free-feed", action="store_true", help="make feed throughput a variable")
    a = ap.parse_args()
    ff = a.free_feed

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".log.jsonl")

    if a.init:
        z0 = np.array(json.loads(Path(a.init).read_text())["z"])
        if ff and len(z0) == K * len(VARS):
            z0 = np.append(z0, (F_AVG - F_RANGE[0]) / (F_RANGE[1] - F_RANGE[0]))
    else:
        z0 = naive_start(ff)
    meta = {"ncol": a.ncol, "cycles": a.cycles, "goal": a.goal, "free_feed": ff}
    es = cma.CMAEvolutionStrategy(z0, a.sigma, {"bounds": [0, 1], "popsize": a.popsize,
                                               "seed": a.seed, "verbose": -9})
    with mp.Pool(a.workers) as pool:
        best = evaluate((z0, a.ncol, a.cycles, a.goal, ff))
        print(f"start J={best['J']:.4f} purity={np.round(best.get('purity', []), 4)}", flush=True)
        for it in range(a.iters):
            X = es.ask()
            results = pool.map(evaluate, [(x, a.ncol, a.cycles, a.goal, ff) for x in X])
            es.tell(X, [r["J"] for r in results])
            with log.open("a") as fh:
                for r in results:
                    fh.write(json.dumps({"iter": it, **r}) + "\n")
            gen_best = min(results, key=lambda r: r["J"])
            if gen_best["J"] < best["J"]:
                best = gen_best
                out.write_text(json.dumps({**best, **meta}, indent=2))
            print(f"it {it:3d}  gen J={gen_best['J']:.4f}  best J={best['J']:.4f}  "
                  f"best purity={np.round(best.get('purity', []), 4)}  feed={best.get('feed', 0):.2e}  "
                  f"sigma={es.sigma:.3f}  t/eval={np.mean([r['dt'] for r in results]):.1f}s", flush=True)
            if es.stop():
                print("cma stop:", es.stop())
                break
    out.write_text(json.dumps({**best, **meta}, indent=2))


if __name__ == "__main__":
    main()
