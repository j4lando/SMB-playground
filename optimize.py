"""
Optimize the per-switch connection matrices of the 4-outlet SMB (smb4.py) with CMA-ES.

Decision variables (per sub-step k of every switch period, K sub-steps):
    frac_k   duration share of the switch period
    m0_k     flow-rate ratio of the column at P0 (sets eluent flow, open loop)
    w3_k     fraction of P0 outflow withdrawn to OUT_T3
    w2_k     fraction of P1 outflow withdrawn to OUT_T2
    w1_k     fraction of P2 outflow withdrawn to OUT_T1
    s_k      feed share (time-averaged feed flow is fixed at F_AVG)
Everything else (columns, isotherm, switch time, feed concentration) is fixed.

Objective: maximise the worst outlet purity (target >= 95 %) at cyclic steady state.

Usage:
    .venv/bin/python optimize.py --ncol 20 --cycles 8 --iters 120 --out results/stage1.json
    .venv/bin/python optimize.py --ncol 40 --cycles 10 --iters 40 --init results/stage1.json \
        --sigma 0.05 --out results/stage2.json
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
from smb4 import Schedule, SubStep, SubStepV2, q_from_m

K = 3                    # sub-steps per switch period
F_AVG = 1.0e-7           # time-averaged feed flow [m^3/s] (tutorial: 0.98e-7)
M0_MAX = 25.0
W_MAX = 0.95
PURITY_TARGET = 0.95
VARS = ["frac", "m0", "w3", "w2", "w1", "feed"]
N_VAR = len(VARS) * K


def deadzone(z, dz=0.1):
    """Map [0,1] -> [0,1] with a flat zero for z < dz so a port can be fully closed."""
    return np.clip((z - dz) / (1 - dz), 0.0, 1.0)


# ---- layout "general": eluent make-up into every column, any split at every node -------------
K2 = 2                   # sub-steps per switch for the general layout
E_MAX = 1.5e-6           # max eluent flow into one position [m^3/s]
E_MIN = 2e-8             # keeps every column flowing
F_RANGE = (0.5e-7, 4e-7) # feed-throughput range when it is a decision variable
VARS2 = ["frac", "feed", "e0", "e1", "e2", "e3", "w0", "w1", "w2", "w3"]


def n_var(layout, free_feed=False):
    if layout == "series":
        return N_VAR
    return K2 * len(VARS2) + (1 if free_feed else 0)


def twosided(z, dz=0.1):
    """[0,1] -> [0,1] with flat 0 below dz and flat 1 above 1-dz (ports can be fully closed/open)."""
    return np.clip((z - dz) / (1 - 2 * dz), 0.0, 1.0)


def decode_general(z, free_feed=False):
    z = np.clip(np.asarray(z, float), 0.0, 1.0)
    f_avg = F_AVG
    if free_feed:
        f_avg = F_RANGE[0] + (F_RANGE[1] - F_RANGE[0]) * z[-1]
        z = z[:-1]
    z = z.reshape(K2, len(VARS2))
    frac = 0.05 + 0.95 * z[:, 0]
    frac = frac / frac.sum()
    share = deadzone(z[:, 1])
    if np.sum(frac * share) <= 1e-12:
        share = np.ones(K2)
    feed = f_avg * share / np.sum(frac * share)
    subs = []
    for k in range(K2):
        e = E_MIN + (E_MAX - E_MIN) * z[k, 2:6]
        w = twosided(z[k, 6:10])
        subs.append(SubStepV2(frac=float(frac[k]), e=e, w=w, F=float(feed[k])))
    return Schedule(subs)


def decode(z, layout="series", free_feed=False):
    if layout == "general":
        return decode_general(z, free_feed)
    z = np.clip(np.asarray(z, float), 0.0, 1.0).reshape(K, len(VARS))
    frac = 0.05 + 0.95 * z[:, 0]
    frac = frac / frac.sum()
    m0 = M0_MAX * z[:, 1]
    w = W_MAX * deadzone(z[:, 2:5])
    share = deadzone(z[:, 5])
    if np.sum(frac * share) <= 1e-12:
        share = np.ones(K)
    feed = F_AVG * share / np.sum(frac * share)
    subs = []
    for k in range(K):
        q0 = q_from_m(m0[k])
        x3 = w[k, 0] * q0
        q1 = q0 - x3
        x2 = w[k, 1] * q1
        q2 = q1 - x2
        x1 = w[k, 2] * q2
        subs.append(SubStep(frac=float(frac[k]), E=float(q0), F=float(feed[k]),
                            X3=float(x3), X2=float(x2), X1=float(x1), rec=0.0))
    return Schedule(subs)


def objective_from_result(res, sch, goal="purity"):
    p = res["purity"]
    if goal == "purity":
        # maximise the worst purity
        j = -float(np.min(p)) - 0.25 * float(np.mean(p))
    else:
        # maximise feed throughput subject to every purity >= target (+ small safety margin)
        f_avg = sum(s.F * s.frac for s in sch.substeps)
        short = np.maximum(0.0, PURITY_TARGET + 0.003 - p)
        j = -f_avg / 1e-7 + 200.0 * float(np.sum(short))
    if res["css_delta"] > 2e-3:
        j += res["css_delta"]
    return j


def evaluate(args):
    z, ncol, cycles, layout, goal, free_feed = args
    t = time.time()
    try:
        sch = decode(z, layout, free_feed)
        res = smb4.simulate(sch, n_cycles=cycles, n_col=ncol)
        j = objective_from_result(res, sch, goal)
        f_avg = sum(s.F * s.frac for s in sch.substeps)
        return dict(z=list(map(float, z)), J=j, purity=res["purity"].tolist(), feed=f_avg,
                    recovery=res["recovery"].tolist(), css=res["css_delta"], dt=time.time() - t)
    except Exception as e:  # solver failure -> bad score
        return dict(z=list(map(float, z)), J=1.0, error=str(e)[:200], dt=time.time() - t)


def initial_guess():
    """Hand-built start: [feed + flush in series] -> [withdraw all] -> [withdraw, no feed]."""
    z = np.zeros((K, len(VARS)))
    # frac, m0/M0_MAX, w3, w2, w1, feed  (w/feed in deadzone coords: 0.1 == closed)
    z[0] = [0.30, 4 / M0_MAX, 0.0, 0.0, 0.0, 1.0]
    z[1] = [0.40, 13 / M0_MAX, 0.55, 0.55, 0.55, 1.0]
    z[2] = [0.30, 13 / M0_MAX, 0.55, 0.55, 0.55, 0.0]
    z[:, 2:6] = np.where(z[:, 2:6] > 0, 0.1 + 0.9 * z[:, 2:6], 0.05)
    return z.ravel()


def initial_guess_general(free_feed=False):
    """Naive start: classic-looking ring, half the outflow of every column withdrawn,
    feed in both sub-steps, equal eluent everywhere. No hand tuning."""
    z = np.tile([0.5, 1.0, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.5], (K2, 1)).ravel()
    return np.append(z, 0.5 / 3.5) if free_feed else z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncol", type=int, default=20)
    ap.add_argument("--cycles", type=int, default=8)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--popsize", type=int, default=18)
    ap.add_argument("--sigma", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--init", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="results/stage1.json")
    ap.add_argument("--layout", choices=["series", "general"], default="series")
    ap.add_argument("--goal", choices=["purity", "throughput"], default="purity")
    ap.add_argument("--free-feed", action="store_true", help="make feed throughput a variable")
    a = ap.parse_args()
    ff = a.free_feed

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".log.jsonl")

    if a.init:
        z0 = np.array(json.loads(Path(a.init).read_text())["z"])
        if ff and len(z0) == n_var(a.layout, False):
            z0 = np.append(z0, (F_AVG - F_RANGE[0]) / (F_RANGE[1] - F_RANGE[0]))
    else:
        z0 = initial_guess_general(ff) if a.layout == "general" else initial_guess()
    meta = {"ncol": a.ncol, "cycles": a.cycles, "layout": a.layout, "goal": a.goal,
            "free_feed": ff}
    es = cma.CMAEvolutionStrategy(z0, a.sigma, {"bounds": [0, 1], "popsize": a.popsize,
                                               "seed": a.seed, "verbose": -9})
    best = None
    pool = mp.Pool(a.workers)
    try:
        start = evaluate((z0, a.ncol, a.cycles, a.layout, a.goal, ff))
        best = start
        print(f"start J={start['J']:.4f} purity={np.round(start.get('purity', []), 4)}", flush=True)
        for it in range(a.iters):
            X = es.ask()
            results = pool.map(evaluate, [(x, a.ncol, a.cycles, a.layout, a.goal, ff) for x in X])
            es.tell(X, [r["J"] for r in results])
            with log.open("a") as fh:
                for r in results:
                    fh.write(json.dumps({"iter": it, **r}) + "\n")
            gen_best = min(results, key=lambda r: r["J"])
            if gen_best["J"] < best["J"]:
                best = gen_best
                out.write_text(json.dumps({**best, **meta}, indent=2))
            print(f"it {it:3d}  gen J={gen_best['J']:.4f}  best J={best['J']:.4f}  "
                  f"best purity={np.round(best.get('purity', []), 4)}  feed={best.get('feed', 0):.2e}  sigma={es.sigma:.3f}  "
                  f"t/eval={np.mean([r['dt'] for r in results]):.1f}s", flush=True)
            if es.stop():
                print("cma stop:", es.stop())
                break
    finally:
        pool.close()
        pool.join()
    out.write_text(json.dumps({**best, **meta}, indent=2))
    print("best:", json.dumps(best["purity"]))


if __name__ == "__main__":
    main()
