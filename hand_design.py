"""
Hand design from ideal (dispersion-free) band positions, encoded as an optimize.py start vector.

Each column's whole outflow goes to the product outlet of its position (fully decoupled,
"sequential" operation). Eluent volume per switch at every position is chosen so that, with no
dispersion, the band of the component to be removed leaves the column completely while the
next-stronger one stays inside:
    retention volume  V_R,i = V_col * (eps + (1 - eps) * H_i)
"""
import json
from pathlib import Path

import numpy as np

import optimize as o
import smb4

ALPHA = 0.3                                   # feed sub-step share of the switch
ts, eps, V = smb4.SWITCH_TIME, smb4.TOTAL_POROSITY, smb4.COL_VOLUME
D = [V * (eps + (1 - eps) * h) for h in smb4.HENRY]
Vf = o.F_AVG * ts                             # feed volume per switch

a = D[0] + 0.6e-5                             # P3: flush R out after feeding, keep T1
b = (D[1] + D[2] - 2 * a - Vf) / 2            # P2: elute T1, keep T2 (equal margins)
c = (D[2] + D[3] - 2 * (a + b) - Vf) / 2      # P1: elute T2, keep T3
d = D[3] - (a + b + c) + 3e-5                 # P0: elute T3 (+ margin for dispersion)


def to_z(e):
    return (np.asarray(e) - o.E_MIN) / (o.E_MAX - o.E_MIN)


# frac encoding: (0.05 + 0.95 zA) / (0.05 + 0.95 zA + 1) = ALPHA with zB = 1
za = (ALPHA / (1 - ALPHA) - 0.05) / 0.95
z = np.array([
    [za, 1.0, *to_z([d / ts, c / ts, b / ts, o.E_MIN]), 1, 1, 1, 1],
    [1.0, 0.0, *to_z([d / ts, c / ts, b / ts, a / ((1 - ALPHA) * ts)]), 1, 1, 1, 1],
]).ravel()

if __name__ == "__main__":
    print("eluent volume per switch at P3, P2, P1, P0 [mL]:", np.round(np.array([a, b, c, d]) * 1e6, 1))
    res = o.evaluate((z, 40, 8, "purity", False))
    print("purity:", np.round(res["purity"], 4))
    Path("results").mkdir(exist_ok=True)
    Path("results/hand_design.json").write_text(json.dumps(
        {**res, "ncol": 40, "cycles": 8, "goal": "purity", "free_feed": False}, indent=2))
