"""
Hand design from ideal (dispersion-free) band positions, encoded in optimize.py's "general" layout.

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


za = 0.3 / 0.7 - 0.05                          # frac encoding: (0.05+0.95 zA)/(0.05+0.95 zA + 1) = 0.3
z = np.zeros((2, len(o.VARS2)))
z[0] = [za / 0.95, 1.0, *to_z([d / ts, c / ts, b / ts, o.E_MIN]), 1, 1, 1, 1]
z[1] = [1.0, 0.0, *to_z([d / ts, c / ts, b / ts, a / ((1 - ALPHA) * ts)]), 1, 1, 1, 1]

if __name__ == "__main__":
    print("volumes per switch a,b,c,d [mL]:", np.round(np.array([a, b, c, d]) * 1e6, 1))
    sch = o.decode_general(z.ravel())
    print([round(s.frac, 4) for s in sch.substeps], [s.F for s in sch.substeps])
    res = o.evaluate((z.ravel(), 40, 8, "general", "purity", False))
    print(res["purity"])
    Path("results").mkdir(exist_ok=True)
    Path("results/hand_design.json").write_text(json.dumps(
        {**res, "ncol": 40, "cycles": 8, "layout": "general", "goal": "purity",
         "free_feed": False}, indent=2))
