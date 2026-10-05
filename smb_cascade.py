"""
Ternary separation in CADET with two directly coupled 4-zone SMB units (isocratic, linear isotherm).

Components: 0 = R (raffinate impurity, barely binds), 1 = T1, 2 = T2 (T1/T2 overlap: H differ by 1.7)

    FEED ─► unit 1 (R | T1+T2) ─ raffinate ─► OUT_R
                               └ extract ───► unit 2 feed port (T1 | T2) ─ raffinate ─► OUT_T1
                                                                         └ extract ───► OUT_T2
    ELUENT feeds zone I of both units.

Units:  0 FEED  1 ELUENT  2 OUT_R  3 OUT_T1  4 OUT_T2  5-8 unit-1 columns  9-12 unit-2 columns

Each unit is a classic closed-loop 4-zone SMB with one column per zone (as in the CADET tutorial).
Positions P0..P3 are zones I..IV in flow direction:
    ELUENT -> P0 -> [extract] -> P1 -> [+feed] -> P2 -> [raffinate] -> P3 -> P0
At every switch the ports advance one column (column at P1 moves to P0, P0 -> P3, ...).

Both units switch together with the same switch time. Unit 1's extract feeds unit 2 directly, so
    Q_extract,1 = Q_feed,2   <=>   (m_I1 - m_II1) V1 = (m_III2 - m_II2) V2
Unit 2 keeps the tutorial columns; unit 1's column volume follows from this constraint, at the
tutorial aspect ratio L/D.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from addict import Dict
from cadet import Cadet

CADET_ROOT = Path(__file__).resolve().parent / ".venv"

COMP_NAMES = ["R", "T1", "T2"]
N_COMP = 3

FEED, ELUENT = 0, 1
OUT = {"R": 2, "T1": 3, "T2": 4}                 # outlet unit per stream (comp j -> OUT index j)
COL0 = (5, 9)                                    # first column unit of unit 1 / unit 2
N_POS = 4

# ---- isotherm: arbitrary ka/kd with H = ka/kd -> R 0.3, T1 6.0, T2 7.7 ------------------------
LIN_KA = [0.6, 9.0, 9.24]
LIN_KD = [2.0, 1.5, 1.2]
HENRY = [a / d for a, d in zip(LIN_KA, LIN_KD)]

# ---- columns: tutorial geometry for unit 2; unit 1 is resized at the same aspect ratio --------
TOTAL_POROSITY = 0.83
COL_DISPERSION = 3.8148e-20
L2 = 0.25
D2 = 0.02                                        # cross section 3.1416e-4 m^2, as in the tutorial
ASPECT = L2 / D2

FEED_CONC = [0.003, 0.003, 0.003]                # mol/m^3 (= mM), like the tutorial
SWITCH_TIME = 180.0                              # s, shared by both units


def col_volume(length, diameter):
    return length * np.pi * diameter ** 2 / 4


def dims_for_volume(volume):
    """(length, diameter) of a column with the given volume at the tutorial aspect ratio."""
    d = (4 * volume / (np.pi * ASPECT)) ** (1 / 3)
    return ASPECT * d, d


V2 = col_volume(L2, D2)


@dataclass
class Design:
    """Zone flow-rate ratios m = (Q t_s - V eps) / (V (1 - eps)) for zones I..IV of each unit."""
    m1: np.ndarray
    m2: np.ndarray
    switch_time: float = SWITCH_TIME

    @property
    def v1(self):
        """Unit-1 column volume that makes unit-1 extract flow equal unit-2 feed flow."""
        return V2 * (self.m2[2] - self.m2[1]) / (self.m1[0] - self.m1[1])

    def zone_flows(self, unit):
        m, v = (self.m1, self.v1) if unit == 1 else (self.m2, V2)
        eps = TOTAL_POROSITY
        return (np.asarray(m) * v * (1 - eps) + v * eps) / self.switch_time

    def port_flows(self, unit):
        qI, qII, qIII, qIV = self.zone_flows(unit)
        return dict(eluent=qI - qIV, extract=qI - qII, feed=qIII - qII, raffinate=qIII - qIV)

    @property
    def feed_flow(self):
        return self.port_flows(1)["feed"]

    def is_feasible(self):
        return all(q > 0 for u in (1, 2) for q in self.port_flows(u).values()) and \
            all(q > 0 for u in (1, 2) for q in self.zone_flows(u))

    def geometry(self):
        L1, D1 = dims_for_volume(self.v1)
        return dict(unit1=dict(length=L1, diameter=D1, volume=self.v1),
                    unit2=dict(length=L2, diameter=D2, volume=V2))


def connections(design: Design, shift: int):
    """Absolute 7-column CADET connection rows for switch index `shift` (same for every cycle)."""
    def col(unit, pos):
        return COL0[unit - 1] + (pos + shift) % N_POS
    rows = []
    for unit, x_dst, r_dst in ((1, col(2, 2), OUT["R"]), (2, OUT["T2"], OUT["T1"])):
        qI, qII, qIII, qIV = design.zone_flows(unit)
        p = design.port_flows(unit)
        unit_rows = [
            (ELUENT, col(unit, 0), p["eluent"]),
            (col(unit, 3), col(unit, 0), qIV),           # recycle
            (col(unit, 0), x_dst, p["extract"]),
            (col(unit, 0), col(unit, 1), qII),
            (col(unit, 1), col(unit, 2), qII),
            (col(unit, 2), r_dst, p["raffinate"]),
            (col(unit, 2), col(unit, 3), qIV),
        ]
        if unit == 1:   # unit 2's feed is unit 1's extract row above
            unit_rows.append((FEED, col(1, 2), p["feed"]))
        rows += unit_rows
    out = []
    for a, b, q in rows:
        out.extend([a, b, -1, -1, -1, -1, float(q)])
    return out


# ---------------------------------------------------------------------------
# CADET model
# ---------------------------------------------------------------------------
def _column_unit(length, diameter, n_col):
    u = Dict()
    u.unit_type = "LUMPED_RATE_MODEL_WITHOUT_PORES"
    u.ncomp = N_COMP
    u.col_dispersion = COL_DISPERSION
    u.col_length = length
    u.total_porosity = TOTAL_POROSITY
    u.velocity = 1
    u.cross_section_area = np.pi * diameter ** 2 / 4
    u.adsorption_model = "LINEAR"
    u.nbound = [1] * N_COMP
    u.adsorption.is_kinetic = 0
    u.adsorption.lin_ka = LIN_KA
    u.adsorption.lin_kd = LIN_KD
    u.init_c = [0.0] * N_COMP
    u.init_q = [0.0] * N_COMP

    d = u.discretization
    d.ncol = n_col
    d.nbound = [1] * N_COMP
    d.use_analytic_jacobian = 1
    d.reconstruction = "WENO"
    d.gs_type = 1
    d.max_krylov = 0
    d.max_restarts = 10
    d.schur_safety = 1e-8
    d.weno.boundary_model = 0
    d.weno.weno_eps = 1e-10
    d.weno.weno_order = 3
    return u


def build_model(design: Design, n_cycles=30, n_col=40, out_dt=0.1, out_last_cycles=2):
    sim = Cadet(install_path=CADET_ROOT)
    m = sim.root.input.model
    n_units = 2 + len(OUT) + 2 * N_POS
    m.nunits = n_units

    for idx in (FEED, ELUENT):
        m[f"unit_{idx:03d}"].unit_type = "INLET"
        m[f"unit_{idx:03d}"].ncomp = N_COMP
        m[f"unit_{idx:03d}"].inlet_type = "PIECEWISE_CUBIC_POLY"
    for idx in OUT.values():
        m[f"unit_{idx:03d}"].unit_type = "OUTLET"
        m[f"unit_{idx:03d}"].ncomp = N_COMP
    geo = design.geometry()
    for unit, g in ((1, geo["unit1"]), (2, geo["unit2"])):
        for k in range(N_POS):
            m[f"unit_{COL0[unit - 1] + k:03d}"] = _column_unit(g["length"], g["diameter"], n_col)

    # --- sections: one per switch ------------------------------------------
    ts = design.switch_time
    nsec = n_cycles * N_POS
    times = np.arange(nsec + 1) * ts
    sim.root.input.solver.sections.nsec = nsec
    sim.root.input.solver.sections.section_times = times
    sim.root.input.solver.sections.section_continuity = [0] * (nsec - 1)

    zeros = [0.0] * N_COMP
    for i in range(nsec):
        for idx, cc in ((FEED, FEED_CONC), (ELUENT, zeros)):
            sec = m[f"unit_{idx:03d}"][f"sec_{i:03d}"]
            sec.const_coeff = cc
            sec.lin_coeff = zeros
            sec.quad_coeff = zeros
            sec.cube_coeff = zeros

    m.connections.connections_include_ports = 1
    m.connections.nswitches = nsec
    for i in range(nsec):
        sw = m.connections[f"switch_{i:03d}"]
        sw.section = i
        sw.connections = connections(design, i % N_POS)

    # --- solver ------------------------------------------------------------
    m.solver.gs_type = 1
    m.solver.max_krylov = 0
    m.solver.max_restarts = 10
    m.solver.schur_safety = 1e-8
    s = sim.root.input.solver
    s.nthreads = 1
    s.consistent_init_mode = 1
    s.time_integrator.abstol = 1e-9
    s.time_integrator.algtol = 1e-12
    s.time_integrator.reltol = 1e-5
    s.time_integrator.init_step_size = 1e-6
    s.time_integrator.max_steps = 1000000

    t_end = times[-1]
    t_start_out = max(0.0, t_end - out_last_cycles * N_POS * ts)
    s.user_solution_times = np.unique(np.concatenate(
        [[0.0], np.arange(t_start_out, t_end, out_dt), times[times >= t_start_out], [t_end]]))

    r = sim.root.input["return"]
    r.split_components_data = 0
    r.split_ports_data = 0
    for idx in range(n_units):
        r[f"unit_{idx:03d}"].write_solution_outlet = 1 if idx in OUT.values() else 0
    r.write_solution_times = 1
    return sim, dict(section_times=times)


def outlet_flows(design: Design):
    return {"R": design.port_flows(1)["raffinate"],
            "T1": design.port_flows(2)["raffinate"],
            "T2": design.port_flows(2)["extract"]}


def cycle_masses(sim, meta, design: Design, cycle: int):
    """Mass (mol) of every component leaving every outlet during `cycle` -> array [outlet, comp]."""
    t = np.asarray(sim.root.output.solution.solution_times)
    t0 = cycle * N_POS * design.switch_time
    mask = (t >= t0 - 1e-9) & (t <= t0 + N_POS * design.switch_time + 1e-9)
    flows = outlet_flows(design)
    masses = np.zeros((len(OUT), N_COMP))
    for j, (name, unit) in enumerate(OUT.items()):
        c = np.asarray(sim.root.output.solution[f"unit_{unit:03d}"].solution_outlet)[mask]
        masses[j] = flows[name] * np.trapezoid(c, t[mask], axis=0)
    return masses


def metrics(masses, design: Design):
    """Purity of each outlet (wrt its target), recovery of each component in its outlet,
    and mass-balance closure (out / in) per component."""
    fed = design.feed_flow * N_POS * design.switch_time * np.asarray(FEED_CONC)
    target = np.diag(masses)
    tot_out = masses.sum(axis=1)
    purity = np.divide(target, tot_out, out=np.zeros(N_COMP), where=tot_out > 0)
    return purity, target / fed, masses.sum(axis=0) / fed


def simulate(design: Design, n_cycles=30, n_col=40, keep=False, timeout=900):
    """Run CADET; return purity / recovery / closure / cyclic-steady-state check (and sim if keep)."""
    sim, meta = build_model(design, n_cycles=n_cycles, n_col=n_col)
    fn = Path(tempfile.gettempdir()) / f"smbc_{os.getpid()}_{uuid.uuid4().hex[:8]}.h5"
    sim.filename = str(fn)
    try:
        sim.save()
        ret = sim.run_simulation(timeout=timeout)   # results are loaded into memory
    finally:
        fn.unlink(missing_ok=True)
    if ret.return_code != 0:
        raise RuntimeError(ret.error_message)
    last = cycle_masses(sim, meta, design, n_cycles - 1)
    prev = cycle_masses(sim, meta, design, n_cycles - 2)
    purity, recovery, closure = metrics(last, design)
    p_prev, _, _ = metrics(prev, design)
    out = dict(purity=purity, recovery=recovery, closure=closure,
               css_delta=float(np.max(np.abs(purity - p_prev))), masses=last)
    if keep:
        out["sim"], out["meta"] = sim, meta
    return out
