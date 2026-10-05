"""
4-component SMB in CADET: 2 inlets (feed, eluent), 4 columns, 4 outlets.

Components: 0 = R (impurity, barely binds), 1 = T1, 2 = T2, 3 = T3 (strongest).

Units:  0 FEED   1 ELUENT   2 OUT_R   3 OUT_T1   4 OUT_T2   5 OUT_T3
        6..9 columns (LUMPED_RATE_MODEL_WITHOUT_PORES, linear isotherm)

Columns sit in a ring of positions P0 -> P1 -> P2 -> P3 -> P0. At every switch the ports
advance one column (column at P1 moves to P0, P0 -> P3, ...), as in the CADET tutorial.
Each position has one product outlet (P0 -> T3, P1 -> T2, P2 -> T1, P3 -> R); feed enters P3
and eluent can enter any position. Each switch period is split into sub-steps, each with its
own connection matrix.
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

COMP_NAMES = ["R", "T1", "T2", "T3"]
N_COMP = 4

FEED, ELUENT = 0, 1
OUT = {"R": 2, "T1": 3, "T2": 4, "T3": 5}          # outlet unit per stream
COL0 = 6
N_POS = 4
POS_OUTLET = ["T3", "T2", "T1", "R"]               # product outlet of each position

# ---- fixed column / isotherm data (tutorial geometry) ----------------------
COL_LENGTH = 0.25
TOTAL_POROSITY = 0.83
CROSS_SECTION = 3.141592653589793e-4
COL_DISPERSION = 3.8148e-20
COL_VOLUME = COL_LENGTH * CROSS_SECTION

# Arbitrarily chosen ka/kd. Henry constant H = ka/kd -> R 0.25, T1 2.5, T2 5.0, T3 10.0
LIN_KA = [0.5, 5.0, 7.5, 12.0]
LIN_KD = [2.0, 2.0, 1.5, 1.2]
HENRY = [a / d for a, d in zip(LIN_KA, LIN_KD)]

FEED_CONC = [0.003, 0.003, 0.003, 0.003]          # mol/m^3 (= mM), like the tutorial
SWITCH_TIME = 180.0                               # s, fixed like the tutorial


@dataclass
class SubStep:
    """One sub-step of a switch period, in the relative (position) frame.

    e[p]  eluent flow into the inlet of position p                     [m^3/s]
    w[p]  fraction of position p's outflow sent to its product outlet  (POS_OUTLET[p]),
          the rest goes to position p+1 (P3 -> P0 closes the ring)
    F     feed flow into P3                                            [m^3/s]
    """
    frac: float          # share of the switch period
    e: np.ndarray
    w: np.ndarray
    F: float

    def column_flows(self):
        # Q_p = e_p + (1 - w_{p-1}) Q_{p-1} (+F at P3): cyclic linear system
        A = np.eye(N_POS)
        b = np.array(self.e, float)
        b[3] += self.F
        for p in range(N_POS):
            A[p, p - 1] -= 1.0 - self.w[p - 1]
        return np.linalg.solve(A, b)

    def outlet_flows(self):
        q = self.column_flows()
        return {POS_OUTLET[p]: q[p] * self.w[p] for p in range(N_POS)}

    def relative_connections(self):
        """(from, to, flow) with symbolic names P0..P3 / FEED / ELUENT / outlet names."""
        q = self.column_flows()
        rows = [("FEED", "P3", self.F)]
        for p in range(N_POS):
            rows.append(("ELUENT", f"P{p}", self.e[p]))
            rows.append((f"P{p}", POS_OUTLET[p], q[p] * self.w[p]))
            rows.append((f"P{p}", f"P{(p + 1) % N_POS}", q[p] * (1.0 - self.w[p])))
        return [row for row in rows if row[2] > 0.0]


@dataclass
class Schedule:
    substeps: list[SubStep]
    switch_time: float = SWITCH_TIME


def connections_for(substep: SubStep, shift: int):
    """Absolute 7-column CADET connection rows for a sub-step at switch index `shift`."""
    def unit(name):
        if name == "FEED":
            return FEED
        if name == "ELUENT":
            return ELUENT
        if name in OUT:
            return OUT[name]
        return COL0 + (int(name[1]) + shift) % N_POS   # column at that position after `shift` switches
    rows = []
    for a, b, q in substep.relative_connections():
        rows.extend([unit(a), unit(b), -1, -1, -1, -1, float(q)])
    return rows


# ---------------------------------------------------------------------------
# CADET model
# ---------------------------------------------------------------------------
def _column_unit(n_col):
    u = Dict()
    u.unit_type = "LUMPED_RATE_MODEL_WITHOUT_PORES"
    u.ncomp = N_COMP
    u.col_dispersion = COL_DISPERSION
    u.col_length = COL_LENGTH
    u.total_porosity = TOTAL_POROSITY
    u.velocity = 1
    u.cross_section_area = CROSS_SECTION
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


def build_model(schedule: Schedule, n_cycles=20, n_col=40, out_dt=1.0, out_last_cycles=2):
    """Return a configured Cadet object plus section metadata."""
    sim = Cadet(install_path=CADET_ROOT)
    m = sim.root.input.model
    n_units = 2 + len(OUT) + N_POS
    m.nunits = n_units

    for idx in (FEED, ELUENT):
        m[f"unit_{idx:03d}"].unit_type = "INLET"
        m[f"unit_{idx:03d}"].ncomp = N_COMP
        m[f"unit_{idx:03d}"].inlet_type = "PIECEWISE_CUBIC_POLY"
    for idx in OUT.values():
        m[f"unit_{idx:03d}"].unit_type = "OUTLET"
        m[f"unit_{idx:03d}"].ncomp = N_COMP
    for k in range(N_POS):
        m[f"unit_{COL0 + k:03d}"] = _column_unit(n_col)

    # --- sections: n_cycles * 4 switches * n_substeps --------------------
    ts = schedule.switch_time
    times = [0.0]
    sec_substep = []   # (cycle, switch, substep index) for each section
    for c in range(n_cycles):
        for sw in range(N_POS):
            t0 = (c * N_POS + sw) * ts
            acc = 0.0
            for k, s in enumerate(schedule.substeps):
                acc += s.frac
                times.append(t0 + acc * ts)
                sec_substep.append((c, sw, k))
    times[-1] = n_cycles * N_POS * ts
    nsec = len(sec_substep)
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

    # --- connections: one switch per section ------------------------------
    m.connections.connections_include_ports = 1
    m.connections.nswitches = nsec
    for i, (c, sw, k) in enumerate(sec_substep):
        sw_grp = m.connections[f"switch_{i:03d}"]
        sw_grp.section = i
        sw_grp.connections = connections_for(schedule.substeps[k], sw)

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

    # output times: dense over the last cycles, always including section edges
    t_end = times[-1]
    t_start_out = max(0.0, t_end - out_last_cycles * N_POS * ts)
    edges = np.array(times)
    s.user_solution_times = np.unique(np.concatenate(
        [[0.0], np.arange(t_start_out, t_end, out_dt), edges[edges >= t_start_out], [t_end]]))

    r = sim.root.input["return"]
    r.split_components_data = 0
    r.split_ports_data = 0
    for idx in range(n_units):
        r[f"unit_{idx:03d}"].write_solution_outlet = 1 if idx in OUT.values() else 0
    r.write_solution_times = 1

    return sim, dict(section_times=edges, sec_substep=sec_substep)


def cycle_masses(sim, meta, schedule: Schedule, cycle: int):
    """Mass (mol) of every component leaving every outlet during `cycle` -> array [outlet, comp]."""
    t = np.asarray(sim.root.output.solution.solution_times)
    st = meta["section_times"]
    masses = np.zeros((len(OUT), N_COMP))
    for i, (c, sw, k) in enumerate(meta["sec_substep"]):
        if c != cycle:
            continue
        mask = (t >= st[i] - 1e-9) & (t <= st[i + 1] + 1e-9)
        flows = schedule.substeps[k].outlet_flows()
        for j, (name, unit) in enumerate(OUT.items()):
            if flows[name] <= 0:
                continue
            cc = np.asarray(sim.root.output.solution[f"unit_{unit:03d}"].solution_outlet)[mask]
            masses[j] += flows[name] * np.trapezoid(cc, t[mask], axis=0)
    return masses


def metrics(masses, schedule: Schedule):
    """Purity of each outlet (wrt its target), recovery of each component in its outlet,
    and mass-balance closure (out / in) per component."""
    feed_vol = sum(s.F * s.frac for s in schedule.substeps) * schedule.switch_time * N_POS
    fed = feed_vol * np.asarray(FEED_CONC)
    target = np.diag(masses)
    tot_out = masses.sum(axis=1)
    purity = np.divide(target, tot_out, out=np.zeros(N_COMP), where=tot_out > 0)
    recovery = np.divide(target, fed, out=np.zeros(N_COMP), where=fed > 0)
    closure = np.divide(masses.sum(axis=0), fed, out=np.zeros(N_COMP), where=fed > 0)
    return purity, recovery, closure


def simulate(schedule: Schedule, n_cycles=20, n_col=40, keep=False, timeout=600):
    """Run CADET; return purity / recovery / closure / cyclic-steady-state check (and sim if keep)."""
    sim, meta = build_model(schedule, n_cycles=n_cycles, n_col=n_col)
    fn = Path(tempfile.gettempdir()) / f"smb4_{os.getpid()}_{uuid.uuid4().hex[:8]}.h5"
    sim.filename = str(fn)
    try:
        sim.save()
        ret = sim.run_simulation(timeout=timeout)   # results are loaded into memory
    finally:
        fn.unlink(missing_ok=True)
    if ret.return_code != 0:
        raise RuntimeError(ret.error_message)
    last = cycle_masses(sim, meta, schedule, n_cycles - 1)
    prev = cycle_masses(sim, meta, schedule, n_cycles - 2)
    purity, recovery, closure = metrics(last, schedule)
    p_prev, _, _ = metrics(prev, schedule)
    out = dict(purity=purity, recovery=recovery, closure=closure,
               css_delta=float(np.max(np.abs(purity - p_prev))), masses=last)
    if keep:
        out["sim"], out["meta"] = sim, meta
    return out
