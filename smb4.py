"""
4-component SMB in CADET: 2 inlets (feed, eluent), 4 columns, 4 outlets.

Components (index): 0 = R  (raffinate impurity, barely binds)
                    1 = T1 (weak target)
                    2 = T2 (medium target)
                    3 = T3 (strong target)

Unit layout (same 2 inlets + 4 columns as the CADET tutorial, but 4 outlets):
    0 FEED (inlet)     1 ELUENT (inlet)
    2 OUT_R            3 OUT_T1          4 OUT_T2          5 OUT_T3
    6..9 columns (LUMPED_RATE_MODEL_WITHOUT_PORES, linear isotherm)

Process description
-------------------
Columns sit in a ring in fixed *positions* P0..P3 (flow goes P0->P1->P2->P3->P0).
At every switch the ports advance one column, so the column at P0 moves to P3,
P1 -> P0, P2 -> P1, P3 -> P2 (same direction as the tutorial).

Port placement (relative to positions):
    ELUENT -> P0 inlet
    P0 outlet -> OUT_T3 (withdraw X3)       -> rest to P1
    P1 outlet -> OUT_T2 (withdraw X2)       -> rest to P2
    P2 outlet -> OUT_T1 (withdraw X1)       -> rest + FEED to P3
    P3 outlet -> OUT_R  (withdraw R)        -> rest recycled to P0

Each switch period is split into sub-steps with their own connection matrix
(flows / which outlets are open). The flows of every sub-step and the sub-step
durations are the decision variables ("optimize over switches via connections").
"""
from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from addict import Dict
from cadet import Cadet

CADET_ROOT = Path(__file__).resolve().parent / ".venv"

COMP_NAMES = ["R", "T1", "T2", "T3"]
N_COMP = 4

FEED, ELUENT = 0, 1
OUT = {"R": 2, "T1": 3, "T2": 4, "T3": 5}          # outlet unit per stream
OUT_TARGET = {2: 0, 3: 1, 4: 2, 5: 3}              # outlet unit -> component it should hold
COL0 = 6
N_COL_UNITS = 4

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


def q_from_m(m, t_s=SWITCH_TIME):
    """Column flow rate (m^3/s) for a dimensionless SMB flow-rate ratio m."""
    return (m * COL_VOLUME * (1 - TOTAL_POROSITY) + COL_VOLUME * TOTAL_POROSITY) / t_s


def m_from_q(q, t_s=SWITCH_TIME):
    return (q * t_s - COL_VOLUME * TOTAL_POROSITY) / (COL_VOLUME * (1 - TOTAL_POROSITY))


# ---------------------------------------------------------------------------
# Sub-step definition (relative frame)
# ---------------------------------------------------------------------------
@dataclass
class SubStep:
    frac: float          # fraction of the switch period
    E: float             # eluent flow into P0                         [m^3/s]
    F: float             # feed flow into P3                           [m^3/s]
    X3: float            # withdrawal P0 outlet -> OUT_T3              [m^3/s]
    X2: float            # withdrawal P1 outlet -> OUT_T2              [m^3/s]
    X1: float            # withdrawal P2 outlet -> OUT_T1              [m^3/s]
    rec: float = 0.0     # recycle P3 outlet -> P0                     [m^3/s]

    def column_flows(self):
        q0 = self.E + self.rec
        q1 = q0 - self.X3
        q2 = q1 - self.X2
        q3 = q2 - self.X1 + self.F
        r = q3 - self.rec
        return np.array([q0, q1, q2, q3]), r

    def relative_connections(self):
        """List of (from, to, flow) using symbolic names P0..P3 / FEED / ELUENT / outlet names."""
        (q0, q1, q2, q3), r = self.column_flows()
        rows = [
            ("ELUENT", "P0", self.E),
            ("P0", "T3", self.X3), ("P0", "P1", q1),
            ("P1", "T2", self.X2), ("P1", "P2", q2),
            ("P2", "T1", self.X1), ("P2", "P3", q2 - self.X1),
            ("FEED", "P3", self.F),
            ("P3", "R", r), ("P3", "P0", self.rec),
        ]
        return [row for row in rows if row[2] > 0.0]

    def outlet_flows(self):
        _, r = self.column_flows()
        return {"R": r, "T1": self.X1, "T2": self.X2, "T3": self.X3}


# Product outlet attached to the outlet of each position (P0..P3)
POS_OUTLET = ["T3", "T2", "T1", "R"]


@dataclass
class SubStepV2:
    """Generalised sub-step: eluent make-up into every position and any split at every node.

    e[p]  eluent flow into the inlet of position p                     [m^3/s]
    w[p]  fraction of position p's outflow sent to its product outlet  (POS_OUTLET[p]),
          the rest goes to position p+1 (P3 -> P0 closes the ring)
    F     feed flow into P3                                            [m^3/s]
    """
    frac: float
    e: np.ndarray
    w: np.ndarray
    F: float

    def column_flows(self):
        # Q_p = e_p + (1 - w_{p-1}) Q_{p-1} (+F at P3): cyclic linear system
        A = np.eye(4)
        b = np.array(self.e, float).copy()
        b[3] += self.F
        for p in range(4):
            A[p, (p - 1) % 4] -= 1.0 - self.w[(p - 1) % 4]
        q = np.linalg.solve(A, b)
        return q, q[3] * self.w[3]

    def relative_connections(self):
        q, _ = self.column_flows()
        rows = [("FEED", "P3", self.F)]
        for p in range(4):
            rows.append(("ELUENT", f"P{p}", self.e[p]))
            rows.append((f"P{p}", POS_OUTLET[p], q[p] * self.w[p]))
            rows.append((f"P{p}", f"P{(p + 1) % 4}", q[p] * (1.0 - self.w[p])))
        return [row for row in rows if row[2] > 0.0]

    def outlet_flows(self):
        q, _ = self.column_flows()
        return {POS_OUTLET[p]: q[p] * self.w[p] for p in range(4)}

    # convenience for reporting
    @property
    def E(self):
        return float(np.sum(self.e))


def connections_for(substep: SubStep, shift: int):
    """Absolute 7-column CADET connection rows for a sub-step at switch index `shift`."""
    def unit(name):
        if name == "FEED":
            return FEED
        if name == "ELUENT":
            return ELUENT
        if name in OUT:
            return OUT[name]
        pos = int(name[1])
        # column sitting at position `pos` after `shift` switches
        return COL0 + (pos + shift) % N_COL_UNITS
    rows = []
    for a, b, q in substep.relative_connections():
        rows.extend([unit(a), unit(b), -1, -1, -1, -1, float(q)])
    return rows


@dataclass
class Schedule:
    substeps: list[SubStep]
    switch_time: float = SWITCH_TIME

    def validate(self, qmin=1e-9):
        fr = sum(s.frac for s in self.substeps)
        if abs(fr - 1.0) > 1e-9:
            raise ValueError(f"sub-step fractions sum to {fr}")
        for s in self.substeps:
            qs, _ = s.column_flows()
            if np.any(qs < qmin) or min(s.outlet_flows().values()) < -1e-15:
                return False
        return True


# ---------------------------------------------------------------------------
# CADET model
# ---------------------------------------------------------------------------
def _column_unit():
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
    return u


def _set_discretization(u, n_col):
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


def build_model(schedule: Schedule, n_cycles=20, n_col=40, out_dt=1.0, out_last_cycles=2):
    """Return a configured Cadet object plus section metadata."""
    sim = Cadet(install_path=CADET_ROOT)
    m = sim.root.input.model
    n_units = 2 + 4 + N_COL_UNITS
    m.nunits = n_units

    for idx in (FEED, ELUENT):
        m[f"unit_{idx:03d}"].unit_type = "INLET"
        m[f"unit_{idx:03d}"].ncomp = N_COMP
        m[f"unit_{idx:03d}"].inlet_type = "PIECEWISE_CUBIC_POLY"
    for idx in OUT.values():
        m[f"unit_{idx:03d}"].unit_type = "OUTLET"
        m[f"unit_{idx:03d}"].ncomp = N_COMP
    for k in range(N_COL_UNITS):
        u = _column_unit()
        _set_discretization(u, n_col)
        m[f"unit_{COL0 + k:03d}"] = u

    # --- sections: n_cycles * 4 switches * n_substeps --------------------
    ts = schedule.switch_time
    times = [0.0]
    sec_substep = []   # (cycle, switch, substep index) for each section
    for c in range(n_cycles):
        for sw in range(N_COL_UNITS):
            t0 = (c * N_COL_UNITS + sw) * ts
            acc = 0.0
            for k, s in enumerate(schedule.substeps):
                if s.frac <= 0:
                    continue
                acc += s.frac
                times.append(t0 + acc * ts)
                sec_substep.append((c, sw, k))
    times[-1] = n_cycles * N_COL_UNITS * ts
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
    t_start_out = max(0.0, t_end - out_last_cycles * N_COL_UNITS * ts)
    grid = np.arange(t_start_out, t_end, out_dt)
    edges = np.array(times)
    grid = np.unique(np.concatenate([[0.0], grid, edges[edges >= t_start_out], [t_end]]))
    s.user_solution_times = grid

    r = sim.root.input["return"]
    r.split_components_data = 0
    r.split_ports_data = 0
    for idx in range(n_units):
        r[f"unit_{idx:03d}"].write_solution_outlet = 1 if idx in OUT.values() else 0
    r.write_solution_times = 1

    meta = dict(section_times=np.array(times), sec_substep=sec_substep, n_cycles=n_cycles)
    return sim, meta


def outlet_flow(substep, outlet_unit: int):
    flows = substep.outlet_flows()
    return {OUT[name]: q for name, q in flows.items()}[outlet_unit]


def cycle_masses(sim, meta, schedule: Schedule, cycle: int):
    """Mass (mol) of every component leaving every outlet during `cycle` -> array [outlet(4), comp(4)]."""
    t = np.asarray(sim.root.output.solution.solution_times)
    st = meta["section_times"]
    masses = np.zeros((4, N_COMP))
    for i, (c, sw, k) in enumerate(meta["sec_substep"]):
        if c != cycle:
            continue
        a, b = st[i], st[i + 1]
        mask = (t >= a - 1e-9) & (t <= b + 1e-9)
        tt = t[mask]
        for j, unit in enumerate(OUT.values()):
            q = outlet_flow(schedule.substeps[k], unit)
            if q <= 0:
                continue
            cc = np.asarray(sim.root.output.solution[f"unit_{unit:03d}"].solution_outlet)[mask]
            masses[j] += q * np.trapezoid(cc, tt, axis=0)
    return masses


def metrics(masses, schedule: Schedule):
    """Purity of each outlet (wrt its target) and recovery of each component in its outlet."""
    fed = np.zeros(N_COMP)
    for s in schedule.substeps:
        fed += s.F * s.frac * schedule.switch_time * N_COL_UNITS * np.asarray(FEED_CONC)
    tot_out = masses.sum(axis=1)
    purity = np.array([masses[j, j] / tot_out[j] if tot_out[j] > 0 else 0.0 for j in range(4)])
    recovery = np.array([masses[j, j] / fed[j] if fed[j] > 0 else 0.0 for j in range(4)])
    closure = masses.sum(axis=0) / np.where(fed > 0, fed, 1)
    return purity, recovery, closure


def simulate(schedule: Schedule, n_cycles=20, n_col=40, keep=False, workdir=None, timeout=600):
    """Run CADET and return dict with purity / recovery / css-check (and sim if keep=True)."""
    sim, meta = build_model(schedule, n_cycles=n_cycles, n_col=n_col)
    workdir = Path(workdir or tempfile.gettempdir())
    fn = workdir / f"smb4_{os.getpid()}_{uuid.uuid4().hex[:8]}.h5"
    sim.filename = str(fn)
    sim.save()
    ret = sim.run_simulation(timeout=timeout)
    if ret.return_code != 0:
        try:
            fn.unlink()
        except FileNotFoundError:
            pass
        raise RuntimeError(ret.error_message)
    last = cycle_masses(sim, meta, schedule, n_cycles - 1)
    prev = cycle_masses(sim, meta, schedule, n_cycles - 2)
    purity, recovery, closure = metrics(last, schedule)
    p_prev, _, _ = metrics(prev, schedule)
    out = dict(purity=purity, recovery=recovery, closure=closure,
               css_delta=float(np.max(np.abs(purity - p_prev))), masses=last)
    if keep:
        out["sim"], out["meta"] = sim, meta
        out["file"] = fn
    else:
        try:
            fn.unlink()
        except FileNotFoundError:
            pass
    return out
