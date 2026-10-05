# 4-component SMB in CADET: optimizing the switch connections

An experiment based on the CADET SMB tutorial. It uses the same 2 inlets (feed and eluent), the same
4 columns (LRM without pores, L = 0.25 m, ε = 0.83, n_col = 40, linear isotherm, 180 s switch time)
and **4 outlets**. Only the connection matrices are optimized. Column, isotherm and switch time are
fixed.

| component | ka | kd | H = ka/kd | outlet |
|---|---|---|---|---|
| R  (impurity, barely binds) | 0.5  | 2.0 | 0.25 | `OUT_R`  (unit 2) |
| T1 | 5.0  | 2.0 | 2.5  | `OUT_T1` (unit 3) |
| T2 | 7.5  | 1.5 | 5.0  | `OUT_T2` (unit 4) |
| T3 | 12.0 | 1.2 | 10.0 | `OUT_T3` (unit 5) |

Feed is 0.003 mM of each component. Units: 0 FEED, 1 ELUENT, 2–5 outlets, 6–9 columns.

## Result

The optimizer converged on a connection schedule that puts every component into its own outlet at
≥ 95 % purity (cyclic steady state, last cycle of 12, n_col = 40):

| outlet | purity | recovery of its component |
|---|---|---|
| OUT_R  | 96.1 % | 99.8 % |
| OUT_T1 | 96.0 % | 98.4 % |
| OUT_T2 | 96.6 % | 99.7 % |
| OUT_T3 | 99.96 % | 100 % |

- Feed throughput is 1.27e-7 m³/s, 27 % above the tutorial's 0.98e-7.
- Average eluent use is about 1.25e-6 m³/s, so the products come out roughly 10× diluted.
- Mass balance closes to within 0.1 %.
- At n_col = 80 the same schedule gives 99.0 / 99.5 / 99.4 / 100 % (`results/mesh80/`). The 40-cell numbers are
  limited by numerical dispersion, so they are conservative.

Files:
- `results/B_throughput_hand_seed_connections.txt`: every connection row, ready to paste.
- `results/B_throughput_hand_seed_outlets.png` and `..._composition.png`: plots.

### What the converged schedule does

Each 180 s switch is split into 2 sub-steps with their own connection matrices. The pattern rotates
one column every switch, the same way as in the tutorial:

```
sub-step 0 (51.7 s):  FEED  -> P3 -> OUT_R       ELUENT -> P2 -> OUT_T1
                      ELUENT -> P1 -> OUT_T2     ELUENT -> P0 -> OUT_T3
sub-step 1 (128.3 s): same routing, feed off, more eluent through P3 to flush R out
```

As a column rotates P3 → P2 → P1 → P0, each position elutes exactly one more component:
- P3 loads the feed and flushes out R.
- P2 elutes T1.
- P1 elutes T2.
- P0 strips T3.

The eluent volume at each position is set so that the next-stronger component stays in the column.

**Main finding:** with 4 columns and a single isocratic eluent, the optimizer chose to **send each
column's whole outflow to its own product outlet**. It sent nothing on to the next column, even
though it was allowed to. So the converged process is a rotating sequential scheme (like the "SSMB"
idea), not classic counter-current SMB. A classic 4-zone ring only makes one binary split, and a
6-zone quaternary SMB would need more than 4 columns.

## How we got there (optimizer runs)

| run | connection set | start point | outcome |
|---|---|---|---|
| stage 1 (removed; see commit `274fe45`) | series ring, ports at fixed nodes, partial withdrawals | hand guess | plateaued at about 72 % worst purity (n_col = 20); stopped |
| A (`results/A_*`) | eluent can enter any column, any split at every node | naive (50 % splits, equal flows) | CMA-ES went from 0.6 % to a 74 % worst-purity local optimum in 75 generations; stopped |
| hand design (`hand_design.py`) | same | ideal band positions from retention volumes | 97.9 / 97.0 / 98.7 / 99.98 % at feed 1.0e-7 |
| **B** (`results/B_*`) | same, feed throughput free | hand design | **maximized throughput subject to ≥ 95 %: result above** |

**Takeaway for doing this in CADET:** a black-box optimizer over connection flows works for
polishing and pushing throughput. Starting from scratch, it gets stuck in a poor local optimum,
because the feasible region is narrow. Every column outlet has to be fully diverted, and each
position's eluent volume has to land within about 10 mL.

The better start point is a physics-based seed, where each position's eluent volume per switch sits
between the retention volumes `V_col·(ε + (1−ε)·H)` of the components it should and shouldn't elute.
The optimizer then refines from that seed.

## Caveats

- With is_kinetic = 0, a linear isotherm and near-zero axial dispersion, band broadening comes almost
  entirely from the n_col = 40 discretization. Purities depend on the mesh (see the 80-cell check).
  Before trusting the margins, add real dispersion and mass-transfer data.
- This is isocratic: elution is driven by eluent volume, not by a salt or pH step. True bind-and-elute
  would need a gradient-capable binding model (for example SMA) and a second eluent composition.
  That would change the inlet count.

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install cadet-core cadet-python numpy scipy matplotlib cma
```

Re-run the throughput optimization:

```bash
.venv/bin/python optimize.py --goal throughput --free-feed --init results/hand_design.json --ncol 40 --cycles 8 --iters 80 --sigma 0.02 --out results/B_throughput_hand_seed.json
```

Validate the result and write plots and connection matrices:

```bash
.venv/bin/python report.py results/B_throughput_hand_seed.json --ncol 40 --cycles 12
```

- `smb4.py`: CADET model builder. It handles relative positions → absolute connections, one CADET
  switch per sub-step, and the purity and recovery calculation at cyclic steady state.
- `optimize.py`: CMA-ES over the per-sub-step connection flows. It runs in parallel, one cadet-cli
  process per evaluation, about 13 s each at n_col = 40.
- `hand_design.py`: physics-based seed.
- `report.py`: full-fidelity re-simulation, plots, and connection dump.
