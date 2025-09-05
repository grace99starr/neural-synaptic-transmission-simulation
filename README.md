# Neurotransmitter Random Walk Simulation

This project implements a **3D random walk simulation** of neurotransmitters (walkers) diffusing across a synaptic cleft.  
The model is based on a discrete-time, discrete-space stochastic process with absorbing/leaking boundary conditions and deterministic binding at a receptor site.

---

## Features

- **3D diffusion in bulk (aqueous space)** with unbiased random steps (±x, ±y, ±z).
- **Reflective boundary at z=0** (the cleft bottom).
- **Membrane dynamics at z=10**:
  - With probability *p_su*, a walker steps back into the bulk.
  - With probability *1 - p_su*, it moves laterally (2D diffusion along the membrane).
- **Deterministic receptor binding**:
  - Any walker arriving at receptor coordinates `(0,0,10)` (or equivalently `(1,1)` in local tile coordinates) binds immediately.
- **Periodic tiling of the membrane**:
  - The cleft is divided into local tiles (`cell_size × cell_size`), repeated in x/y.
  - If a walker crosses beyond `num_cells_x` or `num_cells_y`, it **leaks** (absorbing boundary).
- **Simulation outcomes tracked**:
  - **Successes** (bound at receptor)
  - **Meanders** (timeouts at `m_max`)
  - **Leaks** (left the membrane grid)
  - **Arrivals** (visited receptor site, bound or not)
- **Cumulative binding curves (CUM(t))**:
  - Track fraction of walkers bound by each step `t`.
  - Plot individual runs or averaged curves.

---

## Parameters

Key parameters you can configure in `Config`:

- `N` — number of walkers per run  
- `m_max` — maximum number of steps allowed per walker  
- `z_max` — cleft height (default 10)  
- `cell_size` — tile width in x and y  
- `num_cells_x`, `num_cells_y` — how many tiles in each direction  
- `p_su` — probability to step back into bulk at membrane  
- `runs` — number of independent simulation runs  
- `seed` — random number generator seed  

---

## Tech Stack
- **Language:** Python 3.10+
- **Libraries:** NumPy, Pandas, Matplotlib
  
---

## Example Outputs

### Summary stats (averaged across runs)
- Mean counts per run: **successes, leaks, meanders, arrivals**
- Fractions: bind fraction, arrival fraction, leaks per walker
- Totals across all runs

### Cumulative binding curves
- **Per-run curves** (visualize stochastic variability)
- **Mean curve** (average across runs)
- Overlays for different values of `p_su`
