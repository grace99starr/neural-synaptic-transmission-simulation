#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ============================================
# Tiled-PCB simulator (absorbing leaks) + sweep stats
# + Cumulative "made it" (bound/N) vs steps m overlays
# Deterministic binding on arrival (no p_sr)
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, List
from IPython.display import display  # for display(df)

# ---------------- Core config ----------------
@dataclass
class Config:
    # Simulation sizes
    N: int = 1000         # walkers per run
    m_max: int = 1000     # max steps per walker
    z_max: int = 10       # membrane plane index

    # PCB tiling: grid of tiles; each tile has local coords 1..cell_size
    cell_size: int = 50
    num_cells_x: int = 10
    num_cells_y: int = 10

    # Membrane dynamics
    p_su: float = 0.12    # at membrane: prob to step back into bulk (z_max -> z_max-1)

    # Replications
    runs: int = 10
    seed: Optional[int] = 42

    def validate(self):
        if self.N <= 0 or self.m_max <= 0:
            raise ValueError("N and m_max must be positive.")
        if self.z_max <= 0:
            raise ValueError("z_max must be >= 1.")
        if self.cell_size <= 1:
            raise ValueError("cell_size must be > 1.")
        if self.num_cells_x <= 0 or self.num_cells_y <= 0:
            raise ValueError("num_cells_x and num_cells_y must be >= 1.")
        if not (0.0 <= self.p_su <= (1.0/6.0) + 1e-12):
            raise ValueError("p_su must be in [0, 1/6].")
        if self.runs <= 0:
            raise ValueError("runs must be positive.")

# ---------------- Moves ----------------
_BULK_MOVES = np.array([
    [ 1,  0,  0],
    [-1,  0,  0],
    [ 0,  1,  0],
    [ 0, -1,  0],
    [ 0,  0,  1],
    [ 0,  0, -1],
], dtype=int)

_MEMBRANE_2D_MOVES = np.array([
    [ 1,  0,  0],
    [-1,  0,  0],
    [ 0,  1,  0],
    [ 0, -1,  0],
], dtype=int)

def _membrane_probs(p_su: float) -> np.ndarray:
    p_lat = (1.0 - p_su) / 4.0
    return np.array([p_lat, p_lat, p_lat, p_lat, p_su], dtype=float)

# ---------------- One run (absorbing leaks) ----------------
def run_once(cfg: Config, seed: Optional[int] = None) -> Dict[str, float]:
    """
    Simulate ONE run and return totals for:
      - successes: walkers that bound at receptor (local (1,1) on z_max)
      - leaks_walkers: walkers that leaked (attempt to step beyond global PCB grid)
      - timeouts: walkers that reached m_max without binding or leaking
      - arrivals: walkers that ever visited receptor coords (bound or not)
      plus fractions: bind_fraction, arrival_fraction
    """
    cfg.validate()
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    mem_probs = _membrane_probs(cfg.p_su)

    # random starts: tile + local coords; all at z=0
    pcb_xs = rng.integers(1, cfg.num_cells_x + 1, size=cfg.N)
    pcb_ys = rng.integers(1, cfg.num_cells_y + 1, size=cfg.N)
    loc_xs = rng.integers(1, cfg.cell_size + 1, size=cfg.N)
    loc_ys = rng.integers(1, cfg.cell_size + 1, size=cfg.N)
    zs     = np.zeros(cfg.N, dtype=int)

    successes = 0
    timeouts = 0
    leaks_walkers = 0
    arrivals_flag = np.zeros(cfg.N, dtype=bool)

    def lateral_with_absorbing_leaks(pcb_x, pcb_y, loc_x, loc_y, dx, dy):
        leaked = False
        # X
        if dx == 1 and loc_x == cfg.cell_size:
            if pcb_x < cfg.num_cells_x: pcb_x += 1; loc_x = 1
            else: leaked = True
        elif dx == -1 and loc_x == 1:
            if pcb_x > 1: pcb_x -= 1; loc_x = cfg.cell_size
            else: leaked = True
        else:
            loc_x = max(1, min(cfg.cell_size, loc_x + dx))
        if leaked:
            return pcb_x, pcb_y, loc_x, loc_y, True
        # Y
        if dy == 1 and loc_y == cfg.cell_size:
            if pcb_y < cfg.num_cells_y: pcb_y += 1; loc_y = 1
            else: leaked = True
        elif dy == -1 and loc_y == 1:
            if pcb_y > 1: pcb_y -= 1; loc_y = cfg.cell_size
            else: leaked = True
        else:
            loc_y = max(1, min(cfg.cell_size, loc_y + dy))
        return pcb_x, pcb_y, loc_x, loc_y, leaked

    for i in range(cfg.N):
        pcb_x = int(pcb_xs[i]); pcb_y = int(pcb_ys[i])
        loc_x = int(loc_xs[i]); loc_y = int(loc_ys[i])
        z = int(zs[i])

        s = 0
        bound = False
        leaked = False
        ever_arrived = False

        while s < cfg.m_max:
            # bind deterministically if on receptor (local (1,1) at membrane)
            if z == cfg.z_max and loc_x == 1 and loc_y == 1:
                ever_arrived = True
                bound = True
                s += 1
                break

            # step
            if z == cfg.z_max:
                choice = rng.choice(5, p=mem_probs)
                if choice == 4:
                    z = z - 1
                else:
                    dx, dy, _ = _MEMBRANE_2D_MOVES[choice]
                    pcb_x, pcb_y, loc_x, loc_y, leaked = lateral_with_absorbing_leaks(
                        pcb_x, pcb_y, loc_x, loc_y, dx, dy
                    )
                    if leaked:
                        s += 1
                        break
            else:
                idx = rng.integers(0, 6)
                dx, dy, dz = _BULK_MOVES[idx]
                if z == 0 and dz == -1: dz = +1
                pcb_x, pcb_y, loc_x, loc_y, leaked = lateral_with_absorbing_leaks(
                    pcb_x, pcb_y, loc_x, loc_y, dx, dy
                )
                if leaked:
                    s += 1
                    break
                z = min(max(z + dz, 0), cfg.z_max)

            s += 1

        # finalize this walker
        if bound: successes += 1
        elif leaked: leaks_walkers += 1
        else: timeouts += 1

        if ever_arrived: arrivals_flag[i] = True

    arrivals = int(arrivals_flag.sum())
    return {
        "successes": float(successes),
        "timeouts": float(timeouts),                # meanders
        "arrivals": float(arrivals),
        "leaks_walkers": float(leaks_walkers),
        "bind_fraction": successes / float(cfg.N),
        "arrival_fraction": arrivals / float(cfg.N),
    }

# ---------------- Average over runs + convenience ----------------
def run_many(cfg: Config) -> Dict[str, float]:
    base_seed = cfg.seed
    acc = {"successes": [], "timeouts": [], "arrivals": [], "leaks_walkers": [],
           "bind_fraction": [], "arrival_fraction": []}
    for r in range(cfg.runs):
        seed_r = (base_seed + 7919*r) if base_seed is not None else None
        res = run_once(cfg, seed=seed_r)
        for k in acc:
            acc[k].append(res[k])

    def mean(a): return float(np.mean(a))
    out = {k + "_mean": mean(v) for k, v in acc.items()}

    # Aliases / derived
    out["leaks_mean"] = out["leaks_walkers_mean"]              # avg walkers leaked per run
    out["meanders_mean"] = out["timeouts_mean"]                # avg walkers timed out per run
    out["leaks_per_walker_mean"] = out["leaks_walkers_mean"] / cfg.N
    # Totals across runs (≈ mean * runs; exact if N is fixed each run)
    out["successes_total"] = out["successes_mean"] * cfg.runs
    out["leaks_total"]     = out["leaks_mean"] * cfg.runs
    out["meanders_total"]  = out["meanders_mean"] * cfg.runs
    out["arrivals_total"]  = out["arrivals_mean"] * cfg.runs
    return out

# ---------------- Cumulative "made it" (bound) vs steps m ----------------
def binding_curve_one_run(cfg: Config, seed: Optional[int] = None) -> np.ndarray:
    """
    Returns curve[m-1] = cumulative # of walkers bound by step m (1..m_max) for ONE run.
    Deterministic binding on arrival.
    """
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    mem_probs = _membrane_probs(cfg.p_su)

    pcb_xs = rng.integers(1, cfg.num_cells_x + 1, size=cfg.N)
    pcb_ys = rng.integers(1, cfg.num_cells_y + 1, size=cfg.N)
    loc_xs = rng.integers(1, cfg.cell_size + 1, size=cfg.N)
    loc_ys = rng.integers(1, cfg.cell_size + 1, size=cfg.N)
    zs     = np.zeros(cfg.N, dtype=int)

    bind_steps: List[int] = []

    def lateral_with_absorbing_leaks(pcb_x, pcb_y, loc_x, loc_y, dx, dy):
        leaked = False
        # X
        if dx == 1 and loc_x == cfg.cell_size:
            if pcb_x < cfg.num_cells_x: pcb_x += 1; loc_x = 1
            else: leaked = True
        elif dx == -1 and loc_x == 1:
            if pcb_x > 1: pcb_x -= 1; loc_x = cfg.cell_size
            else: leaked = True
        else:
            loc_x = max(1, min(cfg.cell_size, loc_x + dx))
        if leaked: return pcb_x, pcb_y, loc_x, loc_y, True
        # Y
        if dy == 1 and loc_y == cfg.cell_size:
            if pcb_y < cfg.num_cells_y: pcb_y += 1; loc_y = 1
            else: leaked = True
        elif dy == -1 and loc_y == 1:
            if pcb_y > 1: pcb_y -= 1; loc_y = cfg.cell_size
            else: leaked = True
        else:
            loc_y = max(1, min(cfg.cell_size, loc_y + dy))
        return pcb_x, pcb_y, loc_x, loc_y, leaked

    for i in range(cfg.N):
        pcb_x = int(pcb_xs[i]); pcb_y = int(pcb_ys[i])
        loc_x = int(loc_xs[i]); loc_y = int(loc_ys[i])
        z = int(zs[i])

        s = 0
        while s < cfg.m_max:
            # bind deterministically if on receptor
            if z == cfg.z_max and loc_x == 1 and loc_y == 1:
                s += 1
                bind_steps.append(s)
                break

            if z == cfg.z_max:
                choice = rng.choice(5, p=mem_probs)
                if choice == 4:
                    z = z - 1
                else:
                    dx, dy, _ = _MEMBRANE_2D_MOVES[choice]
                    pcb_x, pcb_y, loc_x, loc_y, leaked = lateral_with_absorbing_leaks(
                        pcb_x, pcb_y, loc_x, loc_y, dx, dy
                    )
                    if leaked:
                        s += 1
                        break
            else:
                idx = rng.integers(0, 6)
                dx, dy, dz = _BULK_MOVES[idx]
                if z == 0 and dz == -1: dz = +1
                pcb_x, pcb_y, loc_x, loc_y, leaked = lateral_with_absorbing_leaks(
                    pcb_x, pcb_y, loc_x, loc_y, dx, dy
                )
                if leaked:
                    s += 1
                    break
                z = min(max(z + dz, 0), cfg.z_max)

            s += 1

    if len(bind_steps) == 0:
        return np.zeros(cfg.m_max, dtype=float)

    # build cumulative curve length m_max
    bind_steps = np.asarray(bind_steps, dtype=int)  # in [1..m_max]
    hist = np.bincount(bind_steps, minlength=cfg.m_max + 1)[1:cfg.m_max+1]
    return np.cumsum(hist).astype(float)

def binding_curve_mean_over_runs(cfg: Config):
    """
    Returns (m_values, mean_curve) where mean_curve is cumulative bound count averaged over runs.
    """
    base_seed = cfg.seed
    curves = []
    for r in range(cfg.runs):
        seed_r = (base_seed + 7919*r) if base_seed is not None else None
        curves.append(binding_curve_one_run(cfg, seed=seed_r))
    curves = np.vstack(curves)
    mean_curve = curves.mean(axis=0)                # absolute counts
    m_vals = np.arange(1, cfg.m_max + 1)
    return m_vals, mean_curve

def plot_cumulative_binding_norm(cfg: Config, label=None):
    """
    Plot cumulative made-it curve normalized by N (fraction) for one cfg averaged over runs.
    """
    m, mean_curve = binding_curve_mean_over_runs(cfg)
    y = mean_curve / cfg.N
    plt.plot(m, y, label=label or f"p_su={cfg.p_su:.3f}")

# ---------------- Parameters (edit here) ----------------
p_su_values = [0.00, 0.04, 0.08, 0.12, 1.0/6.0]

N = 1000
m_max = 500
z_max = 10
cell_size = 50      # tile size
num_cells_x = 10    # tiles across x
num_cells_y = 10    # tiles across y
runs = 10
seed = 123

# ---------- Sweep only over p_su (deterministic binding; no p_sr) ----------
records = []
for p_su in p_su_values:
    if not (0 <= p_su <= 1/6 + 1e-12):
        raise ValueError(f"p_su={p_su} must be in [0, 1/6].")
    cfg = Config(
        N=N, m_max=m_max, z_max=z_max,
        cell_size=cell_size, num_cells_x=num_cells_x, num_cells_y=num_cells_y,
        p_su=p_su,
        runs=runs, seed=seed
    )
    res = run_many(cfg)
    records.append({
        "p_su": p_su,
        # per-run means
        "successes_mean": res["successes_mean"],
        "leaks_mean": res["leaks_mean"],
        "meanders_mean": res["meanders_mean"],      # timeouts
        "arrivals_mean": res["arrivals_mean"],
        "bind_fraction_mean": res["bind_fraction_mean"],
        "arrival_fraction_mean": res["arrival_fraction_mean"],
        "leaks_per_walker_mean": res["leaks_per_walker_mean"],
        # totals across runs (≈ mean * runs)
        "successes_total": res["successes_total"],
        "leaks_total": res["leaks_total"],
        "meanders_total": res["meanders_total"],
        "arrivals_total": res["arrivals_total"],
    })

df = pd.DataFrame.from_records(records).sort_values("p_su").reset_index(drop=True)
display(df)

# ---------- Pretty print ----------
print("=== SUMMARY (means per run and totals across runs) — deterministic binding ===")
for _, r in df.iterrows():
    print(
        f"p_su={r['p_su']:.3f} | "
        f"successes_mean={r['successes_mean']:.1f}, "
        f"leaks_mean={r['leaks_mean']:.1f}, "
        f"meanders_mean={r['meanders_mean']:.1f}, "
        f"arrivals_mean={r['arrivals_mean']:.1f} || "
        f"successes_total≈{r['successes_total']:.0f}, "
        f"leaks_total≈{r['leaks_total']:.0f}, "
        f"meanders_total≈{r['meanders_total']:.0f}, "
        f"arrivals_total≈{r['arrivals_total']:.0f}"
    )

# ---------------- Cumulative made-it (bound/N) vs m overlays ----------------
PSU_LIST_FOR_OVERLAY = p_su_values  # reuse the same grid

plt.figure(figsize=(8,5))
for p_su in PSU_LIST_FOR_OVERLAY:
    cfg = Config(
        N=N, m_max=m_max, z_max=z_max,
        cell_size=cell_size, num_cells_x=num_cells_x, num_cells_y=num_cells_y,
        p_su=p_su,
        runs=runs, seed=seed
    )
    plot_cumulative_binding_norm(cfg, label=f"p_su={p_su:.3f}")
plt.xlabel("m (steps)")
plt.ylabel("Cumulative made it (fraction of N)")
plt.title(f"Cumulative bound fraction vs steps — varying p_su\n"
          f"Tiles: {num_cells_x}×{num_cells_y}, cell_size={cell_size}, N={N}, runs={runs}")
plt.legend(title="p_su")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


import os
out_dir = "Downloads/research_local"
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "simulation_summary.csv")
df.to_csv(path, index=False)
print(f"Saved to {path}")


# In[ ]:


# Collect cumulative binding curves for each p_su
records_cum = []
for p_su in p_su_values:
    cfg = Config(
        N=N, m_max=m_max, z_max=z_max,
        cell_size=cell_size, num_cells_x=num_cells_x, num_cells_y=num_cells_y,
        p_su=p_su,
        runs=runs, seed=seed
    )
    m_vals, mean_curve = binding_curve_mean_over_runs(cfg)
    frac_curve = mean_curve / cfg.N   # normalize to fraction of walkers
    for m, frac in zip(m_vals, frac_curve):
        records_cum.append({"p_su": p_su, "m": m, "cumulative_fraction": frac})

df_cum = pd.DataFrame.from_records(records_cum)


# In[ ]:


out_dir = "Downloads/research_local"
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "simulation_data.csv")
df_cum.to_csv(path, index=False)
print(f"Saved to {path}")


# In[ ]:




