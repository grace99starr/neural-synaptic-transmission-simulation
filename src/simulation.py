import numpy as np
from typing import Optional, Dict, List

from src.config import Config
from src.moves import BULK_MOVES, MEMBRANE_2D_MOVES, membrane_probs

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
