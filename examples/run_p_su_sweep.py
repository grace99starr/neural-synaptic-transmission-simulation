import os
import pandas as pd
import matplotlib.pyplot as plt

from src.config import Config
from src.simulation import run_many, binding_curve_mean_over_runs
from src.plotting import plot_cumulative_binding_norm

p_su_values = [0.00, 0.04, 0.08, 0.12, 1.0/6.0]

N = 1000
m_max = 500
z_max = 10
cell_size = 50
num_cells_x = 10
num_cells_y = 10
runs = 10
seed = 123

records = []

for p_su in p_su_values:
    cfg = Config(
        N=N,
        m_max=m_max,
        z_max=z_max,
        cell_size=cell_size,
        num_cells_x=num_cells_x,
        num_cells_y=num_cells_y,
        p_su=p_su,
        runs=runs,
        seed=seed,
    )

    res = run_many(cfg)

    records.append({
        "p_su": p_su,
        "successes_mean": res["successes_mean"],
        "leaks_mean": res["leaks_mean"],
        "meanders_mean": res["meanders_mean"],
        "arrivals_mean": res["arrivals_mean"],
        "bind_fraction_mean": res["bind_fraction_mean"],
        "arrival_fraction_mean": res["arrival_fraction_mean"],
        "leaks_per_walker_mean": res["leaks_per_walker_mean"],
        "successes_total": res["successes_total"],
        "leaks_total": res["leaks_total"],
        "meanders_total": res["meanders_total"],
        "arrivals_total": res["arrivals_total"],
    })

df = pd.DataFrame.from_records(records)
print(df)

os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/simulation_summary.csv", index=False)

plt.figure(figsize=(8, 5))

for p_su in p_su_values:
    cfg = Config(
        N=N,
        m_max=m_max,
        z_max=z_max,
        cell_size=cell_size,
        num_cells_x=num_cells_x,
        num_cells_y=num_cells_y,
        p_su=p_su,
        runs=runs,
        seed=seed,
    )
    plot_cumulative_binding_norm(cfg, label=f"p_su={p_su:.3f}")

plt.xlabel("m (steps)")
plt.ylabel("Cumulative made it (fraction of N)")
plt.title("Cumulative bound fraction vs steps — varying p_su")
plt.legend(title="p_su")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/cumulative_binding.png", dpi=300)
plt.show()