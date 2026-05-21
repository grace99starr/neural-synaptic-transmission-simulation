import matplotlib.pyplot as plt
from src.simulation import binding_curve_mean_over_runs
from src.config import Config

def plot_cumulative_binding_norm(cfg: Config, label=None):
    m, mean_curve = binding_curve_mean_over_runs(cfg)
    y = mean_curve / cfg.N
    plt.plot(m, y, label=label or f"p_su={cfg.p_su:.3f}")