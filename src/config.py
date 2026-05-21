from dataclasses import dataclass
from typing import Optional, Dict, List

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