import numpy as np

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