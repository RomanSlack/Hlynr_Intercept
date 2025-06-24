"""Mathematical utilities for AegisIntercept"""

import numpy as np
from typing import Tuple


def distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Calculate Euclidean distance between two positions."""
    return np.linalg.norm(pos1 - pos2)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))