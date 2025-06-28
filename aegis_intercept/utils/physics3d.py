"""3D Physics Utilities for AegisIntercept"""

import numpy as np

def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two 3D points."""
    return np.linalg.norm(p1 - p2)

def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp a value within a specified range."""
    return max(min_value, min(value, max_value))

def linear_drag(velocity: np.ndarray, drag_coefficient: float) -> np.ndarray:
    """Calculate linear drag force."""
    return -drag_coefficient * velocity
