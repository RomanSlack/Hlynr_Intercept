"""Mathematical utilities for AegisIntercept"""

import numpy as np
from typing import Tuple


def distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Calculate Euclidean distance between two positions."""
    return np.linalg.norm(pos1 - pos2)


def normalize_vector(vector: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize a vector to unit length with numerical stability.
    
    Args:
        vector: Input vector to normalize
        epsilon: Minimum norm threshold to avoid division by zero
        
    Returns:
        Normalized vector, or zero vector if input norm is below epsilon
    """
    norm = np.linalg.norm(vector)
    if norm < epsilon:
        return np.zeros_like(vector)
    return vector / norm


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0, epsilon: float = 1e-12) -> float:
    """
    Safely divide two numbers, avoiding division by zero.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value to return if denominator is near zero
        epsilon: Minimum denominator threshold
        
    Returns:
        Result of division or default value
    """
    if abs(denominator) < epsilon:
        return default
    return numerator / denominator


def sanitize_array(arr: np.ndarray, max_val: float = 1e6) -> np.ndarray:
    """
    Sanitize array by replacing NaN/Inf values and clamping extreme values.
    
    Args:
        arr: Input array
        max_val: Maximum allowed absolute value
        
    Returns:
        Sanitized array
    """
    # Replace NaN and Inf values
    arr = np.nan_to_num(arr, nan=0.0, posinf=max_val, neginf=-max_val)
    
    # Clamp extreme values
    arr = np.clip(arr, -max_val, max_val)
    
    return arr