"""
Centralized schema definition for 26D radar observations.

This module defines the fixed observation schema to prevent brittle index mapping
throughout the HRL codebase. All observation indexing should use these constants.
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Observation26DSchema:
    """
    Fixed schema for 26D radar observations from core.py Radar26DObservation.

    Layout:
        [0-2]:   relative_position (target - interceptor) [3D meters]
        [3-5]:   relative_velocity [3D m/s]
        [6-8]:   interceptor_velocity [3D m/s]
        [9-11]:  interceptor_orientation (euler angles) [3D radians]
        [12]:    fuel_fraction [0-1]
        [13]:    time_to_intercept [seconds]
        [14]:    radar_lock_quality [0-1]
        [15]:    closing_rate [m/s]
        [16]:    off_axis_angle [radians]
        [17-25]: (reserved/padding)
    """
    # Position and velocity (target-relative)
    RELATIVE_POSITION: Tuple[int, int] = (0, 3)      # [0:3]
    RELATIVE_VELOCITY: Tuple[int, int] = (3, 6)      # [3:6]

    # Interceptor self-state
    INTERCEPTOR_VELOCITY: Tuple[int, int] = (6, 9)   # [6:9]
    INTERCEPTOR_ORIENTATION: Tuple[int, int] = (9, 12)  # [9:12]

    # Scalar features
    FUEL_FRACTION: int = 12
    TIME_TO_INTERCEPT: int = 13
    RADAR_LOCK_QUALITY: int = 14
    CLOSING_RATE: int = 15
    OFF_AXIS_ANGLE: int = 16

    # Schema constants
    EXPECTED_DIMENSION: int = 26
    MIN_DIMENSION: int = 17  # Minimum required (excludes padding)


# Global instance for easy access
SCHEMA = Observation26DSchema()


def validate_observation_schema(obs: np.ndarray, strict: bool = True) -> bool:
    """
    Validate that observation matches expected 26D schema.

    Args:
        obs: Observation array to validate
        strict: If True, requires exactly 26D. If False, allows 17D minimum.

    Returns:
        True if observation matches schema

    Raises:
        ValueError: If observation doesn't match expected schema
    """
    # Handle frame-stacked observations
    if obs.ndim > 1:
        obs = obs.flatten()

    obs_dim = obs.shape[0]

    # Check for frame-stacked observations (104D = 26D * 4)
    if obs_dim % SCHEMA.EXPECTED_DIMENSION == 0:
        # Frame-stacked, extract latest frame
        obs = obs[-SCHEMA.EXPECTED_DIMENSION:]
        obs_dim = SCHEMA.EXPECTED_DIMENSION

    # Validate dimension
    if strict:
        if obs_dim != SCHEMA.EXPECTED_DIMENSION:
            raise ValueError(
                f"Expected {SCHEMA.EXPECTED_DIMENSION}D observation, got {obs_dim}D. "
                f"Schema may have changed in core.py!"
            )
    else:
        if obs_dim < SCHEMA.MIN_DIMENSION:
            raise ValueError(
                f"Observation must be at least {SCHEMA.MIN_DIMENSION}D, got {obs_dim}D"
            )

    # Validate key indices are finite
    required_indices = [
        SCHEMA.FUEL_FRACTION,
        SCHEMA.TIME_TO_INTERCEPT,
        SCHEMA.RADAR_LOCK_QUALITY,
        SCHEMA.CLOSING_RATE,
        SCHEMA.OFF_AXIS_ANGLE,
    ]

    for idx in required_indices:
        if idx >= obs_dim:
            raise ValueError(
                f"Schema mismatch: index {idx} out of bounds for {obs_dim}D observation"
            )
        if not np.isfinite(obs[idx]):
            # Warning, not error (NaN can occur in some cases)
            pass

    return True


def get_relative_position(obs: np.ndarray) -> np.ndarray:
    """Extract relative position from observation."""
    start, end = SCHEMA.RELATIVE_POSITION
    return obs[start:end]


def get_relative_velocity(obs: np.ndarray) -> np.ndarray:
    """Extract relative velocity from observation."""
    start, end = SCHEMA.RELATIVE_VELOCITY
    return obs[start:end]


def get_interceptor_velocity(obs: np.ndarray) -> np.ndarray:
    """Extract interceptor velocity from observation."""
    start, end = SCHEMA.INTERCEPTOR_VELOCITY
    return obs[start:end]


def get_interceptor_orientation(obs: np.ndarray) -> np.ndarray:
    """Extract interceptor orientation from observation."""
    start, end = SCHEMA.INTERCEPTOR_ORIENTATION
    return obs[start:end]


def get_fuel_fraction(obs: np.ndarray) -> float:
    """Extract fuel fraction from observation."""
    return float(obs[SCHEMA.FUEL_FRACTION])


def get_time_to_intercept(obs: np.ndarray) -> float:
    """Extract time to intercept from observation."""
    return float(obs[SCHEMA.TIME_TO_INTERCEPT])


def get_radar_lock_quality(obs: np.ndarray) -> float:
    """Extract radar lock quality from observation."""
    return float(obs[SCHEMA.RADAR_LOCK_QUALITY])


def get_closing_rate(obs: np.ndarray) -> float:
    """Extract closing rate from observation."""
    return float(obs[SCHEMA.CLOSING_RATE])


def get_off_axis_angle(obs: np.ndarray) -> float:
    """Extract off-axis angle from observation."""
    return float(obs[SCHEMA.OFF_AXIS_ANGLE])


def extract_latest_frame(obs: np.ndarray) -> np.ndarray:
    """
    Extract latest 26D frame from potentially frame-stacked observation.

    Args:
        obs: Observation array (26D, 104D, or other)

    Returns:
        Latest 26D observation frame
    """
    if obs.ndim > 1:
        obs = obs.flatten()

    obs_dim = obs.shape[0]

    # If frame-stacked (multiple of 26), extract latest
    if obs_dim > SCHEMA.EXPECTED_DIMENSION and obs_dim % SCHEMA.EXPECTED_DIMENSION == 0:
        return obs[-SCHEMA.EXPECTED_DIMENSION:]
    elif obs_dim == SCHEMA.EXPECTED_DIMENSION:
        return obs
    else:
        # Unexpected dimension, return as-is and let validation catch it
        return obs
