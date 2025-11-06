"""
Convert 26D radar observations to abstract state for high-level policy.
"""
import numpy as np
from typing import Dict, Optional, Any

from .observation_schema import (
    SCHEMA,
    extract_latest_frame,
    get_relative_position,
    get_fuel_fraction,
    get_time_to_intercept,
    get_radar_lock_quality,
    get_closing_rate,
    get_off_axis_angle,
)


def abstract_observation(full_obs: np.ndarray) -> np.ndarray:
    """
    Convert 26D/104D observation to abstract state for selector policy.

    Uses centralized schema from observation_schema.py for index safety.

    Args:
        full_obs: Either 26D base observation or 104D frame-stacked (26*4)

    Returns:
        abstract_state: 7D vector:
            [0] distance_to_target (normalized to [0, 1])
            [1] closing_rate (normalized to [-1, 1])
            [2] radar_lock_quality (0-1)
            [3] fuel_fraction (0-1)
            [4] off_axis_angle (normalized to [-1, 1])
            [5] time_to_intercept_estimate (normalized to [0, 1])
            [6] relative_altitude (normalized to [-1, 1])
    """
    # Extract latest frame using schema-aware function
    obs = extract_latest_frame(full_obs)

    # 1. Distance to target (normalized by max detection range)
    relative_position = get_relative_position(obs)
    distance = np.linalg.norm(relative_position)
    distance_norm = np.clip(distance / 5000.0, 0, 1)  # 5km max radar range

    # 2. Closing rate (normalized, centered at 0)
    closing_rate = get_closing_rate(obs)
    closing_rate_norm = np.clip(closing_rate / 500.0, -1, 1)  # +/-500 m/s

    # 3. Radar lock quality (already 0-1)
    lock_quality = get_radar_lock_quality(obs)

    # 4. Fuel fraction (already 0-1)
    fuel = get_fuel_fraction(obs)

    # 5. Off-axis angle (normalized)
    off_axis = get_off_axis_angle(obs)
    off_axis_norm = np.clip(off_axis / np.pi, -1, 1)  # +/- pi radians

    # 6. Time to intercept estimate (normalized)
    tti = get_time_to_intercept(obs)
    tti_norm = np.clip(tti / 10.0, 0, 1)  # 0-10 seconds

    # 7. Relative altitude (derived from position)
    rel_altitude = relative_position[2]  # Z component
    altitude_norm = np.clip(rel_altitude / 1000.0, -1, 1)  # +/-1km

    abstract_state = np.array([
        distance_norm,
        closing_rate_norm,
        lock_quality,
        fuel,
        off_axis_norm,
        tti_norm,
        altitude_norm,
    ], dtype=np.float32)

    return abstract_state


def extract_env_state_for_transitions(
    full_obs: np.ndarray,
    env_info: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Extract environment state features for forced transition checks.

    Prefers high-fidelity values from env_info when available, falls back to
    observation-derived values otherwise (Risk #7 mitigation).

    Args:
        full_obs: 26D or 104D observation
        env_info: Optional info dict from environment step

    Returns:
        Dict with keys: 'lock_quality', 'distance', 'fuel', 'closing_rate'
    """
    # Extract latest frame using schema
    obs = extract_latest_frame(full_obs)

    # Use env_info values when available for higher fidelity
    if env_info is not None:
        lock_quality = env_info.get('radar_lock_quality', get_radar_lock_quality(obs))
        fuel = env_info.get('fuel_fraction', get_fuel_fraction(obs))
        closing_rate = env_info.get('closing_rate', get_closing_rate(obs))

        # Distance might be in env_info as actual_distance
        if 'actual_distance' in env_info:
            distance = float(env_info['actual_distance'])
        elif 'miss_distance' in env_info:
            # Use miss distance as proxy if available
            distance = float(env_info['miss_distance'])
        else:
            # Fall back to observation
            relative_position = get_relative_position(obs)
            distance = float(np.linalg.norm(relative_position))
    else:
        # Fall back to observation-derived values
        relative_position = get_relative_position(obs)
        distance = float(np.linalg.norm(relative_position))
        lock_quality = get_radar_lock_quality(obs)
        fuel = get_fuel_fraction(obs)
        closing_rate = get_closing_rate(obs)

    return {
        'lock_quality': float(lock_quality),
        'distance': float(distance),
        'fuel': float(fuel),
        'closing_rate': float(closing_rate),
    }


def is_valid_abstract_state(abstract_state: np.ndarray) -> bool:
    """
    Validate abstract state dimensions and value ranges.

    Args:
        abstract_state: 7D abstract state vector

    Returns:
        True if valid

    Raises:
        ValueError: If abstract state is invalid
    """
    if abstract_state.shape != (7,):
        raise ValueError(f"Expected 7D abstract state, got {abstract_state.shape}")

    # Check all values are finite
    if not np.all(np.isfinite(abstract_state)):
        raise ValueError("Abstract state contains non-finite values")

    # Check normalization bounds
    # Distance [0], fuel [3], TTI [5] should be in [0, 1]
    for idx in [0, 3, 5]:
        if not (0 <= abstract_state[idx] <= 1):
            raise ValueError(
                f"Abstract state[{idx}] = {abstract_state[idx]} out of [0, 1] range"
            )

    # Lock quality [2] should be in [0, 1]
    if not (0 <= abstract_state[2] <= 1):
        raise ValueError(
            f"Lock quality = {abstract_state[2]} out of [0, 1] range"
        )

    # Closing rate [1], off-axis [4], altitude [6] should be in [-1, 1]
    for idx in [1, 4, 6]:
        if not (-1 <= abstract_state[idx] <= 1):
            raise ValueError(
                f"Abstract state[{idx}] = {abstract_state[idx]} out of [-1, 1] range"
            )

    return True
