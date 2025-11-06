"""
Convert 26D radar observations to abstract state for high-level policy.
"""
import numpy as np
from typing import Dict


def abstract_observation(full_obs: np.ndarray) -> np.ndarray:
    """
    Convert 26D/104D observation to abstract state for selector policy.

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
    # Handle frame-stacked observations (extract latest frame)
    if len(full_obs.shape) > 1 or full_obs.shape[0] > 26:
        # Assuming frame stacking: last 26 elements are most recent
        obs = full_obs[-26:] if full_obs.ndim == 1 else full_obs.flatten()[-26:]
    else:
        obs = full_obs

    # Observation layout (from core.py):
    # [0-2]:   relative_position (target - interceptor)
    # [3-5]:   relative_velocity
    # [6-8]:   interceptor_velocity
    # [9-11]:  interceptor_orientation (euler)
    # [12]:    fuel_fraction
    # [13]:    time_to_intercept
    # [14]:    radar_lock_quality
    # [15]:    closing_rate
    # [16]:    off_axis_angle

    # 1. Distance to target (normalized by max detection range)
    relative_position = obs[0:3]
    distance = np.linalg.norm(relative_position)
    distance_norm = np.clip(distance / 5000.0, 0, 1)  # 5km max radar range

    # 2. Closing rate (normalized, centered at 0)
    closing_rate = obs[15]
    closing_rate_norm = np.clip(closing_rate / 500.0, -1, 1)  # +/-500 m/s

    # 3. Radar lock quality (already 0-1)
    lock_quality = obs[14]

    # 4. Fuel fraction (already 0-1)
    fuel = obs[12]

    # 5. Off-axis angle (normalized)
    off_axis = obs[16]
    off_axis_norm = np.clip(off_axis / np.pi, -1, 1)  # +/- pi radians

    # 6. Time to intercept estimate (normalized)
    tti = obs[13]
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


def extract_env_state_for_transitions(full_obs: np.ndarray) -> Dict[str, float]:
    """
    Extract environment state features for forced transition checks.

    Args:
        full_obs: 26D or 104D observation

    Returns:
        Dict with keys: 'lock_quality', 'distance', 'fuel', 'closing_rate'
    """
    # Handle frame stacking
    if len(full_obs.shape) > 1 or full_obs.shape[0] > 26:
        obs = full_obs[-26:] if full_obs.ndim == 1 else full_obs.flatten()[-26:]
    else:
        obs = full_obs

    distance = np.linalg.norm(obs[0:3])

    return {
        'lock_quality': float(obs[14]),
        'distance': float(distance),
        'fuel': float(obs[12]),
        'closing_rate': float(obs[15]),
    }
