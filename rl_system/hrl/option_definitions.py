"""
Shared constants and enumerations for HRL options.
"""
from enum import IntEnum
from typing import Dict, Any, Optional
import yaml
from pathlib import Path


class Option(IntEnum):
    """High-level options for missile defense."""
    SEARCH = 0      # Wide-area scanning for target acquisition
    TRACK = 1       # Maintain lock and close distance
    TERMINAL = 2    # Final intercept guidance


# Option metadata
OPTION_METADATA: Dict[Option, Dict[str, Any]] = {
    Option.SEARCH: {
        'name': 'Search',
        'description': 'Wide-area scanning, large angular changes',
        'expected_duration': 200,  # steps
        'forced_exit_conditions': ['radar_lock_acquired'],
        'color': '#FF6B6B',  # Red for visualization
        'default_reward_weight': 1.0,
    },
    Option.TRACK: {
        'name': 'Track',
        'description': 'Maintain radar lock, approach target',
        'expected_duration': 500,
        'forced_exit_conditions': ['radar_lock_lost', 'close_range_threshold'],
        'color': '#4ECDC4',  # Teal
        'default_reward_weight': 1.0,
    },
    Option.TERMINAL: {
        'name': 'Terminal',
        'description': 'Final guidance, high precision',
        'expected_duration': 100,
        'forced_exit_conditions': ['intercept_complete', 'miss_imminent'],
        'color': '#95E1D3',  # Light green
        'default_reward_weight': 2.0,  # Higher weight for critical phase
    },
}


# Physical thresholds for forced transitions
FORCED_TRANSITION_THRESHOLDS = {
    'radar_lock_quality_min': 0.3,      # Below this, lose track → SEARCH
    'radar_lock_quality_search': 0.7,   # Above this, exit search → TRACK
    'close_range_threshold': 200.0,     # meters - trigger TERMINAL (increased to match long-range training)
    'terminal_fuel_min': 0.1,           # Min fuel to enter terminal
    'miss_imminent_distance': 400.0,    # If distance increasing beyond this, abort (increased for long-range)
}


def get_option_name(option: Option) -> str:
    """Get human-readable name for option."""
    return OPTION_METADATA[option]['name']


def get_option_color(option: Option) -> str:
    """Get color for visualization."""
    return OPTION_METADATA[option]['color']


def get_expected_duration(option: Option) -> int:
    """Get expected duration in steps."""
    return OPTION_METADATA[option]['expected_duration']


# Hysteresis bands for forced transitions (prevents thrashing near thresholds)
HYSTERESIS_BANDS = {
    'lock_acquire_threshold': 0.75,      # Enter TRACK (higher than basic threshold)
    'lock_maintain_threshold': 0.55,     # Exit TRACK (lower, provides hysteresis)
    'terminal_enter_distance': 200.0,    # Enter TERMINAL (meters) - matches long-range training
    'terminal_exit_distance': 250.0,     # Exit TERMINAL (meters, provides hysteresis)
}


# Minimum dwell times per option (prevents rapid switching)
MIN_DWELL_STEPS = {
    Option.SEARCH: 50,      # At least 0.5 seconds at 100Hz
    Option.TRACK: 50,       # At least 0.5 seconds
    Option.TERMINAL: 30,    # At least 0.3 seconds (shorter for time-critical phase)
}


def load_thresholds_from_config(config_path: str) -> Dict[str, Any]:
    """
    Load transition thresholds from HRL configuration file.

    Args:
        config_path: Path to YAML config file (e.g., 'configs/hrl/hrl_base.yaml')

    Returns:
        Dictionary containing threshold values

    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If required keys are missing from config
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Extract HRL section
    if 'hrl' not in config:
        raise KeyError(f"'hrl' section not found in {config_path}")

    hrl_config = config['hrl']

    # Extract transitions section
    if 'transitions' in hrl_config:
        return hrl_config['transitions']
    else:
        # Fall back to default thresholds
        return FORCED_TRANSITION_THRESHOLDS


def get_min_dwell(option: Option) -> int:
    """Get minimum dwell time for an option."""
    return MIN_DWELL_STEPS[option]
