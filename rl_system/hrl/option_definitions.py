"""
Shared constants and enumerations for HRL options.
"""
from enum import IntEnum
from typing import Dict, Any


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
    'close_range_threshold': 100.0,     # meters - trigger TERMINAL
    'terminal_fuel_min': 0.1,           # Min fuel to enter terminal
    'miss_imminent_distance': 200.0,    # If distance increasing beyond this, abort
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
