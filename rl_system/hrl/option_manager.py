"""
Option switching logic and forced transitions.
"""
import numpy as np
from typing import Optional, Dict, Any

from .option_definitions import Option, FORCED_TRANSITION_THRESHOLDS


class OptionManager:
    """
    Manages option switching logic, including forced transitions.

    Forced transitions occur when environment state requires option change
    (e.g., radar lock lost, close range reached).
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        enable_forced: bool = True,
    ):
        """
        Args:
            thresholds: Override default forced transition thresholds
            enable_forced: Enable forced transitions (disable for testing)
        """
        self.thresholds = thresholds or FORCED_TRANSITION_THRESHOLDS
        self.enable_forced = enable_forced

        # Statistics tracking
        self.transition_counts = {
            'forced': 0,
            'selector': 0,
            'timeout': 0,
        }

    def get_forced_transition(
        self,
        current_option: Option,
        env_state: Dict[str, Any],
    ) -> Optional[Option]:
        """
        Check if environment state forces an option change.

        Args:
            current_option: Current active option
            env_state: Environment state dict with keys:
                - lock_quality: float [0, 1]
                - distance: float (meters)
                - fuel: float [0, 1]
                - closing_rate: float (m/s)

        Returns:
            Required option, or None if no forced transition
        """
        if not self.enable_forced:
            return None

        lock_quality = env_state.get('lock_quality', 0.0)
        distance = env_state.get('distance', float('inf'))
        fuel = env_state.get('fuel', 1.0)

        # Rule 1: Lost radar lock → SEARCH
        if current_option in [Option.TRACK, Option.TERMINAL]:
            if lock_quality < self.thresholds['radar_lock_quality_min']:
                self.transition_counts['forced'] += 1
                return Option.SEARCH

        # Rule 2: Acquired lock during search → TRACK
        if current_option == Option.SEARCH:
            if lock_quality >= self.thresholds['radar_lock_quality_search']:
                self.transition_counts['forced'] += 1
                return Option.TRACK

        # Rule 3: Close range → TERMINAL
        if current_option == Option.TRACK:
            if distance < self.thresholds['close_range_threshold']:
                if fuel >= self.thresholds['terminal_fuel_min']:
                    self.transition_counts['forced'] += 1
                    return Option.TERMINAL

        # Rule 4: Out of terminal range → TRACK
        if current_option == Option.TERMINAL:
            if distance > self.thresholds['miss_imminent_distance']:
                # Miss imminent, return to tracking
                self.transition_counts['forced'] += 1
                return Option.TRACK

        return None

    def should_timeout_option(
        self,
        current_option: Option,
        steps_in_option: int,
    ) -> bool:
        """
        Check if option has been active too long (timeout).

        Args:
            current_option: Current option
            steps_in_option: Steps since option started

        Returns:
            True if option should timeout
        """
        # Get expected duration
        from .option_definitions import get_expected_duration
        expected_duration = get_expected_duration(current_option)

        # Allow 2x expected duration before forcing timeout
        max_duration = expected_duration * 2

        if steps_in_option > max_duration:
            self.transition_counts['timeout'] += 1
            return True

        return False

    def get_statistics(self) -> Dict[str, int]:
        """Get transition statistics."""
        return self.transition_counts.copy()

    def reset_statistics(self):
        """Reset statistics counters."""
        self.transition_counts = {
            'forced': 0,
            'selector': 0,
            'timeout': 0,
        }
