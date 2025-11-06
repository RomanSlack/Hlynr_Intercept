"""
Option switching logic and forced transitions.

Phase 2: Enhanced with hysteresis bands and min-dwell enforcement.
"""
import numpy as np
from typing import Optional, Dict, Any

from .option_definitions import (
    Option,
    FORCED_TRANSITION_THRESHOLDS,
    HYSTERESIS_BANDS,
    MIN_DWELL_STEPS,
    get_expected_duration,
)


class OptionManager:
    """
    Manages option switching logic, including forced transitions.

    Phase 2 enhancements:
        - Hysteresis bands to prevent thrashing near thresholds
        - Min-dwell time enforcement to prevent rapid switching
        - Step tracking for option duration analysis

    Forced transitions occur when environment state requires option change
    (e.g., radar lock lost, close range reached).
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        enable_forced: bool = True,
        enable_hysteresis: bool = True,
        enable_min_dwell: bool = True,
    ):
        """
        Args:
            thresholds: Override default forced transition thresholds
            enable_forced: Enable forced transitions (disable for testing)
            enable_hysteresis: Use hysteresis bands (disable for testing)
            enable_min_dwell: Enforce min-dwell times (disable for testing)
        """
        self.thresholds = thresholds or FORCED_TRANSITION_THRESHOLDS
        self.enable_forced = enable_forced
        self.enable_hysteresis = enable_hysteresis
        self.enable_min_dwell = enable_min_dwell

        # Current option tracking
        self.current_option = Option.SEARCH  # Start in SEARCH
        self.steps_in_current_option = 0

        # Statistics tracking
        self.transition_counts = {
            'forced': 0,
            'selector': 0,
            'timeout': 0,
            'min_dwell_blocked': 0,  # Track how many transitions were blocked
        }

    def get_forced_transition(
        self,
        current_option: Option,
        env_state: Dict[str, Any],
    ) -> Optional[Option]:
        """
        Check if environment state forces an option change.

        Phase 2: Uses hysteresis bands and respects min-dwell times.

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

        # Determine forced transition based on rules
        forced_option = None

        # Rule 1: Lost radar lock → SEARCH
        if current_option in [Option.TRACK, Option.TERMINAL]:
            # Use hysteresis: different threshold for losing lock
            threshold = (
                HYSTERESIS_BANDS['lock_maintain_threshold']
                if self.enable_hysteresis
                else self.thresholds['radar_lock_quality_min']
            )
            if lock_quality < threshold:
                forced_option = Option.SEARCH

        # Rule 2: Acquired lock during search → TRACK
        if current_option == Option.SEARCH:
            # Use hysteresis: higher threshold for acquiring lock
            threshold = (
                HYSTERESIS_BANDS['lock_acquire_threshold']
                if self.enable_hysteresis
                else self.thresholds['radar_lock_quality_search']
            )
            if lock_quality >= threshold:
                forced_option = Option.TRACK

        # Rule 3: Close range → TERMINAL
        if current_option == Option.TRACK:
            # Use hysteresis: tighter threshold for entering terminal
            threshold = (
                HYSTERESIS_BANDS['terminal_enter_distance']
                if self.enable_hysteresis
                else self.thresholds['close_range_threshold']
            )
            if distance < threshold and fuel >= self.thresholds['terminal_fuel_min']:
                forced_option = Option.TERMINAL

        # Rule 4: Out of terminal range → TRACK
        if current_option == Option.TERMINAL:
            # Use hysteresis: wider threshold for exiting terminal
            threshold = (
                HYSTERESIS_BANDS['terminal_exit_distance']
                if self.enable_hysteresis
                else self.thresholds['miss_imminent_distance']
            )
            if distance > threshold:
                forced_option = Option.TRACK

        # No forced transition needed
        if forced_option is None:
            return None

        # Check min-dwell time (prevent rapid switching)
        if self.enable_min_dwell:
            min_dwell = MIN_DWELL_STEPS.get(current_option, 0)
            if self.steps_in_current_option < min_dwell:
                # Allow critical transitions (e.g., lock lost, fuel critical)
                is_critical = self._is_critical_transition(
                    current_option, forced_option, env_state
                )
                if not is_critical:
                    # Block transition, haven't been in option long enough
                    self.transition_counts['min_dwell_blocked'] += 1
                    return None

        # Transition is allowed
        self.transition_counts['forced'] += 1
        return forced_option

    def _is_critical_transition(
        self,
        current_option: Option,
        forced_option: Option,
        env_state: Dict[str, Any],
    ) -> bool:
        """
        Determine if a forced transition is critical (bypass min-dwell).

        Critical transitions:
            - Losing radar lock (safety)
            - Fuel critical (< 10%)
            - Extreme distance changes

        Args:
            current_option: Current option
            forced_option: Proposed forced option
            env_state: Environment state

        Returns:
            True if transition is critical and should bypass min-dwell
        """
        lock_quality = env_state.get('lock_quality', 0.0)
        fuel = env_state.get('fuel', 1.0)

        # Critical: Lock lost (safety-critical)
        if forced_option == Option.SEARCH and current_option != Option.SEARCH:
            if lock_quality < 0.2:  # Very poor lock
                return True

        # Critical: Fuel critical
        if fuel < 0.1:
            return True

        # Not critical
        return False

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
        expected_duration = get_expected_duration(current_option)

        # Allow 2x expected duration before forcing timeout
        max_duration = expected_duration * 2

        if steps_in_option > max_duration:
            self.transition_counts['timeout'] += 1
            return True

        return False

    def update_step_count(self):
        """
        Increment step counter for current option.

        Call this every simulation step to track option duration.
        """
        self.steps_in_current_option += 1

    def record_switch(self, new_option: Option, switch_type: str = 'selector'):
        """
        Record an option switch (reset counters, update stats).

        Args:
            new_option: The option being switched to
            switch_type: Type of switch ('selector', 'forced', 'timeout')
        """
        self.current_option = new_option
        self.steps_in_current_option = 0

        # Update statistics
        if switch_type in self.transition_counts:
            self.transition_counts[switch_type] += 1

    def get_statistics(self) -> Dict[str, int]:
        """Get transition statistics."""
        return self.transition_counts.copy()

    def reset_statistics(self):
        """Reset statistics counters."""
        self.transition_counts = {
            'forced': 0,
            'selector': 0,
            'timeout': 0,
            'min_dwell_blocked': 0,
        }

    def reset(self):
        """Reset manager state (call on episode reset)."""
        self.current_option = Option.SEARCH
        self.steps_in_current_option = 0
