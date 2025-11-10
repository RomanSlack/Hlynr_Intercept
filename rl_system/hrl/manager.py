"""
Hierarchical Manager coordinates selector and specialists.

Phase 2: Enhanced with debugging metadata and proper coordination.
"""
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .option_definitions import Option, FORCED_TRANSITION_THRESHOLDS
from .selector_policy import SelectorPolicy
from .specialist_policies import SpecialistPolicy
from .option_manager import OptionManager
from .observation_abstraction import abstract_observation


@dataclass
class HRLState:
    """Current state of the hierarchical controller."""
    current_option: Option
    prev_option: Option  # Track previous option for debugging
    steps_in_option: int
    lstm_states: Dict[Option, Any]  # Per-specialist LSTM states
    last_decision_step: int
    total_steps: int


class HierarchicalManager:
    """
    Coordinates selector and specialists during inference/training.

    Phase 2: Enhanced with detailed metadata for debugging and analysis.
    """

    def __init__(
        self,
        selector: Optional[SelectorPolicy] = None,
        specialists: Optional[Dict[Option, SpecialistPolicy]] = None,
        action_dim: int = 6,
        decision_interval: int = 100,
        enable_forced_transitions: bool = True,
        enable_hysteresis: bool = True,
        enable_min_dwell: bool = True,
        default_option: Option = Option.SEARCH,
    ):
        """
        Args:
            selector: SelectorPolicy instance (or None for default "rules" mode)
            specialists: Dict of SpecialistPolicy instances (or None for stubs)
            action_dim: Action space dimension (6D continuous)
            decision_interval: Steps between high-level decisions (typically 100)
            enable_forced_transitions: Allow forced option changes
            enable_hysteresis: Use hysteresis bands in option manager
            enable_min_dwell: Enforce min-dwell times
            default_option: Initial option on reset
        """
        self.action_dim = action_dim
        self.decision_interval = decision_interval
        self.enable_forced_transitions = enable_forced_transitions
        self.default_option = default_option

        # Create policies (use provided or defaults)
        self.selector = selector if selector is not None else SelectorPolicy(mode="rules")
        self.specialists = specialists if specialists is not None else {
            Option.SEARCH: SpecialistPolicy(Option.SEARCH),
            Option.TRACK: SpecialistPolicy(Option.TRACK),
            Option.TERMINAL: SpecialistPolicy(Option.TERMINAL),
        }

        # Option switching logic
        self.option_manager = OptionManager(
            thresholds=FORCED_TRANSITION_THRESHOLDS,
            enable_forced=enable_forced_transitions,
            enable_hysteresis=enable_hysteresis,
            enable_min_dwell=enable_min_dwell,
        )

        # State tracking
        self.state = None
        self.reset()

    def _due_for_selector(self) -> bool:
        """
        Return True exactly when the selector should be called.
        With total_steps incremented at the end of select_action(),
        the 10th call after reset should trigger if decision_interval=10.
        """
        return (self.state.total_steps + 1) % self.decision_interval == 0

    def reset(self):
        """Reset to initial state."""
        self.state = HRLState(
            current_option=self.default_option,
            prev_option=self.default_option,  # Same as current initially
            steps_in_option=0,
            lstm_states={opt: None for opt in Option},
            last_decision_step=0,
            total_steps=0,
        )

        # Reset option manager
        self.option_manager.reset()

        # Reset all specialist LSTM states
        for specialist in self.specialists.values():
            if hasattr(specialist, 'reset_lstm'):
                specialist.reset_lstm()

    def select_action(
        self,
        full_obs: np.ndarray,
        env_state: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Select action using hierarchical policy.

        Phase 2: Returns detailed metadata for debugging.

        Args:
            full_obs: 26D or 104D observation
            env_state: Environment state for forced transitions (optional)
            deterministic: Use deterministic policies

        Returns:
            action: 6D continuous action from active specialist
            info: Enhanced debug information with option switching details
        """
        # Compute abstract state for selector (useful for debugging)
        abstract_state = abstract_observation(full_obs)

        # 1. Check for forced transitions
        forced_option = None
        if self.enable_forced_transitions and env_state is not None:
            forced_option = self.option_manager.get_forced_transition(
                current_option=self.state.current_option,
                env_state=env_state,
            )

        # 2. Check if selector should decide
        option_switched = False
        switch_reason = "continue"
        selector_choice = None

        if forced_option is not None:
            new_option = forced_option
            option_switched = (new_option != self.state.current_option)
            switch_reason = "forced"
        elif self.selector is not None and self._due_for_selector():
            # Selector decision at cadence; feed 7D abstract state
            selector_choice = int(self.selector.predict(abstract_state, deterministic))
            # Clamp to valid option range [0, 1, 2] in case model outputs invalid values
            selector_choice = max(0, min(2, selector_choice))
            new_option = Option(selector_choice)
            option_switched = (new_option != self.state.current_option)
            switch_reason = "selector" if option_switched else "continue"
            # Record when we made the decision (optional but helpful for info/debug)
            self.state.last_decision_step = self.state.total_steps + 1
        else:
            new_option = self.state.current_option
            switch_reason = "continue"

        # 3. Handle option switching
        prev_option = self.state.current_option
        if option_switched:
            self._switch_option(new_option, switch_reason)

        # 4. Execute specialist
        active_specialist = self.specialists[self.state.current_option]
        action = active_specialist.predict(full_obs, deterministic=deterministic)

        # 5. Update state and option manager
        self.state.steps_in_option += 1
        self.state.total_steps += 1
        self.option_manager.update_step_count()

        # 6. Build enhanced info dict
        info = {
            # Core option info
            'hrl/option': self.state.current_option.name,
            'hrl/option_index': int(self.state.current_option),
            'hrl/steps_in_option': self.state.steps_in_option,

            # Switch information
            'hrl/option_switched': option_switched,
            'hrl/switch_reason': switch_reason,
            'hrl/prev_option': prev_option.name,
            'hrl/new_option': new_option.name if option_switched else self.state.current_option.name,

            # Transition details
            'hrl/forced_transition': forced_option is not None,
            'hrl/selector_choice': selector_choice if selector_choice is not None else int(self.state.current_option),

            # State snapshots (for debugging)
            'hrl/abstract_state': abstract_state.tolist(),  # 7D vector
            'hrl/env_state': env_state if env_state is not None else {},

            # Timing
            'hrl/total_steps': self.state.total_steps,
            'hrl/last_decision_step': self.state.last_decision_step,
        }

        return action, info

    def should_switch_option(self) -> bool:
        """Check if it's time for selector decision."""
        return (self.state.total_steps - self.state.last_decision_step) >= self.decision_interval

    def _switch_option(self, new_option: Option, switch_reason: str = 'selector'):
        """Handle option switching."""
        old_option = self.state.current_option

        # Reset old specialist LSTM
        if hasattr(self.specialists[old_option], 'reset_lstm'):
            self.specialists[old_option].reset_lstm()

        # Update state
        self.state.prev_option = old_option
        self.state.current_option = new_option
        self.state.steps_in_option = 0
        self.state.last_decision_step = self.state.total_steps

        # Notify option manager of switch
        self.option_manager.record_switch(new_option, switch_type=switch_reason)

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'current_option': self.state.current_option.name,
            'prev_option': self.state.prev_option.name,
            'total_steps': self.state.total_steps,
            'steps_in_current_option': self.state.steps_in_option,
            'transition_stats': self.option_manager.get_statistics(),
        }
