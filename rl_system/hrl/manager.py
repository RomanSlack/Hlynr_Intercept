"""
Hierarchical Manager coordinates selector and specialists.

Phase 1 stub: Basic coordination logic with no-op specialists.
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
    steps_in_option: int
    lstm_states: Dict[Option, Any]  # Per-specialist LSTM states
    last_decision_step: int
    total_steps: int


class HierarchicalManager:
    """
    Coordinates selector and specialists during inference/training.

    Phase 1 stub: Manages option switching with no-op specialists.
    """

    def __init__(
        self,
        action_dim: int = 6,  # For stub: just need action dimension
        decision_interval: int = 100,
        enable_forced_transitions: bool = True,
        default_option: Option = Option.SEARCH,
    ):
        """
        Args:
            action_dim: Action space dimension (6D continuous)
            decision_interval: Steps between high-level decisions
            enable_forced_transitions: Allow forced option changes
            default_option: Initial option on reset
        """
        self.action_dim = action_dim
        self.decision_interval = decision_interval
        self.enable_forced_transitions = enable_forced_transitions
        self.default_option = default_option

        # Create stub policies
        self.selector = SelectorPolicy(mode="fixed")
        self.specialists = {
            Option.SEARCH: SpecialistPolicy(Option.SEARCH),
            Option.TRACK: SpecialistPolicy(Option.TRACK),
            Option.TERMINAL: SpecialistPolicy(Option.TERMINAL),
        }

        # Option switching logic
        self.option_manager = OptionManager(
            thresholds=FORCED_TRANSITION_THRESHOLDS,
            enable_forced=enable_forced_transitions,
        )

        # State tracking
        self.state = None
        self.reset()

    def reset(self):
        """Reset to initial state."""
        self.state = HRLState(
            current_option=self.default_option,
            steps_in_option=0,
            lstm_states={opt: None for opt in Option},
            last_decision_step=0,
            total_steps=0,
        )

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
        Select action using hierarchical policy (stub: returns zeros).

        Args:
            full_obs: 26D or 104D observation
            env_state: Environment state for forced transitions
            deterministic: Use deterministic policies

        Returns:
            action: 6D continuous action (zeros from specialist stub)
            info: Debug information
        """
        # 1. Check for forced transitions
        forced_option = None
        if self.enable_forced_transitions and env_state is not None:
            forced_option = self.option_manager.get_forced_transition(
                current_option=self.state.current_option,
                env_state=env_state,
            )

        # 2. Check if selector should decide
        option_switched = False
        if forced_option is not None:
            new_option = forced_option
            option_switched = True
            switch_reason = "forced"
        elif self.should_switch_option():
            # Selector decision
            abstract_state = abstract_observation(full_obs)
            new_option_idx = self.selector.predict(abstract_state, deterministic)
            new_option = Option(new_option_idx)
            option_switched = (new_option != self.state.current_option)
            switch_reason = "selector"
        else:
            new_option = self.state.current_option
            switch_reason = "continue"

        # 3. Handle option switching
        if option_switched:
            self._switch_option(new_option)

        # 4. Execute specialist (stub: returns zeros)
        active_specialist = self.specialists[self.state.current_option]
        action = active_specialist.predict(full_obs, deterministic)

        # 5. Update state
        self.state.steps_in_option += 1
        self.state.total_steps += 1

        # 6. Build info
        info = {
            'hrl/option': self.state.current_option.name,
            'hrl/option_index': int(self.state.current_option),
            'hrl/steps_in_option': self.state.steps_in_option,
            'hrl/option_switched': option_switched,
            'hrl/switch_reason': switch_reason,
            'hrl/forced_transition': forced_option is not None,
        }

        return action, info

    def should_switch_option(self) -> bool:
        """Check if it's time for selector decision."""
        return (self.state.total_steps - self.state.last_decision_step) >= self.decision_interval

    def _switch_option(self, new_option: Option):
        """Handle option switching."""
        old_option = self.state.current_option

        # Reset old specialist LSTM
        if hasattr(self.specialists[old_option], 'reset_lstm'):
            self.specialists[old_option].reset_lstm()

        # Update state
        self.state.current_option = new_option
        self.state.steps_in_option = 0
        self.state.last_decision_step = self.state.total_steps

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'current_option': self.state.current_option.name,
            'total_steps': self.state.total_steps,
            'steps_in_current_option': self.state.steps_in_option,
            'transition_stats': self.option_manager.get_statistics(),
        }
