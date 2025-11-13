"""
Gymnasium wrappers for hierarchical RL.

CRITICAL: This wrapper manages observation persistence for HRL manager.
The manager needs the current observation to make decisions, but Gymnasium
doesn't provide it during step(). We store _last_obs to solve this.
"""
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional

from .manager import HierarchicalManager
from .observation_abstraction import extract_env_state_for_transitions, abstract_observation
from .selector_policy import Option


class HRLActionWrapper(gym.Wrapper):
    """
    Wrapper to use HierarchicalManager for action selection.

    Supports two modes:
    1. Autonomous mode (selector=internal): Manager uses its own selector policy
    2. Training mode (selector=None): Accepts discrete option indices from external selector
    """

    def __init__(
        self,
        env: gym.Env,
        manager: HierarchicalManager,
        return_hrl_info: bool = True,
        training_selector: bool = False,
    ):
        """
        Args:
            env: Base InterceptEnvironment
            manager: HierarchicalManager instance
            return_hrl_info: Include HRL info in step() info dict
            training_selector: If True, wrapper expects discrete option indices as actions.
                             If False, manager uses its own selector (autonomous mode).
        """
        super().__init__(env)
        self.manager = manager
        self.return_hrl_info = return_hrl_info
        self.training_selector = training_selector
        self._last_obs = None

        # Track transition statistics for selector training mode
        if training_selector:
            self._num_switches = 0
            self._forced_transitions = 0
            self._selector_decisions = 0

        # Override action space for selector training mode
        if training_selector:
            self.action_space = gym.spaces.Discrete(3)  # SEARCH, TRACK, TERMINAL

    def reset(self, **kwargs):
        """Reset environment and HRL manager."""
        obs, info = self.env.reset(**kwargs)
        self.manager.reset()
        self._last_obs = obs

        # Reset training statistics
        if self.training_selector:
            self._num_switches = 0
            self._forced_transitions = 0
            self._selector_decisions = 0

        return obs, info

    def step(self, action=None):
        """
        Step with hierarchical action selection.

        Args:
            action: If training_selector=True, expects discrete option index (0, 1, or 2).
                   If training_selector=False, action is ignored (manager selects internally).

        Returns:
            Standard Gymnasium step output with HRL info

        Note:
            Uses stored _last_obs for manager decision. After env step,
            updates _last_obs with next_obs for the next iteration.
        """
        # CRITICAL: Use stored observation from previous step
        # This is the current observation needed for manager decision
        if self._last_obs is None:
            raise RuntimeError(
                "HRLActionWrapper.step() called before reset(). "
                "Must call reset() first to initialize observation."
            )

        obs = self._last_obs

        # Handle frame-stacked observations: flatten (4, 26) → (104,)
        # FrameStackObservation returns shape (stack_size, obs_dim) which needs flattening
        # for specialists that expect (104,) flattened observations
        if obs.ndim > 1:
            obs = obs.flatten()

        # Get environment state for forced transitions
        env_state = extract_env_state_for_transitions(obs, env_info=None)

        if self.training_selector:
            # TRAINING MODE: Use external selector's discrete action
            if action is None:
                raise ValueError(
                    "training_selector=True but no action provided. "
                    "Expected discrete option index (0, 1, or 2)."
                )

            # Convert discrete index to Option enum
            option_index = int(action)
            if option_index not in [0, 1, 2]:
                raise ValueError(f"Invalid option index: {option_index}. Expected 0, 1, or 2.")

            selected_option = Option(option_index)

            # Check for forced transitions (override selector choice if necessary)
            forced_option = None
            if self.manager.enable_forced_transitions:
                forced_option = self.manager.option_manager.get_forced_transition(
                    current_option=self.manager.state.current_option,
                    env_state=env_state,
                )

            if forced_option is not None:
                selected_option = forced_option
                forced_transition = True
            else:
                forced_transition = False

            # Update manager state and track statistics
            if self.manager.state.current_option != selected_option:
                self._num_switches += 1
                if forced_transition:
                    self._forced_transitions += 1
                else:
                    self._selector_decisions += 1

            self.manager.state.current_option = selected_option
            self.manager.state.steps_in_option = 0

            # Get action from the selected specialist
            specialist = self.manager.specialists[selected_option]
            specialist_action = specialist.predict(obs, deterministic=True)

            # Create HRL info
            hrl_info = {
                'hrl/option': selected_option.name,
                'hrl/option_index': selected_option.value,
                'hrl/forced_transition': forced_transition,
                'hrl/num_switches': self._num_switches,
                'hrl/forced_transitions': self._forced_transitions,
                'hrl/selector_decisions': self._selector_decisions,
            }

            final_action = specialist_action
        else:
            # AUTONOMOUS MODE: Manager uses its own selector
            final_action, hrl_info = self.manager.select_action(obs, env_state)

        # Execute in base environment
        next_obs, reward, terminated, truncated, info = self.env.step(final_action)

        # CRITICAL: Update stored observation for next step
        self._last_obs = next_obs

        # Add HRL info
        if self.return_hrl_info:
            info.update(hrl_info)

        return next_obs, reward, terminated, truncated, info


class AbstractObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to convert 26D/104D observations to abstract 7D state for selector training.

    This wrapper is used when training the selector policy, which operates on
    abstract strategic information rather than raw sensor data.
    """

    def __init__(self, env: gym.Env):
        """
        Args:
            env: Base environment (should output 26D base or 104D frame-stacked obs)
        """
        super().__init__(env)

        # Override observation space to abstract state dimension
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),  # Abstract state dimension
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Convert full observation to abstract state."""
        return abstract_observation(obs)
