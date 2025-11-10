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


class HRLActionWrapper(gym.Wrapper):
    """
    Wrapper to use HierarchicalManager for action selection.

    Phase 1 stub: Ignores incoming actions and uses HRL manager (which returns zeros).
    """

    def __init__(
        self,
        env: gym.Env,
        manager: HierarchicalManager,
        return_hrl_info: bool = True,
    ):
        """
        Args:
            env: Base InterceptEnvironment
            manager: HierarchicalManager instance
            return_hrl_info: Include HRL info in step() info dict
        """
        super().__init__(env)
        self.manager = manager
        self.return_hrl_info = return_hrl_info
        self._last_obs = None

    def reset(self, **kwargs):
        """Reset environment and HRL manager."""
        obs, info = self.env.reset(**kwargs)
        self.manager.reset()
        self._last_obs = obs
        return obs, info

    def step(self, action=None):
        """
        Step with hierarchical action selection.

        Args:
            action: Ignored - actions come from HRL manager

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

        # Get action from manager - will use env_state for forced transitions
        # (env_info will be passed after we have it from env.step)
        env_state = extract_env_state_for_transitions(obs, env_info=None)
        action, hrl_info = self.manager.select_action(obs, env_state)

        # Execute in base environment
        next_obs, reward, terminated, truncated, info = self.env.step(action)

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
