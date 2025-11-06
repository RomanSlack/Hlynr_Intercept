"""
Gymnasium wrappers for hierarchical RL.

Phase 1 stub: Basic wrapper structure with pass-through.
"""
import numpy as np
import gymnasium as gym
from typing import Dict, Any, Optional

from .manager import HierarchicalManager
from .observation_abstraction import extract_env_state_for_transitions


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
        Step with hierarchical action selection (stub: ignores input action).

        Args:
            action: Ignored - actions come from HRL manager

        Returns:
            Standard Gymnasium step output with HRL info
        """
        # Get action from manager
        obs = self._last_obs if self._last_obs is not None else np.zeros(self.observation_space.shape, dtype=np.float32)
        env_state = extract_env_state_for_transitions(obs)
        action, hrl_info = self.manager.select_action(obs, env_state)

        # Execute in base environment
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = next_obs

        # Add HRL info
        if self.return_hrl_info:
            info.update(hrl_info)

        return next_obs, reward, terminated, truncated, info
