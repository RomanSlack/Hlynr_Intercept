"""Basic tests for the Aegis3DInterceptEnv environment."""

import pytest
import numpy as np
import gymnasium as gym
from aegis_intercept.envs.aegis_3d_env import Aegis3DInterceptEnv

class TestAegis3DInterceptEnv:
    """Test cases for the 3D intercept environment."""

    def test_env_creation(self):
        """Test that environment can be created successfully."""
        env = Aegis3DInterceptEnv()
        assert env is not None
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)

    def test_observation_space(self):
        """Test observation space properties."""
        env = Aegis3DInterceptEnv(world_size=100.0)
        obs_space = env.observation_space
        assert obs_space.shape == (13,)
        assert obs_space.dtype == np.float32

    def test_action_space(self):
        """Test action space properties."""
        env = Aegis3DInterceptEnv()
        action_space = env.action_space
        assert action_space.shape == (3,)
        assert action_space.dtype == np.float32
        assert np.all(action_space.low == -1.0)
        assert np.all(action_space.high == 1.0)

    def test_reset(self):
        """Test that reset returns valid observation."""
        env = Aegis3DInterceptEnv()
        observation, info = env.reset(seed=42)
        assert observation.shape == env.observation_space.shape
        assert observation.dtype == np.float32
        assert env.observation_space.contains(observation)
        assert isinstance(info, dict)

    def test_step(self):
        """Test that step progresses environment correctly."""
        env = Aegis3DInterceptEnv()
        env.reset(seed=42)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        assert observation.shape == env.observation_space.shape
        assert observation.dtype == np.float32
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert env.observation_space.contains(observation)
