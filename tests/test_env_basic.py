"""Basic tests for the Aegis2DInterceptEnv environment."""

import pytest
import numpy as np
import gymnasium as gym
from aegis_intercept.envs import Aegis2DInterceptEnv


class TestAegis2DInterceptEnv:
    """Test cases for the 2D intercept environment."""
    
    def test_env_creation(self):
        """Test that environment can be created successfully."""
        env = Aegis2DInterceptEnv()
        assert env is not None
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)
    
    def test_observation_space(self):
        """Test observation space properties."""
        env = Aegis2DInterceptEnv(world_size=100.0)
        obs_space = env.observation_space
        
        # Should be 8-dimensional: [interceptor_pos(2), interceptor_vel(2), missile_pos(2), missile_vel(2)]
        assert obs_space.shape == (8,)
        assert obs_space.dtype == np.float32
        
        # Bounds should be symmetric around world size
        assert np.all(obs_space.low == -100.0)
        assert np.all(obs_space.high == 100.0)
    
    def test_action_space(self):
        """Test action space properties."""
        env = Aegis2DInterceptEnv()
        action_space = env.action_space
        
        # Should be 2-dimensional: [delta_vx, delta_vy]
        assert action_space.shape == (2,)
        assert action_space.dtype == np.float32
        
        # Actions should be in [-1, 1] range
        assert np.all(action_space.low == -1.0)
        assert np.all(action_space.high == 1.0)
    
    def test_reset(self):
        """Test that reset returns valid observation."""
        env = Aegis2DInterceptEnv()
        observation, info = env.reset(seed=42)
        
        # Check observation shape and type
        assert observation.shape == env.observation_space.shape
        assert observation.dtype == np.float32
        
        # Check that observation is within bounds
        assert env.observation_space.contains(observation)
        
        # Check info dictionary
        assert isinstance(info, dict)
        assert "interceptor_pos" in info
        assert "missile_pos" in info
        assert "defended_point" in info
        assert "step_count" in info
        
        # Step count should be reset
        assert info["step_count"] == 0
    
    def test_step(self):
        """Test that step progresses environment correctly."""
        env = Aegis2DInterceptEnv()
        env.reset(seed=42)
        
        # Sample a random action
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Check return types and shapes
        assert observation.shape == env.observation_space.shape
        assert observation.dtype == np.float32
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Check that observation is still within bounds
        assert env.observation_space.contains(observation)
        
        # Step count should have incremented
        assert info["step_count"] == 1
    
    def test_multiple_steps(self):
        """Test multiple consecutive steps."""
        env = Aegis2DInterceptEnv(max_steps=10)
        env.reset(seed=42)
        
        for i in range(5):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            assert env.observation_space.contains(observation)
            assert info["step_count"] == i + 1
            
            if terminated or truncated:
                break
    
    def test_deterministic_reset(self):
        """Test that reset with same seed produces same initial state."""
        env1 = Aegis2DInterceptEnv()
        env2 = Aegis2DInterceptEnv()
        
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        # Should be identical with same seed
        np.testing.assert_array_equal(obs1, obs2)
    
    def test_reward_bounds(self):
        """Test that rewards are within expected ranges."""
        env = Aegis2DInterceptEnv()
        env.reset(seed=42)
        
        rewards = []
        for _ in range(20):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            
            if terminated or truncated:
                break
        
        # Most step rewards should be small and negative (efficiency penalty)
        # Terminal rewards can be -1 (miss) or +1 (intercept) or -0.5 (out of bounds)
        for reward in rewards[:-1]:  # Exclude potential terminal reward
            assert reward <= 0.01  # Small positive rewards for progress are allowed
            assert reward >= -0.02  # Step penalty shouldn't be too large
    
    def test_action_clipping(self):
        """Test that actions outside [-1,1] are handled properly."""
        env = Aegis2DInterceptEnv()
        env.reset(seed=42)
        
        # Test extreme actions
        extreme_action = np.array([10.0, -10.0])
        observation, reward, terminated, truncated, info = env.step(extreme_action)
        
        # Should not crash and should return valid observation
        assert env.observation_space.contains(observation)
    
    def test_environment_termination(self):
        """Test various termination conditions."""
        env = Aegis2DInterceptEnv(max_steps=5)  # Short episode for testing
        env.reset(seed=42)
        
        terminated = False
        truncated = False
        step_count = 0
        
        while not (terminated or truncated) and step_count < 10:
            action = env.action_space.sample()
            _, _, terminated, truncated, info = env.step(action)
            step_count += 1
        
        # Should terminate within max_steps or earlier due to other conditions
        assert terminated or truncated
        assert step_count <= 5


if __name__ == "__main__":
    pytest.main([__file__])