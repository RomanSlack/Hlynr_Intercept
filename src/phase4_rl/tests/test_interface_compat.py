"""
Interface compatibility tests for Phase 4 RL components.

This module tests the compatibility between RadarEnv and existing environment
interfaces, ensuring consistent action/observation shapes and behavior.
"""

import pytest
import numpy as np
import gymnasium as gym

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radar_env import RadarEnv
from config import ConfigLoader
from scenarios import ScenarioLoader


class TestRadarEnvInterface:
    """Test cases for RadarEnv interface compatibility."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.config = ConfigLoader()._config
    
    def test_gymnasium_interface_compliance(self):
        """Test that RadarEnv properly implements Gymnasium interface."""
        env = RadarEnv(config=self.config)
        
        # Test that it's a proper Gym environment
        assert isinstance(env, gym.Env)
        
        # Test required attributes exist
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'render')
        assert hasattr(env, 'close')
        
        # Test spaces are properly defined
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)
    
    def test_observation_space_shape(self):
        """Test observation space shape consistency."""
        # Test with different entity configurations
        configs = [
            {'num_missiles': 1, 'num_interceptors': 1},
            {'num_missiles': 2, 'num_interceptors': 2},
            {'num_missiles': 3, 'num_interceptors': 2},
        ]
        
        for config_override in configs:
            test_config = self.config.copy()
            test_config['environment'].update(config_override)
            
            env = RadarEnv(config=test_config)
            
            # Observation space should be properly sized
            assert env.observation_space.shape[0] > 0
            assert len(env.observation_space.shape) == 1  # Should be 1D
            
            # Test that observation matches space
            obs, _ = env.reset()
            assert obs.shape == env.observation_space.shape
            assert env.observation_space.contains(obs)
    
    def test_action_space_shape(self):
        """Test action space shape consistency."""
        configs = [
            {'num_missiles': 1, 'num_interceptors': 1},
            {'num_missiles': 2, 'num_interceptors': 2},
            {'num_missiles': 3, 'num_interceptors': 2},
        ]
        
        for config_override in configs:
            test_config = self.config.copy()
            test_config['environment'].update(config_override)
            
            env = RadarEnv(config=test_config)
            
            # Action space should scale with number of interceptors
            expected_action_dim = config_override['num_interceptors'] * 6  # 6 actions per interceptor
            assert env.action_space.shape[0] == expected_action_dim
            
            # Test random action sampling
            action = env.action_space.sample()
            assert env.action_space.contains(action)
    
    def test_reset_functionality(self):
        """Test reset method functionality."""
        env = RadarEnv(config=self.config)
        
        # Test reset returns correct format
        obs, info = env.reset()
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert obs.shape == env.observation_space.shape
        
        # Test multiple resets
        for _ in range(3):
            obs2, info2 = env.reset()
            assert obs2.shape == obs.shape
            assert isinstance(info2, dict)
    
    def test_step_functionality(self):
        """Test step method functionality."""
        env = RadarEnv(config=self.config)
        obs, _ = env.reset()
        
        # Test with valid action
        action = env.action_space.sample()
        step_result = env.step(action)
        
        assert len(step_result) == 5  # obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = step_result
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
        
        assert obs.shape == env.observation_space.shape
    
    def test_scenario_integration(self):
        """Test integration with different scenarios."""
        scenario_loader = ScenarioLoader()
        scenarios = scenario_loader.list_scenarios()
        
        for scenario_name in scenarios[:2]:  # Test first 2 scenarios
            try:
                scenario_config = scenario_loader.create_environment_config(
                    scenario_name, self.config
                )
                
                env = RadarEnv(config=scenario_config, scenario_name=scenario_name)
                
                # Test that environment works with scenario
                obs, info = env.reset()
                assert 'scenario' in info
                assert info['scenario'] == scenario_name
                
                # Test step functionality
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                assert obs.shape == env.observation_space.shape
                
            except Exception as e:
                pytest.fail(f"Scenario {scenario_name} failed: {e}")


class TestEnvironmentComparison:
    """Test compatibility between RadarEnv and existing environments."""
    
    def test_action_observation_shape_consistency(self):
        """Test that environments have consistent action/observation shapes."""
        # This test assumes there might be other environment implementations
        # For now, we'll test RadarEnv with different configurations
        
        configs = [
            {'num_missiles': 1, 'num_interceptors': 1},
            {'num_missiles': 2, 'num_interceptors': 2},
        ]
        
        envs = []
        for config_override in configs:
            test_config = ConfigLoader()._config.copy()
            test_config['environment'].update(config_override)
            env = RadarEnv(config=test_config)
            envs.append(env)
        
        # All environments should use the same data types
        for env in envs:
            assert isinstance(env.observation_space, gym.spaces.Box)
            assert isinstance(env.action_space, gym.spaces.Box)
            assert env.observation_space.dtype == np.float32
            assert env.action_space.dtype == np.float32
    
    def test_multi_entity_scaling(self):
        """Test that environments scale properly with entity count."""
        base_config = ConfigLoader()._config
        
        # Test with increasing entity counts
        entity_configs = [
            (1, 1),  # 1 missile, 1 interceptor
            (2, 2),  # 2 missiles, 2 interceptors  
            (3, 3),  # 3 missiles, 3 interceptors
        ]
        
        prev_obs_dim = None
        prev_action_dim = None
        
        for num_missiles, num_interceptors in entity_configs:
            test_config = base_config.copy()
            test_config['environment']['num_missiles'] = num_missiles
            test_config['environment']['num_interceptors'] = num_interceptors
            
            env = RadarEnv(config=test_config)
            
            obs_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            
            # Observation dimension should increase with entity count
            if prev_obs_dim is not None:
                assert obs_dim > prev_obs_dim
            
            # Action dimension should scale with interceptor count
            if prev_action_dim is not None:
                assert action_dim > prev_action_dim
            
            # Action dimension should be exactly 6 * num_interceptors
            assert action_dim == 6 * num_interceptors
            
            prev_obs_dim = obs_dim
            prev_action_dim = action_dim


class TestEnvironmentStability:
    """Test environment stability and error handling."""
    
    def test_invalid_actions(self):
        """Test environment handles invalid actions gracefully."""
        env = RadarEnv()
        env.reset()
        
        # Test with action outside bounds
        invalid_action = np.ones(env.action_space.shape[0]) * 2.0  # Outside [-1, 1]
        
        # Environment should clip or handle invalid actions
        try:
            obs, reward, terminated, truncated, info = env.step(invalid_action)
            # If it doesn't raise an error, check that it returns valid results
            assert obs.shape == env.observation_space.shape
            assert isinstance(reward, (int, float, np.number))
        except Exception as e:
            # If it raises an error, it should be a meaningful one
            assert "action" in str(e).lower() or "bound" in str(e).lower()
    
    def test_episode_termination(self):
        """Test episode termination conditions."""
        env = RadarEnv()
        env.reset()
        
        # Run episode until termination
        max_steps = 1000
        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        # Episode should terminate within reasonable time
        assert step < max_steps - 1 or truncated
    
    def test_render_modes(self):
        """Test different render modes."""
        # Test render mode None (no rendering)
        env = RadarEnv(render_mode=None)
        env.reset()
        action = env.action_space.sample()
        env.step(action)
        
        # Should not raise errors
        result = env.render()
        assert result is None or isinstance(result, np.ndarray)
        
        env.close()
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test with invalid configuration
        invalid_config = {
            'environment': {
                'num_missiles': -1,  # Invalid
                'num_interceptors': 0  # Invalid
            }
        }
        
        # Environment should handle invalid configurations gracefully
        try:
            env = RadarEnv(config=invalid_config)
            # If it doesn't raise an error, it should use reasonable defaults
            assert env.num_missiles > 0
            assert env.num_interceptors > 0
        except ValueError:
            # Should raise meaningful error for invalid config
            pass


class TestEnvironmentSeeding:
    """Test environment seeding and reproducibility."""
    
    def test_reset_seeding(self):
        """Test that seeding produces reproducible results."""
        env = RadarEnv()
        
        # Reset with same seed should produce same initial observations
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        
        np.testing.assert_array_equal(obs1, obs2)
    
    def test_step_reproducibility(self):
        """Test that seeded environments behave reproducibly."""
        env1 = RadarEnv()
        env2 = RadarEnv()
        
        # Reset both with same seed
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        # Take same action in both
        action = env1.action_space.sample()
        
        result1 = env1.step(action)
        result2 = env2.step(action)
        
        # Results should be identical (or very close due to floating point)
        np.testing.assert_array_almost_equal(result1[0], result2[0], decimal=5)
        assert abs(result1[1] - result2[1]) < 1e-6  # Rewards should be very close


if __name__ == '__main__':
    pytest.main([__file__])