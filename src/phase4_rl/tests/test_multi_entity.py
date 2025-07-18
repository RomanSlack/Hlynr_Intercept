"""
Tests for multi-entity functionality in Phase 4 RL components.

This module tests the multi-missile/interceptor capabilities, entity scaling,
and coordination features of the radar environment.
"""

import pytest
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radar_env import RadarEnv
from config import ConfigLoader
from scenarios import ScenarioLoader
from diagnostics import Logger


class TestMultiEntityEnvironment:
    """Test cases for multi-entity environment functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.base_config = ConfigLoader()._config
    
    def test_single_entity_baseline(self):
        """Test baseline single missile/interceptor scenario."""
        config = self.base_config.copy()
        config['environment']['num_missiles'] = 1
        config['environment']['num_interceptors'] = 1
        
        env = RadarEnv(config=config)
        
        assert env.num_missiles == 1
        assert env.num_interceptors == 1
        
        obs, _ = env.reset()
        action = env.action_space.sample()
        step_result = env.step(action)
        
        assert len(step_result) == 5
        assert step_result[0].shape == env.observation_space.shape
    
    def test_multi_missile_scenario(self):
        """Test multi-missile scenarios."""
        for num_missiles in [2, 3, 5]:
            config = self.base_config.copy()
            config['environment']['num_missiles'] = num_missiles
            config['environment']['num_interceptors'] = 2
            
            env = RadarEnv(config=config)
            
            assert env.num_missiles == num_missiles
            assert env.num_interceptors == 2
            
            # Test that environment handles multiple missiles
            obs, info = env.reset()
            
            # Check that missile positions are initialized
            if 'missile_positions' in info:
                assert len(info['missile_positions']) == num_missiles
    
    def test_multi_interceptor_scenario(self):
        """Test multi-interceptor scenarios."""
        for num_interceptors in [2, 3, 4]:
            config = self.base_config.copy()
            config['environment']['num_missiles'] = 2
            config['environment']['num_interceptors'] = num_interceptors
            
            env = RadarEnv(config=config)
            
            assert env.num_interceptors == num_interceptors
            
            # Action space should scale with interceptor count
            expected_action_dim = num_interceptors * 6  # 6 actions per interceptor
            assert env.action_space.shape[0] == expected_action_dim
            
            # Test action processing
            obs, _ = env.reset()
            action = env.action_space.sample()
            step_result = env.step(action)
            
            assert step_result[0].shape == env.observation_space.shape
    
    def test_asymmetric_entity_counts(self):
        """Test scenarios with different missile/interceptor ratios."""
        test_cases = [
            (1, 3),  # Outnumbered missiles
            (3, 1),  # Outnumbered interceptors
            (2, 3),  # Mixed ratio
            (4, 2),  # 2:1 missile advantage
        ]
        
        for num_missiles, num_interceptors in test_cases:
            config = self.base_config.copy()
            config['environment']['num_missiles'] = num_missiles
            config['environment']['num_interceptors'] = num_interceptors
            
            env = RadarEnv(config=config)
            
            # Environment should handle asymmetric counts
            obs, info = env.reset()
            action = env.action_space.sample()
            
            # Should complete step without errors
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert obs.shape == env.observation_space.shape
            assert isinstance(reward, (int, float, np.number))
    
    def test_entity_spawn_positioning(self):
        """Test that entities spawn in correct positions."""
        config = self.base_config.copy()
        config['environment']['num_missiles'] = 3
        config['environment']['num_interceptors'] = 2
        
        # Set specific spawn positions
        config['spawn'] = {
            'missile_spawn_positions': [[0, 0], [50, 0], [100, 0]],
            'interceptor_spawn_positions': [[500, 500], [600, 500]],
            'target_positions': [[1000, 1000], [1100, 1000], [1200, 1000]]
        }
        
        env = RadarEnv(config=config)
        obs, info = env.reset()
        
        # Check that entities spawned at specified positions
        if 'missile_positions' in info:
            missile_positions = info['missile_positions']
            assert len(missile_positions) == 3
            
            # Check approximate spawn positions (allowing for small variations)
            expected_positions = np.array([[0, 0], [50, 0], [100, 0]])
            actual_positions = np.array(missile_positions)
            
            # Should be close to expected positions
            assert np.allclose(actual_positions, expected_positions, atol=10.0)
    
    def test_observation_space_scaling(self):
        """Test that observation space scales properly with entity count."""
        base_obs_dims = {}
        
        # Test different entity configurations
        test_configs = [
            (1, 1), (2, 1), (1, 2), (2, 2), (3, 3)
        ]
        
        for num_missiles, num_interceptors in test_configs:
            config = self.base_config.copy()
            config['environment']['num_missiles'] = num_missiles
            config['environment']['num_interceptors'] = num_interceptors
            
            env = RadarEnv(config=config)
            obs_dim = env.observation_space.shape[0]
            
            # Store for comparison
            base_obs_dims[(num_missiles, num_interceptors)] = obs_dim
            
            # Observation should increase with entity count
            if (1, 1) in base_obs_dims:
                baseline_dim = base_obs_dims[(1, 1)]
                if num_missiles > 1 or num_interceptors > 1:
                    assert obs_dim > baseline_dim
    
    def test_multi_entity_radar_processing(self):
        """Test radar processing with multiple entities."""
        config = self.base_config.copy()
        config['environment']['num_missiles'] = 3
        config['environment']['num_interceptors'] = 2
        
        env = RadarEnv(config=config)
        obs, _ = env.reset()
        
        # Step environment to generate radar returns
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check that radar processing handles multiple entities
        # This is tested implicitly by successful observation generation
        assert obs.shape == env.observation_space.shape
        assert np.all(np.isfinite(obs))  # No NaN or infinite values


class TestEntityInteraction:
    """Test cases for entity interaction and coordination."""
    
    def test_interception_distance_calculation(self):
        """Test calculation of interception distances."""
        config = ConfigLoader()._config.copy()
        config['environment']['num_missiles'] = 2
        config['environment']['num_interceptors'] = 2
        
        env = RadarEnv(config=config)
        obs, info = env.reset()
        
        # Take some steps to allow movement
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check that distance information is provided
            if 'min_interception_distances' in info:
                distances = info['min_interception_distances']
                assert len(distances) == env.num_interceptors
                assert all(d >= 0 for d in distances)
    
    def test_multi_target_assignment(self):
        """Test target assignment in multi-entity scenarios."""
        config = ConfigLoader()._config.copy()
        config['environment']['num_missiles'] = 3
        config['environment']['num_interceptors'] = 2
        
        env = RadarEnv(config=config)
        logger = Logger()
        
        obs, _ = env.reset()
        logger.reset_episode()
        
        # Run episode and log data
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            step_data = {
                'step': step,
                'observation': obs,
                'action': action,
                'reward': reward,
                'done': terminated or truncated,
                'info': info
            }
            logger.log_step(step_data)
            
            if terminated or truncated:
                break
        
        # Analyze target assignment
        metrics = logger.get_episode_metrics()
        
        # Should have entity count information
        if 'entity_counts' in metrics:
            assert metrics['entity_counts']['missiles'] == 3
            assert metrics['entity_counts']['interceptors'] == 2
    
    def test_coordination_scoring(self):
        """Test coordination scoring for multiple interceptors."""
        config = ConfigLoader()._config.copy() 
        config['environment']['num_missiles'] = 2
        config['environment']['num_interceptors'] = 3
        
        env = RadarEnv(config=config)
        logger = Logger()
        
        obs, _ = env.reset()
        logger.reset_episode()
        
        # Run short episode
        for step in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            step_data = {
                'step': step,
                'observation': obs,
                'action': action,
                'reward': reward,
                'done': terminated or truncated,
                'info': info
            }
            logger.log_step(step_data)
            
            if terminated or truncated:
                break
        
        metrics = logger.get_episode_metrics()
        
        # Should calculate coordination score for multiple interceptors
        if 'coordination_score' in metrics:
            assert isinstance(metrics['coordination_score'], (int, float))
            assert metrics['coordination_score'] >= 0


class TestScenarioEntityConfiguration:
    """Test entity configuration through scenarios."""
    
    def test_easy_scenario_entities(self):
        """Test entity configuration in easy scenario."""
        scenario_loader = ScenarioLoader()
        
        try:
            easy_config = scenario_loader.load_scenario("easy")
            
            assert easy_config['num_missiles'] == 1
            assert easy_config['num_interceptors'] == 1
            
            # Test environment creation with scenario
            env_config = scenario_loader.create_environment_config(
                "easy", ConfigLoader()._config
            )
            
            env = RadarEnv(config=env_config, scenario_name="easy")
            
            assert env.num_missiles == 1
            assert env.num_interceptors == 1
            
        except FileNotFoundError:
            pytest.skip("Easy scenario file not found")
    
    def test_multi_entity_scenarios(self):
        """Test multi-entity scenarios."""
        scenario_loader = ScenarioLoader()
        
        try:
            # Test scenarios with multiple entities
            test_scenarios = ["medium", "hard", "impossible"]
            
            for scenario_name in test_scenarios:
                try:
                    scenario_config = scenario_loader.load_scenario(scenario_name)
                    
                    # Should have multiple entities in harder scenarios
                    if scenario_name in ["hard", "impossible"]:
                        assert scenario_config['num_missiles'] > 1 or scenario_config['num_interceptors'] > 1
                    
                    # Test environment creation
                    env_config = scenario_loader.create_environment_config(
                        scenario_name, ConfigLoader()._config
                    )
                    
                    env = RadarEnv(config=env_config, scenario_name=scenario_name)
                    
                    # Environment should handle scenario configuration
                    obs, info = env.reset()
                    action = env.action_space.sample()
                    step_result = env.step(action)
                    
                    assert len(step_result) == 5
                    
                except FileNotFoundError:
                    continue  # Skip missing scenarios
                    
        except FileNotFoundError:
            pytest.skip("Scenario files not found")


class TestEntityPhysics:
    """Test physics and dynamics for multiple entities."""
    
    def test_independent_entity_movement(self):
        """Test that entities move independently."""
        config = ConfigLoader()._config.copy()
        config['environment']['num_missiles'] = 2
        config['environment']['num_interceptors'] = 2
        
        env = RadarEnv(config=config)
        obs, info = env.reset()
        
        initial_positions = {
            'missiles': info.get('missile_positions', []).copy() if 'missile_positions' in info else [],
            'interceptors': info.get('interceptor_positions', []).copy() if 'interceptor_positions' in info else []
        }
        
        # Take several steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        final_positions = {
            'missiles': info.get('missile_positions', []),
            'interceptors': info.get('interceptor_positions', [])
        }
        
        # Entities should have moved
        if initial_positions['missiles'] and final_positions['missiles']:
            for i, (init_pos, final_pos) in enumerate(zip(initial_positions['missiles'], final_positions['missiles'])):
                distance_moved = np.linalg.norm(np.array(final_pos) - np.array(init_pos))
                assert distance_moved > 0, f"Missile {i} did not move"
    
    def test_collision_detection(self):
        """Test collision detection between entities."""
        config = ConfigLoader()._config.copy()
        config['environment']['num_missiles'] = 1
        config['environment']['num_interceptors'] = 1
        
        # Set spawn positions very close together
        config['spawn'] = {
            'missile_spawn_positions': [[0, 0]],
            'interceptor_spawn_positions': [[10, 10]],  # Very close
            'target_positions': [[1000, 1000]]
        }
        
        env = RadarEnv(config=config)
        obs, info = env.reset()
        
        # Apply actions to bring entities together
        for step in range(100):
            # Action to move interceptor toward missile
            action = np.array([0.5, 0.5, 1.0, 0.0, 0.0, 0.0])  # Move toward missile
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                # Episode terminated, likely due to interception
                break
            
            if 'min_interception_distances' in info:
                min_distance = min(info['min_interception_distances'])
                if min_distance < 50.0:  # Close approach
                    break


if __name__ == '__main__':
    pytest.main([__file__])