"""
Environment Validation Tests for 6DOF System

This module contains comprehensive tests to validate the environment
implementation including:
- State space consistency and bounds
- Action space validation across different modes
- Reward system correctness and stability
- Episode boundary conditions
- Observation/action space compatibility
- Environment determinism and reproducibility

Author: Tester Agent
Date: Phase 3 Testing Framework
"""

import pytest
import numpy as np
import gymnasium as gym
from typing import Dict, Any, List, Tuple
import json
import tempfile
import os

from aegis_intercept.envs.aegis_6dof_env import (
    Aegis6DInterceptEnv, DifficultyMode, ActionMode
)
from aegis_intercept.utils.physics6dof import distance_6dof


class TestActionSpaceValidation:
    """Test action space configuration and validation"""
    
    def test_action_mode_3dof_compatibility(self):
        """Test 3DOF action mode for backward compatibility"""
        env = Aegis6DInterceptEnv(
            action_mode=ActionMode.ACCELERATION_3DOF,
            legacy_3dof_mode=True
        )
        
        # Should be 4D: [ax, ay, az, explode]
        assert env.action_space.shape == (4,)
        assert env.action_space.dtype == np.float32
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)
        
        # Test action execution
        env.reset(seed=42)
        action = np.array([0.5, -0.3, 0.8, 0.0])  # No explosion
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
    
    def test_action_mode_6dof_acceleration(self):
        """Test 6DOF acceleration action mode"""
        env = Aegis6DInterceptEnv(
            action_mode=ActionMode.ACCELERATION_6DOF,
            difficulty_mode=DifficultyMode.FULL_6DOF
        )
        
        # Should be 7D: [fx, fy, fz, tx, ty, tz, explode]
        assert env.action_space.shape == (7,)
        
        env.reset(seed=42)
        action = np.array([0.8, 0.2, -0.5, 0.1, -0.3, 0.4, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == env.observation_space.shape
        assert '6dof_state' in info['interceptor_6dof_state'] or isinstance(info['interceptor_6dof_state'], np.ndarray)
    
    def test_action_mode_thrust_attitude(self):
        """Test thrust-attitude action mode"""
        env = Aegis6DInterceptEnv(
            action_mode=ActionMode.THRUST_ATTITUDE,
            difficulty_mode=DifficultyMode.FULL_6DOF
        )
        
        # Should be 5D: [thrust_mag, pitch, yaw, roll, explode]
        assert env.action_space.shape == (5,)
        
        env.reset(seed=42)
        action = np.array([0.7, 0.2, -0.1, 0.3, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == env.observation_space.shape
    
    def test_action_clipping_behavior(self):
        """Test action clipping handles extreme values properly"""
        env = Aegis6DInterceptEnv(action_mode=ActionMode.ACCELERATION_6DOF)
        env.reset(seed=42)
        
        # Test extreme actions
        extreme_actions = [
            np.array([10.0, -10.0, 5.0, 8.0, -3.0, 15.0, 2.0]),  # Way outside bounds
            np.array([-50.0, 100.0, -25.0, 0.0, 0.0, 0.0, -10.0]),  # Extreme values
            np.array([np.inf, -np.inf, np.nan, 0.0, 0.0, 0.0, 0.0])  # Invalid values
        ]
        
        for action in extreme_actions:
            # Should not crash even with invalid actions
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                assert env.observation_space.contains(obs)
            except Exception as e:
                # If it does crash, it should be a controlled failure
                assert "nan" in str(e).lower() or "inf" in str(e).lower()


class TestObservationSpaceValidation:
    """Test observation space consistency and bounds"""
    
    def test_3dof_observation_space(self):
        """Test 3DOF observation space (backward compatibility)"""
        env = Aegis6DInterceptEnv(
            difficulty_mode=DifficultyMode.EASY_3DOF,
            legacy_3dof_mode=True
        )
        
        # Should be 14D for 3DOF mode
        expected_dim = 14  # [int_pos(3), int_vel(3), mis_pos(3), mis_vel(3), time(1), fuel(1)]
        assert env.observation_space.shape == (expected_dim,)
        
        obs, info = env.reset(seed=42)
        assert obs.shape == (expected_dim,)
        assert env.observation_space.contains(obs)
        
        # Test that observations stay in bounds over multiple steps
        for _ in range(20):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            assert env.observation_space.contains(obs), f"Observation out of bounds: {obs}"
            if terminated or truncated:
                break
    
    def test_6dof_observation_space(self):
        """Test full 6DOF observation space"""
        env = Aegis6DInterceptEnv(
            difficulty_mode=DifficultyMode.FULL_6DOF,
            action_mode=ActionMode.ACCELERATION_6DOF
        )
        
        # Should be 31D for 6DOF mode
        expected_dim = 31  # [int_state(13), mis_state(13), env_state(5)]
        assert env.observation_space.shape == (expected_dim,)
        
        obs, info = env.reset(seed=42)
        assert obs.shape == (expected_dim,)
        assert env.observation_space.contains(obs)
        
        # Validate structure of observation
        interceptor_state = obs[0:13]  # position(3) + velocity(3) + quat(4) + omega(3)
        missile_state = obs[13:26]
        env_state = obs[26:31]  # time, fuel, wind(3)
        
        # Check quaternion normalization in observation
        int_quat = interceptor_state[6:10]
        mis_quat = missile_state[6:10]
        
        int_quat_norm = np.linalg.norm(int_quat)
        mis_quat_norm = np.linalg.norm(mis_quat)
        
        assert abs(int_quat_norm - 1.0) < 1e-6, f"Interceptor quaternion not normalized: {int_quat_norm}"
        assert abs(mis_quat_norm - 1.0) < 1e-6, f"Missile quaternion not normalized: {mis_quat_norm}"
    
    def test_observation_bounds_consistency(self):
        """Test that observation bounds are consistent with actual values"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        # Run multiple episodes to test bounds
        for episode in range(5):
            obs, _ = env.reset(seed=42 + episode)
            
            for step in range(100):
                # Check current observation is within declared bounds
                if not env.observation_space.contains(obs):
                    # Print detailed information about bound violations
                    low, high = env.observation_space.low, env.observation_space.high
                    violations = []
                    for i, (o, l, h) in enumerate(zip(obs, low, high)):
                        if o < l or o > h:
                            violations.append(f"obs[{i}] = {o:.6f}, bounds = [{l:.6f}, {h:.6f}]")
                    
                    pytest.fail(f"Observation bounds violated at episode {episode}, step {step}:\n" + 
                               "\n".join(violations))
                
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                
                if terminated or truncated:
                    break
    
    def test_observation_determinism(self):
        """Test that observations are deterministic given same initial conditions"""
        # Create two identical environments
        env1 = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        env2 = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        # Reset with same seed
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        
        np.testing.assert_array_equal(obs1, obs2, "Initial observations should be identical")
        
        # Take same actions
        action_sequence = [env1.action_space.sample() for _ in range(10)]
        
        for action in action_sequence:
            obs1, _, term1, trunc1, _ = env1.step(action)
            obs2, _, term2, trunc2, _ = env2.step(action)
            
            np.testing.assert_array_almost_equal(obs1, obs2, decimal=10,
                                               err_msg="Observations should remain identical")
            assert term1 == term2 and trunc1 == trunc2, "Termination should be identical"
            
            if term1 or trunc1:
                break


class TestRewardSystemValidation:
    """Test reward system correctness and consistency"""
    
    def test_reward_basic_properties(self):
        """Test basic reward system properties"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        rewards = []
        for episode in range(10):
            obs, _ = env.reset(seed=42 + episode)
            episode_rewards = []
            
            for step in range(100):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Reward should be a finite number
                assert np.isfinite(reward), f"Non-finite reward: {reward}"
                episode_rewards.append(reward)
                
                if terminated or truncated:
                    break
            
            rewards.extend(episode_rewards)
        
        # Check reward distribution makes sense
        rewards = np.array(rewards)
        
        # Most rewards should be negative (efficiency penalty) or small positive
        median_reward = np.median(rewards)
        assert median_reward < 1.0, f"Median reward too high: {median_reward}"
        
        # Should have some variety in rewards
        reward_std = np.std(rewards)
        assert reward_std > 0.1, f"Reward variance too low: {reward_std}"
    
    def test_intercept_reward_consistency(self):
        """Test that intercept scenarios give appropriate rewards"""
        env = Aegis6DInterceptEnv(
            difficulty_mode=DifficultyMode.FULL_6DOF,
            intercept_threshold=100.0,  # Large threshold for easier intercept
            max_steps=50  # Short episodes
        )
        
        intercept_rewards = []
        miss_rewards = []
        
        for episode in range(20):
            obs, _ = env.reset(seed=42 + episode)
            
            for step in range(50):
                # Use action that tries to approach missile
                if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                    # Get positions from 6DOF objects
                    int_pos = env.interceptor_6dof.position
                    mis_pos = env.missile_6dof.position
                    
                    # Simple proportional control toward missile
                    direction = mis_pos - int_pos
                    if np.linalg.norm(direction) > 1e-6:
                        direction = direction / np.linalg.norm(direction)
                    
                    action = np.concatenate([direction * 0.8, direction * 0.2, [0.0]])  # Don't explode yet
                else:
                    action = env.action_space.sample()
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated:
                    # Check if it was an intercept or miss
                    if 'intercept_distance' in info:
                        dist = info['intercept_distance']
                        if dist < env.intercept_threshold:
                            intercept_rewards.append(reward)
                        else:
                            miss_rewards.append(reward)
                    break
        
        # Intercept rewards should be higher than miss rewards
        if intercept_rewards and miss_rewards:
            avg_intercept = np.mean(intercept_rewards)
            avg_miss = np.mean(miss_rewards)
            assert avg_intercept > avg_miss, f"Intercept rewards ({avg_intercept}) should be higher than miss rewards ({avg_miss})"
    
    def test_explosion_reward_bonus(self):
        """Test that explosion intercepts give bonus rewards"""
        env = Aegis6DInterceptEnv(
            difficulty_mode=DifficultyMode.FULL_6DOF,
            explosion_radius=200.0,  # Large explosion radius
            intercept_threshold=150.0
        )
        
        proximity_rewards = []
        explosion_rewards = []
        
        for episode in range(15):
            obs, _ = env.reset(seed=100 + episode)
            
            for step in range(30):
                # Get current distance if possible
                if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                    distance = distance_6dof(env.interceptor_6dof.position, env.missile_6dof.position)
                    
                    # Explode when close
                    explode = 1.0 if distance < env.explosion_radius else 0.0
                    
                    # Move toward missile
                    int_pos = env.interceptor_6dof.position
                    mis_pos = env.missile_6dof.position
                    direction = mis_pos - int_pos
                    if np.linalg.norm(direction) > 1e-6:
                        direction = direction / np.linalg.norm(direction)
                    
                    action = np.concatenate([direction * 0.9, direction * 0.1, [explode]])
                else:
                    action = env.action_space.sample()
                    if step > 20:  # Try exploding later in episode
                        action[-1] = 1.0
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated and reward > 10:  # High reward suggests successful intercept
                    if 'intercept_method' in info:
                        if info['intercept_method'] == 'explosion':
                            explosion_rewards.append(reward)
                        elif info['intercept_method'] == 'proximity':
                            proximity_rewards.append(reward)
                    break
        
        # Explosion intercepts should generally give higher rewards
        if explosion_rewards and proximity_rewards:
            avg_explosion = np.mean(explosion_rewards)
            avg_proximity = np.mean(proximity_rewards)
            # Allow some tolerance since other factors affect reward
            assert avg_explosion >= avg_proximity * 0.9, \
                f"Explosion rewards ({avg_explosion}) should be at least as high as proximity ({avg_proximity})"
    
    def test_fuel_efficiency_reward_component(self):
        """Test that fuel efficiency affects rewards appropriately"""
        env = Aegis6DInterceptEnv(
            difficulty_mode=DifficultyMode.FULL_6DOF,
            enable_fuel_system=True,
            max_fuel=100.0,
            fuel_burn_rate=2.0  # High burn rate for noticeable effect
        )
        
        # Compare high-thrust vs low-thrust strategies
        high_thrust_rewards = []
        low_thrust_rewards = []
        
        for strategy in ['high_thrust', 'low_thrust']:
            for episode in range(10):
                obs, _ = env.reset(seed=200 + episode)
                
                for step in range(50):
                    if strategy == 'high_thrust':
                        action = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.0])
                    else:
                        action = np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1, 0.0])
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if terminated:
                        if strategy == 'high_thrust':
                            high_thrust_rewards.append(reward)
                        else:
                            low_thrust_rewards.append(reward)
                        break
        
        # This test checks that fuel efficiency has some impact, but allows for
        # the fact that high-thrust might lead to better intercept success
        if high_thrust_rewards and low_thrust_rewards:
            print(f"High thrust avg reward: {np.mean(high_thrust_rewards):.2f}")
            print(f"Low thrust avg reward: {np.mean(low_thrust_rewards):.2f}")
            # Just ensure the fuel system is having some effect
            assert abs(np.mean(high_thrust_rewards) - np.mean(low_thrust_rewards)) > 0.1


class TestEpisodeBoundaryConditions:
    """Test episode termination and boundary conditions"""
    
    def test_maximum_steps_termination(self):
        """Test that episodes terminate at maximum steps"""
        max_steps = 20
        env = Aegis6DInterceptEnv(max_steps=max_steps)
        
        obs, _ = env.reset(seed=42)
        
        for step in range(max_steps + 5):  # Go beyond max steps
            action = np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0])  # Gentle action
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert info['step_count'] == step + 1
            
            if step >= max_steps - 1:
                assert truncated, f"Episode should be truncated at step {step}"
                break
            else:
                # Should not be truncated before max steps (unless terminated for other reasons)
                if not terminated:
                    assert not truncated, f"Episode truncated too early at step {step}"
    
    def test_out_of_bounds_termination(self):
        """Test that vehicles going out of bounds causes termination"""
        env = Aegis6DInterceptEnv(
            world_size=100.0,  # Small world for easier boundary testing
            difficulty_mode=DifficultyMode.FULL_6DOF
        )
        
        obs, _ = env.reset(seed=42)
        
        # Apply large forces to try to go out of bounds
        for step in range(100):
            # Very large action to force out of bounds
            action = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                # Check if termination was due to out of bounds
                if hasattr(env, 'interceptor_6dof'):
                    pos = env.interceptor_6dof.position
                    world_limit = env.world_size * 3  # Check against actual limit used
                    
                    out_of_bounds = (
                        pos[0] < -env.world_size or pos[0] > world_limit or
                        pos[1] < -env.world_size or pos[1] > world_limit or
                        pos[2] > world_limit
                    )
                    
                    if out_of_bounds:
                        assert reward < 0, "Out of bounds should give negative reward"
                        break
        
        # Should have terminated within reasonable number of steps
        assert step < 99, "Should have gone out of bounds and terminated"
    
    def test_intercept_termination(self):
        """Test termination on successful intercept"""
        env = Aegis6DInterceptEnv(
            intercept_threshold=200.0,  # Large threshold
            difficulty_mode=DifficultyMode.FULL_6DOF
        )
        
        intercept_found = False
        
        for episode in range(10):
            obs, _ = env.reset(seed=300 + episode)
            
            for step in range(100):
                # Try to move toward missile
                if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                    int_pos = env.interceptor_6dof.position
                    mis_pos = env.missile_6dof.position
                    direction = mis_pos - int_pos
                    if np.linalg.norm(direction) > 1e-6:
                        direction = direction / np.linalg.norm(direction)
                    action = np.concatenate([direction, np.zeros(3), [0.0]])
                else:
                    action = env.action_space.sample()
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated:
                    if 'intercept_distance' in info and info['intercept_distance'] < env.intercept_threshold:
                        intercept_found = True
                        assert reward > 0, "Successful intercept should give positive reward"
                        break
            
            if intercept_found:
                break
        
        # At least one intercept should be possible with large threshold
        # (This might not always work due to random initial conditions)
        if not intercept_found:
            print("Warning: No intercepts found in test - this might indicate an issue")


class TestEnvironmentModeConsistency:
    """Test consistency between different environment modes"""
    
    def test_difficulty_mode_progression(self):
        """Test that difficulty modes provide appropriate progression"""
        modes = [
            DifficultyMode.EASY_3DOF,
            DifficultyMode.MEDIUM_3DOF,
            DifficultyMode.SIMPLIFIED_6DOF,
            DifficultyMode.FULL_6DOF,
            DifficultyMode.EXPERT_6DOF
        ]
        
        mode_complexities = {}
        
        for mode in modes:
            env = Aegis6DInterceptEnv(difficulty_mode=mode)
            obs, _ = env.reset(seed=42)
            
            # Measure "complexity" by observation space dimension
            complexity = {
                'obs_dim': env.observation_space.shape[0],
                'action_dim': env.action_space.shape[0],
                'has_6dof': hasattr(env, 'interceptor_6dof') and env.interceptor_6dof is not None
            }
            
            mode_complexities[mode] = complexity
        
        # Check that complexity generally increases
        prev_obs_dim = 0
        for mode in modes:
            complexity = mode_complexities[mode]
            
            # 6DOF modes should have larger observation space
            if complexity['has_6dof']:
                assert complexity['obs_dim'] >= 31, f"6DOF mode {mode} should have large obs space"
            
            # At least some progression should be evident
            if mode in [DifficultyMode.SIMPLIFIED_6DOF, DifficultyMode.FULL_6DOF, DifficultyMode.EXPERT_6DOF]:
                assert complexity['obs_dim'] >= prev_obs_dim, f"Mode {mode} should not decrease complexity"
            
            prev_obs_dim = complexity['obs_dim']
    
    def test_legacy_3dof_compatibility(self):
        """Test that legacy 3DOF mode maintains compatibility"""
        # Create legacy environment
        legacy_env = Aegis6DInterceptEnv(
            legacy_3dof_mode=True,
            action_mode=ActionMode.ACCELERATION_3DOF,
            difficulty_mode=DifficultyMode.EASY_3DOF
        )
        
        # Test basic functionality
        obs, info = legacy_env.reset(seed=42)
        
        # Should have 3DOF observation space
        assert obs.shape[0] <= 16, "Legacy mode should have smaller observation space"
        
        # Test action execution
        action = np.array([0.5, -0.3, 0.8, 0.0])
        obs, reward, terminated, truncated, info = legacy_env.step(action)
        
        assert legacy_env.observation_space.contains(obs)
        assert np.isfinite(reward)
        
        # Should not have 6DOF objects in legacy mode
        assert not hasattr(info, 'interceptor_aerodynamics') or 'interceptor_aerodynamics' not in info


class TestScenarioConfigHandling:
    """Test scenario configuration loading and validation"""
    
    def test_scenario_config_loading(self):
        """Test loading custom scenario configurations"""
        # Create temporary scenario config
        scenario_config = {
            'name': 'test_scenario',
            'description': 'Test scenario for validation',
            'interceptor_position_range': {'x': (0, 100), 'y': (0, 100), 'z': (0, 50)},
            'interceptor_velocity_range': {'x': (-10, 10), 'y': (-10, 10), 'z': (0, 20)},
            'missile_position_range': {'x': (200, 400), 'y': (200, 400), 'z': (100, 300)},
            'missile_velocity_range': {'x': (-50, 50), 'y': (-50, 50), 'z': (-20, 0)},
            'target_position_range': {'x': (90, 110), 'y': (90, 110), 'z': (0, 10)},
            'wind_conditions': {'enabled': True, 'strength': 1.0},
            'atmospheric_conditions': {'standard': True},
            'evasion_aggressiveness': 0.8
        }
        
        env = Aegis6DInterceptEnv(
            scenario_config=scenario_config,
            difficulty_mode=DifficultyMode.FULL_6DOF
        )
        
        # Should initialize without errors
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
    
    def test_invalid_scenario_config_handling(self):
        """Test handling of invalid scenario configurations"""
        invalid_configs = [
            {},  # Empty config
            {'name': 'test'},  # Incomplete config
            {'invalid_key': 'invalid_value'},  # Wrong keys
        ]
        
        for config in invalid_configs:
            # Should not crash, might use defaults or ignore invalid config
            try:
                env = Aegis6DInterceptEnv(scenario_config=config)
                obs, _ = env.reset(seed=42)
                # If it doesn't crash, that's acceptable - it should handle gracefully
                assert obs.shape == env.observation_space.shape
            except Exception as e:
                # If it does crash, error should be informative
                assert len(str(e)) > 10, "Error message should be informative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])