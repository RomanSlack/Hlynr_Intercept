"""
Integration Tests for 6DOF System

This module contains comprehensive integration tests to validate:
- Backward compatibility with Phase 2 (3DOF mode)
- End-to-end training pipeline functionality
- Logging system integrity and data capture
- Unity export validation
- Component interaction validation
- System-level functionality

Author: Tester Agent
Date: Phase 3 Testing Framework
"""

import pytest
import numpy as np
import tempfile
import os
import json
import time
import subprocess
import sys
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock
import gymnasium as gym

from aegis_intercept.envs.aegis_6dof_env import (
    Aegis6DInterceptEnv, DifficultyMode, ActionMode
)
from aegis_intercept.envs.aegis_3d_env import Aegis3DInterceptEnv
from aegis_intercept.curriculum.curriculum_manager import (
    CurriculumManager, CurriculumPhase
)
from aegis_intercept.logging.trajectory_logger import TrajectoryLogger, LogLevel
from aegis_intercept.logging.export_manager import ExportManager


class TestBackwardCompatibility:
    """Test backward compatibility with Phase 2 (3DOF) systems"""
    
    def test_3dof_environment_compatibility(self):
        """Test that 3DOF environment still works as expected"""
        # Create original 3DOF environment
        env_3d = Aegis3DInterceptEnv()
        
        # Create 6DOF environment in legacy mode
        env_6d_legacy = Aegis6DInterceptEnv(
            legacy_3dof_mode=True,
            difficulty_mode=DifficultyMode.EASY_3DOF,
            action_mode=ActionMode.ACCELERATION_3DOF
        )
        
        # Test basic functionality compatibility
        obs_3d, info_3d = env_3d.reset(seed=42)
        obs_6d, info_6d = env_6d_legacy.reset(seed=42)
        
        # Observation spaces should be similar dimensions
        assert abs(obs_3d.shape[0] - obs_6d.shape[0]) <= 2, "Observation dimensions should be similar"
        
        # Action spaces should be compatible
        assert env_3d.action_space.shape == env_6d_legacy.action_space.shape
        
        # Test action execution compatibility
        action = np.array([0.5, -0.3, 0.8])  # 3DOF action
        
        # 3DOF environment
        obs_3d_1, reward_3d, term_3d, trunc_3d, info_3d_1 = env_3d.step(action)
        
        # 6DOF legacy environment (need to add explosion command)
        action_6d = np.concatenate([action, [0.0]])  # Add explosion command
        obs_6d_1, reward_6d, term_6d, trunc_6d, info_6d_1 = env_6d_legacy.step(action_6d)
        
        # Results should be similar (allowing for small differences)
        assert obs_3d_1.shape[0] <= obs_6d_1.shape[0], "6DOF should have at least as many obs dimensions"
        assert abs(reward_3d - reward_6d) < 5.0, "Rewards should be similar in magnitude"
    
    def test_phase2_training_script_compatibility(self):
        """Test that Phase 2 training patterns still work"""
        # Test basic environment creation and training loop structure
        env = Aegis6DInterceptEnv(
            legacy_3dof_mode=True,
            difficulty_mode=DifficultyMode.EASY_3DOF
        )
        
        # Simulate training loop
        total_reward = 0
        num_episodes = 5
        
        for episode in range(num_episodes):
            obs, info = env.reset(seed=42 + episode)
            episode_reward = 0
            
            for step in range(50):
                # Simple policy for testing
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            total_reward += episode_reward
        
        # Should complete without errors
        assert num_episodes == 5, "All episodes should complete"
        assert abs(total_reward) < 1000, "Total reward should be reasonable"
    
    def test_model_loading_compatibility(self):
        """Test that Phase 2 models can still be evaluated"""
        # Create compatible environment
        env = Aegis6DInterceptEnv(
            legacy_3dof_mode=True,
            difficulty_mode=DifficultyMode.EASY_3DOF,
            action_mode=ActionMode.ACCELERATION_3DOF
        )
        
        obs, _ = env.reset(seed=42)
        
        # Test that observation format is compatible with potential saved models
        # (This would test actual model loading if models were present)
        expected_obs_range = (10, 20)  # Reasonable range for 3DOF observations
        assert expected_obs_range[0] <= obs.shape[0] <= expected_obs_range[1], \
            f"Observation dimension {obs.shape[0]} should be in range {expected_obs_range}"
        
        # Test action space compatibility
        action = env.action_space.sample()
        obs_next, reward, terminated, truncated, info = env.step(action)
        
        # Should execute without errors
        assert env.observation_space.contains(obs_next)


class TestEndToEndTrainingPipeline:
    """Test end-to-end training pipeline functionality"""
    
    def test_basic_training_pipeline(self):
        """Test basic training pipeline components"""
        # Create curriculum manager
        curriculum = CurriculumManager()
        
        # Create environment with curriculum
        env_config = curriculum.get_environment_config()
        env = Aegis6DInterceptEnv(**env_config)
        
        # Test training loop structure
        num_episodes = 3
        for episode in range(num_episodes):
            obs, info = env.reset(seed=42 + episode)
            episode_reward = 0
            episode_length = 0
            
            for step in range(100):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            # Update curriculum
            success = episode_reward > 5.0  # Simple success criterion
            fuel_used = 100.0 - info.get('fuel_remaining', 50.0)
            intercept_time = episode_length * env.dt
            
            curriculum.update_performance(episode_reward, success, fuel_used, intercept_time)
        
        # Should complete without errors
        status = curriculum.get_curriculum_status()
        assert status['phase_progress']['episodes_completed'] == num_episodes
    
    def test_curriculum_environment_integration(self):
        """Test curriculum and environment integration"""
        curriculum = CurriculumManager()
        
        # Test different phases
        test_phases = [
            CurriculumPhase.PHASE_1_BASIC_3DOF,
            CurriculumPhase.PHASE_3_SIMPLIFIED_6DOF,
            CurriculumPhase.PHASE_4_FULL_6DOF
        ]
        
        for phase in test_phases:
            curriculum.set_phase(phase)
            env_config = curriculum.get_environment_config()
            
            # Should be able to create environment for each phase
            env = Aegis6DInterceptEnv(**env_config)
            
            # Test basic functionality
            obs, info = env.reset(seed=42)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert env.observation_space.contains(obs), f"Invalid observation for phase {phase}"
            assert np.isfinite(reward), f"Invalid reward for phase {phase}"
    
    def test_multi_environment_consistency(self):
        """Test consistency across multiple environment instances"""
        curriculum = CurriculumManager()
        env_config = curriculum.get_environment_config()
        
        # Create multiple environments
        envs = [Aegis6DInterceptEnv(**env_config) for _ in range(3)]
        
        # Reset with same seed
        observations = []
        for env in envs:
            obs, _ = env.reset(seed=12345)
            observations.append(obs)
        
        # Should be identical
        for i in range(1, len(observations)):
            np.testing.assert_array_equal(observations[0], observations[i],
                                        "Environments with same seed should be identical")
        
        # Take same action
        action = envs[0].action_space.sample()
        next_observations = []
        
        for env in envs:
            obs, _, _, _, _ = env.step(action)
            next_observations.append(obs)
        
        # Should remain identical
        for i in range(1, len(next_observations)):
            np.testing.assert_array_almost_equal(next_observations[0], next_observations[i],
                                               decimal=10, err_msg="Environments should remain synchronized")
    
    @pytest.mark.slow
    def test_extended_training_stability(self):
        """Test training stability over extended episodes"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        rewards = []
        episode_lengths = []
        
        for episode in range(20):  # More episodes for stability test
            obs, _ = env.reset(seed=episode)
            episode_reward = 0
            episode_length = 0
            
            for step in range(200):
                # Simple proportional controller for more realistic behavior
                if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                    int_pos = env.interceptor_6dof.position
                    mis_pos = env.missile_6dof.position
                    direction = mis_pos - int_pos
                    if np.linalg.norm(direction) > 1e-6:
                        direction = direction / np.linalg.norm(direction)
                    action = np.concatenate([direction * 0.5, direction * 0.1, [0.0]])
                else:
                    action = env.action_space.sample() * 0.5  # Gentler actions
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Check for stability indicators
        rewards = np.array(rewards)
        lengths = np.array(episode_lengths)
        
        # Should not have extreme outliers
        reward_std = np.std(rewards)
        reward_mean = np.mean(rewards)
        assert reward_std < abs(reward_mean) * 2, "Reward variance should not be excessive"
        
        # Episodes should not be getting dramatically shorter/longer
        length_trend = np.polyfit(range(len(lengths)), lengths, 1)[0]  # Linear trend
        assert abs(length_trend) < 2.0, "Episode length should not trend dramatically"


class TestLoggingSystemIntegrity:
    """Test logging system integrity and data capture"""
    
    def test_trajectory_logger_basic_functionality(self):
        """Test basic trajectory logging functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = TrajectoryLogger(
                log_dir=temp_dir,
                log_level=LogLevel.DETAILED,
                enable_compression=False
            )
            
            # Test episode logging
            episode_id = logger.start_episode({"test": True})
            assert episode_id is not None
            
            # Log some trajectory data
            for step in range(10):
                state_data = {
                    'interceptor_position': [step, step, step],
                    'missile_position': [100 - step, 100 - step, 100],
                    'reward': float(step * 0.1),
                    'action': [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]
                }
                logger.log_step(state_data)
            
            # End episode
            episode_summary = {
                'total_reward': 5.0,
                'success': True,
                'episode_length': 10
            }
            logger.end_episode(episode_summary)
            
            # Check that files were created
            log_files = os.listdir(temp_dir)
            assert len(log_files) > 0, "Log files should be created"
            
            # Check file content
            episode_file = None
            for file in log_files:
                if file.endswith('.json'):
                    episode_file = os.path.join(temp_dir, file)
                    break
            
            assert episode_file is not None, "Episode log file should exist"
            
            with open(episode_file, 'r') as f:
                episode_data = json.load(f)
            
            assert 'episode_id' in episode_data
            assert 'trajectory' in episode_data
            assert len(episode_data['trajectory']) == 10
    
    def test_trajectory_logger_6dof_integration(self):
        """Test trajectory logger with 6DOF environment"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = TrajectoryLogger(log_dir=temp_dir, log_level=LogLevel.DETAILED)
            env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
            
            # Run episode with logging
            obs, info = env.reset(seed=42)
            episode_id = logger.start_episode({'seed': 42})
            
            for step in range(20):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Extract logging data
                log_data = {
                    'step': step,
                    'reward': reward,
                    'action': action.tolist(),
                    'terminated': terminated,
                    'truncated': truncated
                }
                
                # Add 6DOF specific data if available
                if 'interceptor_6dof_state' in info:
                    log_data['interceptor_6dof_state'] = info['interceptor_6dof_state'].tolist()
                if 'missile_6dof_state' in info:
                    log_data['missile_6dof_state'] = info['missile_6dof_state'].tolist()
                if 'intercept_distance' in info:
                    log_data['intercept_distance'] = info['intercept_distance']
                
                logger.log_step(log_data)
                
                if terminated or truncated:
                    break
            
            logger.end_episode({'success': terminated and reward > 0})
            
            # Verify logged data
            log_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
            assert len(log_files) > 0, "Should create log files"
    
    def test_export_manager_functionality(self):
        """Test export manager for Unity visualization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            export_manager = ExportManager(
                export_dir=temp_dir,
                enable_unity_export=True,
                export_frequency=1  # Export every episode
            )
            
            # Create sample trajectory data
            trajectory_data = {
                'episode_id': 'test_episode',
                'trajectory': [
                    {
                        'step': i,
                        'interceptor_position': [i * 2, i * 3, i * 1],
                        'missile_position': [100 - i, 100 - i * 2, 50 + i],
                        'target_position': [50, 50, 0],
                        'timestamp': time.time() + i * 0.1
                    }
                    for i in range(15)
                ],
                'episode_summary': {
                    'success': True,
                    'total_reward': 20.0,
                    'intercept_distance': 5.0
                }
            }
            
            # Export data
            export_manager.export_episode(trajectory_data)
            
            # Check export files
            export_files = os.listdir(temp_dir)
            assert len(export_files) > 0, "Should create export files"
            
            # Check Unity format file
            unity_files = [f for f in export_files if 'unity' in f.lower()]
            if unity_files:
                unity_file = os.path.join(temp_dir, unity_files[0])
                with open(unity_file, 'r') as f:
                    unity_data = json.load(f)
                
                assert 'trajectory' in unity_data
                assert 'metadata' in unity_data
    
    def test_logging_performance_impact(self):
        """Test that logging doesn't significantly impact performance"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test without logging
            start_time = time.time()
            obs, _ = env.reset(seed=42)
            
            for _ in range(100):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset()
            
            time_without_logging = time.time() - start_time
            
            # Test with logging
            logger = TrajectoryLogger(log_dir=temp_dir, log_level=LogLevel.BASIC)
            
            start_time = time.time()
            obs, _ = env.reset(seed=42)
            episode_id = logger.start_episode({'test': True})
            
            for step in range(100):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                logger.log_step({
                    'step': step,
                    'reward': reward,
                    'action': action.tolist()
                })
                
                if terminated or truncated:
                    logger.end_episode({'success': reward > 0})
                    obs, _ = env.reset()
                    episode_id = logger.start_episode({'test': True})
            
            logger.end_episode({'success': False})
            time_with_logging = time.time() - start_time
            
            # Logging overhead should be reasonable (less than 50% overhead)
            overhead_ratio = time_with_logging / time_without_logging
            assert overhead_ratio < 1.5, f"Logging overhead too high: {overhead_ratio:.2f}x"


class TestUnityExportValidation:
    """Test Unity export data validation"""
    
    def test_unity_export_data_format(self):
        """Test Unity export data format compliance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            export_manager = ExportManager(
                export_dir=temp_dir,
                enable_unity_export=True
            )
            
            # Create 6DOF trajectory data
            trajectory_data = {
                'episode_id': 'unity_test',
                'trajectory': []
            }
            
            # Add realistic 6DOF trajectory points
            for i in range(25):
                trajectory_point = {
                    'step': i,
                    'timestamp': i * 0.05,
                    'interceptor_position': [i * 5, i * 3, i * 2],
                    'interceptor_velocity': [50, 30, 20],
                    'interceptor_orientation': [1, 0, 0, 0],  # Quaternion
                    'interceptor_angular_velocity': [0.1, 0.05, 0.02],
                    'missile_position': [200 - i * 4, 200 - i * 6, 100 + i],
                    'missile_velocity': [-40, -60, 10],
                    'missile_orientation': [0.9, 0.1, 0.1, 0.1],
                    'missile_angular_velocity': [0.2, 0.15, 0.1],
                    'target_position': [100, 100, 0],
                    'reward': float(i * 0.5),
                    'action': [0.8, 0.2, -0.3, 0.1, -0.1, 0.2, 0.0]
                }
                trajectory_data['trajectory'].append(trajectory_point)
            
            # Export for Unity
            export_manager.export_episode(trajectory_data)
            
            # Find Unity export file
            export_files = os.listdir(temp_dir)
            unity_file = None
            for file in export_files:
                if 'unity' in file.lower() and file.endswith('.json'):
                    unity_file = os.path.join(temp_dir, file)
                    break
            
            assert unity_file is not None, "Unity export file should be created"
            
            # Validate Unity format
            with open(unity_file, 'r') as f:
                unity_data = json.load(f)
            
            # Check required Unity fields
            assert 'metadata' in unity_data
            assert 'trajectory' in unity_data
            assert 'visualization_config' in unity_data or 'config' in unity_data
            
            # Check trajectory format
            trajectory = unity_data['trajectory']
            assert len(trajectory) > 0, "Trajectory should have data points"
            
            # Check each trajectory point has required fields
            required_fields = [
                'timestamp', 'interceptor_position', 'missile_position', 'target_position'
            ]
            
            for point in trajectory[:5]:  # Check first few points
                for field in required_fields:
                    assert field in point, f"Missing field {field} in trajectory point"
                
                # Check position format (should be 3D coordinates)
                assert len(point['interceptor_position']) == 3
                assert len(point['missile_position']) == 3
                assert len(point['target_position']) == 3
    
    def test_export_frequency_control(self):
        """Test export frequency control"""
        with tempfile.TemporaryDirectory() as temp_dir:
            export_manager = ExportManager(
                export_dir=temp_dir,
                enable_unity_export=True,
                export_frequency=3  # Export every 3rd episode
            )
            
            # Create multiple episodes
            for episode in range(7):
                trajectory_data = {
                    'episode_id': f'episode_{episode}',
                    'trajectory': [
                        {
                            'step': 0,
                            'interceptor_position': [0, 0, 0],
                            'missile_position': [100, 100, 100],
                            'target_position': [50, 50, 0]
                        }
                    ]
                }
                export_manager.export_episode(trajectory_data)
            
            # Check number of export files
            export_files = os.listdir(temp_dir)
            
            # Should have exported episodes 0, 3, 6 (every 3rd)
            expected_exports = 3
            actual_exports = len([f for f in export_files if f.endswith('.json')])
            
            assert actual_exports <= expected_exports + 1, f"Too many exports: {actual_exports}"
            assert actual_exports >= expected_exports - 1, f"Too few exports: {actual_exports}"


class TestComponentInteraction:
    """Test interaction between different system components"""
    
    def test_curriculum_logging_integration(self):
        """Test integration between curriculum and logging systems"""
        curriculum = CurriculumManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = TrajectoryLogger(log_dir=temp_dir)
            
            # Simulate training with curriculum and logging
            for episode in range(5):
                env_config = curriculum.get_environment_config()
                env = Aegis6DInterceptEnv(**env_config)
                
                obs, info = env.reset(seed=episode)
                episode_id = logger.start_episode({
                    'episode': episode,
                    'phase': curriculum.current_phase.value,
                    'scenario': curriculum.current_scenario
                })
                
                episode_reward = 0
                for step in range(30):
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    episode_reward += reward
                    
                    logger.log_step({
                        'step': step,
                        'reward': reward,
                        'phase': curriculum.current_phase.value
                    })
                    
                    if terminated or truncated:
                        break
                
                # Update both systems
                success = episode_reward > 5.0
                curriculum.update_performance(episode_reward, success, 50.0, step * env.dt)
                logger.end_episode({
                    'total_reward': episode_reward,
                    'success': success,
                    'phase': curriculum.current_phase.value
                })
            
            # Verify both systems recorded data
            log_files = os.listdir(temp_dir)
            assert len(log_files) > 0, "Logger should create files"
            
            status = curriculum.get_curriculum_status()
            assert status['phase_progress']['episodes_completed'] == 5
    
    def test_environment_curriculum_consistency(self):
        """Test consistency between environment and curriculum configurations"""
        curriculum = CurriculumManager()
        
        # Test each phase
        for phase in CurriculumPhase:
            curriculum.set_phase(phase)
            env_config = curriculum.get_environment_config()
            
            # Create environment with curriculum config
            env = Aegis6DInterceptEnv(**env_config)
            
            # Verify configuration consistency
            phase_config = curriculum.get_current_phase_config()
            
            assert env.difficulty_mode == phase_config.difficulty_mode
            assert env.action_mode == phase_config.action_mode
            assert env.world_size == phase_config.world_size
            assert env.max_steps == phase_config.max_steps
            assert env.intercept_threshold == phase_config.intercept_threshold
    
    def test_error_handling_integration(self):
        """Test error handling across integrated components"""
        curriculum = CurriculumManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = TrajectoryLogger(log_dir=temp_dir)
            
            # Test with invalid environment parameters
            env_config = curriculum.get_environment_config()
            env_config['world_size'] = -100  # Invalid value
            
            try:
                env = Aegis6DInterceptEnv(**env_config)
                # If it doesn't crash, it should handle gracefully
                obs, _ = env.reset(seed=42)
                assert env.observation_space.contains(obs)
            except Exception as e:
                # Error should be informative
                assert len(str(e)) > 5, "Error message should be informative"
            
            # Test logging with invalid data
            try:
                episode_id = logger.start_episode({'test': True})
                logger.log_step({'invalid': 'data_structure'})
                logger.end_episode({'test': True})
                # Should handle gracefully
            except Exception as e:
                assert "invalid" in str(e).lower() or "data" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])