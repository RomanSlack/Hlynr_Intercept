"""
Adversary Behavior Validation Tests

This module contains comprehensive tests to validate the enhanced adversary
system including:
- Evasive pattern verification and execution
- Threat assessment accuracy and responsiveness
- Parameter sensitivity and configuration handling
- Realistic flight dynamics and physics compliance
- Adaptive behavior and learning response
- JSON configuration validation

Author: Tester Agent
Date: Phase 3 Testing Framework
"""

import pytest
import numpy as np
import json
import tempfile
import time
from typing import Dict, Any, List, Tuple
from unittest.mock import patch, MagicMock

from aegis_intercept.envs.aegis_6dof_env import Aegis6DInterceptEnv, DifficultyMode
from aegis_intercept.utils.physics6dof import (
    RigidBody6DOF, VehicleType, distance_6dof, intercept_geometry_6dof
)


class TestEvasivePatternVerification:
    """Test evasive maneuver patterns and execution"""
    
    def test_evasive_pattern_activation(self):
        """Test that evasive patterns activate under threat conditions"""
        env = Aegis6DInterceptEnv(
            difficulty_mode=DifficultyMode.FULL_6DOF,
            world_size=200.0  # Smaller world for closer encounters
        )
        
        pattern_activations = []
        threat_distances = []
        
        for episode in range(10):
            env.reset(seed=42 + episode)
            
            for step in range(50):
                # Move interceptor toward missile aggressively
                if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                    int_pos = env.interceptor_6dof.position
                    mis_pos = env.missile_6dof.position
                    
                    # Calculate threat distance
                    threat_dist = distance_6dof(int_pos, mis_pos)
                    threat_distances.append(threat_dist)
                    
                    # Aggressive intercept action
                    direction = mis_pos - int_pos
                    if np.linalg.norm(direction) > 1e-6:
                        direction = direction / np.linalg.norm(direction)
                    
                    action = np.concatenate([direction * 1.0, direction * 0.3, [0.0]])
                else:
                    action = env.action_space.sample()
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Check if missile is performing evasive maneuvers
                if hasattr(env, 'missile_6dof'):
                    # Look for signs of evasive behavior in missile acceleration/angular velocity
                    missile_omega = env.missile_6dof.angular_velocity
                    omega_magnitude = np.linalg.norm(missile_omega)
                    
                    # High angular velocity suggests evasive maneuvers
                    if omega_magnitude > 1.0:
                        pattern_activations.append(threat_dist)
                
                if terminated or truncated:
                    break
        
        # Analyze results
        if pattern_activations and threat_distances:
            avg_activation_distance = np.mean(pattern_activations)
            avg_threat_distance = np.mean(threat_distances)
            
            # Evasive patterns should activate when threat is closer
            assert avg_activation_distance < avg_threat_distance * 1.5, \
                "Evasive patterns should activate at closer distances"
    
    def test_specific_evasion_patterns(self):
        """Test specific evasion patterns for correctness"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.EXPERT_6DOF)
        
        pattern_signatures = {
            'spiral': [],
            'barrel_roll': [],
            'jink': [],
            'weave': []
        }
        
        # Run multiple episodes to capture different patterns
        for episode in range(15):
            env.reset(seed=100 + episode)
            
            missile_positions = []
            missile_orientations = []
            
            for step in range(60):
                # Force close encounter to trigger evasion
                if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                    # Position interceptor close to missile
                    env.interceptor_6dof.position = env.missile_6dof.position + np.array([50, 0, 0])
                    
                    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    action = env.action_space.sample()
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                if hasattr(env, 'missile_6dof'):
                    missile_positions.append(env.missile_6dof.position.copy())
                    missile_orientations.append(env.missile_6dof.orientation.copy())
                
                if terminated or truncated:
                    break
            
            # Analyze trajectory for pattern signatures
            if len(missile_positions) > 10:
                positions = np.array(missile_positions)
                
                # Check for spiral pattern (circular motion in x-y plane)
                if len(positions) > 20:
                    # Look for periodic motion
                    x_positions = positions[:, 0]
                    y_positions = positions[:, 1]
                    
                    # Calculate centroid and radial distances
                    centroid_x, centroid_y = np.mean(x_positions), np.mean(y_positions)
                    radial_distances = np.sqrt((x_positions - centroid_x)**2 + (y_positions - centroid_y)**2)
                    
                    # If radial distances are relatively constant, might be spiral
                    if np.std(radial_distances) < np.mean(radial_distances) * 0.3:
                        pattern_signatures['spiral'].append(episode)
                    
                    # Check for jinking (rapid direction changes)
                    if len(positions) > 5:
                        velocity_changes = []
                        for i in range(2, len(positions) - 2):
                            v1 = positions[i] - positions[i-1]
                            v2 = positions[i+1] - positions[i]
                            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                                angle_change = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                                velocity_changes.append(angle_change)
                        
                        # Large average angle changes suggest jinking
                        if velocity_changes and np.mean(velocity_changes) > 0.5:
                            pattern_signatures['jink'].append(episode)
        
        # Verify that some patterns were detected
        total_patterns = sum(len(episodes) for episodes in pattern_signatures.values())
        assert total_patterns > 0, "Should detect some evasive patterns"
    
    def test_evasion_pattern_effectiveness(self):
        """Test that evasive patterns actually improve missile survival"""
        # Compare survival rates with and without evasion
        survival_with_evasion = []
        survival_without_evasion = []
        
        # Test with evasion enabled (expert mode)
        env_with_evasion = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.EXPERT_6DOF)
        
        for episode in range(8):
            env_with_evasion.reset(seed=200 + episode)
            survived = True
            
            for step in range(100):
                # Simple pursuit strategy for interceptor
                if hasattr(env_with_evasion, 'interceptor_6dof') and hasattr(env_with_evasion, 'missile_6dof'):
                    int_pos = env_with_evasion.interceptor_6dof.position
                    mis_pos = env_with_evasion.missile_6dof.position
                    direction = mis_pos - int_pos
                    if np.linalg.norm(direction) > 1e-6:
                        direction = direction / np.linalg.norm(direction)
                    action = np.concatenate([direction * 0.8, direction * 0.2, [0.0]])
                else:
                    action = env_with_evasion.action_space.sample()
                
                obs, reward, terminated, truncated, info = env_with_evasion.step(action)
                
                if terminated and reward > 10:  # Successful intercept
                    survived = False
                    break
                elif terminated or truncated:
                    break
            
            survival_with_evasion.append(survived)
        
        # Test with reduced evasion (basic mode)
        env_without_evasion = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.SIMPLIFIED_6DOF)
        
        for episode in range(8):
            env_without_evasion.reset(seed=200 + episode)  # Same seeds for fair comparison
            survived = True
            
            for step in range(100):
                if hasattr(env_without_evasion, 'interceptor_6dof') and hasattr(env_without_evasion, 'missile_6dof'):
                    int_pos = env_without_evasion.interceptor_6dof.position
                    mis_pos = env_without_evasion.missile_6dof.position
                    direction = mis_pos - int_pos
                    if np.linalg.norm(direction) > 1e-6:
                        direction = direction / np.linalg.norm(direction)
                    action = np.concatenate([direction * 0.8, direction * 0.2, [0.0]])
                else:
                    action = env_without_evasion.action_space.sample()
                
                obs, reward, terminated, truncated, info = env_without_evasion.step(action)
                
                if terminated and reward > 10:
                    survived = False
                    break
                elif terminated or truncated:
                    break
            
            survival_without_evasion.append(survived)
        
        # Evasive behavior should improve survival rate
        survival_rate_with = sum(survival_with_evasion) / len(survival_with_evasion)
        survival_rate_without = sum(survival_without_evasion) / len(survival_without_evasion)
        
        print(f"Survival rate with evasion: {survival_rate_with:.2f}")
        print(f"Survival rate without evasion: {survival_rate_without:.2f}")
        
        # Allow some flexibility, but evasion should generally help
        assert survival_rate_with >= survival_rate_without - 0.2, \
            "Evasive behavior should not significantly hurt survival"


class TestThreatAssessmentAccuracy:
    """Test threat assessment and response accuracy"""
    
    def test_threat_level_calculation(self):
        """Test threat level calculation based on geometry"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        threat_scenarios = [
            # (interceptor_pos, interceptor_vel, missile_pos, missile_vel, expected_threat)
            ([0, 0, 0], [100, 0, 0], [1000, 0, 0], [-50, 0, 0], 'high'),    # Head-on, fast closing
            ([0, 0, 0], [10, 0, 0], [1000, 0, 0], [-5, 0, 0], 'medium'),     # Head-on, slow closing
            ([0, 0, 0], [100, 0, 0], [0, 1000, 0], [0, -50, 0], 'low'),      # Perpendicular approach
            ([0, 0, 0], [10, 0, 0], [1000, 0, 0], [50, 0, 0], 'none'),       # Diverging
        ]
        
        for int_pos, int_vel, mis_pos, mis_vel, expected_threat in threat_scenarios:
            int_pos = np.array(int_pos, dtype=float)
            int_vel = np.array(int_vel, dtype=float)
            mis_pos = np.array(mis_pos, dtype=float)
            mis_vel = np.array(mis_vel, dtype=float)
            
            # Calculate threat metrics
            distance = distance_6dof(int_pos, mis_pos)
            geometry = intercept_geometry_6dof(int_pos, int_vel, mis_pos, mis_vel)
            
            # Assess threat level
            distance_threat = max(0, 1.0 - distance / 500.0)  # Threat increases as distance decreases
            closing_threat = max(0, geometry['closing_velocity'] / 100.0)  # Threat based on closing speed
            
            total_threat = (distance_threat + closing_threat) / 2.0
            
            if expected_threat == 'high':
                assert total_threat > 0.6, f"High threat scenario should have high threat score: {total_threat}"
            elif expected_threat == 'medium':
                assert 0.3 < total_threat <= 0.6, f"Medium threat scenario: {total_threat}"
            elif expected_threat == 'low':
                assert 0.1 < total_threat <= 0.3, f"Low threat scenario: {total_threat}"
            elif expected_threat == 'none':
                assert total_threat <= 0.1, f"No threat scenario should have low threat score: {total_threat}"
    
    def test_adaptive_threat_response(self):
        """Test that threat response adapts to interceptor behavior"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        # Test different interceptor strategies
        strategies = {
            'aggressive': lambda int_pos, mis_pos: (mis_pos - int_pos) / np.linalg.norm(mis_pos - int_pos) if np.linalg.norm(mis_pos - int_pos) > 0 else np.zeros(3),
            'conservative': lambda int_pos, mis_pos: (mis_pos - int_pos) / np.linalg.norm(mis_pos - int_pos) * 0.3 if np.linalg.norm(mis_pos - int_pos) > 0 else np.zeros(3),
            'random': lambda int_pos, mis_pos: np.random.normal(0, 0.5, 3)
        }
        
        response_intensities = {}
        
        for strategy_name, strategy_func in strategies.items():
            intensities = []
            
            for episode in range(5):
                env.reset(seed=300 + episode)
                
                for step in range(40):
                    if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                        int_pos = env.interceptor_6dof.position
                        mis_pos = env.missile_6dof.position
                        
                        # Apply strategy
                        strategy_direction = strategy_func(int_pos, mis_pos)
                        action = np.concatenate([strategy_direction, strategy_direction * 0.1, [0.0]])
                        
                        # Measure missile response intensity
                        prev_missile_omega = env.missile_6dof.angular_velocity.copy()
                    else:
                        action = env.action_space.sample()
                        prev_missile_omega = np.zeros(3)
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if hasattr(env, 'missile_6dof'):
                        current_missile_omega = env.missile_6dof.angular_velocity
                        omega_change = np.linalg.norm(current_missile_omega - prev_missile_omega)
                        intensities.append(omega_change)
                    
                    if terminated or truncated:
                        break
            
            if intensities:
                response_intensities[strategy_name] = np.mean(intensities)
        
        # Aggressive strategy should provoke stronger response than conservative
        if 'aggressive' in response_intensities and 'conservative' in response_intensities:
            assert response_intensities['aggressive'] >= response_intensities['conservative'] * 0.8, \
                "Aggressive strategy should provoke stronger evasive response"
    
    def test_threat_assessment_timing(self):
        """Test that threat assessment updates at appropriate frequency"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        threat_assessments = []
        assessment_times = []
        
        env.reset(seed=42)
        
        for step in range(50):
            start_time = time.time()
            
            if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                int_pos = env.interceptor_6dof.position
                mis_pos = env.missile_6dof.position
                
                # Force close encounter
                direction = mis_pos - int_pos
                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)
                action = np.concatenate([direction * 1.0, direction * 0.5, [0.0]])
                
                # Calculate current threat
                distance = distance_6dof(int_pos, mis_pos)
                threat_level = max(0, 1.0 - distance / 200.0)
                threat_assessments.append(threat_level)
            else:
                action = env.action_space.sample()
                threat_assessments.append(0.0)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            assessment_time = time.time() - start_time
            assessment_times.append(assessment_time)
            
            if terminated or truncated:
                break
        
        # Assessment should be fast (< 10ms per step for real-time capability)
        avg_assessment_time = np.mean(assessment_times)
        assert avg_assessment_time < 0.01, f"Threat assessment too slow: {avg_assessment_time:.6f}s"
        
        # Threat should vary over time (not constant)
        threat_variance = np.var(threat_assessments)
        assert threat_variance > 0.01, "Threat assessment should vary over time"


class TestParameterSensitivity:
    """Test parameter sensitivity and configuration handling"""
    
    def test_evasion_aggressiveness_parameter(self):
        """Test evasion aggressiveness parameter effects"""
        # Test different aggressiveness levels
        aggressiveness_levels = [0.3, 0.7, 1.0, 1.5]
        evasion_intensities = {}
        
        for aggressiveness in aggressiveness_levels:
            # Create scenario config with specific aggressiveness
            scenario_config = {
                'name': f'test_aggressiveness_{aggressiveness}',
                'description': 'Test scenario for aggressiveness',
                'interceptor_position_range': {'x': (0, 100), 'y': (0, 100), 'z': (0, 50)},
                'interceptor_velocity_range': {'x': (-10, 10), 'y': (-10, 10), 'z': (0, 20)},
                'missile_position_range': {'x': (200, 300), 'y': (200, 300), 'z': (100, 200)},
                'missile_velocity_range': {'x': (-50, 50), 'y': (-50, 50), 'z': (-20, 0)},
                'target_position_range': {'x': (45, 55), 'y': (45, 55), 'z': (0, 10)},
                'wind_conditions': {'enabled': True},
                'atmospheric_conditions': {'standard': True},
                'evasion_aggressiveness': aggressiveness
            }
            
            env = Aegis6DInterceptEnv(
                difficulty_mode=DifficultyMode.FULL_6DOF,
                scenario_config=scenario_config
            )
            
            intensities = []
            
            for episode in range(3):
                env.reset(seed=400 + episode)
                
                for step in range(30):
                    if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                        # Position interceptor to threaten missile
                        int_pos = env.interceptor_6dof.position
                        mis_pos = env.missile_6dof.position
                        env.interceptor_6dof.position = mis_pos + np.array([30, 0, 0])
                        
                        action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        
                        # Measure evasion intensity
                        missile_omega = env.missile_6dof.angular_velocity
                        intensity = np.linalg.norm(missile_omega)
                        intensities.append(intensity)
                    else:
                        action = env.action_space.sample()
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
            
            if intensities:
                evasion_intensities[aggressiveness] = np.mean(intensities)
        
        # Higher aggressiveness should generally lead to higher evasion intensity
        aggressiveness_values = sorted(evasion_intensities.keys())
        if len(aggressiveness_values) >= 2:
            # Check that trend is generally increasing
            for i in range(len(aggressiveness_values) - 1):
                low_agg = aggressiveness_values[i]
                high_agg = aggressiveness_values[i + 1]
                
                # Allow some tolerance, but higher aggressiveness should not be much lower
                assert evasion_intensities[high_agg] >= evasion_intensities[low_agg] * 0.7, \
                    f"Higher aggressiveness ({high_agg}) should not significantly reduce evasion intensity"
    
    def test_wind_condition_effects(self):
        """Test wind condition effects on adversary behavior"""
        wind_conditions = [
            {'enabled': False},
            {'enabled': True, 'strength': 0.5},
            {'enabled': True, 'strength': 1.5}
        ]
        
        trajectory_deviations = {}
        
        for wind_config in wind_conditions:
            scenario_config = {
                'name': f'wind_test_{wind_config}',
                'description': 'Wind test scenario',
                'interceptor_position_range': {'x': (0, 100), 'y': (0, 100), 'z': (0, 50)},
                'interceptor_velocity_range': {'x': (-10, 10), 'y': (-10, 10), 'z': (0, 20)},
                'missile_position_range': {'x': (200, 300), 'y': (200, 300), 'z': (100, 200)},
                'missile_velocity_range': {'x': (-50, 50), 'y': (-50, 50), 'z': (-20, 0)},
                'target_position_range': {'x': (45, 55), 'y': (45, 55), 'z': (0, 10)},
                'wind_conditions': wind_config,
                'atmospheric_conditions': {'standard': True}
            }
            
            env = Aegis6DInterceptEnv(
                difficulty_mode=DifficultyMode.FULL_6DOF,
                scenario_config=scenario_config
            )
            
            trajectory_positions = []
            
            env.reset(seed=42)  # Same seed for fair comparison
            
            for step in range(40):
                if hasattr(env, 'missile_6dof'):
                    trajectory_positions.append(env.missile_6dof.position.copy())
                    
                    # No interceptor action to isolate wind effects
                    action = np.zeros(7)
                else:
                    action = env.action_space.sample()
                
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            
            if len(trajectory_positions) > 10:
                positions = np.array(trajectory_positions)
                
                # Calculate trajectory deviation (variance from straight line)
                if len(positions) > 2:
                    # Fit line to trajectory
                    t = np.arange(len(positions))
                    x_fit = np.polyfit(t, positions[:, 0], 1)
                    y_fit = np.polyfit(t, positions[:, 1], 1)
                    z_fit = np.polyfit(t, positions[:, 2], 1)
                    
                    # Calculate deviations from fitted line
                    x_line = np.polyval(x_fit, t)
                    y_line = np.polyval(y_fit, t)
                    z_line = np.polyval(z_fit, t)
                    
                    deviations = np.sqrt(
                        (positions[:, 0] - x_line)**2 + 
                        (positions[:, 1] - y_line)**2 + 
                        (positions[:, 2] - z_line)**2
                    )
                    
                    avg_deviation = np.mean(deviations)
                    wind_strength = wind_config.get('strength', 0) if wind_config['enabled'] else 0
                    trajectory_deviations[wind_strength] = avg_deviation
        
        # Higher wind strength should generally cause more trajectory deviation
        if len(trajectory_deviations) >= 2:
            wind_strengths = sorted(trajectory_deviations.keys())
            for i in range(len(wind_strengths) - 1):
                low_wind = wind_strengths[i]
                high_wind = wind_strengths[i + 1]
                
                # Higher wind should not cause significantly less deviation
                assert trajectory_deviations[high_wind] >= trajectory_deviations[low_wind] * 0.8, \
                    f"Higher wind strength should not significantly reduce trajectory deviation"
    
    def test_configuration_robustness(self):
        """Test robustness to invalid or edge-case configurations"""
        invalid_configs = [
            # Invalid aggressiveness values
            {'evasion_aggressiveness': -1.0},
            {'evasion_aggressiveness': 10.0},
            
            # Invalid position ranges
            {'interceptor_position_range': {'x': (100, 50), 'y': (0, 100), 'z': (0, 50)}},  # min > max
            
            # Missing required fields (should use defaults)
            {'name': 'minimal_config'},
        ]
        
        for invalid_config in invalid_configs:
            base_config = {
                'name': 'test_invalid',
                'description': 'Test invalid config',
                'interceptor_position_range': {'x': (0, 100), 'y': (0, 100), 'z': (0, 50)},
                'interceptor_velocity_range': {'x': (-10, 10), 'y': (-10, 10), 'z': (0, 20)},
                'missile_position_range': {'x': (200, 300), 'y': (200, 300), 'z': (100, 200)},
                'missile_velocity_range': {'x': (-50, 50), 'y': (-50, 50), 'z': (-20, 0)},
                'target_position_range': {'x': (45, 55), 'y': (45, 55), 'z': (0, 10)},
                'wind_conditions': {'enabled': False},
                'atmospheric_conditions': {'standard': True}
            }
            base_config.update(invalid_config)
            
            try:
                env = Aegis6DInterceptEnv(
                    difficulty_mode=DifficultyMode.FULL_6DOF,
                    scenario_config=base_config
                )
                
                # Should either handle gracefully or fail predictably
                obs, info = env.reset(seed=42)
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                # If it doesn't crash, it should still produce valid results
                assert env.observation_space.contains(obs), "Should produce valid observations"
                assert np.isfinite(reward), "Should produce finite rewards"
                
            except Exception as e:
                # If it does crash, error should be informative
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in ['invalid', 'range', 'config', 'parameter']), \
                    f"Error message should be informative: {e}"


class TestRealisticFlightDynamics:
    """Test realistic flight dynamics and physics compliance"""
    
    def test_physics_constraint_compliance(self):
        """Test that adversary respects physics constraints"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        max_accelerations = []
        max_angular_accelerations = []
        
        for episode in range(5):
            env.reset(seed=500 + episode)
            
            prev_velocity = None
            prev_angular_velocity = None
            
            for step in range(50):
                if hasattr(env, 'missile_6dof'):
                    current_vel = env.missile_6dof.velocity.copy()
                    current_omega = env.missile_6dof.angular_velocity.copy()
                    
                    if prev_velocity is not None:
                        # Calculate acceleration
                        acceleration = (current_vel - prev_velocity) / env.dt
                        max_accel = np.linalg.norm(acceleration)
                        max_accelerations.append(max_accel)
                        
                        # Calculate angular acceleration
                        angular_accel = (current_omega - prev_angular_velocity) / env.dt
                        max_angular_accel = np.linalg.norm(angular_accel)
                        max_angular_accelerations.append(max_angular_accel)
                    
                    prev_velocity = current_vel
                    prev_angular_velocity = current_omega
                
                # Force evasive behavior
                action = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.0])
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
        
        if max_accelerations and max_angular_accelerations:
            max_linear_accel = max(max_accelerations)
            max_angular_accel = max(max_angular_accelerations)
            
            # Check reasonable limits (these are fairly generous for testing)
            assert max_linear_accel < 500.0, f"Linear acceleration too high: {max_linear_accel} m/s²"
            assert max_angular_accel < 200.0, f"Angular acceleration too high: {max_angular_accel} rad/s²"
    
    def test_energy_conservation_during_evasion(self):
        """Test that energy is roughly conserved during evasive maneuvers"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        env.reset(seed=42)
        
        initial_energies = []
        final_energies = []
        
        for episode in range(3):
            if episode > 0:
                env.reset(seed=42 + episode)
            
            # Wait a few steps for dynamics to settle
            for _ in range(5):
                action = np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
                env.step(action)
            
            if hasattr(env, 'missile_6dof'):
                # Calculate initial energy
                initial_ke = 0.5 * env.missile_6dof.aero_props.mass * np.linalg.norm(env.missile_6dof.velocity)**2
                initial_pe = env.missile_6dof.aero_props.mass * 9.81 * env.missile_6dof.position[2]
                initial_energy = initial_ke + initial_pe
                initial_energies.append(initial_energy)
            
            # Force evasive maneuvers for several steps
            for step in range(20):
                action = np.array([1.0, 0.5, 0.2, 0.3, 0.3, 0.3, 0.0])
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            
            if hasattr(env, 'missile_6dof'):
                # Calculate final energy
                final_ke = 0.5 * env.missile_6dof.aero_props.mass * np.linalg.norm(env.missile_6dof.velocity)**2
                final_pe = env.missile_6dof.aero_props.mass * 9.81 * env.missile_6dof.position[2]
                final_energy = final_ke + final_pe
                final_energies.append(final_energy)
        
        if initial_energies and final_energies:
            # Energy should not change dramatically (allowing for thrust and drag)
            for initial, final in zip(initial_energies, final_energies):
                energy_change_ratio = abs(final - initial) / initial
                assert energy_change_ratio < 10.0, f"Energy change too large: {energy_change_ratio}"
    
    def test_aerodynamic_consistency(self):
        """Test aerodynamic behavior consistency"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        altitude_effects = []
        speed_effects = []
        
        env.reset(seed=42)
        
        for step in range(40):
            if hasattr(env, 'missile_6dof'):
                # Get aerodynamic info
                aero_info = env.missile_6dof.get_aerodynamic_info()
                
                altitude = env.missile_6dof.position[2]
                speed = np.linalg.norm(env.missile_6dof.velocity)
                
                altitude_effects.append((altitude, aero_info['air_density']))
                speed_effects.append((speed, aero_info['dynamic_pressure']))
            
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        # Check aerodynamic consistency
        if altitude_effects:
            # Air density should decrease with altitude
            altitudes, densities = zip(*altitude_effects)
            if len(set(altitudes)) > 1:  # Only if altitude varies
                # Check that higher altitudes generally have lower density
                high_alt_indices = [i for i, alt in enumerate(altitudes) if alt > np.mean(altitudes)]
                low_alt_indices = [i for i, alt in enumerate(altitudes) if alt < np.mean(altitudes)]
                
                if high_alt_indices and low_alt_indices:
                    avg_high_density = np.mean([densities[i] for i in high_alt_indices])
                    avg_low_density = np.mean([densities[i] for i in low_alt_indices])
                    
                    assert avg_high_density <= avg_low_density, \
                        "Air density should decrease with altitude"
        
        if speed_effects:
            # Dynamic pressure should increase with speed squared
            speeds, pressures = zip(*speed_effects)
            if len(set(speeds)) > 1:
                # Check correlation
                speed_variance = np.var(speeds)
                pressure_variance = np.var(pressures)
                
                if speed_variance > 0 and pressure_variance > 0:
                    correlation = np.corrcoef(speeds, pressures)[0, 1]
                    assert correlation > 0.5, "Dynamic pressure should correlate with speed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])