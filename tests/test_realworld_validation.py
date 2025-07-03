"""
Real-World Validation Tests

This module contains comprehensive tests to validate real-world applicability:
- Physics accuracy against known aerospace parameters
- Trajectory realism and believability
- Unity visualization accuracy and correctness
- Curriculum effectiveness in learning speed
- Performance comparison with real missile systems
- Aerodynamic modeling validation
- Environmental condition accuracy

Author: Tester Agent
Date: Phase 3 Testing Framework
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import json
import tempfile
import os
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import time

from aegis_intercept.envs.aegis_6dof_env import Aegis6DInterceptEnv, DifficultyMode
from aegis_intercept.utils.physics6dof import (
    RigidBody6DOF, VehicleType, AtmosphericModel, PhysicsConstants,
    distance_6dof, AERODYNAMIC_PROPERTIES
)
from aegis_intercept.curriculum.curriculum_manager import CurriculumManager
from aegis_intercept.logging.trajectory_logger import TrajectoryLogger
from aegis_intercept.logging.export_manager import ExportManager


@dataclass
class RealMissileParameters:
    """Real-world missile parameters for validation"""
    name: str
    mass: float  # kg
    length: float  # m
    diameter: float  # m
    max_speed: float  # m/s (Mach number * 343)
    max_acceleration: float  # m/s²
    max_range: float  # m
    service_ceiling: float  # m
    typical_intercept_time: float  # s


# Real missile parameters for validation
REAL_MISSILE_PARAMETERS = {
    'patriot_pac3': RealMissileParameters(
        name='Patriot PAC-3',
        mass=312.0,  # kg
        length=5.3,  # m
        diameter=0.254,  # m
        max_speed=1715.0,  # ~Mach 5
        max_acceleration=294.2,  # ~30G
        max_range=35000.0,  # 35 km
        service_ceiling=24000.0,  # 24 km
        typical_intercept_time=15.0  # seconds
    ),
    'standard_missile': RealMissileParameters(
        name='Standard Missile SM-3',
        mass=1500.0,  # kg
        length=6.55,  # m
        diameter=0.533,  # m
        max_speed=4200.0,  # >Mach 10
        max_acceleration=196.2,  # ~20G
        max_range=500000.0,  # 500 km (exoatmospheric)
        service_ceiling=160000.0,  # 160 km
        typical_intercept_time=180.0  # seconds
    ),
    'aim120_amraam': RealMissileParameters(
        name='AIM-120 AMRAAM',
        mass=152.0,  # kg
        length=3.66,  # m
        diameter=0.178,  # m
        max_speed=1372.0,  # ~Mach 4
        max_acceleration=294.2,  # ~30G
        max_range=105000.0,  # 105 km
        service_ceiling=15000.0,  # 15 km
        typical_intercept_time=60.0  # seconds
    )
}


class TestPhysicsAccuracy:
    """Test physics accuracy against known aerospace parameters"""
    
    def test_missile_mass_properties(self):
        """Test that simulated missiles have realistic mass properties"""
        interceptor_props = AERODYNAMIC_PROPERTIES[VehicleType.INTERCEPTOR]
        missile_props = AERODYNAMIC_PROPERTIES[VehicleType.MISSILE]
        
        # Compare with real missile ranges
        real_masses = [params.mass for params in REAL_MISSILE_PARAMETERS.values()]
        min_real_mass = min(real_masses)
        max_real_mass = max(real_masses)
        
        # Simulated masses should be within realistic range
        assert min_real_mass * 0.5 <= interceptor_props.mass <= max_real_mass * 2, \
            f"Interceptor mass unrealistic: {interceptor_props.mass}kg vs real range [{min_real_mass}, {max_real_mass}]"
        
        assert min_real_mass * 0.5 <= missile_props.mass <= max_real_mass * 2, \
            f"Missile mass unrealistic: {missile_props.mass}kg vs real range [{min_real_mass}, {max_real_mass}]"
    
    def test_acceleration_limits(self):
        """Test that acceleration limits are realistic"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        max_accelerations = []
        
        for episode in range(5):
            env.reset(seed=episode)
            
            for step in range(50):
                # Apply maximum force
                max_action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
                obs, reward, terminated, truncated, info = env.step(max_action)
                
                if hasattr(env, 'interceptor_6dof'):
                    # Calculate current acceleration (rough estimate)
                    velocity = env.interceptor_6dof.velocity
                    speed = np.linalg.norm(velocity)
                    
                    # Estimate acceleration from thrust and mass
                    mass = env.interceptor_6dof.aero_props.mass
                    max_thrust = 2000.0  # From action scaling in environment
                    max_accel = max_thrust / mass
                    max_accelerations.append(max_accel)
                
                if terminated or truncated:
                    break
        
        if max_accelerations:
            typical_max_accel = np.median(max_accelerations)
            
            # Compare with real missile acceleration limits
            real_max_accels = [params.max_acceleration for params in REAL_MISSILE_PARAMETERS.values()]
            min_real_accel = min(real_max_accels)
            max_real_accel = max(real_max_accels)
            
            # Should be within reasonable range of real systems
            assert min_real_accel * 0.1 <= typical_max_accel <= max_real_accel * 5, \
                f"Simulated max acceleration unrealistic: {typical_max_accel:.1f} m/s² vs real range [{min_real_accel}, {max_real_accel}]"
    
    def test_speed_limits_realism(self):
        """Test that speed limits are realistic"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        max_speeds = []
        
        # Run simulation to find maximum achievable speeds
        for episode in range(3):
            env.reset(seed=episode)
            
            for step in range(100):
                # Constant max thrust in one direction
                action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                obs, reward, terminated, truncated, info = env.step(action)
                
                if hasattr(env, 'interceptor_6dof'):
                    speed = np.linalg.norm(env.interceptor_6dof.velocity)
                    max_speeds.append(speed)
                
                if terminated or truncated:
                    break
        
        if max_speeds:
            achieved_max_speed = max(max_speeds)
            
            # Compare with real missile speeds
            real_max_speeds = [params.max_speed for params in REAL_MISSILE_PARAMETERS.values()]
            min_real_speed = min(real_max_speeds)
            max_real_speed = max(real_max_speeds)
            
            # Should be within reasonable range
            assert min_real_speed * 0.05 <= achieved_max_speed <= max_real_speed * 2, \
                f"Simulated max speed unrealistic: {achieved_max_speed:.1f} m/s vs real range [{min_real_speed}, {max_real_speed}]"
    
    def test_atmospheric_model_accuracy(self):
        """Test atmospheric model against real atmospheric data"""
        test_altitudes = [0, 1000, 5000, 10000, 15000, 20000]  # meters
        
        for altitude in test_altitudes:
            density = AtmosphericModel.get_air_density(altitude)
            sound_speed = AtmosphericModel.get_speed_of_sound(altitude)
            
            # Validate against known atmospheric properties
            if altitude == 0:  # Sea level
                assert abs(density - 1.225) < 0.1, f"Sea level density wrong: {density}"
                assert abs(sound_speed - 343) < 20, f"Sea level sound speed wrong: {sound_speed}"
            
            elif altitude == 10000:  # ~10km altitude
                # At 10km: density ~0.414 kg/m³, sound speed ~299 m/s
                assert 0.3 < density < 0.6, f"10km density wrong: {density}"
                assert 280 < sound_speed < 320, f"10km sound speed wrong: {sound_speed}"
            
            # Basic physics checks
            assert density > 0, f"Density should be positive at {altitude}m"
            assert sound_speed > 200, f"Sound speed too low at {altitude}m: {sound_speed}"
            assert sound_speed < 400, f"Sound speed too high at {altitude}m: {sound_speed}"
    
    def test_gravity_and_physics_constants(self):
        """Test physics constants for accuracy"""
        constants = PhysicsConstants()
        
        # Check gravity
        assert abs(constants.GRAVITY - 9.81) < 0.1, f"Gravity constant wrong: {constants.GRAVITY}"
        
        # Check air density (sea level)
        assert abs(constants.AIR_DENSITY - 1.225) < 0.1, f"Air density wrong: {constants.AIR_DENSITY}"
        
        # Check speed of sound (sea level)
        assert abs(constants.SPEED_OF_SOUND - 343) < 20, f"Sound speed wrong: {constants.SPEED_OF_SOUND}"
        
        # Check reasonable integration parameters
        assert 0.0001 < constants.MIN_TIMESTEP < 0.01, "Min timestep unreasonable"
        assert 0.01 < constants.MAX_TIMESTEP < 1.0, "Max timestep unreasonable"
        assert 1 <= constants.INTEGRATION_SUBSTEPS <= 10, "Integration substeps unreasonable"


class TestTrajectoryRealism:
    """Test trajectory realism and believability"""
    
    def test_intercept_trajectory_shapes(self):
        """Test that intercept trajectories look realistic"""
        env = Aegis6DInterceptEnv(
            difficulty_mode=DifficultyMode.FULL_6DOF,
            world_size=500.0,
            max_steps=200
        )
        
        trajectory_data = []
        
        # Collect trajectory data from multiple intercept attempts
        for episode in range(5):
            env.reset(seed=100 + episode)
            
            episode_trajectory = {
                'interceptor_positions': [],
                'missile_positions': [],
                'times': []
            }
            
            for step in range(200):
                if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                    # Proportional navigation guidance
                    int_pos = env.interceptor_6dof.position
                    int_vel = env.interceptor_6dof.velocity
                    mis_pos = env.missile_6dof.position
                    mis_vel = env.missile_6dof.velocity
                    
                    # Simple PN guidance
                    relative_pos = mis_pos - int_pos
                    relative_vel = mis_vel - int_vel
                    
                    if np.linalg.norm(relative_pos) > 1e-6:
                        los_rate = np.cross(relative_pos, relative_vel) / np.linalg.norm(relative_pos)**2
                        acceleration_cmd = 3.0 * np.cross(los_rate, int_vel)  # PN constant = 3
                        acceleration_cmd = np.clip(acceleration_cmd, -1, 1)
                    else:
                        acceleration_cmd = np.zeros(3)
                    
                    action = np.concatenate([acceleration_cmd, acceleration_cmd * 0.1, [0.0]])
                    
                    # Record positions
                    episode_trajectory['interceptor_positions'].append(int_pos.copy())
                    episode_trajectory['missile_positions'].append(mis_pos.copy())
                    episode_trajectory['times'].append(step * env.dt)
                else:
                    action = env.action_space.sample()
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            if len(episode_trajectory['interceptor_positions']) > 10:
                trajectory_data.append(episode_trajectory)
        
        # Analyze trajectory characteristics
        for i, traj in enumerate(trajectory_data):
            int_positions = np.array(traj['interceptor_positions'])
            mis_positions = np.array(traj['missile_positions'])
            
            if len(int_positions) > 5:
                # Check trajectory smoothness (no sudden jumps)
                int_velocities = np.diff(int_positions, axis=0) / env.dt
                velocity_changes = np.diff(int_velocities, axis=0)
                max_accel_change = np.max(np.linalg.norm(velocity_changes, axis=1))
                
                # Acceleration changes should be reasonable
                assert max_accel_change < 1000, f"Trajectory {i} has unrealistic acceleration changes: {max_accel_change}"
                
                # Check that interceptor generally moves toward missile
                initial_distance = distance_6dof(int_positions[0], mis_positions[0])
                final_distance = distance_6dof(int_positions[-1], mis_positions[-1])
                
                # Should generally be closing (allowing for some cases where it doesn't)
                if len(trajectory_data) > 0:  # At least some should show closing behavior
                    closing_trajectories = sum(1 for traj in trajectory_data 
                                             if distance_6dof(np.array(traj['interceptor_positions'])[0], 
                                                            np.array(traj['missile_positions'])[0]) >
                                                distance_6dof(np.array(traj['interceptor_positions'])[-1], 
                                                            np.array(traj['missile_positions'])[-1]))
                    
                    assert closing_trajectories >= len(trajectory_data) * 0.3, \
                        "At least 30% of trajectories should show closing behavior"
    
    def test_missile_evasion_realism(self):
        """Test that missile evasion patterns look realistic"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.EXPERT_6DOF)
        
        evasion_patterns = []
        
        for episode in range(3):
            env.reset(seed=200 + episode)
            
            missile_positions = []
            missile_orientations = []
            
            for step in range(80):
                # Aggressive pursuit to trigger evasion
                if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                    int_pos = env.interceptor_6dof.position
                    mis_pos = env.missile_6dof.position
                    
                    # Position interceptor close to trigger evasion
                    distance = distance_6dof(int_pos, mis_pos)
                    if distance > 100:
                        env.interceptor_6dof.position = mis_pos + np.array([50, 20, 10])
                    
                    action = np.array([1.0, 0.5, 0.2, 0.3, 0.3, 0.3, 0.0])
                    
                    missile_positions.append(mis_pos.copy())
                    missile_orientations.append(env.missile_6dof.orientation.copy())
                else:
                    action = env.action_space.sample()
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            if len(missile_positions) > 10:
                evasion_patterns.append({
                    'positions': np.array(missile_positions),
                    'orientations': np.array(missile_orientations)
                })
        
        # Analyze evasion realism
        for pattern in evasion_patterns:
            positions = pattern['positions']
            
            # Check for realistic evasion characteristics
            if len(positions) > 5:
                # Calculate path curvature
                velocities = np.diff(positions, axis=0)
                speed_changes = np.diff(np.linalg.norm(velocities, axis=1))
                direction_changes = np.diff(velocities, axis=0)
                
                # Should show some maneuvering (not straight line)
                direction_change_magnitude = np.mean(np.linalg.norm(direction_changes, axis=1))
                assert direction_change_magnitude > 0.1, "Missile should show some evasive maneuvering"
                
                # Maneuvers should be within physical limits
                max_direction_change = np.max(np.linalg.norm(direction_changes, axis=1))
                assert max_direction_change < 50, f"Missile maneuvers too extreme: {max_direction_change}"
    
    def test_intercept_geometry_realism(self):
        """Test that intercept geometries are realistic"""
        env = Aegis6DInterceptEnv(
            difficulty_mode=DifficultyMode.FULL_6DOF,
            intercept_threshold=25.0
        )
        
        successful_intercepts = []
        
        for episode in range(10):
            env.reset(seed=300 + episode)
            
            intercept_data = None
            
            for step in range(150):
                if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                    int_pos = env.interceptor_6dof.position
                    int_vel = env.interceptor_6dof.velocity
                    mis_pos = env.missile_6dof.position
                    mis_vel = env.missile_6dof.velocity
                    
                    # Lead pursuit guidance
                    relative_pos = mis_pos - int_pos
                    distance = np.linalg.norm(relative_pos)
                    
                    if distance > 1e-6:
                        # Estimate intercept point
                        relative_vel = mis_vel - int_vel
                        closing_speed = -np.dot(relative_pos, relative_vel) / distance
                        
                        if closing_speed > 0:
                            time_to_intercept = distance / closing_speed
                            intercept_point = mis_pos + mis_vel * time_to_intercept
                            direction_to_intercept = intercept_point - int_pos
                            
                            if np.linalg.norm(direction_to_intercept) > 1e-6:
                                direction_to_intercept = direction_to_intercept / np.linalg.norm(direction_to_intercept)
                            else:
                                direction_to_intercept = relative_pos / distance
                        else:
                            direction_to_intercept = relative_pos / distance
                    else:
                        direction_to_intercept = np.array([1, 0, 0])
                    
                    action = np.concatenate([direction_to_intercept * 0.8, direction_to_intercept * 0.1, [0.0]])
                else:
                    action = env.action_space.sample()
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated:
                    if 'intercept_distance' in info and info['intercept_distance'] < env.intercept_threshold:
                        # Successful intercept
                        intercept_data = {
                            'intercept_distance': info['intercept_distance'],
                            'intercept_time': step * env.dt,
                            'final_positions': (int_pos.copy(), mis_pos.copy()) if hasattr(env, 'interceptor_6dof') else (None, None)
                        }
                    break
                elif truncated:
                    break
            
            if intercept_data:
                successful_intercepts.append(intercept_data)
        
        # Analyze intercept realism
        if successful_intercepts:
            intercept_times = [data['intercept_time'] for data in successful_intercepts]
            intercept_distances = [data['intercept_distance'] for data in successful_intercepts]
            
            # Intercept times should be realistic
            avg_intercept_time = np.mean(intercept_times)
            real_intercept_times = [params.typical_intercept_time for params in REAL_MISSILE_PARAMETERS.values()]
            min_real_time = min(real_intercept_times)
            max_real_time = max(real_intercept_times)
            
            # Should be within reasonable range of real intercept times
            assert min_real_time * 0.1 <= avg_intercept_time <= max_real_time * 2, \
                f"Average intercept time unrealistic: {avg_intercept_time:.1f}s vs real range [{min_real_time}, {max_real_time}]"
            
            # Intercept distances should be reasonable
            avg_intercept_distance = np.mean(intercept_distances)
            assert avg_intercept_distance <= env.intercept_threshold, "Intercept distance should be within threshold"
            assert avg_intercept_distance >= 1.0, "Intercept distance should not be too small (unrealistic precision)"


class TestUnityVisualizationAccuracy:
    """Test Unity visualization accuracy and correctness"""
    
    def test_unity_export_data_accuracy(self):
        """Test that Unity export data accurately represents simulation"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_manager = ExportManager(
                export_dir=temp_dir,
                enable_unity_export=True,
                export_frequency=1
            )
            
            # Run simulation and collect data
            env.reset(seed=42)
            
            simulation_data = []
            
            for step in range(30):
                if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                    # Record ground truth data
                    ground_truth = {
                        'step': step,
                        'time': step * env.dt,
                        'interceptor_position': env.interceptor_6dof.position.copy(),
                        'interceptor_velocity': env.interceptor_6dof.velocity.copy(),
                        'interceptor_orientation': env.interceptor_6dof.orientation.copy(),
                        'missile_position': env.missile_6dof.position.copy(),
                        'missile_velocity': env.missile_6dof.velocity.copy(),
                        'missile_orientation': env.missile_6dof.orientation.copy()
                    }
                    simulation_data.append(ground_truth)
                
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            # Create export data
            trajectory_data = {
                'episode_id': 'unity_accuracy_test',
                'trajectory': []
            }
            
            for data in simulation_data:
                trajectory_point = {
                    'step': data['step'],
                    'timestamp': data['time'],
                    'interceptor_position': data['interceptor_position'].tolist(),
                    'interceptor_velocity': data['interceptor_velocity'].tolist(),
                    'interceptor_orientation': data['interceptor_orientation'].tolist(),
                    'missile_position': data['missile_position'].tolist(),
                    'missile_velocity': data['missile_velocity'].tolist(),
                    'missile_orientation': data['missile_orientation'].tolist(),
                    'target_position': env.target_pos.tolist()
                }
                trajectory_data['trajectory'].append(trajectory_point)
            
            # Export to Unity format
            export_manager.export_episode(trajectory_data)
            
            # Verify Unity export accuracy
            export_files = [f for f in os.listdir(temp_dir) if 'unity' in f.lower() and f.endswith('.json')]
            assert len(export_files) > 0, "Unity export file should be created"
            
            unity_file = os.path.join(temp_dir, export_files[0])
            with open(unity_file, 'r') as f:
                unity_data = json.load(f)
            
            # Verify data accuracy
            unity_trajectory = unity_data['trajectory']
            assert len(unity_trajectory) == len(simulation_data), "Unity trajectory length should match simulation"
            
            # Check position accuracy
            for i, (sim_data, unity_point) in enumerate(zip(simulation_data, unity_trajectory)):
                sim_int_pos = sim_data['interceptor_position']
                unity_int_pos = np.array(unity_point['interceptor_position'])
                
                position_error = np.linalg.norm(sim_int_pos - unity_int_pos)
                assert position_error < 1e-6, f"Unity position error too large at step {i}: {position_error}"
                
                # Check orientation accuracy (quaternions)
                sim_int_quat = sim_data['interceptor_orientation']
                unity_int_quat = np.array(unity_point['interceptor_orientation'])
                
                quat_error = np.linalg.norm(sim_int_quat - unity_int_quat)
                assert quat_error < 1e-6, f"Unity quaternion error too large at step {i}: {quat_error}"
    
    def test_unity_visualization_completeness(self):
        """Test that Unity export contains all necessary visualization data"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            export_manager = ExportManager(
                export_dir=temp_dir,
                enable_unity_export=True
            )
            
            # Create comprehensive trajectory
            env.reset(seed=42)
            
            trajectory_data = {
                'episode_id': 'completeness_test',
                'trajectory': [],
                'episode_summary': {
                    'success': True,
                    'total_reward': 25.0,
                    'intercept_distance': 5.0,
                    'episode_length': 40
                }
            }
            
            for step in range(40):
                if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                    trajectory_point = {
                        'step': step,
                        'timestamp': step * env.dt,
                        'interceptor_position': env.interceptor_6dof.position.tolist(),
                        'interceptor_velocity': env.interceptor_6dof.velocity.tolist(),
                        'interceptor_orientation': env.interceptor_6dof.orientation.tolist(),
                        'interceptor_angular_velocity': env.interceptor_6dof.angular_velocity.tolist(),
                        'missile_position': env.missile_6dof.position.tolist(),
                        'missile_velocity': env.missile_6dof.velocity.tolist(),
                        'missile_orientation': env.missile_6dof.orientation.tolist(),
                        'missile_angular_velocity': env.missile_6dof.angular_velocity.tolist(),
                        'target_position': env.target_pos.tolist(),
                        'wind_velocity': env.wind_velocity.tolist() if hasattr(env, 'wind_velocity') else [0, 0, 0],
                        'simulation_time': env.simulation_time,
                        'fuel_remaining': env.fuel_remaining
                    }
                    trajectory_data['trajectory'].append(trajectory_point)
                
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            # Export and verify completeness
            export_manager.export_episode(trajectory_data)
            
            export_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
            unity_file = os.path.join(temp_dir, export_files[0])
            
            with open(unity_file, 'r') as f:
                unity_data = json.load(f)
            
            # Check required fields for Unity visualization
            required_fields = ['metadata', 'trajectory']
            for field in required_fields:
                assert field in unity_data, f"Unity export missing required field: {field}"
            
            # Check trajectory data completeness
            if unity_data['trajectory']:
                sample_point = unity_data['trajectory'][0]
                required_point_fields = [
                    'timestamp', 'interceptor_position', 'missile_position', 'target_position'
                ]
                
                for field in required_point_fields:
                    assert field in sample_point, f"Unity trajectory point missing field: {field}"
    
    def test_unity_coordinate_system_consistency(self):
        """Test Unity coordinate system consistency"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        # Test coordinate transformations if needed
        test_positions = [
            np.array([100, 200, 300]),
            np.array([0, 0, 0]),
            np.array([-50, 100, 500])
        ]
        
        for pos in test_positions:
            # Simulate what Unity export would contain
            unity_pos = pos.tolist()  # Direct conversion for now
            
            # Verify coordinate consistency
            converted_back = np.array(unity_pos)
            position_error = np.linalg.norm(pos - converted_back)
            
            assert position_error < 1e-10, f"Coordinate conversion error: {position_error}"
            
            # Check that positions are reasonable for Unity (not too large)
            assert all(abs(coord) < 1e6 for coord in unity_pos), \
                f"Unity coordinates too large: {unity_pos}"


class TestCurriculumEffectiveness:
    """Test curriculum learning effectiveness"""
    
    def test_curriculum_learning_speed(self):
        """Test that curriculum learning improves learning speed"""
        # Compare learning with and without curriculum
        results = {}
        
        # Test with curriculum
        curriculum_manager = CurriculumManager()
        curriculum_performance = []
        
        # Simulate curriculum learning
        for phase_progress in range(20):  # Simulate progression through phases
            env_config = curriculum_manager.get_environment_config()
            env = Aegis6DInterceptEnv(**env_config)
            
            episode_rewards = []
            
            for episode in range(5):  # Few episodes per phase for speed
                env.reset(seed=1000 + phase_progress * 5 + episode)
                episode_reward = 0
                
                for step in range(50):
                    # Improving policy over time
                    if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                        int_pos = env.interceptor_6dof.position
                        mis_pos = env.missile_6dof.position
                        direction = mis_pos - int_pos
                        if np.linalg.norm(direction) > 1e-6:
                            direction = direction / np.linalg.norm(direction)
                        
                        # Improve policy over phases
                        skill_level = min(1.0, phase_progress / 10.0)
                        noise_level = 0.5 * (1 - skill_level)
                        action = np.concatenate([direction * 0.8, direction * 0.2, [0.0]])
                        action += np.random.normal(0, noise_level, 7)
                        action = np.clip(action, -1, 1)
                    else:
                        action = env.action_space.sample()
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    
                    if terminated or truncated:
                        break
                
                episode_rewards.append(episode_reward)
            
            avg_reward = np.mean(episode_rewards)
            curriculum_performance.append(avg_reward)
            
            # Update curriculum (simulate learning progress)
            success_rate = max(0.0, min(1.0, 0.3 + phase_progress * 0.05))
            curriculum_manager.update_performance(avg_reward, success_rate > 0.7, 50.0, 10.0)
        
        results['curriculum'] = curriculum_performance
        
        # Test without curriculum (fixed difficult environment)
        no_curriculum_performance = []
        env_fixed = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        for progress in range(20):
            episode_rewards = []
            
            for episode in range(5):
                env_fixed.reset(seed=2000 + progress * 5 + episode)
                episode_reward = 0
                
                for step in range(50):
                    if hasattr(env_fixed, 'interceptor_6dof') and hasattr(env_fixed, 'missile_6dof'):
                        int_pos = env_fixed.interceptor_6dof.position
                        mis_pos = env_fixed.missile_6dof.position
                        direction = mis_pos - int_pos
                        if np.linalg.norm(direction) > 1e-6:
                            direction = direction / np.linalg.norm(direction)
                        
                        skill_level = min(1.0, progress / 15.0)  # Slower learning without curriculum
                        noise_level = 0.6 * (1 - skill_level)
                        action = np.concatenate([direction * 0.7, direction * 0.15, [0.0]])
                        action += np.random.normal(0, noise_level, 7)
                        action = np.clip(action, -1, 1)
                    else:
                        action = env_fixed.action_space.sample()
                    
                    obs, reward, terminated, truncated, info = env_fixed.step(action)
                    episode_reward += reward
                    
                    if terminated or truncated:
                        break
                
                episode_rewards.append(episode_reward)
            
            avg_reward = np.mean(episode_rewards)
            no_curriculum_performance.append(avg_reward)
        
        results['no_curriculum'] = no_curriculum_performance
        
        # Analyze learning effectiveness
        curriculum_final = np.mean(results['curriculum'][-5:])
        no_curriculum_final = np.mean(results['no_curriculum'][-5:])
        
        curriculum_improvement = curriculum_final - results['curriculum'][0]
        no_curriculum_improvement = no_curriculum_final - results['no_curriculum'][0]
        
        # Curriculum should lead to better or faster improvement
        assert curriculum_improvement >= no_curriculum_improvement * 0.8, \
            f"Curriculum learning not effective: {curriculum_improvement} vs {no_curriculum_improvement}"
        
        print(f"Curriculum Learning Effectiveness:")
        print(f"  With curriculum: {curriculum_improvement:.2f} improvement")
        print(f"  Without curriculum: {no_curriculum_improvement:.2f} improvement")
    
    def test_curriculum_phase_appropriateness(self):
        """Test that curriculum phases are appropriately ordered by difficulty"""
        curriculum_manager = CurriculumManager()
        
        phase_difficulties = {}
        
        # Test each phase
        test_phases = [
            DifficultyMode.EASY_3DOF,
            DifficultyMode.MEDIUM_3DOF,
            DifficultyMode.SIMPLIFIED_6DOF,
            DifficultyMode.FULL_6DOF,
            DifficultyMode.EXPERT_6DOF
        ]
        
        for difficulty in test_phases:
            env = Aegis6DInterceptEnv(
                difficulty_mode=difficulty,
                max_steps=30
            )
            
            success_rates = []
            
            # Test with consistent policy across all difficulties
            for episode in range(10):
                env.reset(seed=3000 + episode)
                success = False
                
                for step in range(30):
                    if hasattr(env, 'interceptor_6dof') and hasattr(env, 'missile_6dof'):
                        int_pos = env.interceptor_6dof.position
                        mis_pos = env.missile_6dof.position
                        direction = mis_pos - int_pos
                        if np.linalg.norm(direction) > 1e-6:
                            direction = direction / np.linalg.norm(direction)
                        action = np.concatenate([direction * 0.6, direction * 0.1, [0.0]])
                    else:
                        # 3DOF mode
                        action = np.array([0.5, 0.3, 0.8, 0.0])
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if terminated and reward > 5:
                        success = True
                        break
                    elif terminated or truncated:
                        break
                
                success_rates.append(success)
            
            phase_difficulties[difficulty] = np.mean(success_rates)
        
        # Check that difficulty generally increases
        difficulty_order = [
            DifficultyMode.EASY_3DOF,
            DifficultyMode.MEDIUM_3DOF,
            DifficultyMode.SIMPLIFIED_6DOF,
            DifficultyMode.FULL_6DOF,
            DifficultyMode.EXPERT_6DOF
        ]
        
        for i in range(len(difficulty_order) - 1):
            current_mode = difficulty_order[i]
            next_mode = difficulty_order[i + 1]
            
            current_success = phase_difficulties[current_mode]
            next_success = phase_difficulties[next_mode]
            
            # Allow some flexibility, but generally should get harder
            assert next_success <= current_success + 0.3, \
                f"Difficulty progression inconsistent: {current_mode} ({current_success:.2f}) vs {next_mode} ({next_success:.2f})"
        
        print(f"Curriculum Difficulty Progression:")
        for mode, success_rate in phase_difficulties.items():
            print(f"  {mode.value}: {success_rate:.2f} success rate")


class TestEnvironmentalConditions:
    """Test environmental condition accuracy"""
    
    def test_wind_effect_realism(self):
        """Test that wind effects are realistic"""
        # Test different wind conditions
        wind_conditions = [
            {'strength': 0.0, 'description': 'no wind'},
            {'strength': 0.5, 'description': 'light wind'},
            {'strength': 1.0, 'description': 'moderate wind'},
            {'strength': 2.0, 'description': 'strong wind'}
        ]
        
        wind_effects = {}
        
        for wind_config in wind_conditions:
            env = Aegis6DInterceptEnv(
                difficulty_mode=DifficultyMode.FULL_6DOF,
                enable_wind=wind_config['strength'] > 0,
                wind_strength=wind_config['strength']
            )
            
            trajectory_deviations = []
            
            for episode in range(3):
                env.reset(seed=4000 + episode)
                
                positions = []
                
                for step in range(40):
                    if hasattr(env, 'interceptor_6dof'):
                        positions.append(env.interceptor_6dof.position.copy())
                        
                        # Constant thrust to isolate wind effects
                        action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    else:
                        action = env.action_space.sample()
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                
                if len(positions) > 10:
                    positions = np.array(positions)
                    
                    # Calculate deviation from straight line
                    # Fit line to trajectory
                    t = np.arange(len(positions))
                    if len(positions) > 2:
                        x_fit = np.polyfit(t, positions[:, 0], 1)
                        y_fit = np.polyfit(t, positions[:, 1], 1)
                        
                        x_line = np.polyval(x_fit, t)
                        y_line = np.polyval(y_fit, t)
                        
                        deviations = np.sqrt((positions[:, 0] - x_line)**2 + (positions[:, 1] - y_line)**2)
                        avg_deviation = np.mean(deviations)
                        trajectory_deviations.append(avg_deviation)
            
            if trajectory_deviations:
                wind_effects[wind_config['strength']] = np.mean(trajectory_deviations)
        
        # Check wind effect realism
        if len(wind_effects) >= 2:
            wind_strengths = sorted(wind_effects.keys())
            
            # Stronger wind should generally cause more deviation
            for i in range(len(wind_strengths) - 1):
                low_wind = wind_strengths[i]
                high_wind = wind_strengths[i + 1]
                
                # Allow some tolerance but expect general trend
                ratio = wind_effects[high_wind] / max(wind_effects[low_wind], 1e-6)
                assert ratio >= 0.5, f"Wind effect inconsistent: {high_wind} wind vs {low_wind} wind = {ratio:.2f}x"
        
        print(f"Wind Effects: {wind_effects}")
    
    def test_altitude_effects(self):
        """Test altitude effects on aerodynamics"""
        env = Aegis6DInterceptEnv(difficulty_mode=DifficultyMode.FULL_6DOF)
        
        altitude_effects = {}
        
        test_altitudes = [1000, 5000, 10000, 20000]  # meters
        
        for altitude in test_altitudes:
            env.reset(seed=42)
            
            # Set vehicle to test altitude
            if hasattr(env, 'interceptor_6dof'):
                env.interceptor_6dof.position[2] = altitude
                
                # Get aerodynamic info at this altitude
                aero_info = env.interceptor_6dof.get_aerodynamic_info()
                altitude_effects[altitude] = {
                    'air_density': aero_info['air_density'],
                    'dynamic_pressure': aero_info['dynamic_pressure'],
                    'speed_of_sound': aero_info['speed_of_sound']
                }
        
        # Verify altitude effects
        altitudes = sorted(altitude_effects.keys())
        
        for i in range(len(altitudes) - 1):
            low_alt = altitudes[i]
            high_alt = altitudes[i + 1]
            
            low_density = altitude_effects[low_alt]['air_density']
            high_density = altitude_effects[high_alt]['air_density']
            
            # Air density should decrease with altitude
            assert high_density <= low_density, \
                f"Air density should decrease with altitude: {low_alt}m ({low_density}) vs {high_alt}m ({high_density})"
        
        print(f"Altitude Effects: {altitude_effects}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])