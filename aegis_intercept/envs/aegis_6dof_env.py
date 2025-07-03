"""
Enhanced 6DOF Missile Intercept Environment for AegisIntercept Phase 3

This environment extends the 3DOF system to full 6 Degrees of Freedom simulation
with quaternion-based orientation, aerodynamic forces/torques, and realistic
missile/interceptor physics. It maintains backward compatibility with 3DOF mode
for curriculum learning.

Features:
- 29-dimensional state space (13D per vehicle + 3D environment)
- 6DOF action space with thrust vectors and attitude commands
- Curriculum learning compatibility (3DOF → 6DOF progression)
- Realistic aerodynamic modeling
- Advanced evasion behaviors
- Comprehensive logging integration

Author: Coder Agent  
Date: Phase 3 Implementation
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, Union
from enum import Enum
import json
import os

from ..utils.physics6dof import (
    RigidBody6DOF, VehicleType, AtmosphericModel, QuaternionUtils,
    distance_6dof, intercept_geometry_6dof, PhysicsConstants
)
from ..utils.physics3d import distance as distance_3d  # For 3DOF compatibility
from ..rendering.viewer3d import Viewer3D


class DifficultyMode(Enum):
    """Difficulty modes for curriculum learning"""
    EASY_3DOF = "easy_3dof"          # Original 3DOF physics
    MEDIUM_3DOF = "medium_3dof"      # 3DOF with enhanced adversary
    SIMPLIFIED_6DOF = "simplified_6dof"  # 6DOF with reduced complexity
    FULL_6DOF = "full_6dof"          # Complete 6DOF simulation
    EXPERT_6DOF = "expert_6dof"      # 6DOF with advanced scenarios


class ActionMode(Enum):
    """Action space modes"""
    ACCELERATION_3DOF = "accel_3dof"     # [ax, ay, az, explode] - 4D
    ACCELERATION_6DOF = "accel_6dof"     # [fx, fy, fz, tx, ty, tz, explode] - 7D
    THRUST_ATTITUDE = "thrust_attitude"   # [thrust_mag, pitch, yaw, roll, explode] - 5D


@gym.envs.registration.register(
    id='Aegis6DIntercept-v0',
    entry_point='aegis_intercept.envs:Aegis6DInterceptEnv',
)
class Aegis6DInterceptEnv(gym.Env):
    """Enhanced 6DOF Missile Intercept Environment"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
    
    def __init__(
        self,
        # Environment parameters
        world_size: float = 300.0,
        max_steps: int = 300,
        dt: float = 0.05,
        
        # Mission parameters
        intercept_threshold: float = 30.0,
        miss_threshold: float = 10.0,
        explosion_radius: float = 50.0,
        
        # Difficulty and curriculum settings
        difficulty_mode: Union[DifficultyMode, str] = DifficultyMode.FULL_6DOF,
        action_mode: Union[ActionMode, str] = ActionMode.ACCELERATION_6DOF,
        
        # Physics parameters
        enable_wind: bool = True,
        enable_atmosphere: bool = True,
        wind_strength: float = 1.0,
        
        # Rendering
        render_mode: Optional[str] = None,
        
        # Compatibility with 3DOF
        legacy_3dof_mode: bool = False,
        
        # Advanced features
        enable_fuel_system: bool = True,
        max_fuel: float = 150.0,
        fuel_burn_rate: float = 0.8,
        
        # Scenario loading
        scenario_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        # Store configuration
        self.world_size = world_size
        self.max_steps = max_steps
        self.dt = dt
        self.intercept_threshold = intercept_threshold
        self.miss_threshold = miss_threshold
        self.explosion_radius = explosion_radius
        self.render_mode = render_mode
        
        # Difficulty and action modes
        if isinstance(difficulty_mode, str):
            self.difficulty_mode = DifficultyMode(difficulty_mode)
        else:
            self.difficulty_mode = difficulty_mode
            
        if isinstance(action_mode, str):
            self.action_mode = ActionMode(action_mode)
        else:
            self.action_mode = action_mode
        
        # Physics settings
        self.enable_wind = enable_wind
        self.enable_atmosphere = enable_atmosphere
        self.wind_strength = wind_strength
        self.legacy_3dof_mode = legacy_3dof_mode
        
        # Fuel system
        self.enable_fuel_system = enable_fuel_system
        self.max_fuel = max_fuel
        self.fuel_burn_rate = fuel_burn_rate
        
        # Load scenario configuration
        self.scenario_config = scenario_config or {}
        
        # Physics constants
        self.physics_constants = PhysicsConstants()
        
        # Initialize 6DOF physics if not in legacy mode
        if not self.legacy_3dof_mode:
            self.interceptor_6dof = None
            self.missile_6dof = None
        
        # Legacy 3DOF state for compatibility
        self.interceptor_pos_3d = np.zeros(3, dtype=np.float32)
        self.interceptor_vel_3d = np.zeros(3, dtype=np.float32)
        self.missile_pos_3d = np.zeros(3, dtype=np.float32)
        self.missile_vel_3d = np.zeros(3, dtype=np.float32)
        self.target_pos = np.zeros(3, dtype=np.float32)
        
        # Environment state
        self.step_count = 0
        self.simulation_time = 0.0
        self.fuel_remaining = self.max_fuel
        self.wind_velocity = np.zeros(3)
        
        # Setup observation and action spaces
        self._setup_spaces()
        
        # Rendering
        self.viewer = None
        self.popup_message = None
        self.popup_timer = 0
        
        # Performance tracking
        self.episode_stats = {}
        
    def _setup_spaces(self):
        """Setup observation and action spaces based on difficulty mode"""
        
        # Action space setup
        if self.action_mode == ActionMode.ACCELERATION_3DOF or self.legacy_3dof_mode:
            # [ax, ay, az, explode] - compatible with original 3DOF
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float32
            )
        elif self.action_mode == ActionMode.ACCELERATION_6DOF:
            # [fx, fy, fz, tx, ty, tz, explode] - full 6DOF forces and torques
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(7,), dtype=np.float32
            )
        elif self.action_mode == ActionMode.THRUST_ATTITUDE:
            # [thrust_magnitude, pitch_cmd, yaw_cmd, roll_cmd, explode]
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(5,), dtype=np.float32
            )
        
        # Observation space setup
        if self.legacy_3dof_mode or self.difficulty_mode == DifficultyMode.EASY_3DOF:
            # Original 3DOF observation space (14D)
            # [interceptor_pos(3), interceptor_vel(3), missile_pos(3), missile_vel(3), time_remaining(1), fuel_remaining(1)]
            low = np.array(
                [0, 0, 0] +  # interceptor_pos
                [-100, -100, -100] +  # interceptor_vel
                [0, 0, 0] +  # missile_pos
                [-100, -100, -100] +  # missile_vel
                [0] +  # time_remaining
                [0],  # fuel_remaining
                dtype=np.float32
            )
            high = np.array(
                [self.world_size * 2] * 3 +  # interceptor_pos
                [100] * 3 +  # interceptor_vel
                [self.world_size * 2] * 3 +  # missile_pos
                [100] * 3 +  # missile_vel
                [1] +  # time_remaining
                [1],  # fuel_remaining
                dtype=np.float32
            )
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
            
        else:
            # Full 6DOF observation space (29D)
            # Interceptor: pos(3) + vel(3) + quat(4) + omega(3) = 13D
            # Missile: pos(3) + vel(3) + quat(4) + omega(3) = 13D  
            # Environment: time(1) + fuel(1) + wind(3) = 3D (but we'll use 5D for consistency)
            # Total: 13 + 13 + 5 = 31D (adjusted for practical use)
            
            obs_dim = 31
            low = np.full(obs_dim, -np.inf, dtype=np.float32)
            high = np.full(obs_dim, np.inf, dtype=np.float32)
            
            # Position bounds
            for i in [0, 1, 2, 13, 14, 15]:  # Interceptor and missile positions
                low[i] = 0
                high[i] = self.world_size * 2
            
            # Velocity bounds
            for i in [3, 4, 5, 16, 17, 18]:  # Interceptor and missile velocities
                low[i] = -200
                high[i] = 200
            
            # Quaternion bounds (normalized)
            for i in [6, 7, 8, 9, 19, 20, 21, 22]:  # Quaternions
                low[i] = -1
                high[i] = 1
            
            # Angular velocity bounds
            for i in [10, 11, 12, 23, 24, 25]:  # Angular velocities
                low[i] = -50
                high[i] = 50
            
            # Environment state bounds
            low[26] = 0  # time_remaining
            high[26] = 1
            low[27] = 0  # fuel_remaining
            high[27] = 1
            low[28:31] = -50  # wind_velocity
            high[28:31] = 50
            
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.simulation_time = 0.0
        self.fuel_remaining = self.max_fuel
        
        # Reset episode statistics
        self.episode_stats = {
            'max_altitude_reached': 0.0,
            'min_distance_achieved': float('inf'),
            'fuel_efficiency': 0.0,
            'intercept_method': None
        }
        
        # Setup target position
        self.target_pos = np.array([
            self.world_size + self.np_random.uniform(-20, 20),
            self.world_size + self.np_random.uniform(-20, 20),
            0.0
        ], dtype=np.float32)
        
        # Initialize based on difficulty mode
        if self.legacy_3dof_mode or self.difficulty_mode == DifficultyMode.EASY_3DOF:
            self._reset_3dof_mode()
        else:
            self._reset_6dof_mode()
        
        # Initialize wind
        if self.enable_wind:
            self.wind_velocity = AtmosphericModel.get_wind_velocity(
                self.target_pos, self.simulation_time
            ) * self.wind_strength
        else:
            self.wind_velocity = np.zeros(3)
        
        return self._get_observation(), self._get_info()
    
    def _reset_3dof_mode(self):
        """Reset in 3DOF compatibility mode"""
        # Setup interceptor (similar to original 3DOF)
        interceptor_offset = self.np_random.uniform(10, 30)
        interceptor_angle = self.np_random.uniform(0, 2 * np.pi)
        self.interceptor_pos_3d = self.target_pos + np.array([
            interceptor_offset * np.cos(interceptor_angle),
            interceptor_offset * np.sin(interceptor_angle),
            0.0
        ], dtype=np.float32)
        self.interceptor_vel_3d = np.array([0.0, 0.0, 8.0], dtype=np.float32)
        
        # Setup missile
        missile_distance = self.np_random.uniform(200, 350)
        missile_angle = self.np_random.uniform(0, 2 * np.pi)
        missile_x = self.target_pos[0] + missile_distance * np.cos(missile_angle)
        missile_y = self.target_pos[1] + missile_distance * np.sin(missile_angle)
        missile_x = np.clip(missile_x, 30, self.world_size * 2 - 30)
        missile_y = np.clip(missile_y, 30, self.world_size * 2 - 30)
        missile_height = self.np_random.uniform(100, 400)
        
        self.missile_pos_3d = np.array([missile_x, missile_y, missile_height], dtype=np.float32)
        direction = (self.target_pos - self.missile_pos_3d) / np.linalg.norm(self.target_pos - self.missile_pos_3d)
        self.missile_vel_3d = direction * 25.0  # missile speed
        
        # Initialize evasion state
        self.missile_evasion_timer = 0
        self.current_evasion_direction = np.zeros(3, dtype=np.float32)
        self.evasion_duration = 0
        self.base_direction = direction.copy()
    
    def _reset_6dof_mode(self):
        """Reset in full 6DOF mode"""
        # Initialize interceptor 6DOF
        interceptor_offset = self.np_random.uniform(20, 50)
        interceptor_angle = self.np_random.uniform(0, 2 * np.pi)
        interceptor_pos = self.target_pos + np.array([
            interceptor_offset * np.cos(interceptor_angle),
            interceptor_offset * np.sin(interceptor_angle),
            5.0  # Start slightly above ground
        ])
        
        interceptor_vel = np.array([0.0, 0.0, 15.0])  # Launch velocity
        interceptor_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        interceptor_omega = np.zeros(3)
        
        self.interceptor_6dof = RigidBody6DOF(
            VehicleType.INTERCEPTOR,
            interceptor_pos, interceptor_vel, interceptor_quat, interceptor_omega
        )
        
        # Initialize missile 6DOF  
        missile_distance = self.np_random.uniform(300, 500)
        missile_angle = self.np_random.uniform(0, 2 * np.pi)
        missile_height = self.np_random.uniform(200, 800)
        
        missile_pos = self.target_pos + np.array([
            missile_distance * np.cos(missile_angle),
            missile_distance * np.sin(missile_angle),
            missile_height
        ])
        
        # Missile initial orientation and velocity toward target
        target_direction = (self.target_pos - missile_pos)
        target_direction = target_direction / np.linalg.norm(target_direction)
        missile_vel = target_direction * 35.0  # Higher speed for 6DOF
        
        # Random missile orientation (quaternion)
        missile_euler = self.np_random.uniform(-0.2, 0.2, 3)  # Small random orientation
        missile_quat = self._euler_to_quaternion(missile_euler)
        missile_omega = self.np_random.uniform(-0.5, 0.5, 3)  # Small initial spin
        
        self.missile_6dof = RigidBody6DOF(
            VehicleType.MISSILE,
            missile_pos, missile_vel, missile_quat, missile_omega
        )
        
        # Copy positions for compatibility
        self.interceptor_pos_3d = interceptor_pos.copy()
        self.interceptor_vel_3d = interceptor_vel.copy()
        self.missile_pos_3d = missile_pos.copy()
        self.missile_vel_3d = missile_vel.copy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        action = np.clip(action, -1.0, 1.0)
        
        # Extract explosion command
        explosion_command = action[-1] > 0.5  # Last element is always explosion
        
        # Process action based on mode
        if self.legacy_3dof_mode or self.difficulty_mode == DifficultyMode.EASY_3DOF:
            reward, terminated, truncated = self._step_3dof_mode(action, explosion_command)
        else:
            reward, terminated, truncated = self._step_6dof_mode(action, explosion_command)
        
        # Update simulation time and step count
        self.simulation_time += self.dt
        self.step_count += 1
        
        # Update episode statistics
        if not self.legacy_3dof_mode and self.interceptor_6dof is not None:
            current_distance = distance_6dof(self.interceptor_6dof.position, self.missile_6dof.position)
            self.episode_stats['max_altitude_reached'] = max(
                self.episode_stats['max_altitude_reached'], 
                self.interceptor_6dof.position[2]
            )
            self.episode_stats['min_distance_achieved'] = min(
                self.episode_stats['min_distance_achieved'], 
                current_distance
            )
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _step_3dof_mode(self, action: np.ndarray, explosion_command: bool) -> Tuple[float, bool, bool]:
        """Execute step in 3DOF compatibility mode"""
        thrust_action = action[:3]
        
        # Fuel system
        if self.enable_fuel_system:
            thrust_magnitude = np.linalg.norm(thrust_action)
            fuel_consumed = thrust_magnitude * self.fuel_burn_rate * self.dt
            self.fuel_remaining = max(0.0, self.fuel_remaining - fuel_consumed)
            
            if self.fuel_remaining <= 0:
                thrust_action = np.zeros(3)
        
        # Simple 3DOF physics (from original environment)
        max_accel = 25.0
        drag_coeff = 0.03
        max_velocity = 50.0
        
        # Apply thrust and drag
        thrust = thrust_action * max_accel
        drag = -drag_coeff * self.interceptor_vel_3d
        self.interceptor_vel_3d += (thrust + drag) * self.dt
        
        # Velocity limiting
        vel_mag = np.linalg.norm(self.interceptor_vel_3d)
        if vel_mag > max_velocity:
            self.interceptor_vel_3d = self.interceptor_vel_3d / vel_mag * max_velocity
        
        # Update position
        self.interceptor_pos_3d += self.interceptor_vel_3d * self.dt
        
        # Simple missile evasion (from original)
        self._update_missile_3dof()
        
        # Check termination conditions
        intercept_dist = distance_3d(self.interceptor_pos_3d, self.missile_pos_3d)
        miss_dist = distance_3d(self.missile_pos_3d, self.target_pos)
        
        exploded = explosion_command and intercept_dist < self.explosion_radius
        intercepted = exploded or intercept_dist < self.intercept_threshold
        missile_hit = miss_dist < self.miss_threshold
        
        # Boundary checks
        out_of_bounds = (
            self.interceptor_pos_3d[2] > self.world_size * 2 or
            self.interceptor_pos_3d[0] < 0 or self.interceptor_pos_3d[0] > self.world_size * 2 or
            self.interceptor_pos_3d[1] < 0 or self.interceptor_pos_3d[1] > self.world_size * 2
        )
        
        terminated = intercepted or missile_hit or out_of_bounds
        truncated = self.step_count >= self.max_steps and not terminated
        
        # Reward calculation (simplified from original)
        if intercepted:
            reward = 15.0 + (self.fuel_remaining / self.max_fuel) * 5.0
            self.episode_stats['intercept_method'] = 'explosion' if exploded else 'proximity'
        elif missile_hit:
            reward = -8.0
        elif out_of_bounds:
            reward = -2.0
        else:
            # Distance-based reward
            reward = max(0, 1.0 - intercept_dist / 200.0) - 0.02
        
        return reward, terminated, truncated
    
    def _step_6dof_mode(self, action: np.ndarray, explosion_command: bool) -> Tuple[float, bool, bool]:
        """Execute step in full 6DOF mode"""
        # Parse control inputs based on action mode
        if self.action_mode == ActionMode.ACCELERATION_6DOF:
            # [fx, fy, fz, tx, ty, tz, explode]
            thrust_force = action[:3] * 2000.0  # Scale to Newtons
            control_torque = action[3:6] * 500.0  # Scale to N⋅m
        elif self.action_mode == ActionMode.THRUST_ATTITUDE:
            # [thrust_magnitude, pitch_cmd, yaw_cmd, roll_cmd, explode]
            thrust_mag = (action[0] + 1.0) * 1000.0  # 0-2000N
            pitch_cmd = action[1] * 0.5  # ±0.5 rad
            yaw_cmd = action[2] * 0.5
            roll_cmd = action[3] * 0.5
            
            # Convert attitude commands to thrust vector and torques
            thrust_force = np.array([thrust_mag, 0, 0])  # Thrust along body x-axis
            control_torque = np.array([roll_cmd * 100, pitch_cmd * 100, yaw_cmd * 100])
        else:
            # Default to force/torque mode
            thrust_force = action[:3] * 1000.0
            control_torque = action[3:6] * 200.0
        
        # Fuel consumption
        if self.enable_fuel_system:
            fuel_consumed = np.linalg.norm(thrust_force) / 1000.0 * self.fuel_burn_rate * self.dt
            self.fuel_remaining = max(0.0, self.fuel_remaining - fuel_consumed)
            
            if self.fuel_remaining <= 0:
                thrust_force = np.zeros(3)
        
        # Set control inputs
        self.interceptor_6dof.set_control_inputs(thrust_force, control_torque)
        
        # Update 6DOF physics
        self.interceptor_6dof.step(self.dt, self.simulation_time, self.wind_velocity)
        
        # Update missile behavior
        self._update_missile_6dof()
        
        # Update 3DOF compatibility positions
        self.interceptor_pos_3d = self.interceptor_6dof.position.copy()
        self.interceptor_vel_3d = self.interceptor_6dof.velocity.copy()
        self.missile_pos_3d = self.missile_6dof.position.copy()
        self.missile_vel_3d = self.missile_6dof.velocity.copy()
        
        # Check termination conditions
        intercept_dist = distance_6dof(self.interceptor_6dof.position, self.missile_6dof.position)
        miss_dist = distance_6dof(self.missile_6dof.position, self.target_pos)
        
        exploded = explosion_command and intercept_dist < self.explosion_radius
        intercepted = exploded or intercept_dist < self.intercept_threshold
        missile_hit = miss_dist < self.miss_threshold
        
        # Advanced termination conditions for 6DOF
        out_of_bounds = (
            self.interceptor_6dof.position[2] > self.world_size * 3 or
            self.interceptor_6dof.position[0] < -self.world_size or 
            self.interceptor_6dof.position[0] > self.world_size * 3 or
            self.interceptor_6dof.position[1] < -self.world_size or 
            self.interceptor_6dof.position[1] > self.world_size * 3
        )
        
        # Structural limits
        g_force = np.linalg.norm(thrust_force) / (self.interceptor_6dof.aero_props.mass * 9.81)
        structural_failure = g_force > 20.0  # 20G limit
        
        terminated = intercepted or missile_hit or out_of_bounds or structural_failure
        truncated = self.step_count >= self.max_steps and not terminated
        
        # Advanced reward calculation for 6DOF
        reward = self._calculate_6dof_reward(
            intercept_dist, miss_dist, intercepted, exploded, 
            missile_hit, out_of_bounds, structural_failure
        )
        
        return reward, terminated, truncated
    
    def _update_missile_3dof(self):
        """Update missile behavior in 3DOF mode (simplified evasion)"""
        # Simple evasive behavior (from original environment)
        self.missile_evasion_timer += 1
        
        if self.missile_evasion_timer % 15 == 0:  # Evasion frequency
            self.evasion_duration = self.np_random.integers(5, 20)
            evasion_direction = self.np_random.normal(0, 1, 3)
            evasion_direction[2] *= 0.3
            
            if np.linalg.norm(evasion_direction) > 0:
                self.current_evasion_direction = evasion_direction / np.linalg.norm(evasion_direction)
        
        # Update base direction toward target
        self.base_direction = (self.target_pos - self.missile_pos_3d) / np.linalg.norm(self.target_pos - self.missile_pos_3d)
        
        # Apply evasive maneuver
        if self.evasion_duration > 0:
            self.evasion_duration -= 1
            evasion_strength = 0.3 * (self.evasion_duration / 20.0)
            final_direction = 0.7 * self.base_direction + evasion_strength * self.current_evasion_direction
            final_direction = final_direction / np.linalg.norm(final_direction)
        else:
            noise = self.np_random.normal(0, 0.05, 3)
            final_direction = self.base_direction + noise
            final_direction = final_direction / np.linalg.norm(final_direction)
        
        # Update missile velocity and position
        missile_speed = 25.0
        self.missile_vel_3d = final_direction * missile_speed
        self.missile_pos_3d += self.missile_vel_3d * self.dt
    
    def _update_missile_6dof(self):
        """Update missile behavior in 6DOF mode (advanced evasion)"""
        # Threat assessment
        interceptor_pos = self.interceptor_6dof.position
        missile_pos = self.missile_6dof.position
        
        threat_distance = distance_6dof(interceptor_pos, missile_pos)
        intercept_geometry = intercept_geometry_6dof(
            interceptor_pos, self.interceptor_6dof.velocity,
            missile_pos, self.missile_6dof.velocity
        )
        
        # Adaptive evasion based on threat level
        threat_level = max(0, 1.0 - threat_distance / 200.0)  # High threat when close
        closing_threat = max(0, intercept_geometry['closing_velocity'] / 100.0)  # High threat when closing fast
        
        # Base guidance toward target
        target_direction = self.target_pos - missile_pos
        target_distance = np.linalg.norm(target_direction)
        if target_distance > 1e-6:
            target_direction = target_direction / target_distance
        
        # Evasive maneuver generation
        evasion_force = np.zeros(3)
        evasion_torque = np.zeros(3)
        
        if threat_level > 0.3 or closing_threat > 0.5:
            # Aggressive evasion
            evasion_magnitude = 800.0 * (threat_level + closing_threat)
            
            # Generate random but coordinated evasion
            if hasattr(self, 'evasion_timer'):
                self.evasion_timer += 1
            else:
                self.evasion_timer = 0
            
            if self.evasion_timer % 20 == 0:  # Change evasion pattern every second
                self.evasion_pattern = self.np_random.choice(['spiral', 'weave', 'jink', 'barrel_roll'])
            
            # Execute evasion pattern
            t = self.evasion_timer * self.dt
            if hasattr(self, 'evasion_pattern'):
                if self.evasion_pattern == 'spiral':
                    evasion_force = evasion_magnitude * np.array([
                        0, np.sin(t * 8), np.cos(t * 8)
                    ])
                    evasion_torque = np.array([200 * np.sin(t * 4), 0, 0])
                    
                elif self.evasion_pattern == 'weave':
                    evasion_force = evasion_magnitude * np.array([
                        0, np.sin(t * 6), 0.3 * np.sin(t * 12)
                    ])
                    
                elif self.evasion_pattern == 'jink':
                    if self.evasion_timer % 10 < 5:
                        evasion_force = evasion_magnitude * np.array([0, 1, 0.5])
                    else:
                        evasion_force = evasion_magnitude * np.array([0, -1, -0.5])
                        
                elif self.evasion_pattern == 'barrel_roll':
                    evasion_torque = np.array([300 * np.sin(t * 6), 100 * np.cos(t * 3), 0])
        
        # Combine guidance and evasion
        guidance_force = 1200.0 * target_direction  # Strong guidance toward target
        total_force = guidance_force + evasion_force
        total_torque = evasion_torque
        
        # Set missile control inputs
        self.missile_6dof.set_control_inputs(total_force, total_torque)
        
        # Update missile physics
        self.missile_6dof.step(self.dt, self.simulation_time, self.wind_velocity)
    
    def _calculate_6dof_reward(self, intercept_dist: float, miss_dist: float,
                              intercepted: bool, exploded: bool, missile_hit: bool,
                              out_of_bounds: bool, structural_failure: bool) -> float:
        """Calculate reward for 6DOF mode with advanced considerations"""
        
        if intercepted:
            # Success rewards
            base_reward = 20.0
            
            # Efficiency bonuses
            fuel_bonus = (self.fuel_remaining / self.max_fuel) * 8.0
            time_bonus = (self.max_steps - self.step_count) / self.max_steps * 5.0
            method_bonus = 3.0 if exploded else 0.0  # Prefer explosion intercepts
            
            # Intercept quality bonus
            if intercept_dist < 10.0:
                precision_bonus = 5.0
            elif intercept_dist < 20.0:
                precision_bonus = 2.0
            else:
                precision_bonus = 0.0
            
            total_reward = base_reward + fuel_bonus + time_bonus + method_bonus + precision_bonus
            return total_reward
            
        elif missile_hit:
            return -15.0
            
        elif structural_failure:
            return -8.0
            
        elif out_of_bounds:
            return -5.0
            
        else:
            # Progressive reward during engagement
            reward = 0.0
            
            # Distance-based reward (closer is better)
            max_dist = 600.0
            distance_reward = max(0, 2.0 * (1.0 - intercept_dist / max_dist))
            
            # Intercept geometry reward
            if hasattr(self, 'interceptor_6dof') and hasattr(self, 'missile_6dof'):
                geometry = intercept_geometry_6dof(
                    self.interceptor_6dof.position, self.interceptor_6dof.velocity,
                    self.missile_6dof.position, self.missile_6dof.velocity
                )
                
                # Reward good intercept geometry
                if geometry['closing_velocity'] > 0:
                    closing_reward = min(1.0, geometry['closing_velocity'] / 50.0)
                else:
                    closing_reward = -0.5  # Penalty for diverging
                
                # Aspect angle reward (head-on is preferred)
                aspect_reward = 0.5 * (1.0 - geometry['aspect_angle'] / np.pi)
                
                geometry_reward = closing_reward + aspect_reward
            else:
                geometry_reward = 0.0
            
            # Fuel efficiency penalty
            fuel_penalty = -0.1 * (1.0 - self.fuel_remaining / self.max_fuel)
            
            # Time pressure
            time_penalty = -0.03
            
            reward = distance_reward + geometry_reward + fuel_penalty + time_penalty
            return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation based on difficulty mode"""
        
        if self.legacy_3dof_mode or self.difficulty_mode == DifficultyMode.EASY_3DOF:
            # 3DOF observation (14D)
            time_remaining = (self.max_steps - self.step_count) / self.max_steps
            fuel_remaining = self.fuel_remaining / self.max_fuel
            
            return np.concatenate([
                self.interceptor_pos_3d,
                self.interceptor_vel_3d,
                self.missile_pos_3d,
                self.missile_vel_3d,
                [time_remaining],
                [fuel_remaining]
            ]).astype(np.float32)
        
        else:
            # Full 6DOF observation (31D)
            time_remaining = (self.max_steps - self.step_count) / self.max_steps
            fuel_remaining = self.fuel_remaining / self.max_fuel
            
            # Get 6DOF state vectors
            interceptor_state = self.interceptor_6dof.get_state_vector()  # 13D
            missile_state = self.missile_6dof.get_state_vector()  # 13D
            
            # Environment state (5D)
            env_state = np.array([
                time_remaining,
                fuel_remaining,
                self.wind_velocity[0],
                self.wind_velocity[1], 
                self.wind_velocity[2]
            ])
            
            return np.concatenate([
                interceptor_state,  # 13D
                missile_state,      # 13D
                env_state          # 5D
            ]).astype(np.float32)  # Total: 31D
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state"""
        info = {
            "step_count": self.step_count,
            "simulation_time": self.simulation_time,
            "fuel_remaining": self.fuel_remaining,
            "difficulty_mode": self.difficulty_mode.value,
            "action_mode": self.action_mode.value,
        }
        
        if not self.legacy_3dof_mode and self.interceptor_6dof is not None:
            info.update({
                "interceptor_6dof_state": self.interceptor_6dof.get_state_vector(),
                "missile_6dof_state": self.missile_6dof.get_state_vector(),
                "intercept_distance": distance_6dof(self.interceptor_6dof.position, self.missile_6dof.position),
                "intercept_geometry": intercept_geometry_6dof(
                    self.interceptor_6dof.position, self.interceptor_6dof.velocity,
                    self.missile_6dof.position, self.missile_6dof.velocity
                ),
                "interceptor_aerodynamics": self.interceptor_6dof.get_aerodynamic_info(),
                "missile_aerodynamics": self.missile_6dof.get_aerodynamic_info(),
                "wind_velocity": self.wind_velocity,
                "episode_stats": self.episode_stats.copy()
            })
        else:
            info.update({
                "interceptor_pos": self.interceptor_pos_3d.copy(),
                "missile_pos": self.missile_pos_3d.copy(),
                "intercept_distance": distance_3d(self.interceptor_pos_3d, self.missile_pos_3d),
            })
        
        return info
    
    def _euler_to_quaternion(self, euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles to quaternion"""
        roll, pitch, yaw = euler
        
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    def render(self, intercepted: bool = False):
        """Render the environment"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = Viewer3D(self.world_size)
            
            # Use 3D positions for rendering (compatible with both modes)
            self.viewer.render(
                self.interceptor_pos_3d, 
                self.missile_pos_3d, 
                self.target_pos, 
                intercepted
            )
    
    def close(self):
        """Close the environment"""
        if self.viewer:
            self.viewer.close()


# Register the environment
gym.envs.registration.register(
    id='Aegis6DIntercept-v0',
    entry_point='aegis_intercept.envs.aegis_6dof_env:Aegis6DInterceptEnv',
)