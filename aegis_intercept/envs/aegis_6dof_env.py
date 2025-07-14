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
import math

from ..utils.physics6dof import (
    RigidBody6DOF, VehicleType, AtmosphericModel, QuaternionUtils,
    distance_6dof, intercept_geometry_6dof, PhysicsConstants
)
from ..utils.physics3d import distance as distance_3d  # For 3DOF compatibility
from ..rendering.viewer3d import Viewer3D
from ..rendering.radar_viewer import RadarPPI, RadarViewer3D
from ..sensors import (
    GroundRadar, MissileRadar, SensorFusionSystem, FusionConfig,
    SensorEnvironment, WeatherConditions, TrackState
)


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
        
        # Realistic sensor parameters
        enable_realistic_sensors: bool = True,
        ground_radar_position: Optional[np.ndarray] = None,
        sensor_update_rate: float = 0.1,  # seconds
        weather_conditions: Union[WeatherConditions, str] = WeatherConditions.CLEAR,
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
        
        # Realistic sensor system
        self.enable_realistic_sensors = enable_realistic_sensors
        self.sensor_update_rate = sensor_update_rate
        self.last_sensor_update = 0.0
        
        # Handle weather conditions parameter
        if isinstance(weather_conditions, str):
            try:
                self.weather_conditions = WeatherConditions(weather_conditions.upper())
            except ValueError:
                self.weather_conditions = WeatherConditions.CLEAR
        else:
            self.weather_conditions = weather_conditions
        
        # Initialize sensor systems
        if self.enable_realistic_sensors:
            self._initialize_sensor_systems(ground_radar_position)
        else:
            # Initialize placeholders for compatibility
            self.ground_radar = None
            self.sensor_fusion = None
            self.sensor_environment = None
            self.missile_radar = None
        
        # Track estimation state (replaces perfect information)
        self.estimated_missile_state: Optional[TrackState] = None
        self.sensor_observation_history = []
        self.tracking_confidence = 0.0
        
        # Physics constants
        self.physics_constants = PhysicsConstants()
        
        # Initialize 6DOF physics if not in legacy mode
        if not self.legacy_3dof_mode:
            self.interceptor_6dof = None
            self.missile_6dof = None
        else:
            # Ensure 6DOF objects are None in 3DOF mode
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
        
        # RCS (Radar Cross Section) models for realistic detection
        self.interceptor_rcs = 0.1  # m² (small missile)
        self.missile_rcs = 2.0      # m² (larger target missile)
        
        # Rendering
        self.viewer = None
        self.radar_viewer = None
        self.radar_3d_viewer = None
        self.popup_message = None
        self.popup_timer = 0
        self.last_render_time = 0.0
        
        # Performance tracking
        self.episode_stats = {}
        
        # Visualization speed control
        self.viz_speed_multiplier = 10.0  # Default speed multiplier
        
    def _initialize_sensor_systems(self, ground_radar_position: Optional[np.ndarray]):
        """Initialize realistic sensor systems"""
        # Ground radar position (default to origin if not specified)
        if ground_radar_position is None:
            ground_radar_position = np.array([self.world_size, self.world_size, 50.0])
            
        # Initialize ground-based tracking radar
        self.ground_radar = GroundRadar(ground_radar_position, "Defense Radar")
        
        # Initialize sensor fusion system
        fusion_config = FusionConfig(
            process_noise_std=3.0,  # Higher noise for missile tracking
            position_init_std=200.0,  # Large initial uncertainty
            velocity_init_std=100.0,
            confirmation_threshold=2,  # Quick confirmation for fast targets
            deletion_threshold=3,      # Quick deletion for lost tracks
            association_gate=5.0       # Larger gate for maneuvering targets
        )
        self.sensor_fusion = SensorFusionSystem(fusion_config)
        
        # Initialize sensor environment
        self.sensor_environment = SensorEnvironment(self.weather_conditions)
        
        # Missile radar (initialized when missile is created)
        self.missile_radar: Optional[MissileRadar] = None
        
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
        
        # Observation space setup - REALISTIC SENSOR-BASED
        if self.enable_realistic_sensors:
            # Sensor-based observation space
            if self.legacy_3dof_mode or self.difficulty_mode == DifficultyMode.EASY_3DOF:
                # 3DOF sensor observation (22D)
                # Own state: pos(3) + vel(3) + fuel(1) + time(1) = 8D
                # Target estimate: pos(3) + vel(3) + confidence(1) + track_age(1) = 8D  
                # Sensor data: last_detection_time(1) + snr(1) + range_rate(1) + measurement_count(1) = 4D
                # Radar status: ground_radar_active(1) + false_alarm_rate(1) = 2D
                obs_dim = 22
            else:
                # 6DOF sensor observation (44D)
                # Own state: pos(3) + vel(3) + quat(4) + omega(3) + fuel(1) + time(1) = 15D
                # Target estimate: pos(3) + vel(3) + accel(3) + confidence(1) + track_age(1) + quality(1) = 12D
                # Sensor measurements: range(1) + azimuth(1) + elevation(1) + doppler(1) + snr(1) = 5D
                # Sensor uncertainties: pos_std(3) + vel_std(3) = 6D  
                # Tracking info: consecutive_misses(1) + measurement_count(1) + last_update_dt(1) = 3D
                # Environment: weather(1) + wind_speed(1) + jamming_level(1) = 3D
                obs_dim = 44
                
            low = np.full(obs_dim, -np.inf, dtype=np.float32)
            high = np.full(obs_dim, np.inf, dtype=np.float32)
            
            # Set reasonable bounds for key variables
            # Position bounds (first 3 elements for own position)
            low[0:3] = 0
            high[0:3] = self.world_size * 2
            
            # Normalized values bounds
            if self.legacy_3dof_mode or self.difficulty_mode == DifficultyMode.EASY_3DOF:
                low[7] = 0; high[7] = 1  # fuel
                low[8] = 0; high[8] = 1  # time
                low[16] = 0; high[16] = 1  # confidence
                low[20] = 0; high[20] = 1  # ground_radar_active
            else:
                low[14] = 0; high[14] = 1  # fuel
                low[15] = 0; high[15] = 1  # time
                low[27] = 0; high[27] = 1  # confidence
                low[43] = 0; high[43] = 1  # jamming_level
                
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
            
        else:
            # Legacy perfect information mode (for debugging/comparison)
            if self.legacy_3dof_mode or self.difficulty_mode == DifficultyMode.EASY_3DOF:
                obs_dim = 14
            else:
                obs_dim = 31
                
            low = np.full(obs_dim, -np.inf, dtype=np.float32)
            high = np.full(obs_dim, np.inf, dtype=np.float32)
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
            
        # Ensure sensor systems are initialized if needed
        if self.enable_realistic_sensors and not hasattr(self, 'ground_radar'):
            self._initialize_sensor_systems(None)
        
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
        
        # Initialize missile radar if using realistic sensors
        if self.enable_realistic_sensors:
            self.missile_radar = MissileRadar(missile_pos, "Missile Seeker")
            
        # Copy positions for compatibility
        self.interceptor_pos_3d = interceptor_pos.copy()
        self.interceptor_vel_3d = interceptor_vel.copy()
        self.missile_pos_3d = missile_pos.copy()
        self.missile_vel_3d = missile_vel.copy()
        
        # Reset sensor tracking
        if self.enable_realistic_sensors:
            self.estimated_missile_state = None
            self.sensor_observation_history.clear()
            self.tracking_confidence = 0.0
            self.last_sensor_update = 0.0
    
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
        """Execute step in 3DOF compatibility mode with sensor-based rewards"""
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
        
        # REALISTIC TERMINATION CONDITIONS - SENSOR BASED ONLY
        
        # Check own platform limits (always known)
        out_of_bounds = (
            self.interceptor_pos_3d[2] > self.world_size * 2 or
            self.interceptor_pos_3d[0] < 0 or self.interceptor_pos_3d[0] > self.world_size * 2 or
            self.interceptor_pos_3d[1] < 0 or self.interceptor_pos_3d[1] > self.world_size * 2
        )
        
        # Explosion/intercept detection - only if we have a valid track
        exploded = False
        intercepted = False
        
        if explosion_command and self.estimated_missile_state is not None:
            # Can only detect explosion if we have sensor contact
            estimated_range = np.linalg.norm(self.estimated_missile_state.position - self.interceptor_pos_3d)
            if estimated_range < self.explosion_radius and self.tracking_confidence > 0.3:
                exploded = True
                intercepted = True
                
        # Check if target hit protected asset (detected by ground sensors)
        missile_hit = False
        if self.enable_realistic_sensors:
            # Only know about missile impact if sensors detect it near target
            true_miss_dist = distance_3d(self.missile_pos_3d, self.target_pos)
            if true_miss_dist < self.miss_threshold:
                # Simulate ground sensors detecting impact
                missile_hit = True
        
        terminated = intercepted or missile_hit or out_of_bounds
        truncated = self.step_count >= self.max_steps and not terminated
        
        # SENSOR-BASED REWARD CALCULATION
        reward = self._calculate_sensor_based_reward_3dof(
            exploded, intercepted, missile_hit, out_of_bounds
        )
        
        return reward, terminated, truncated
    
    def _step_6dof_mode(self, action: np.ndarray, explosion_command: bool) -> Tuple[float, bool, bool]:
        """Execute step in full 6DOF mode with sensor-based rewards"""
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
        
        # REALISTIC TERMINATION CONDITIONS - SENSOR BASED ONLY
        
        # Own platform limits (always known)
        out_of_bounds = (
            self.interceptor_6dof.position[2] > self.world_size * 3 or
            self.interceptor_6dof.position[0] < -self.world_size or 
            self.interceptor_6dof.position[0] > self.world_size * 3 or
            self.interceptor_6dof.position[1] < -self.world_size or 
            self.interceptor_6dof.position[1] > self.world_size * 3
        )
        
        # Structural limits (own platform - always known)
        g_force = np.linalg.norm(thrust_force) / (self.interceptor_6dof.aero_props.mass * 9.81)
        structural_failure = g_force > 20.0  # 20G limit
        
        # Explosion/intercept detection - only if we have sensor track
        exploded = False
        intercepted = False
        
        if explosion_command and self.estimated_missile_state is not None:
            # Can only detect explosion if we have valid track with sufficient confidence
            estimated_range = np.linalg.norm(
                self.estimated_missile_state.position - self.interceptor_6dof.position
            )
            
            # Require high confidence and close range for explosion detection
            if (estimated_range < self.explosion_radius and 
                self.tracking_confidence > 0.5):
                exploded = True
                intercepted = True
                
        # Missile hit detection - only through ground sensors
        missile_hit = False
        if self.enable_realistic_sensors:
            # Simulate ground-based impact detection system
            true_miss_dist = distance_6dof(self.missile_6dof.position, self.target_pos)
            if true_miss_dist < self.miss_threshold:
                # Ground sensors detect impact
                missile_hit = True
        
        terminated = intercepted or missile_hit or out_of_bounds or structural_failure
        truncated = self.step_count >= self.max_steps and not terminated
        
        # SENSOR-BASED REWARD CALCULATION
        reward = self._calculate_sensor_based_reward_6dof(
            exploded, intercepted, missile_hit, out_of_bounds, structural_failure
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
        
        # Update missile radar position if it exists
        if self.missile_radar is not None:
            self.missile_radar.update_position(self.missile_6dof.position)
    
    def _calculate_sensor_based_reward_3dof(self, exploded: bool, intercepted: bool,
                                           missile_hit: bool, out_of_bounds: bool) -> float:
        """Calculate reward based only on sensor-available information (3DOF mode)"""
        
        if intercepted:
            # Success rewards - only based on known information
            base_reward = 15.0
            
            # Fuel efficiency bonus (own platform - always known)
            fuel_bonus = (self.fuel_remaining / self.max_fuel) * 5.0
            
            # Time efficiency bonus (own platform - always known)
            time_bonus = (self.max_steps - self.step_count) / self.max_steps * 3.0
            
            # Method bonus for explosion (sensor detected)
            method_bonus = 3.0 if exploded else 1.0
            
            # Track quality bonus (sensor-based)
            track_quality_bonus = 0.0
            if self.estimated_missile_state is not None:
                track_quality_bonus = self.tracking_confidence * 2.0
            
            total_reward = base_reward + fuel_bonus + time_bonus + method_bonus + track_quality_bonus
            self.episode_stats['intercept_method'] = 'explosion' if exploded else 'proximity'
            return total_reward
            
        elif missile_hit:
            # Mission failure - detected by ground sensors
            return -10.0
            
        elif out_of_bounds:
            # Own platform violation - always known
            return -3.0
            
        else:
            # Progressive reward based on sensor information only
            reward = 0.0
            
            # Track maintenance reward
            if self.estimated_missile_state is not None and self.tracking_confidence > 0.3:
                # Reward for maintaining track
                track_reward = 0.5 * self.tracking_confidence
                
                # Sensor update frequency reward
                if self.sensor_observation_history:
                    recent_updates = sum(1 for obs in self.sensor_observation_history[-5:] 
                                       if obs['detections'] > 0)
                    update_reward = 0.1 * recent_updates
                else:
                    update_reward = 0.0
                    
                reward += track_reward + update_reward
            else:
                # Penalty for losing track
                reward -= 0.2
                
            # Fuel efficiency penalty (always known)
            fuel_penalty = -0.05 * (1.0 - self.fuel_remaining / self.max_fuel)
            
            # Time pressure
            time_penalty = -0.02
            
            reward += fuel_penalty + time_penalty
            return reward
    
    def _calculate_sensor_based_reward_6dof(self, exploded: bool, intercepted: bool,
                                           missile_hit: bool, out_of_bounds: bool,
                                           structural_failure: bool) -> float:
        """Calculate reward based only on sensor-available information (6DOF mode)"""
        
        if intercepted:
            # Success rewards - based on sensor-confirmed intercept
            base_reward = 25.0
            
            # Fuel efficiency bonus (own platform - always known)
            fuel_bonus = (self.fuel_remaining / self.max_fuel) * 8.0
            
            # Time efficiency bonus (own platform - always known)
            time_bonus = (self.max_steps - self.step_count) / self.max_steps * 5.0
            
            # Method bonus for confirmed explosion
            method_bonus = 5.0 if exploded else 2.0
            
            # Tracking quality bonus (sensor performance)
            track_quality_bonus = 0.0
            if self.estimated_missile_state is not None:
                # Reward high-confidence tracks
                track_quality_bonus = self.tracking_confidence * 4.0
                
                # Reward track stability (low uncertainty)
                if hasattr(self.estimated_missile_state, 'covariance'):
                    pos_uncertainty = np.trace(self.estimated_missile_state.covariance[0:3, 0:3])
                    stability_bonus = max(0, 2.0 - pos_uncertainty / 1000.0)
                    track_quality_bonus += stability_bonus
                    
            total_reward = base_reward + fuel_bonus + time_bonus + method_bonus + track_quality_bonus
            return total_reward
            
        elif missile_hit:
            # Mission failure - detected by ground impact sensors
            return -20.0
            
        elif structural_failure:
            # Own platform failure - always known
            return -10.0
            
        elif out_of_bounds:
            # Own platform violation - always known
            return -5.0
            
        else:
            # Progressive reward based on sensor performance and tracking
            reward = 0.0
            
            # Track quality rewards
            if self.estimated_missile_state is not None:
                confidence_reward = 0.8 * self.tracking_confidence
                
                # Reward for recent measurements
                measurement_age = (self.simulation_time - 
                                 self.estimated_missile_state.last_update_time)
                freshness_reward = max(0, 0.3 * (1.0 - measurement_age / 2.0))
                
                # Reward track consistency (low consecutive misses)
                miss_penalty = -0.1 * self.estimated_missile_state.consecutive_misses
                
                # Sensor geometry reward (if we can estimate it)
                geometry_reward = 0.0
                if (hasattr(self, 'sensor_observation_history') and 
                    self.sensor_observation_history):
                    # Reward active sensor engagement
                    recent_detections = sum(obs['detections'] for obs in 
                                          self.sensor_observation_history[-3:])
                    geometry_reward = min(0.5, recent_detections * 0.1)
                    
                reward += (confidence_reward + freshness_reward + 
                          miss_penalty + geometry_reward)
            else:
                # No track - significant penalty
                reward -= 0.5
                
            # Fuel efficiency penalty (always known)
            fuel_penalty = -0.1 * (1.0 - self.fuel_remaining / self.max_fuel)
            
            # Time pressure
            time_penalty = -0.03
            
            reward += fuel_penalty + time_penalty
            return reward
    
    def _update_sensor_systems(self):
        """Update realistic sensor systems and target tracking"""
        if not self.enable_realistic_sensors:
            return
            
        current_time = self.simulation_time
        
        # Only update sensors at specified rate
        if current_time - self.last_sensor_update < self.sensor_update_rate:
            return
            
        self.last_sensor_update = current_time
        
        # Get target position and velocity (works for both 3DOF and 6DOF)
        if self.missile_6dof is not None:
            target_position = self.missile_6dof.position
            target_velocity = self.missile_6dof.velocity
        elif hasattr(self, 'missile_pos_3d') and self.missile_pos_3d is not None:
            # Fall back to 3DOF positions
            target_position = self.missile_pos_3d
            target_velocity = self.missile_vel_3d
        else:
            # No target available yet (during initialization)
            return
            
        # Check if we have valid sensor systems
        if (not hasattr(self, 'ground_radar') or self.ground_radar is None or
            not hasattr(self, 'sensor_environment') or self.sensor_environment is None or
            not hasattr(self, 'sensor_fusion') or self.sensor_fusion is None):
            return
        
        # Get environmental effects
        env_effects = self.sensor_environment.calculate_sensor_degradation(
            self.ground_radar.position, target_position,
            self.ground_radar.config.frequency, 
            1.0 / self.ground_radar.config.pulse_width
        )
        
        # Ground radar detection attempt
        ground_detection = self.ground_radar.detect_target(
            target_position,
            target_velocity,
            self.missile_rcs,
            current_time,
            atmospheric_effects={'path_loss': env_effects['atmospheric_loss_db']},
            clutter_effects=self.sensor_environment.get_clutter_effects(
                self.ground_radar.position,
                (target_position - self.ground_radar.position) / 
                np.linalg.norm(target_position - self.ground_radar.position),
                self.ground_radar.beamwidth,
                np.linalg.norm(target_position - self.ground_radar.position)
            )
        )
        
        # Process detections through sensor fusion
        detections = []
        if ground_detection is not None:
            detections.append(ground_detection)
            
        # Add false alarms
        false_alarms = self.ground_radar.generate_false_alarms(current_time, 0.1)
        detections.extend(false_alarms)
        
        # Update tracking
        updated_tracks = self.sensor_fusion.process_radar_detections(
            detections, self.ground_radar.position, current_time
        )
        
        # Get best target estimate
        best_track = self.sensor_fusion.get_best_target_estimate()
        if best_track is not None:
            self.estimated_missile_state = best_track
            self.tracking_confidence = best_track.confidence
        else:
            # No valid tracks - use last known state with reduced confidence
            if self.estimated_missile_state is not None:
                self.tracking_confidence = max(0.0, self.tracking_confidence - 0.1)
                
        # Store observation history
        obs_data = {
            'timestamp': current_time,
            'detections': len(detections),
            'tracks': len(updated_tracks),
            'best_track_confidence': self.tracking_confidence,
            'environmental_effects': env_effects
        }
        self.sensor_observation_history.append(obs_data)
        
        # Keep only recent history
        if len(self.sensor_observation_history) > 100:
            self.sensor_observation_history = self.sensor_observation_history[-50:]
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation based on sensor mode"""
        
        if self.enable_realistic_sensors:
            # Update sensor systems
            self._update_sensor_systems()
            
            # Build sensor-based observation
            return self._get_sensor_observation()
        else:
            # Legacy perfect information mode
            return self._get_perfect_observation()
            
    def _get_perfect_observation(self) -> np.ndarray:
        """Get perfect information observation (legacy mode)"""
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
            
    def _get_sensor_observation(self) -> np.ndarray:
        """Get realistic sensor-based observation"""
        
        time_remaining = (self.max_steps - self.step_count) / self.max_steps
        fuel_remaining = self.fuel_remaining / self.max_fuel
        
        if self.legacy_3dof_mode or self.difficulty_mode == DifficultyMode.EASY_3DOF:
            # 3DOF sensor observation (22D)
            
            # Own state (8D) - perfect knowledge of own platform
            own_state = np.array([
                self.interceptor_pos_3d[0], self.interceptor_pos_3d[1], self.interceptor_pos_3d[2],
                self.interceptor_vel_3d[0], self.interceptor_vel_3d[1], self.interceptor_vel_3d[2],
                fuel_remaining,
                time_remaining
            ])
            
            # Target estimate (8D) - from sensor fusion
            if self.estimated_missile_state is not None:
                target_estimate = np.array([
                    self.estimated_missile_state.position[0],
                    self.estimated_missile_state.position[1], 
                    self.estimated_missile_state.position[2],
                    self.estimated_missile_state.velocity[0],
                    self.estimated_missile_state.velocity[1],
                    self.estimated_missile_state.velocity[2],
                    self.tracking_confidence,
                    min(1.0, (self.simulation_time - self.estimated_missile_state.creation_time) / 10.0)
                ])
            else:
                # No track - return zeros with low confidence
                target_estimate = np.zeros(8)
                target_estimate[6] = 0.0  # confidence
                
            # Sensor data (4D)
            last_detection_time = 0.0
            snr = 0.0
            range_rate = 0.0
            measurement_count = 0
            
            if self.sensor_observation_history:
                last_obs = self.sensor_observation_history[-1]
                last_detection_time = min(1.0, (self.simulation_time - last_obs['timestamp']) / 5.0)
                snr = 0.5  # Normalized SNR estimate
                if self.estimated_missile_state is not None:
                    measurement_count = min(1.0, self.estimated_missile_state.measurement_count / 20.0)
                    
            sensor_data = np.array([last_detection_time, snr, range_rate, measurement_count])
            
            # Radar status (2D)
            radar_status = np.array([
                1.0 if self.ground_radar.is_active else 0.0,
                0.1  # Normalized false alarm rate
            ])
            
            return np.concatenate([own_state, target_estimate, sensor_data, radar_status]).astype(np.float32)
            
        else:
            # 6DOF sensor observation (44D)
            
            # Own state (15D) - perfect knowledge of own platform
            interceptor_state = self.interceptor_6dof.get_state_vector()  # 13D
            own_state = np.concatenate([
                interceptor_state,
                [fuel_remaining, time_remaining]
            ])
            
            # Target estimate (12D) - from sensor fusion
            if self.estimated_missile_state is not None:
                target_estimate = np.array([
                    self.estimated_missile_state.position[0],
                    self.estimated_missile_state.position[1],
                    self.estimated_missile_state.position[2],
                    self.estimated_missile_state.velocity[0],
                    self.estimated_missile_state.velocity[1],
                    self.estimated_missile_state.velocity[2],
                    self.estimated_missile_state.acceleration[0],
                    self.estimated_missile_state.acceleration[1],
                    self.estimated_missile_state.acceleration[2],
                    self.tracking_confidence,
                    min(1.0, (self.simulation_time - self.estimated_missile_state.creation_time) / 20.0),
                    1.0 if self.estimated_missile_state.quality.value == 'confirmed' else 0.5
                ])
            else:
                target_estimate = np.zeros(12)
                
            # Sensor measurements (5D) - latest raw measurements
            sensor_measurements = np.zeros(5)
            if self.sensor_observation_history and self.estimated_missile_state is not None:
                # Simulate latest measurements
                rel_pos = self.estimated_missile_state.position - self.ground_radar.position
                range_est = np.linalg.norm(rel_pos)
                azimuth_est = np.degrees(np.arctan2(rel_pos[1], rel_pos[0]))
                elevation_est = np.degrees(np.arcsin(rel_pos[2] / range_est)) if range_est > 0 else 0
                doppler_est = 0.0  # Simplified
                snr_est = 15.0  # Simplified
                
                sensor_measurements = np.array([
                    range_est / 100000.0,  # Normalized range
                    azimuth_est / 180.0,   # Normalized azimuth  
                    elevation_est / 90.0,  # Normalized elevation
                    doppler_est / 100.0,   # Normalized doppler
                    min(1.0, snr_est / 30.0)  # Normalized SNR
                ])
                
            # Sensor uncertainties (6D)
            if self.estimated_missile_state is not None:
                pos_cov = np.diag(self.estimated_missile_state.covariance[0:3, 0:3])
                vel_cov = np.diag(self.estimated_missile_state.covariance[3:6, 3:6])
                pos_std = np.sqrt(np.maximum(pos_cov, 1.0)) / 1000.0  # Normalized
                vel_std = np.sqrt(np.maximum(vel_cov, 1.0)) / 100.0   # Normalized
                uncertainties = np.concatenate([pos_std, vel_std])
            else:
                uncertainties = np.ones(6) * 0.9  # High uncertainty
                
            # Tracking info (3D)
            if self.estimated_missile_state is not None:
                tracking_info = np.array([
                    min(1.0, self.estimated_missile_state.consecutive_misses / 5.0),
                    min(1.0, self.estimated_missile_state.measurement_count / 50.0),
                    min(1.0, (self.simulation_time - self.estimated_missile_state.last_update_time) / 2.0)
                ])
            else:
                tracking_info = np.array([1.0, 0.0, 1.0])  # High misses, no measurements, old data
                
            # Environment effects (3D)
            weather_code = {'clear': 0.0, 'light_rain': 0.3, 'heavy_rain': 0.7, 'storm': 1.0}
            env_effects = np.array([
                weather_code.get(self.weather_conditions.value, 0.0),
                self.wind_speed / 20.0,  # Normalized wind speed
                0.0  # Jamming level (not implemented yet)
            ])
            
            return np.concatenate([
                own_state,           # 15D
                target_estimate,     # 12D
                sensor_measurements, # 5D
                uncertainties,       # 6D
                tracking_info,       # 3D
                env_effects         # 3D
            ]).astype(np.float32)   # Total: 44D
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about current state"""
        info = {
            "step_count": self.step_count,
            "simulation_time": self.simulation_time,
            "fuel_remaining": self.fuel_remaining,
            "difficulty_mode": self.difficulty_mode.value,
            "action_mode": self.action_mode.value,
            "enable_realistic_sensors": self.enable_realistic_sensors,
        }
        
        # Add sensor-specific information
        if self.enable_realistic_sensors:
            info.update({
                "tracking_confidence": self.tracking_confidence,
                "estimated_missile_state": self.estimated_missile_state is not None,
                "sensor_update_rate": self.sensor_update_rate,
                "weather_conditions": self.weather_conditions.value,
            })
            
            # Add sensor performance metrics
            if hasattr(self, 'ground_radar'):
                radar_stats = self.ground_radar.get_performance_stats()
                info.update({
                    "sensor_detections": radar_stats['total_detections'],
                    "false_alarms": radar_stats['false_alarms'],
                    "detection_rate": radar_stats['detection_rate'],
                })
                
            # Add fusion system metrics
            if hasattr(self, 'sensor_fusion'):
                fusion_stats = self.sensor_fusion.get_fusion_statistics()
                info.update({
                    "active_tracks": fusion_stats['active_tracks'],
                    "confirmed_tracks": fusion_stats['confirmed_tracks'],
                    "association_rate": fusion_stats['association_rate'],
                    "tracking_accuracy": fusion_stats['association_rate'],  # Proxy for accuracy
                })
        
        if not self.legacy_3dof_mode and self.interceptor_6dof is not None:
            if self.enable_realistic_sensors:
                # In realistic mode, only include information that would be available to sensors
                info.update({
                    "interceptor_6dof_state": self.interceptor_6dof.get_state_vector(),  # Own platform always known
                    "wind_velocity": self.wind_velocity,  # Environmental sensors
                    "episode_stats": self.episode_stats.copy()
                })
                
                # Only include target information if we have a valid track
                if self.estimated_missile_state is not None:
                    info.update({
                        "estimated_missile_position": self.estimated_missile_state.position,
                        "estimated_missile_velocity": self.estimated_missile_state.velocity,
                        "estimate_uncertainty": np.diag(self.estimated_missile_state.covariance[0:3, 0:3]),
                        "track_quality": self.estimated_missile_state.quality.value,
                    })
            else:
                # Legacy perfect information mode
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
            if self.enable_realistic_sensors:
                # 3DOF realistic mode
                info.update({
                    "interceptor_pos": self.interceptor_pos_3d.copy(),  # Own platform always known
                })
                
                if self.estimated_missile_state is not None:
                    info.update({
                        "estimated_missile_pos": self.estimated_missile_state.position,
                        "intercept_distance_estimate": np.linalg.norm(
                            self.estimated_missile_state.position - self.interceptor_pos_3d
                        ),
                    })
            else:
                # Legacy perfect information mode
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
    
    def render(self, intercepted: bool = False, render_mode: str = "god_view"):
        """Render the environment with multiple visualization modes
        
        Args:
            intercepted: Whether intercept occurred
            render_mode: 'god_view', 'radar_ppi', 'radar_3d', or 'both'
        """
        if self.render_mode == "human":
            current_time = self.simulation_time
            dt = current_time - self.last_render_time
            self.last_render_time = current_time
            
            if render_mode in ["god_view", "both"]:
                # Traditional god-view visualization
                if self.viewer is None:
                    speed = getattr(self, 'viz_speed_multiplier', 10.0)
                    self.viewer = Viewer3D(self.world_size, speed_multiplier=speed)
                
                self.viewer.render(
                    self.interceptor_pos_3d, 
                    self.missile_pos_3d, 
                    self.target_pos, 
                    intercepted
                )
            
            if render_mode in ["radar_ppi", "both"] and self.enable_realistic_sensors:
                # Realistic radar PPI display
                if self.radar_viewer is None:
                    self.radar_viewer = RadarPPI(max_range=self.ground_radar.config.max_range)
                
                # Update radar sweep
                self.radar_viewer.update_sweep(dt)
                
                # Add detections only when radar beam sweeps over targets
                self._add_radar_detections_to_viewer()
                
                # Get sensor status for display
                sensor_info = self._get_sensor_status_for_display()
                
                # Render radar display
                self.radar_viewer.render(self.ground_radar.position, sensor_info)
            
            if render_mode in ["radar_3d"] and self.enable_realistic_sensors:
                # 3D radar coverage visualization
                if self.radar_3d_viewer is None:
                    self.radar_3d_viewer = RadarViewer3D(self.world_size, self.ground_radar.position)
                
                # Get estimated tracks for display
                estimated_tracks = [self.estimated_missile_state] if self.estimated_missile_state else []
                
                # Get recent detections
                recent_detections = self._get_recent_detections_for_display()
                
                # Update 3D display
                self.radar_3d_viewer.update(estimated_tracks, self.interceptor_pos_3d, recent_detections)
    
    def _add_radar_detections_to_viewer(self):
        """Add realistic radar detections to the PPI viewer"""
        if not hasattr(self, 'ground_radar') or self.ground_radar is None:
            return
            
        # Check if radar beam is currently illuminating the target
        if self.missile_6dof is not None:
            target_pos = self.missile_6dof.position
        else:
            target_pos = self.missile_pos_3d
            
        # Calculate target azimuth from radar
        relative_pos = target_pos - self.ground_radar.position
        target_azimuth = math.degrees(math.atan2(relative_pos[1], relative_pos[0]))
        if target_azimuth < 0:
            target_azimuth += 360
            
        # Check if beam is currently on target (within beamwidth)
        beam_diff = abs(target_azimuth - self.radar_viewer.current_azimuth)
        if beam_diff > 180:
            beam_diff = 360 - beam_diff
            
        beamwidth = math.degrees(self.ground_radar.beamwidth)
        
        if beam_diff <= beamwidth / 2:
            # Radar is illuminating target - attempt detection
            detection = self.ground_radar.detect_target(
                target_pos,
                self.missile_6dof.velocity if self.missile_6dof else self.missile_vel_3d,
                self.missile_rcs,
                self.simulation_time
            )
            
            if detection is not None:
                # Convert to global coordinates for viewer
                det_range = detection.range
                det_azimuth = math.radians(detection.azimuth)
                det_elevation = math.radians(detection.elevation)
                
                global_pos = self.ground_radar.position + np.array([
                    det_range * math.cos(det_elevation) * math.cos(det_azimuth),
                    det_range * math.cos(det_elevation) * math.sin(det_azimuth),
                    det_range * math.sin(det_elevation)
                ])
                
                # Add to radar viewer
                track_id = None
                if self.estimated_missile_state is not None:
                    track_id = self.estimated_missile_state.track_id
                    
                self.radar_viewer.add_detection(
                    global_pos, detection.snr, self.simulation_time,
                    'target', track_id
                )
        
        # Add false alarms occasionally
        if np.random.random() < 0.01:  # 1% chance per frame
            false_alarms = self.ground_radar.generate_false_alarms(self.simulation_time, 0.1)
            for fa in false_alarms:
                # Convert false alarm to global coordinates
                fa_range = fa.range
                fa_azimuth = math.radians(fa.azimuth)
                fa_elevation = math.radians(fa.elevation)
                
                fa_pos = self.ground_radar.position + np.array([
                    fa_range * math.cos(fa_elevation) * math.cos(fa_azimuth),
                    fa_range * math.cos(fa_elevation) * math.sin(fa_azimuth),
                    fa_range * math.sin(fa_elevation)
                ])
                
                self.radar_viewer.add_detection(
                    fa_pos, fa.snr, self.simulation_time, 'false_alarm'
                )
    
    def _get_sensor_status_for_display(self) -> dict:
        """Get sensor status information for radar display"""
        if not hasattr(self, 'ground_radar') or self.ground_radar is None:
            return {}
            
        radar_stats = self.ground_radar.get_performance_stats()
        
        return {
            'radar_active': self.ground_radar.is_active,
            'detection_rate': radar_stats.get('detection_rate', 0.0),
            'false_alarm_rate': radar_stats.get('false_alarm_rate', 0.0),
            'active_tracks': len(self.sensor_fusion.active_tracks) if hasattr(self, 'sensor_fusion') else 0,
            'weather': self.weather_conditions.value if hasattr(self, 'weather_conditions') else 'clear'
        }
    
    def _get_recent_detections_for_display(self) -> list:
        """Get recent detections for 3D display"""
        # Return recent detections from radar viewer if available
        if hasattr(self, 'radar_viewer') and self.radar_viewer is not None:
            recent_time = self.simulation_time - 10.0  # Last 10 seconds
            return [{
                'position': det['position'],
                'type': det['type']
            } for det in self.radar_viewer.detections 
            if det['timestamp'] > recent_time]
        return []
    
    def close(self):
        """Close the environment"""
        if self.viewer:
            self.viewer.close()
        if hasattr(self, 'radar_viewer') and self.radar_viewer:
            self.radar_viewer.close()
        if hasattr(self, 'radar_3d_viewer') and self.radar_3d_viewer:
            self.radar_3d_viewer.close()


# Register the environment
gym.register(
    id='Aegis6DIntercept-v0',
    entry_point='aegis_intercept.envs.aegis_6dof_env:Aegis6DInterceptEnv',
)