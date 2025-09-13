"""
Multi-entity radar environment for Phase 4 RL training.

This module provides the RadarEnv class that supports multiple missiles and interceptors
with configurable scenarios and radar processing capabilities.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, List, Tuple, Optional, Union
import pygame
import math

try:
    from .config import get_config
    from .scenarios import get_scenario_loader
    from .radar_observations import Radar17DObservation
except ImportError:
    # Fallback for direct execution
    from config import get_config
    from scenarios import get_scenario_loader
    from radar_observations import Radar17DObservation


class RadarEnv(gym.Env):
    """
    Multi-entity radar environment supporting variable numbers of missiles and interceptors.
    
    This environment processes ground-based and on-board radar returns to provide
    observations for RL agents controlling multiple interceptors against multiple missiles.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 scenario_name: Optional[str] = None,
                 num_missiles: Optional[int] = None,
                 num_interceptors: Optional[int] = None,
                 render_mode: Optional[str] = None):
        """
        Initialize RadarEnv.
        
        Args:
            config: Environment configuration dictionary
            scenario_name: Name of scenario to load (overrides config)
            num_missiles: Number of missiles (overrides config)
            num_interceptors: Number of interceptors (overrides config)
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        
        # Load configuration
        self.config_loader = get_config()
        if config is None:
            config = self.config_loader.get_environment_config()
        
        # Load scenario if specified
        if scenario_name is not None:
            scenario_loader = get_scenario_loader()
            scenario_config = scenario_loader.create_environment_config(
                scenario_name, self.config_loader._config
            )
            config.update(scenario_config)
        
        self.config = config
        
        # Override entity counts if specified
        if num_missiles is not None:
            if 'environment' not in self.config:
                self.config['environment'] = {}
            self.config['environment']['num_missiles'] = num_missiles
        if num_interceptors is not None:
            if 'environment' not in self.config:
                self.config['environment'] = {}
            self.config['environment']['num_interceptors'] = num_interceptors
        
        # Environment parameters - get from environment section or top level
        env_config = self.config.get('environment', self.config)
        self.num_missiles = env_config.get('num_missiles', 1)
        self.num_interceptors = env_config.get('num_interceptors', 1)
        self.max_episode_steps = env_config.get('max_episode_steps', 1000)
        
        # Radar configuration
        self.radar_config = self.config.get('radar_config', self.config.get('radar', {}))
        self.radar_range = self.radar_config.get('range', 1000.0)
        self.radar_noise = self.radar_config.get('noise_level', 0.05)
        self.radar_update_rate = self.radar_config.get('update_rate', 10)
        
        # Spawn configuration
        self.spawn_config = self.config.get('spawn_config', self.config.get('spawn', {}))
        
        # Environment state - 6DOF 3D
        self.missile_positions = np.zeros((self.num_missiles, 3), dtype=np.float64)
        self.missile_velocities = np.zeros((self.num_missiles, 3), dtype=np.float64)
        # Initialize orientations with identity quaternions [w,x,y,z]
        self.missile_orientations = np.tile([1.0, 0.0, 0.0, 0.0], (self.num_missiles, 1)).astype(np.float64)
        self.missile_angular_velocities = np.zeros((self.num_missiles, 3), dtype=np.float64)
        
        self.interceptor_positions = np.zeros((self.num_interceptors, 3), dtype=np.float64)
        self.interceptor_velocities = np.zeros((self.num_interceptors, 3), dtype=np.float64)
        # Initialize orientations with identity quaternions [w,x,y,z]
        self.interceptor_orientations = np.tile([1.0, 0.0, 0.0, 0.0], (self.num_interceptors, 1)).astype(np.float64)
        self.interceptor_fuel = np.full(self.num_interceptors, 100.0, dtype=np.float64)  # Fuel levels
        self.interceptor_angular_velocities = np.zeros((self.num_interceptors, 3), dtype=np.float64)
        self.interceptor_active = np.ones(self.num_interceptors, dtype=bool)  # Active status
        
        self.target_positions = np.zeros((self.num_missiles, 3), dtype=np.float64)
        
        # Initialize orientations to identity quaternions
        self.missile_orientations[:, 0] = 1.0  # w = 1
        self.interceptor_orientations[:, 0] = 1.0  # w = 1
        
        # Additional 6DOF state
        self.interceptor_fuel = np.full(self.num_interceptors, 100.0, dtype=np.float64)  # kg
        self.interceptor_active = np.ones(self.num_interceptors, dtype=bool)
        self.missile_active = np.ones(self.num_missiles, dtype=bool)
        
        # Physical parameters
        self.gravity = np.array([0.0, 0.0, -9.81])  # ENU coordinates
        self.air_density = 1.225  # kg/m³ at sea level
        self.interceptor_mass = 10.0  # kg
        self.missile_mass = 100.0  # kg
        
        # Simulation parameters
        self.dt = 0.1  # Time step in seconds
        self.current_step = 0
        self.episode_length = 0
        
        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_size = 800
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Scenario-specific configuration
        self.scenario_config = self.config.get('scenario', {})
        self.adversary_config = self.scenario_config.get('adversary_config', {})
        self.success_criteria = self.scenario_config.get('success_criteria', {})
        
    def _setup_spaces(self):
        """Setup observation and action spaces based on entity configuration."""
        
        # Special case: 1v1 scenarios use 17D radar observations
        if self.num_missiles == 1 and self.num_interceptors == 1:
            # Use 17D radar observation space for 1v1 scenarios
            self.use_radar_17d = True
            self.radar_observer = Radar17DObservation(
                max_range=10000.0,
                max_velocity=1000.0
            )
            
            # Set observation space to exactly 17 dimensions
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(17,),
                dtype=np.float32
            )
        else:
            # Use original observation space for multi-entity scenarios
            self.use_radar_17d = False
            
            # Base observation dimensions per entity (3D)
            missile_obs_dim = 9  # pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, target_x, target_y, target_z
            interceptor_obs_dim = 10  # pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, fuel, status, target_missile_id, distance_to_target
            
            # Radar return dimensions
            ground_radar_dim = 4 * self.num_missiles  # detected_x, detected_y, confidence, range for each missile
            onboard_radar_dim = 3 * self.num_interceptors  # local_detections, bearing, range for each interceptor
            
            # Environmental data
            env_data_dim = 6  # wind_x, wind_y, atmospheric_density, temperature, visibility, em_interference
            
            # Relative positioning matrix (interceptor-missile pairs)
            relative_pos_dim = self.num_interceptors * self.num_missiles * 3  # distance, bearing, relative_velocity
            
            total_obs_dim = (
                self.num_missiles * missile_obs_dim +
                self.num_interceptors * interceptor_obs_dim +
                ground_radar_dim +
                onboard_radar_dim +
                env_data_dim +
                relative_pos_dim
            )
            
            # Observation space: normalized values between -1 and 1
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(total_obs_dim,),
                dtype=np.float32
            )
        
        # Action space: continuous control for each interceptor
        # Actions per interceptor: thrust_x, thrust_y, thrust_magnitude, steering_angle, special_action, confidence
        actions_per_interceptor = 6
        total_action_dim = self.num_interceptors * actions_per_interceptor
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(total_action_dim,),
            dtype=np.float32
        )
        
        # Store dimensions for easy access
        if hasattr(self, 'use_radar_17d') and self.use_radar_17d:
            self.obs_dim = 17
        else:
            self.obs_dim = total_obs_dim
        self.action_dim = total_action_dim
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Seed the radar observer if using 17D observations
        if hasattr(self, 'use_radar_17d') and self.use_radar_17d and hasattr(self, 'radar_observer'):
            if seed is not None:
                self.radar_observer.seed(seed)
        
        self.current_step = 0
        self.episode_length = 0
        
        # Initialize entity positions based on spawn configuration
        self._initialize_positions()
        
        # Initialize velocities
        self._initialize_velocities()
        
        # Reset environmental conditions
        self._reset_environment()
        
        # Generate initial observation
        observation = self._get_observation()
        
        info = {
            'num_missiles': self.num_missiles,
            'num_interceptors': self.num_interceptors,
            'scenario': self.scenario_config.get('name', 'default'),
            'difficulty': self.scenario_config.get('difficulty_level', 1)
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one environment step.
        
        Args:
            action: Action array for all interceptors
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        self.episode_length += 1
        
        # Process actions for each interceptor
        self._process_actions(action)
        
        # Update missile dynamics (adversary behavior)
        self._update_missiles()
        
        # Update interceptor dynamics
        self._update_interceptors()
        
        # Update environmental conditions
        self._update_environment()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.episode_length >= self.max_episode_steps
        
        # Generate observation
        observation = self._get_observation()
        
        # Compile info dictionary
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _initialize_positions(self):
        """Initialize entity positions based on spawn configuration."""
        
        # Get spawn positions from configuration
        if 'missile_spawn_positions' in self.spawn_config:
            missile_spawns = np.array(self.spawn_config['missile_spawn_positions'], dtype=np.float64)
            if len(missile_spawns) >= self.num_missiles:
                self.missile_positions = missile_spawns[:self.num_missiles].copy().astype(np.float64)
            else:
                # Fall back to random spawning if not enough positions specified
                self._random_missile_spawn()
        else:
            self._random_missile_spawn()
        
        if 'interceptor_spawn_positions' in self.spawn_config:
            interceptor_spawns = np.array(self.spawn_config['interceptor_spawn_positions'], dtype=np.float64)
            if len(interceptor_spawns) >= self.num_interceptors:
                self.interceptor_positions = interceptor_spawns[:self.num_interceptors].copy().astype(np.float64)
            else:
                self._random_interceptor_spawn()
        else:
            self._random_interceptor_spawn()
        
        if 'target_positions' in self.spawn_config:
            target_spawns = np.array(self.spawn_config['target_positions'], dtype=np.float64)
            if len(target_spawns) >= self.num_missiles:
                self.target_positions = target_spawns[:self.num_missiles].copy().astype(np.float64)
            else:
                self._random_target_spawn()
        else:
            self._random_target_spawn()
    
    def _random_missile_spawn(self):
        """Randomly spawn missiles in designated 3D area."""
        spawn_area = self.spawn_config.get('missile_spawn_area', [[-100, -100, 100], [100, 100, 500]])
        for i in range(self.num_missiles):
            # Handle both 2D and 3D spawn areas
            if len(spawn_area[0]) == 2:
                # 2D area, add default altitude
                pos_2d = self.np_random.uniform(spawn_area[0], spawn_area[1])
                self.missile_positions[i] = [pos_2d[0], pos_2d[1], 300.0]  # Default altitude
            else:
                # 3D area
                self.missile_positions[i] = self.np_random.uniform(spawn_area[0], spawn_area[1])
    
    def _random_interceptor_spawn(self):
        """Randomly spawn interceptors in designated 3D area."""
        spawn_area = self.spawn_config.get('interceptor_spawn_area', [[400, 400, 50], [600, 600, 200]])
        for i in range(self.num_interceptors):
            # Handle both 2D and 3D spawn areas
            if len(spawn_area[0]) == 2:
                # 2D area, add default altitude
                pos_2d = self.np_random.uniform(spawn_area[0], spawn_area[1])
                self.interceptor_positions[i] = [pos_2d[0], pos_2d[1], 100.0]  # Default altitude
            else:
                # 3D area
                self.interceptor_positions[i] = self.np_random.uniform(spawn_area[0], spawn_area[1])
    
    def _random_target_spawn(self):
        """Randomly spawn targets in designated 3D area."""
        target_area = self.spawn_config.get('target_area', [[800, 800, 0], [1000, 1000, 10]])
        for i in range(self.num_missiles):
            # Handle both 2D and 3D target areas
            if len(target_area[0]) == 2:
                # 2D area, add default altitude
                pos_2d = self.np_random.uniform(target_area[0], target_area[1])
                self.target_positions[i] = [pos_2d[0], pos_2d[1], 5.0]  # Ground level target
            else:
                # 3D area
                self.target_positions[i] = self.np_random.uniform(target_area[0], target_area[1])
    
    def _initialize_velocities(self):
        """Initialize entity velocities."""
        # Missiles start with velocity toward their targets
        for i in range(self.num_missiles):
            direction = self.target_positions[i] - self.missile_positions[i]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-8:
                direction = direction / direction_norm
            else:
                direction = np.array([1.0, 0.0, 0.0])  # Default direction if target == position
            
            base_speed = 50.0  # Base missile speed
            speed_multiplier = self.adversary_config.get('speed_multiplier', 1.0)
            self.missile_velocities[i] = direction * base_speed * speed_multiplier
        
        # Interceptors start stationary
        self.interceptor_velocities.fill(0.0)
    
    def _reset_environment(self):
        """Reset environmental conditions."""
        # This will be expanded based on scenario configuration
        pass
    
    def _process_actions(self, action: np.ndarray):
        """Process actions for all interceptors."""
        # Ensure action is a numpy array
        action = np.asarray(action, dtype=np.float32)
        
        # Handle scalar or 1D case
        if action.ndim == 0:
            action = np.array([action])
        
        actions_per_interceptor = self.action_dim // self.num_interceptors
        expected_actions = self.num_interceptors * actions_per_interceptor
        
        # Ensure we have the right number of actions
        if len(action) != expected_actions:
            # Pad or truncate to expected size
            if len(action) < expected_actions:
                action = np.pad(action, (0, expected_actions - len(action)), 'constant', constant_values=0)
            else:
                action = action[:expected_actions]
        
        for i in range(self.num_interceptors):
            start_idx = i * actions_per_interceptor
            end_idx = start_idx + actions_per_interceptor
            interceptor_action = action[start_idx:end_idx]
            
            # Apply action to interceptor i
            self._apply_interceptor_action(i, interceptor_action)
    
    def _apply_interceptor_action(self, interceptor_id: int, action: np.ndarray):
        """Apply 6DOF action to specific interceptor."""
        if not self.interceptor_active[interceptor_id]:
            return
        
        # Extract 6DOF action components: [thrust_x, thrust_y, thrust_z, torque_x, torque_y, torque_z]
        thrust_body = np.array([
            action[0] * 1000.0,  # Body-frame thrust X (N)
            action[1] * 1000.0,  # Body-frame thrust Y (N) 
            action[2] * 1000.0   # Body-frame thrust Z (N)
        ])
        
        torque_body = np.array([
            action[3] * 50.0,    # Torque about X-axis (Nm)
            action[4] * 50.0,    # Torque about Y-axis (Nm)
            action[5] * 50.0     # Torque about Z-axis (Nm)
        ])
        
        # Convert body-frame thrust to world-frame using quaternion
        q = self.interceptor_orientations[interceptor_id]
        thrust_world = self._rotate_vector_by_quaternion(thrust_body, q)
        
        # Apply forces (F = ma, so a = F/m)
        mass = self.interceptor_mass
        acceleration = thrust_world / mass + self.gravity
        
        # Update velocity (Euler integration)
        self.interceptor_velocities[interceptor_id] += acceleration * self.dt
        
        # Apply angular dynamics
        # Simplified: torque directly affects angular velocity (assuming identity inertia tensor)
        inertia_inv = 1.0 / 10.0  # 1/I, simplified
        angular_acceleration = torque_body * inertia_inv
        self.interceptor_angular_velocities[interceptor_id] += angular_acceleration * self.dt
        
        # Apply limits
        max_velocity = 300.0  # m/s
        max_angular_velocity = 10.0  # rad/s
        
        velocity_norm = np.linalg.norm(self.interceptor_velocities[interceptor_id])
        if velocity_norm > max_velocity:
            self.interceptor_velocities[interceptor_id] *= max_velocity / velocity_norm
            
        angular_velocity_norm = np.linalg.norm(self.interceptor_angular_velocities[interceptor_id])
        if angular_velocity_norm > max_angular_velocity:
            self.interceptor_angular_velocities[interceptor_id] *= max_angular_velocity / angular_velocity_norm
        
        # Consume fuel based on thrust magnitude
        fuel_consumption_rate = 0.01  # kg/s per 1000N
        thrust_magnitude = np.linalg.norm(thrust_body)
        fuel_used = fuel_consumption_rate * (thrust_magnitude / 1000.0) * self.dt
        self.interceptor_fuel[interceptor_id] = max(0.0, self.interceptor_fuel[interceptor_id] - fuel_used)
        
        # Deactivate if out of fuel
        if self.interceptor_fuel[interceptor_id] <= 0.0:
            self.interceptor_active[interceptor_id] = False
    
    def _rotate_vector_by_quaternion(self, vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """Rotate a vector by a quaternion (q = [w, x, y, z])."""
        w, x, y, z = quaternion
        v = vector
        
        # Quaternion rotation: v' = q * v * q*
        # Using the formula: v' = v + 2 * cross(q_vec, cross(q_vec, v) + w * v)
        q_vec = np.array([x, y, z])
        cross1 = np.cross(q_vec, v)
        cross2 = np.cross(q_vec, cross1 + w * v)
        rotated = v + 2 * cross2
        
        return rotated
    
    def _integrate_quaternion(self, quaternion: np.ndarray, angular_velocity: np.ndarray, dt: float) -> np.ndarray:
        """Integrate quaternion using angular velocity."""
        w, x, y, z = quaternion
        wx, wy, wz = angular_velocity
        
        # Quaternion derivative: dq/dt = 0.5 * q * [0, wx, wy, wz]
        dq = 0.5 * np.array([
            -wx*x - wy*y - wz*z,
             wx*w + wz*y - wy*z,
             wy*w - wz*x + wx*z,
             wz*w + wy*x - wx*y
        ])
        
        # Euler integration
        new_q = quaternion + dq * dt
        
        # Normalize quaternion
        norm = np.linalg.norm(new_q)
        if norm > 1e-6:
            new_q /= norm
        else:
            new_q = np.array([1.0, 0.0, 0.0, 0.0])  # Reset to identity if degenerate
        
        return new_q
    
    def _update_missiles(self):
        """Update missile positions and behaviors in 3D."""
        for i in range(self.num_missiles):
            if not self.missile_active[i]:
                continue
                
            # Apply gravity
            self.missile_velocities[i] += self.gravity * self.dt
            
            # Apply adversary AI based on scenario configuration
            evasion_iq = self.adversary_config.get('evasion_iq', 0.2)
            maneuver_freq = self.adversary_config.get('maneuver_frequency', 0.1)
            
            if self.np_random.random() < maneuver_freq * evasion_iq:
                # Apply 3D evasive maneuver
                maneuver_strength = 50.0 * evasion_iq
                maneuver = self.np_random.uniform(-maneuver_strength, maneuver_strength, 3)
                maneuver[2] *= 0.5  # Reduce vertical maneuvering
                self.missile_velocities[i] += maneuver * self.dt
            
            # Apply air resistance
            drag_coefficient = 0.005
            velocity_magnitude = np.linalg.norm(self.missile_velocities[i])
            if velocity_magnitude > 0:
                drag_force = -drag_coefficient * self.missile_velocities[i] * velocity_magnitude
                drag_acceleration = drag_force / self.missile_mass
                self.missile_velocities[i] += drag_acceleration * self.dt
            
            # Update position
            self.missile_positions[i] += self.missile_velocities[i] * self.dt
            
            # Update missile orientation to face velocity direction (simplified)
            if velocity_magnitude > 1.0:
                velocity_normalized = self.missile_velocities[i] / velocity_magnitude
                self.missile_orientations[i] = self._velocity_to_quaternion(velocity_normalized)
    
    def _velocity_to_quaternion(self, velocity_normalized: np.ndarray) -> np.ndarray:
        """Convert velocity direction to quaternion (simplified)."""
        # Simplified orientation - just return identity for now
        # In a full implementation, would compute proper quaternion from velocity direction
        return np.array([1.0, 0.0, 0.0, 0.0])
    
    def _update_interceptors(self):
        """Update interceptor positions and orientations."""
        for i in range(self.num_interceptors):
            if not self.interceptor_active[i]:
                continue
                
            # Update position
            self.interceptor_positions[i] += self.interceptor_velocities[i] * self.dt
            
            # Update orientation using angular velocity
            self.interceptor_orientations[i] = self._integrate_quaternion(
                self.interceptor_orientations[i],
                self.interceptor_angular_velocities[i],
                self.dt
            )
            
            # Apply air resistance
            drag_coefficient = 0.01
            drag_force = -drag_coefficient * self.interceptor_velocities[i] * np.linalg.norm(self.interceptor_velocities[i])
            drag_acceleration = drag_force / self.interceptor_mass
            self.interceptor_velocities[i] += drag_acceleration * self.dt
            
            # Apply angular damping
            angular_damping = 0.95
            self.interceptor_angular_velocities[i] *= angular_damping
    
    def _update_environment(self):
        """Update environmental conditions."""
        # This will include wind effects, atmospheric changes, etc.
        pass
    
    def _get_observation(self) -> np.ndarray:
        """Generate current observation with radar processing."""
        
        # Use 17D observation for 1v1 scenarios
        if hasattr(self, 'use_radar_17d') and self.use_radar_17d:
            # Get interceptor state
            interceptor_state = {
                'position': self.interceptor_positions[0],
                'velocity': self.interceptor_velocities[0],
                'orientation': self.interceptor_orientations[0],
                'fuel': self.interceptor_fuel[0] if hasattr(self, 'interceptor_fuel') else 100.0
            }
            
            # Get missile state
            missile_state = {
                'position': self.missile_positions[0],
                'velocity': self.missile_velocities[0]
            }
            
            # Get radar quality (simulated based on range)
            rel_pos = missile_state['position'] - interceptor_state['position']
            range_to_target = np.linalg.norm(rel_pos)
            radar_quality = np.clip(1.0 - range_to_target / self.radar_observer.max_range, 0.1, 1.0)
            
            # Compute 17D observation
            obs = self.radar_observer.compute_observation(
                interceptor_state=interceptor_state,
                missile_state=missile_state,
                radar_quality=radar_quality,
                radar_noise=self.radar_noise
            )
            
            return obs
        
        # Original observation generation for multi-entity scenarios
        obs_components = []
        
        # Process radar returns
        ground_radar_returns = self._process_ground_radar()
        onboard_radar_returns = self._process_onboard_radar()
        
        # Missile observations
        for i in range(self.num_missiles):
            missile_obs = np.concatenate([
                self.missile_positions[i] / 1000.0,  # Normalize positions
                self.missile_velocities[i] / 200.0,  # Normalize velocities
                self.target_positions[i] / 1000.0    # Target positions
            ])
            obs_components.append(missile_obs)
        
        # Interceptor observations
        for i in range(self.num_interceptors):
            # Find closest missile for targeting info
            distances = [np.linalg.norm(self.interceptor_positions[i] - self.missile_positions[j]) 
                        for j in range(self.num_missiles)]
            closest_missile = np.argmin(distances)
            
            interceptor_obs = np.concatenate([
                self.interceptor_positions[i] / 1000.0,  # Position
                self.interceptor_velocities[i] / 200.0,  # Velocity
                [1.0, 1.0],  # Fuel and status (simplified)
                [closest_missile / max(1, self.num_missiles - 1)],  # Target ID normalized
                [distances[closest_missile] / 1000.0]  # Distance to target
            ])
            obs_components.append(interceptor_obs)
        
        # Add radar returns
        obs_components.extend([ground_radar_returns, onboard_radar_returns])
        
        # Environmental data
        env_data = np.array([0.0, 0.0, 1.0, 0.5, 1.0, 0.1])  # Simplified environmental state
        obs_components.append(env_data)
        
        # Relative positioning matrix
        relative_positions = []
        for i in range(self.num_interceptors):
            for j in range(self.num_missiles):
                distance = np.linalg.norm(self.interceptor_positions[i] - self.missile_positions[j])
                rel_pos = self.missile_positions[j] - self.interceptor_positions[i]
                bearing = np.arctan2(rel_pos[1], rel_pos[0]) / np.pi  # Normalize to [-1, 1]
                rel_velocity = np.linalg.norm(self.missile_velocities[j] - self.interceptor_velocities[i])
                
                relative_positions.extend([
                    distance / 1000.0,
                    bearing,
                    rel_velocity / 200.0
                ])
        
        obs_components.append(np.array(relative_positions))
        
        # Concatenate all observation components
        observation = np.concatenate(obs_components)
        
        # Ensure observation has correct dimensions
        if len(observation) != self.obs_dim:
            # Pad or truncate to match expected dimensions
            if len(observation) < self.obs_dim:
                observation = np.pad(observation, (0, self.obs_dim - len(observation)))
            else:
                observation = observation[:self.obs_dim]
        
        return observation.astype(np.float32)
    
    def _process_ground_radar(self) -> np.ndarray:
        """Process ground-based radar returns."""
        radar_returns = []
        
        for i in range(self.num_missiles):
            # Simulate radar detection with noise (3D position but radar returns 2D projection)
            detected_pos_3d = self.missile_positions[i] + self.np_random.normal(0, self.radar_noise * 10, 3)
            distance = np.linalg.norm(self.missile_positions[i])  # Full 3D distance
            confidence = max(0.0, 1.0 - distance / self.radar_range - self.np_random.uniform(0, self.radar_noise))
            
            # Ground radar typically provides 2D position (X,Y) with range
            radar_returns.extend([
                detected_pos_3d[0] / 1000.0,  # X position
                detected_pos_3d[1] / 1000.0,  # Y position  
                confidence,                   # Detection confidence
                distance / self.radar_range   # Normalized range
            ])
        
        return np.array(radar_returns)
    
    def _process_onboard_radar(self) -> np.ndarray:
        """Process onboard radar returns for each interceptor."""
        radar_returns = []
        
        for i in range(self.num_interceptors):
            # Count detections within onboard radar range
            onboard_range = self.radar_config.get('onboard_radar_range', 200.0)
            detections = 0
            avg_bearing = 0.0
            avg_range = 0.0
            
            for j in range(self.num_missiles):
                distance = np.linalg.norm(self.interceptor_positions[i] - self.missile_positions[j])
                if distance <= onboard_range:
                    detections += 1
                    rel_pos = self.missile_positions[j] - self.interceptor_positions[i]
                    bearing = np.arctan2(rel_pos[1], rel_pos[0])
                    avg_bearing += bearing
                    avg_range += distance
            
            if detections > 0:
                avg_bearing /= detections
                avg_range /= detections
            
            radar_returns.extend([
                detections / max(1, self.num_missiles),
                avg_bearing / np.pi,
                avg_range / onboard_range
            ])
        
        return np.array(radar_returns)
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current step."""
        reward = 0.0
        
        # Reward for interceptor-missile proximity
        for i in range(self.num_interceptors):
            for j in range(self.num_missiles):
                distance = np.linalg.norm(self.interceptor_positions[i] - self.missile_positions[j])
                if distance < 50.0:  # Successful interception
                    reward += 100.0
                else:
                    # Reward for getting closer
                    reward += max(0, 200.0 - distance) * 0.01
        
        # Penalty for time (encourage efficiency)
        reward -= 0.1
        
        # Penalty for excessive velocity (encourage fuel efficiency)
        for i in range(self.num_interceptors):
            velocity_penalty = np.linalg.norm(self.interceptor_velocities[i]) * 0.001
            reward -= velocity_penalty
        
        return reward
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        
        # Check for successful interceptions
        for i in range(self.num_interceptors):
            for j in range(self.num_missiles):
                distance = np.linalg.norm(self.interceptor_positions[i] - self.missile_positions[j])
                interception_threshold = self.success_criteria.get('interception_distance', 50.0)
                if distance < interception_threshold:
                    return True
        
        # Check if missiles reached targets
        for i in range(self.num_missiles):
            distance_to_target = np.linalg.norm(self.missile_positions[i] - self.target_positions[i])
            if distance_to_target < 20.0:
                return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get information dictionary for current step."""
        info = {
            'step': self.current_step,
            'episode_length': self.episode_length,
            'missile_positions': self.missile_positions.copy(),
            'interceptor_positions': self.interceptor_positions.copy(),
            'missile_velocities': self.missile_velocities.copy(),
            'interceptor_velocities': self.interceptor_velocities.copy(),
        }
        
        # Add interception distances
        min_distances = []
        for i in range(self.num_interceptors):
            distances = [np.linalg.norm(self.interceptor_positions[i] - self.missile_positions[j]) 
                        for j in range(self.num_missiles)]
            min_distances.append(min(distances) if distances else float('inf'))
        
        info['min_interception_distances'] = min_distances
        
        return info
    
    def render(self, mode: Optional[str] = None):
        """
        Render the environment.
        
        Args:
            mode: Render mode ('human' or 'rgb_array')
        """
        if mode is None:
            mode = self.render_mode
        
        if mode == "human":
            return self._render_human()
        elif mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render environment for human viewing."""
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Draw entities
        scale = self.screen_size / 1200.0  # Scale factor for display
        
        # Draw missiles (red circles)
        for pos in self.missile_positions:
            screen_pos = ((pos + 600) * scale).astype(int)
            pygame.draw.circle(self.screen, (255, 0, 0), screen_pos, 5)
        
        # Draw interceptors (blue circles)
        for pos in self.interceptor_positions:
            screen_pos = ((pos + 600) * scale).astype(int)
            pygame.draw.circle(self.screen, (0, 0, 255), screen_pos, 5)
        
        # Draw targets (green squares)
        for pos in self.target_positions:
            screen_pos = ((pos + 600) * scale).astype(int)
            pygame.draw.rect(self.screen, (0, 255, 0), 
                           (screen_pos[0] - 3, screen_pos[1] - 3, 6, 6))
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def _render_rgb_array(self):
        """Render environment as RGB array."""
        # Create a simple RGB array representation
        img = np.zeros((self.screen_size, self.screen_size, 3), dtype=np.uint8)
        
        # This is a simplified implementation
        # In practice, you would create a more detailed visualization
        
        return img
    
    def close(self):
        """Clean up rendering resources."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None