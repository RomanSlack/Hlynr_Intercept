"""
6-DOF Missile Intercept Environment for AegisIntercept Phase 3.

This environment provides a complete 6-DOF simulation with quaternion-based
attitude dynamics, realistic atmospheric effects, and advanced adversary behaviors.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, List
from pyquaternion import Quaternion

from ..physics import RigidBody6DOF, AtmosphericModel, DragModel, WindField
from ..utils.maths import distance, normalize_vector, clamp, sanitize_array


class Aegis6DOFEnv(gym.Env):
    """
    6-DOF continuous environment for missile interception with full physics.
    
    This environment simulates realistic missile engagement scenarios with:
    - Full 6-DOF rigid body dynamics using quaternions
    - Atmospheric effects (drag, wind, density variation)
    - Configurable adversary behaviors
    - Dense reward shaping for effective learning
    - Scalable observation space (≤32 dimensions)
    
    The interceptor must engage and destroy the adversary before it reaches
    the defended target while managing fuel consumption and flight envelope.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    
    def __init__(self,
                 render_mode: Optional[str] = None,
                 max_episode_steps: int = 1000,
                 time_step: float = 0.02,
                 world_scale: float = 1000.0,
                 wind_config: Optional[Dict[str, Any]] = None,
                 curriculum_level: str = "easy"):
        """
        Initialize 6-DOF environment.
        
        Args:
            render_mode: Rendering mode ("human" or "rgb_array")
            max_episode_steps: Maximum steps per episode
            time_step: Physics time step in seconds
            world_scale: World scale in meters (engagement arena size)
            wind_config: Wind field configuration
            curriculum_level: Difficulty level ("easy", "medium", "hard", "impossible")
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.time_step = time_step
        self.world_scale = world_scale
        self.curriculum_level = curriculum_level
        
        # Physics models
        self.atmosphere = AtmosphericModel()
        self.wind_field = WindField(wind_config)
        
        # Interceptor configuration
        self.interceptor = RigidBody6DOF(
            mass=150.0,  # kg
            inertia=np.diag([10.0, 50.0, 50.0]),  # kg⋅m² (missile-like)
            gravity=9.80665
        )
        
        # Adversary configuration
        self.adversary = RigidBody6DOF(
            mass=200.0,  # kg
            inertia=np.diag([15.0, 60.0, 60.0]),  # kg⋅m²
            gravity=9.80665
        )
        
        # Drag models
        self.interceptor_drag = DragModel(
            reference_area=0.02,  # m²
            cd_subsonic=0.4,
            cd_supersonic=0.6
        )
        
        self.adversary_drag = DragModel(
            reference_area=0.03,  # m²
            cd_subsonic=0.5,
            cd_supersonic=0.7
        )
        
        # Target location (defended point)
        self.target_position = np.array([0.0, 0.0, 0.0])
        
        # Environment state
        self.current_step = 0
        self.episode_time = 0.0
        self.interceptor_fuel = 1.0  # Normalized fuel level
        self.fuel_consumption_rate = 0.001  # Per time step
        
        # Mission parameters based on curriculum level
        self.mission_params = self._get_mission_params(curriculum_level)
        
        # Action space: [thrust_x, thrust_y, thrust_z, torque_x, torque_y, torque_z]
        # All values normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )
        
        # Observation space: interceptor (13) + adversary (13) + environment (8) = 34
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(34,),
            dtype=np.float32
        )
        
        # Rendering
        self.viewer = None
        self.trajectory_history = []
        
        # Performance tracking
        self.episode_metrics = {
            'min_distance': np.inf,
            'fuel_used': 0.0,
            'time_to_intercept': 0.0,
            'success': False
        }
    
    def _get_mission_params(self, level: str) -> Dict[str, Any]:
        """Get mission parameters based on curriculum level."""
        params = {
            'easy': {
                'spawn_separation': 500.0,  # m
                'adversary_max_thrust': 2000.0,  # N
                'interceptor_max_thrust': 5000.0,  # N
                'interceptor_max_torque': 500.0,  # N⋅m
                'adversary_evasion_aggressiveness': 0.2,
                'wind_severity': 0.1,
                'kill_distance': 5.0  # m
            },
            'medium': {
                'spawn_separation': 800.0,
                'adversary_max_thrust': 3000.0,
                'interceptor_max_thrust': 4500.0,
                'interceptor_max_torque': 450.0,
                'adversary_evasion_aggressiveness': 0.5,
                'wind_severity': 0.2,
                'kill_distance': 3.0
            },
            'hard': {
                'spawn_separation': 1200.0,
                'adversary_max_thrust': 4000.0,
                'interceptor_max_thrust': 4000.0,
                'interceptor_max_torque': 400.0,
                'adversary_evasion_aggressiveness': 0.8,
                'wind_severity': 0.3,
                'kill_distance': 2.0
            },
            'impossible': {
                'spawn_separation': 1500.0,
                'adversary_max_thrust': 5000.0,
                'interceptor_max_thrust': 3500.0,
                'interceptor_max_torque': 350.0,
                'adversary_evasion_aggressiveness': 1.0,
                'wind_severity': 0.4,
                'kill_distance': 1.5
            }
        }
        return params.get(level, params['easy'])
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset time and step counters
        self.current_step = 0
        self.episode_time = 0.0
        self.interceptor_fuel = 1.0
        
        # Reset trajectory history
        self.trajectory_history = []
        
        # Reset episode metrics
        self.episode_metrics = {
            'min_distance': np.inf,
            'fuel_used': 0.0,
            'time_to_intercept': 0.0,
            'success': False
        }
        
        # Generate realistic engagement scenario positions
        separation = self.mission_params['spawn_separation']
        
        # Adversary spawns at much greater distance for realistic engagement
        adversary_angle = np.random.uniform(0, 2 * np.pi)
        adversary_elevation = np.random.uniform(-np.pi/8, np.pi/8)  # Reduced elevation range
        adversary_distance = separation + np.random.uniform(-200, 200)  # 1300-1700m for easy
        
        adversary_pos = np.array([
            adversary_distance * np.cos(adversary_angle) * np.cos(adversary_elevation),
            adversary_distance * np.sin(adversary_angle) * np.cos(adversary_elevation),
            adversary_distance * np.sin(adversary_elevation) + 1200.0  # Higher altitude
        ])
        
        # Adversary initial velocity toward target (balanced speed)
        adversary_vel_direction = normalize_vector(self.target_position - adversary_pos)
        adversary_speed = np.random.uniform(180, 250)  # m/s - balanced
        adversary_vel = adversary_speed * adversary_vel_direction
        
        # Interceptor spawns at favorable position for intercept
        # Place interceptor to create intercept geometry rather than head-on
        interceptor_angle = adversary_angle + np.random.uniform(-np.pi/3, np.pi/3)  # ±60° from adversary
        interceptor_distance = separation * 0.8 + np.random.uniform(-100, 100)  # 1200-1400m
        
        interceptor_pos = np.array([
            interceptor_distance * np.cos(interceptor_angle),
            interceptor_distance * np.sin(interceptor_angle),
            1000.0 + np.random.uniform(-200, 200)  # Variable altitude
        ])
        
        # Interceptor initial velocity for intercept course (higher speed)
        # Calculate intercept point for initial velocity direction
        time_to_intercept = np.random.uniform(15.0, 25.0)  # 15-25 second engagements
        predicted_adversary_pos = adversary_pos + adversary_vel * time_to_intercept
        intercept_direction = normalize_vector(predicted_adversary_pos - interceptor_pos)
        
        interceptor_speed = np.random.uniform(200, 300)  # m/s - higher speed
        interceptor_vel = interceptor_speed * intercept_direction
        
        # Reset rigid bodies
        self.interceptor.reset(
            position=interceptor_pos,
            velocity=interceptor_vel,
            orientation=Quaternion(axis=[0, 0, 1], angle=0),
            angular_velocity=np.zeros(3)
        )
        
        self.adversary.reset(
            position=adversary_pos,
            velocity=adversary_vel,
            orientation=Quaternion(axis=[0, 0, 1], angle=0),
            angular_velocity=np.zeros(3)
        )
        
        # Reset wind field
        self.wind_field.reset_turbulence(seed)
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one simulation step."""
        # Clamp action to valid range
        action = np.clip(action, -1.0, 1.0)
        
        # Extract forces and torques from action
        thrust_cmd = action[0:3]  # Normalized thrust commands
        torque_cmd = action[3:6]  # Normalized torque commands
        
        # Convert to actual forces and torques
        max_thrust = self.mission_params['interceptor_max_thrust']
        max_torque = self.mission_params['interceptor_max_torque']
        
        # Body-frame thrust force
        thrust_force = thrust_cmd * max_thrust
        # Convert to world frame
        thrust_world = self.interceptor.body_to_world(thrust_force)
        
        # Body-frame torque
        torque = torque_cmd * max_torque
        
        # Apply atmospheric drag and wind effects
        interceptor_pos = self.interceptor.get_position()
        interceptor_vel = self.interceptor.get_velocity()
        
        # Calculate drag force
        drag_force = self.interceptor_drag.get_drag_force(interceptor_vel, interceptor_pos[2])
        
        # Calculate wind force
        wind_force = self.wind_field.get_wind_force(
            interceptor_vel, interceptor_pos, self.episode_time, self.interceptor_drag
        )
        
        # Total force on interceptor
        total_force = thrust_world + drag_force + wind_force
        
        # Apply numerical stability limits to prevent explosion
        MAX_FORCE = 50000.0  # 50kN maximum total force
        MAX_TORQUE = 1000.0  # 1000 N⋅m maximum torque
        
        total_force = sanitize_array(total_force, max_val=MAX_FORCE)
        torque = sanitize_array(torque, max_val=MAX_TORQUE)
        
        # Apply forces to interceptor
        self.interceptor.apply_force_torque(total_force, torque, self.time_step)
        
        # Update adversary (autonomous behavior)
        self._update_adversary()
        
        # Update fuel consumption
        thrust_magnitude = np.linalg.norm(thrust_force)
        fuel_used = self.fuel_consumption_rate * (thrust_magnitude / max_thrust)
        self.interceptor_fuel = max(0.0, self.interceptor_fuel - fuel_used)
        self.episode_metrics['fuel_used'] += fuel_used
        
        # Update time
        self.episode_time += self.time_step
        self.current_step += 1
        
        # Calculate distances
        interceptor_pos = self.interceptor.get_position()
        adversary_pos = self.adversary.get_position()
        
        intercept_distance = distance(interceptor_pos, adversary_pos)
        target_distance = distance(adversary_pos, self.target_position)
        
        # Update metrics
        self.episode_metrics['min_distance'] = min(
            self.episode_metrics['min_distance'], intercept_distance
        )
        
        # Store trajectory point
        self.trajectory_history.append({
            'time': self.episode_time,
            'interceptor_pos': interceptor_pos.copy(),
            'adversary_pos': adversary_pos.copy(),
            'interceptor_vel': self.interceptor.get_velocity().copy(),
            'adversary_vel': self.adversary.get_velocity().copy(),
            'distance': intercept_distance,
            'fuel': self.interceptor_fuel
        })
        
        # Calculate reward
        reward = self._calculate_reward(intercept_distance, target_distance, action)
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Success: interceptor close enough to adversary
        if intercept_distance <= self.mission_params['kill_distance']:
            terminated = True
            reward += 100.0
            self.episode_metrics['success'] = True
            self.episode_metrics['time_to_intercept'] = self.episode_time
        
        # Failure: adversary reaches target
        elif target_distance <= 10.0:
            terminated = True
            reward -= 100.0
        
        # Failure: interceptor out of fuel
        elif self.interceptor_fuel <= 0.0:
            terminated = True
            reward -= 50.0
        
        # Failure: interceptor out of bounds or crashed
        elif (np.linalg.norm(interceptor_pos) > self.world_scale or 
              interceptor_pos[2] < 0):
            terminated = True
            reward -= 50.0
        
        # Truncation: maximum episode length
        elif self.current_step >= self.max_episode_steps:
            truncated = True
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _update_adversary(self):
        """Update adversary missile with autonomous behavior."""
        # Get adversary state
        adversary_pos = self.adversary.get_position()
        adversary_vel = self.adversary.get_velocity()
        interceptor_pos = self.interceptor.get_position()
        
        # Proportional navigation toward target
        target_direction = normalize_vector(self.target_position - adversary_pos)
        desired_vel = target_direction * 200.0  # Desired speed
        
        # Add evasion behavior when interceptor is close
        intercept_distance = distance(adversary_pos, interceptor_pos)
        evasion_threshold = 300.0  # Start evasion when interceptor within 300m
        
        if intercept_distance < evasion_threshold:
            # Calculate evasion force
            evasion_direction = normalize_vector(adversary_pos - interceptor_pos)
            evasion_strength = self.mission_params['adversary_evasion_aggressiveness']
            evasion_strength *= (1.0 - intercept_distance / evasion_threshold)
            
            # Add evasion component
            desired_vel += evasion_direction * evasion_strength * 100.0
        
        # Calculate required force
        vel_error = desired_vel - adversary_vel
        force_cmd = vel_error * self.adversary.mass * 0.5  # Proportional control
        
        # Limit force magnitude
        max_force = self.mission_params['adversary_max_thrust']
        force_magnitude = np.linalg.norm(force_cmd)
        if force_magnitude > max_force:
            force_cmd = force_cmd * (max_force / force_magnitude)
        
        # Apply atmospheric effects
        drag_force = self.adversary_drag.get_drag_force(adversary_vel, adversary_pos[2])
        wind_force = self.wind_field.get_wind_force(
            adversary_vel, adversary_pos, self.episode_time, self.adversary_drag
        )
        
        total_force = force_cmd + drag_force + wind_force
        
        # Apply forces (no torque for simple adversary)
        self.adversary.apply_force_torque(total_force, np.zeros(3), self.time_step)
    
    def _calculate_reward(self, intercept_distance: float, target_distance: float, action: np.ndarray) -> float:
        """Calculate dense reward with proper guidance shaping."""
        reward = 0.0
        
        # Get current state for guidance calculations
        interceptor_pos = self.interceptor.get_position()
        interceptor_vel = self.interceptor.get_velocity()
        adversary_pos = self.adversary.get_position()
        adversary_vel = self.adversary.get_velocity()
        
        # 1. Primary distance reward (linear + scaled components)
        # Linear component for large distances
        if intercept_distance > 1000.0:
            distance_reward = 2.0 * (2000.0 - intercept_distance) / 1000.0  # Linear from 2km to 1km
        else:
            # Exponential for close range with better scaling
            distance_reward = 2.0 + 3.0 * np.exp(-intercept_distance / 200.0)  # Exponential below 1km
        
        reward += distance_reward
        
        # 2. Approach angle reward (Proportional Navigation principle)
        relative_pos = adversary_pos - interceptor_pos
        relative_vel = adversary_vel - interceptor_vel
        closing_rate = -np.dot(relative_pos, relative_vel) / max(np.linalg.norm(relative_pos), 1.0)
        
        if closing_rate > 0:  # Approaching
            approach_reward = 1.0 * min(closing_rate / 100.0, 2.0)  # Reward up to 200 m/s closing
            reward += approach_reward
        
        # 3. Velocity alignment reward
        if np.linalg.norm(relative_pos) > 1.0:
            desired_direction = normalize_vector(relative_pos)
            current_direction = normalize_vector(interceptor_vel)
            alignment = np.dot(current_direction, desired_direction)
            alignment_reward = 0.5 * max(alignment, 0.0)  # Reward positive alignment
            reward += alignment_reward
        
        # 4. Reduced fuel penalty (was too aggressive)
        fuel_usage = np.linalg.norm(action[0:3]) / 3.0  # Normalized thrust usage
        fuel_penalty = -0.005 * fuel_usage  # Reduced by 10x
        reward += fuel_penalty
        
        # 5. Reduced control effort penalty
        control_penalty = -0.001 * np.sum(np.square(action))  # Reduced by 10x
        reward += control_penalty
        
        # 6. Target protection bonus (scaled by distance)
        target_protection_bonus = 0.1 * min(target_distance / 500.0, 2.0)
        reward += target_protection_bonus
        
        # 7. Time penalty (encourage quick intercept but not overwhelming)
        time_penalty = -0.0005  # Reduced by 2x
        reward += time_penalty
        
        # 8. Progress reward (reward for reducing distance over time)
        if hasattr(self, '_last_intercept_distance'):
            distance_change = self._last_intercept_distance - intercept_distance
            progress_reward = 5.0 * distance_change / self.time_step  # Reward rate of closure
            reward += progress_reward
        
        # Store for next step
        self._last_intercept_distance = intercept_distance
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector with numerical stability."""
        # Interceptor state (13 dimensions)
        interceptor_state = self.interceptor.get_full_state()
        
        # Adversary state (13 dimensions)
        adversary_state = self.adversary.get_full_state()
        
        # Environment features (8 dimensions - expanded for better guidance)
        interceptor_pos = self.interceptor.get_position()
        adversary_pos = self.adversary.get_position()
        interceptor_vel = self.interceptor.get_velocity()
        adversary_vel = self.adversary.get_velocity()
        
        # Relative position and velocity
        relative_pos = adversary_pos - interceptor_pos
        relative_vel = adversary_vel - interceptor_vel
        
        # Calculate additional guidance features
        intercept_distance = np.linalg.norm(relative_pos)
        closing_rate = -np.dot(relative_pos, relative_vel) / max(intercept_distance, 1.0)
        
        # Additional features
        time_normalized = clamp(self.episode_time / (self.max_episode_steps * self.time_step), 0.0, 1.0)
        fuel_level = clamp(self.interceptor_fuel, 0.0, 1.0)
        
        # Improved normalization with better scales
        env_features = np.array([
            clamp(relative_pos[0] / 5000.0, -2.0, 2.0),  # Position: 5km scale for better resolution
            clamp(relative_pos[1] / 5000.0, -2.0, 2.0),
            clamp(relative_pos[2] / 5000.0, -2.0, 2.0),
            clamp(intercept_distance / 2500.0, 0.0, 2.0),  # Raw distance (0-5km)
            clamp(closing_rate / 200.0, -2.0, 2.0),  # Closing rate (±400 m/s)
            clamp(np.linalg.norm(relative_vel) / 1000.0, 0.0, 2.0),  # Relative speed
            time_normalized,
            fuel_level
        ])
        
        # Process interceptor state with improved normalization
        interceptor_obs = interceptor_state.copy()
        interceptor_obs[0:3] = np.clip(interceptor_obs[0:3] / 5000.0, -2.0, 2.0)  # Position: 5km scale
        interceptor_obs[3:6] = np.clip(interceptor_obs[3:6] / 1000.0, -1.5, 1.5)  # Velocity: 1000 m/s scale
        # Quaternion [6:10] already normalized [-1,1]
        interceptor_obs[10:13] = np.clip(interceptor_obs[10:13] / 20.0, -1.5, 1.5)  # Angular velocity: 20 rad/s scale
        
        # Process adversary state with improved normalization
        adversary_obs = adversary_state.copy()
        adversary_obs[0:3] = np.clip(adversary_obs[0:3] / 5000.0, -2.0, 2.0)      # Position: 5km scale
        adversary_obs[3:6] = np.clip(adversary_obs[3:6] / 1000.0, -1.5, 1.5)      # Velocity: 1000 m/s scale
        # Quaternion [6:10] already normalized [-1,1]
        adversary_obs[10:13] = np.clip(adversary_obs[10:13] / 20.0, -1.5, 1.5)    # Angular velocity: 20 rad/s scale
        
        # Concatenate all features
        observation = np.concatenate([interceptor_obs, adversary_obs, env_features])
        
        # Apply final sanitization to remove any NaN/Inf values
        observation = sanitize_array(observation, max_val=10.0)
        
        return observation.astype(np.float32)
    
    def _get_info(self) -> Dict:
        """Get additional information about the current state."""
        interceptor_pos = self.interceptor.get_position()
        adversary_pos = self.adversary.get_position()
        
        return {
            'intercept_distance': distance(interceptor_pos, adversary_pos),
            'target_distance': distance(adversary_pos, self.target_position),
            'fuel_remaining': self.interceptor_fuel,
            'episode_time': self.episode_time,
            'current_step': self.current_step,
            'episode_metrics': self.episode_metrics.copy()
        }
    
    def render(self):
        """Render the environment (placeholder for visualization)."""
        if self.render_mode == "human":
            # This will be implemented in the visualization module
            pass
        elif self.render_mode == "rgb_array":
            # Return RGB array for video recording
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def get_trajectory_data(self) -> List[Dict]:
        """Get trajectory history for analysis."""
        return self.trajectory_history.copy()
    
    def set_curriculum_level(self, level: str):
        """Update curriculum difficulty level."""
        self.curriculum_level = level
        self.mission_params = self._get_mission_params(level)