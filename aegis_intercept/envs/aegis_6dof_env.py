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
                 time_step: float = 0.01,
                 world_scale: float = 5000.0,
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
                'spawn_separation': 1500.0,  # m - increased for realistic engagement
                'adversary_max_thrust': 3000.0,  # N
                'interceptor_max_thrust': 10000.0,  # N - doubled for better authority
                'interceptor_max_torque': 1000.0,  # N⋅m - doubled
                'adversary_evasion_aggressiveness': 0.2,
                'wind_severity': 0.1,
                'kill_distance': 5.0  # m
            },
            'medium': {
                'spawn_separation': 2000.0,  # m - increased
                'adversary_max_thrust': 4000.0,
                'interceptor_max_thrust': 9000.0,  # N - doubled
                'interceptor_max_torque': 900.0,  # N⋅m - doubled
                'adversary_evasion_aggressiveness': 0.5,
                'wind_severity': 0.2,
                'kill_distance': 3.0
            },
            'hard': {
                'spawn_separation': 2500.0,  # m - increased
                'adversary_max_thrust': 5000.0,
                'interceptor_max_thrust': 8000.0,  # N - doubled
                'interceptor_max_torque': 800.0,  # N⋅m - doubled
                'adversary_evasion_aggressiveness': 0.8,
                'wind_severity': 0.3,
                'kill_distance': 2.0
            },
            'impossible': {
                'spawn_separation': 3000.0,  # m - increased
                'adversary_max_thrust': 6000.0,
                'interceptor_max_thrust': 7000.0,  # N - doubled
                'interceptor_max_torque': 700.0,  # N⋅m - doubled
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
        
        # Initialize progress tracking for reward function
        self._last_intercept_distance = None
        
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
        termination_reason = "ongoing"
        
        # Success: interceptor close enough to adversary
        if intercept_distance <= self.mission_params['kill_distance']:
            terminated = True
            reward += 100.0
            self.episode_metrics['success'] = True
            self.episode_metrics['time_to_intercept'] = self.episode_time
            termination_reason = "intercept_success"
        
        # Failure: adversary reaches target
        elif target_distance <= 10.0:
            terminated = True
            reward -= 100.0
            termination_reason = "adversary_reached_target"
        
        # Failure: interceptor out of fuel
        elif self.interceptor_fuel <= 0.0:
            terminated = True
            reward -= 50.0
            termination_reason = "out_of_fuel"
        
        # Failure: interceptor out of bounds or crashed
        elif (np.linalg.norm(interceptor_pos) > self.world_scale or 
              interceptor_pos[2] < 0):
            terminated = True
            # Severe penalty for ground collision to prevent deliberate crashes
            if interceptor_pos[2] < 0:
                reward -= 200.0  # Much higher penalty for crashing
            else:
                reward -= 50.0   # Regular penalty for out of bounds
            
            # More specific termination reasons
            if np.linalg.norm(interceptor_pos) > self.world_scale:
                termination_reason = f"out_of_bounds_distance_{np.linalg.norm(interceptor_pos):.1f}m_limit_{self.world_scale}m"
            else:
                termination_reason = f"ground_collision_altitude_{interceptor_pos[2]:.1f}m"
        
        # Truncation: maximum episode length
        elif self.current_step >= self.max_episode_steps:
            truncated = True
            termination_reason = "max_steps_reached"
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add termination reason to info
        info['termination_reason'] = termination_reason
        
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
        """Calculate balanced reward preventing exploitation while maintaining guidance."""
        reward = 0.0
        
        # Get current state for guidance calculations
        interceptor_pos = self.interceptor.get_position()
        interceptor_vel = self.interceptor.get_velocity()
        adversary_pos = self.adversary.get_position()
        adversary_vel = self.adversary.get_velocity()
        
        # 1. Primary distance reward (capped and balanced)
        if intercept_distance > 1000.0:
            distance_reward = 1.0 * (1500.0 - intercept_distance) / 500.0  # Max 1.0 reward
        else:
            # Exponential for close range but capped
            distance_reward = 1.0 + 2.0 * np.exp(-intercept_distance / 150.0)  # Max ~3.0
        
        distance_reward = max(0.0, min(distance_reward, 3.0))  # Hard cap at 3.0
        reward += distance_reward
        
        # 2. Approach reward (capped to prevent exploitation)
        relative_pos = adversary_pos - interceptor_pos
        relative_vel = adversary_vel - interceptor_vel
        closing_rate = -np.dot(relative_pos, relative_vel) / max(np.linalg.norm(relative_pos), 1.0)
        
        if closing_rate > 0:  # Approaching
            # Cap approach reward and scale by distance (closer = more important)
            approach_reward = 0.5 * min(closing_rate / 150.0, 1.0)  # Max 0.5 reward
            if intercept_distance < 500.0:  # Bonus for close approach
                approach_reward *= 1.5
            reward += approach_reward
        
        # 3. Intercept course reward (proportional navigation)
        if intercept_distance > 50.0:  # Only when not too close
            desired_direction = normalize_vector(relative_pos)
            current_direction = normalize_vector(interceptor_vel)
            alignment = np.dot(current_direction, desired_direction)
            alignment_reward = 0.3 * max(alignment, 0.0)  # Max 0.3 reward
            reward += alignment_reward
        
        # 4. Fuel efficiency penalty (stronger when inefficient)
        fuel_usage = np.linalg.norm(action[0:3]) / 3.0  # Normalized thrust usage
        
        # Progressive fuel penalty - higher penalty when fuel is low
        fuel_efficiency_penalty = -0.01 * fuel_usage * (2.0 - self.interceptor_fuel)
        reward += fuel_efficiency_penalty
        
        # 5. Control smoothness penalty
        control_penalty = -0.002 * np.sum(np.square(action))
        reward += control_penalty
        
        # 6. Target protection bonus (reasonable scale)
        if target_distance > 200.0:
            protection_bonus = 0.05 * min(target_distance / 1000.0, 1.0)
            reward += protection_bonus
        
        # 7. Time incentive (encourage efficiency)
        time_penalty = -0.001
        reward += time_penalty
        
        # 8. Progress reward (CAPPED to prevent exploitation)
        if hasattr(self, '_last_intercept_distance') and self._last_intercept_distance is not None:
            distance_change = self._last_intercept_distance - intercept_distance
            # Cap progress reward and require minimum distance change
            if abs(distance_change) > 0.1:  # Ignore tiny changes
                progress_reward = np.clip(distance_change / self.time_step * 0.002, -0.5, 0.5)  # Max ±0.5
                reward += progress_reward
        
        # 9. Close approach bonus (reward getting very close)
        if intercept_distance < 100.0:
            close_bonus = 2.0 * np.exp(-intercept_distance / 30.0)  # Exponential bonus for <100m
            reward += close_bonus
        
        # 10. Fuel conservation bonus (reward having fuel remaining)
        if self.interceptor_fuel > 0.3:  # Bonus for having >30% fuel
            fuel_bonus = 0.1 * (self.interceptor_fuel - 0.3)
            reward += fuel_bonus
        
        # 11. Altitude discipline rewards and penalties
        interceptor_altitude = interceptor_pos[2]
        
        if interceptor_altitude < 10.0:  # Very low altitude - danger zone
            altitude_penalty = -1.0 * (10.0 - interceptor_altitude) / 10.0  # Up to -1.0 penalty
            reward += altitude_penalty
        elif interceptor_altitude > 50.0:  # Safe altitude
            altitude_bonus = 0.05  # Small bonus for staying high
            reward += altitude_bonus
        
        # 12. Severe penalty for getting too close to ground (prevention)
        if 0 < interceptor_altitude < 5.0:  # About to crash
            near_crash_penalty = -2.0 * (5.0 - interceptor_altitude) / 5.0  # Up to -2.0 penalty
            reward += near_crash_penalty
        
        # Store for next step
        self._last_intercept_distance = intercept_distance
        
        # Total reward should typically be in range [-2, 10] to prevent exploitation
        return np.clip(reward, -8.0, 10.0)  # Expanded lower bound for altitude penalties
    
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