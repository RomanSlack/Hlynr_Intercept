"""
Multi-Interceptor Environment for AegisIntercept Phase 3.

This module extends the 6-DOF environment to support multiple interceptors
operating under a shared policy with coordinated observations.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
from pyquaternion import Quaternion

from .aegis_6dof_env import Aegis6DOFEnv
from ..physics import RigidBody6DOF, DragModel, WindField
from ..utils.maths import distance, normalize_vector


class MultiInterceptorEnv(Aegis6DOFEnv):
    """
    Multi-interceptor 6-DOF environment.
    
    This environment extends the single-interceptor case to support multiple
    interceptors operating cooperatively against a single adversary. All
    interceptors share the same policy but receive individual observations
    stacked together.
    """
    
    def __init__(self,
                 n_interceptors: int = 2,
                 coordination_radius: float = 200.0,
                 collision_avoidance: bool = True,
                 **kwargs):
        """
        Initialize multi-interceptor environment.
        
        Args:
            n_interceptors: Number of interceptors
            coordination_radius: Radius for coordination bonus (meters)
            collision_avoidance: Enable collision avoidance between interceptors
            **kwargs: Arguments passed to parent Aegis6DOFEnv
        """
        self.n_interceptors = max(1, min(n_interceptors, 4))  # Limit to reasonable range
        self.coordination_radius = coordination_radius
        self.collision_avoidance = collision_avoidance
        
        # Call parent initialization
        super().__init__(**kwargs)
        
        # Override action and observation spaces for multi-interceptor
        self._setup_multi_interceptor_spaces()
        
        # Initialize multiple interceptors
        self.interceptors = []
        self.interceptor_drags = []
        self.interceptor_fuels = []
        
        for i in range(self.n_interceptors):
            # Create interceptor physics
            interceptor = RigidBody6DOF(
                mass=150.0 + np.random.uniform(-10, 10),  # Slight mass variation
                inertia=np.diag([10.0, 50.0, 50.0]) * (1.0 + np.random.uniform(-0.1, 0.1)),
                gravity=9.80665
            )
            self.interceptors.append(interceptor)
            
            # Create drag model for each interceptor
            drag_model = DragModel(
                reference_area=0.02 * (1.0 + np.random.uniform(-0.1, 0.1)),
                cd_subsonic=0.4,
                cd_supersonic=0.6
            )
            self.interceptor_drags.append(drag_model)
            
            # Initialize fuel
            self.interceptor_fuels.append(1.0)
        
        # Multi-interceptor specific metrics
        self.coordination_bonus = 0.0
        self.collision_penalties = 0.0
        self.min_interceptor_separation = np.inf
        
        print(f"Multi-interceptor environment initialized with {self.n_interceptors} interceptors")
    
    def _setup_multi_interceptor_spaces(self):
        """Setup action and observation spaces for multiple interceptors."""
        # Action space: [interceptor_1_actions, interceptor_2_actions, ...]
        # Each interceptor has 6 actions: [thrust_x, thrust_y, thrust_z, torque_x, torque_y, torque_z]
        single_action_dim = 6
        total_action_dim = single_action_dim * self.n_interceptors
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(total_action_dim,),
            dtype=np.float32
        )
        
        # Observation space: stacked observations for all interceptors + adversary + environment
        # Each interceptor: 13 (state) + 6 (relative info) = 19 dimensions
        # Adversary: 13 dimensions
        # Environment: 6 dimensions
        # Coordination info: n_interceptors * 3 (relative positions between interceptors)
        interceptor_obs_dim = 19 * self.n_interceptors
        adversary_obs_dim = 13
        environment_obs_dim = 6
        coordination_obs_dim = self.n_interceptors * 3
        
        total_obs_dim = interceptor_obs_dim + adversary_obs_dim + environment_obs_dim + coordination_obs_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment with multiple interceptors."""
        # Reset base environment (this handles adversary and target)
        _, info = super().reset(seed, options)
        
        # Reset interceptor fuels
        self.interceptor_fuels = [1.0] * self.n_interceptors
        
        # Reset multi-interceptor metrics
        self.coordination_bonus = 0.0
        self.collision_penalties = 0.0
        self.min_interceptor_separation = np.inf
        
        # Generate spawn positions for multiple interceptors
        self._spawn_multiple_interceptors()
        
        # Get multi-interceptor observation
        observation = self._get_multi_interceptor_observation()
        
        return observation, info
    
    def _spawn_multiple_interceptors(self):
        """Spawn multiple interceptors in formation."""
        # Get adversary position for reference
        adversary_pos = self.adversary.get_position()
        
        # Base interceptor position (from parent class logic)
        base_separation = self.mission_params['spawn_separation']
        base_angle = np.random.uniform(0, 2 * np.pi)
        base_distance = base_separation * 0.7 + np.random.uniform(-50, 50)
        
        base_position = np.array([
            base_distance * np.cos(base_angle),
            base_distance * np.sin(base_angle),
            800.0 + np.random.uniform(-100, 100)
        ])
        
        # Formation parameters
        formation_spacing = 100.0  # Distance between interceptors in formation
        formation_patterns = ['line', 'wedge', 'diamond', 'box']
        formation = np.random.choice(formation_patterns)
        
        # Generate positions based on formation
        positions = self._generate_formation_positions(base_position, formation, formation_spacing)
        
        # Initialize each interceptor
        for i, interceptor in enumerate(self.interceptors):
            position = positions[i]
            
            # Initial velocity toward adversary
            intercept_direction = normalize_vector(adversary_pos - position)
            interceptor_speed = np.random.uniform(100, 200)
            velocity = interceptor_speed * intercept_direction
            
            # Add formation-specific velocity adjustments
            velocity += self._get_formation_velocity_adjustment(i, formation)
            
            # Reset interceptor physics
            interceptor.reset(
                position=position,
                velocity=velocity,
                orientation=Quaternion(axis=[0, 0, 1], angle=0),
                angular_velocity=np.zeros(3)
            )
    
    def _generate_formation_positions(self, base_pos: np.ndarray, formation: str, spacing: float) -> List[np.ndarray]:
        """Generate formation positions for interceptors."""
        positions = []
        
        if formation == 'line':
            # Line formation along Y-axis
            for i in range(self.n_interceptors):
                offset = (i - (self.n_interceptors - 1) / 2) * spacing
                pos = base_pos + np.array([0, offset, 0])
                positions.append(pos)
        
        elif formation == 'wedge':
            # V-shaped wedge formation
            positions.append(base_pos)  # Leader at front
            for i in range(1, self.n_interceptors):
                side = (-1) ** i  # Alternate sides
                y_offset = side * (i // 2 + 1) * spacing * 0.7
                x_offset = -abs(y_offset) * 0.5  # Behind leader
                pos = base_pos + np.array([x_offset, y_offset, 0])
                positions.append(pos)
        
        elif formation == 'diamond':
            # Diamond formation
            if self.n_interceptors >= 4:
                positions.append(base_pos)  # Center
                positions.append(base_pos + np.array([spacing, 0, 0]))     # Right
                positions.append(base_pos + np.array([-spacing, 0, 0]))    # Left
                positions.append(base_pos + np.array([0, spacing, 0]))     # Front
                # Add remaining behind center
                for i in range(4, self.n_interceptors):
                    pos = base_pos + np.array([0, -spacing * (i - 3), 0])
                    positions.append(pos)
            else:
                # Fall back to line for fewer interceptors
                return self._generate_formation_positions(base_pos, 'line', spacing)
        
        elif formation == 'box':
            # Box/square formation
            box_size = spacing
            box_positions = [
                np.array([box_size/2, box_size/2, 0]),
                np.array([-box_size/2, box_size/2, 0]),
                np.array([-box_size/2, -box_size/2, 0]),
                np.array([box_size/2, -box_size/2, 0])
            ]
            
            for i in range(self.n_interceptors):
                if i < len(box_positions):
                    pos = base_pos + box_positions[i]
                else:
                    # Stack additional interceptors vertically
                    pos = base_pos + box_positions[i % len(box_positions)] + np.array([0, 0, 50 * (i // len(box_positions))])
                positions.append(pos)
        
        return positions[:self.n_interceptors]
    
    def _get_formation_velocity_adjustment(self, interceptor_idx: int, formation: str) -> np.ndarray:
        """Get velocity adjustment to maintain formation."""
        # Small random adjustments to break perfect symmetry
        base_adjustment = np.random.normal(0, 5, 3)
        
        if formation == 'wedge' and interceptor_idx == 0:
            # Leader gets slight speed boost
            base_adjustment[0] += 10.0
        
        return base_adjustment
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one simulation step with multiple interceptors."""
        # Parse actions for each interceptor
        actions_per_interceptor = 6
        interceptor_actions = []
        
        for i in range(self.n_interceptors):
            start_idx = i * actions_per_interceptor
            end_idx = start_idx + actions_per_interceptor
            interceptor_actions.append(action[start_idx:end_idx])
        
        # Update each interceptor
        self._update_multiple_interceptors(interceptor_actions)
        
        # Update adversary (same as single interceptor case)
        self._update_adversary()
        
        # Update time and counters
        self.episode_time += self.time_step
        self.current_step += 1
        
        # Calculate multi-interceptor metrics
        self._calculate_multi_interceptor_metrics()
        
        # Calculate reward
        reward = self._calculate_multi_interceptor_reward(interceptor_actions)
        
        # Check termination conditions
        terminated, truncated = self._check_multi_interceptor_termination()
        
        # Get observation and info
        observation = self._get_multi_interceptor_observation()
        info = self._get_multi_interceptor_info()
        
        return observation, reward, terminated, truncated, info
    
    def _update_multiple_interceptors(self, actions: List[np.ndarray]):
        """Update physics for all interceptors."""
        for i, (interceptor, action) in enumerate(zip(self.interceptors, actions)):
            # Clamp action
            action = np.clip(action, -1.0, 1.0)
            
            # Extract forces and torques
            thrust_cmd = action[0:3]
            torque_cmd = action[3:6]
            
            # Convert to actual forces
            max_thrust = self.mission_params['interceptor_max_thrust']
            max_torque = self.mission_params['interceptor_max_torque']
            
            thrust_force = thrust_cmd * max_thrust
            thrust_world = interceptor.body_to_world(thrust_force)
            torque = torque_cmd * max_torque
            
            # Apply atmospheric effects
            interceptor_pos = interceptor.get_position()
            interceptor_vel = interceptor.get_velocity()
            
            drag_force = self.interceptor_drags[i].get_drag_force(interceptor_vel, interceptor_pos[2])
            wind_force = self.wind_field.get_wind_force(
                interceptor_vel, interceptor_pos, self.episode_time, self.interceptor_drags[i]
            )
            
            # Collision avoidance force
            avoidance_force = np.zeros(3)
            if self.collision_avoidance:
                avoidance_force = self._calculate_collision_avoidance_force(i)
            
            # Total force
            total_force = thrust_world + drag_force + wind_force + avoidance_force
            
            # Apply forces
            interceptor.apply_force_torque(total_force, torque, self.time_step)
            
            # Update fuel
            thrust_magnitude = np.linalg.norm(thrust_force)
            fuel_used = self.fuel_consumption_rate * (thrust_magnitude / max_thrust)
            self.interceptor_fuels[i] = max(0.0, self.interceptor_fuels[i] - fuel_used)
    
    def _calculate_collision_avoidance_force(self, interceptor_idx: int) -> np.ndarray:
        """Calculate collision avoidance force for an interceptor."""
        current_interceptor = self.interceptors[interceptor_idx]
        current_pos = current_interceptor.get_position()
        
        avoidance_force = np.zeros(3)
        collision_threshold = 50.0  # Minimum separation distance
        max_avoidance_force = 1000.0  # Maximum avoidance force
        
        for i, other_interceptor in enumerate(self.interceptors):
            if i == interceptor_idx:
                continue
            
            other_pos = other_interceptor.get_position()
            separation_vector = current_pos - other_pos
            separation_distance = np.linalg.norm(separation_vector)
            
            if separation_distance < collision_threshold:
                # Calculate repulsive force
                if separation_distance > 1e-6:
                    force_direction = separation_vector / separation_distance
                    force_magnitude = max_avoidance_force * (1.0 - separation_distance / collision_threshold) ** 2
                    avoidance_force += force_direction * force_magnitude
        
        return avoidance_force
    
    def _calculate_multi_interceptor_metrics(self):
        """Calculate metrics specific to multi-interceptor scenarios."""
        # Calculate minimum separation between interceptors
        min_separation = np.inf
        for i in range(self.n_interceptors):
            for j in range(i + 1, self.n_interceptors):
                pos_i = self.interceptors[i].get_position()
                pos_j = self.interceptors[j].get_position()
                separation = distance(pos_i, pos_j)
                min_separation = min(min_separation, separation)
        
        self.min_interceptor_separation = min_separation
        
        # Calculate coordination bonus (interceptors working together)
        adversary_pos = self.adversary.get_position()
        interceptor_positions = [interceptor.get_position() for interceptor in self.interceptors]
        
        # Check if interceptors are approaching from different angles
        angles_to_adversary = []
        for pos in interceptor_positions:
            vector_to_adversary = adversary_pos - pos
            if np.linalg.norm(vector_to_adversary) > 1e-6:
                angle = np.arctan2(vector_to_adversary[1], vector_to_adversary[0])
                angles_to_adversary.append(angle)
        
        if len(angles_to_adversary) >= 2:
            # Calculate angle spread
            angles = np.array(angles_to_adversary)
            angle_diffs = []
            for i in range(len(angles)):
                for j in range(i + 1, len(angles)):
                    diff = abs(angles[i] - angles[j])
                    diff = min(diff, 2 * np.pi - diff)  # Wrap around
                    angle_diffs.append(diff)
            
            if angle_diffs:
                mean_angle_spread = np.mean(angle_diffs)
                # Bonus for good angle spread (multi-axis attack)
                self.coordination_bonus = min(1.0, mean_angle_spread / np.pi)
        
        # Calculate collision penalties
        if self.min_interceptor_separation < 30.0:
            self.collision_penalties += 0.1
    
    def _calculate_multi_interceptor_reward(self, actions: List[np.ndarray]) -> float:
        """Calculate reward for multi-interceptor scenario."""
        # Get distances from all interceptors to adversary
        adversary_pos = self.adversary.get_position()
        interceptor_distances = []
        
        for interceptor in self.interceptors:
            interceptor_pos = interceptor.get_position()
            dist = distance(interceptor_pos, adversary_pos)
            interceptor_distances.append(dist)
        
        # Use minimum distance for main reward shaping
        min_distance = min(interceptor_distances)
        
        # Base reward (similar to single interceptor)
        reward = 0.0
        
        # Dense distance reward
        k = 0.01
        distance_reward = np.exp(-k * min_distance)
        reward += distance_reward
        
        # Fuel penalties for all interceptors
        fuel_penalty = 0.0
        for i, action in enumerate(actions):
            fuel_usage = np.linalg.norm(action[0:3]) / 3.0
            fuel_penalty += -0.05 * fuel_usage / self.n_interceptors  # Normalize by number of interceptors
        reward += fuel_penalty
        
        # Control effort penalties
        control_penalty = 0.0
        for action in actions:
            control_penalty += -0.01 * np.sum(np.square(action)) / self.n_interceptors
        reward += control_penalty
        
        # Multi-interceptor specific rewards
        
        # Coordination bonus
        reward += 0.1 * self.coordination_bonus
        
        # Collision penalty
        if self.min_interceptor_separation < 30.0:
            reward -= 0.5
        
        # Formation maintenance bonus
        formation_bonus = self._calculate_formation_bonus()
        reward += formation_bonus * 0.05
        
        # Multiple threat bonus (multiple interceptors close to adversary)
        close_interceptors = sum(1 for dist in interceptor_distances if dist < 100.0)
        if close_interceptors > 1:
            reward += 0.2 * (close_interceptors - 1)
        
        return reward
    
    def _calculate_formation_bonus(self) -> float:
        """Calculate bonus for maintaining good formation."""
        if self.n_interceptors < 2:
            return 0.0
        
        # Calculate formation coherence
        positions = [interceptor.get_position() for interceptor in self.interceptors]
        center = np.mean(positions, axis=0)
        
        distances_from_center = [distance(pos, center) for pos in positions]
        formation_variance = np.var(distances_from_center)
        
        # Lower variance means better formation
        max_good_variance = 10000.0  # 100m standard deviation
        formation_bonus = max(0.0, 1.0 - formation_variance / max_good_variance)
        
        return formation_bonus
    
    def _check_multi_interceptor_termination(self) -> Tuple[bool, bool]:
        """Check termination conditions for multi-interceptor scenario."""
        # Check if any interceptor successfully intercepted
        adversary_pos = self.adversary.get_position()
        kill_distance = self.mission_params['kill_distance']
        
        for interceptor in self.interceptors:
            interceptor_pos = interceptor.get_position()
            if distance(interceptor_pos, adversary_pos) <= kill_distance:
                return True, False  # Success!
        
        # Check standard termination conditions
        target_distance = distance(adversary_pos, self.target_position)
        
        # Failure conditions
        if target_distance <= 10.0:
            return True, False  # Adversary reached target
        
        # Check if all interceptors are out of fuel
        if all(fuel <= 0.0 for fuel in self.interceptor_fuels):
            return True, False  # All out of fuel
        
        # Check if any interceptor is out of bounds
        world_bound = self.world_scale
        for interceptor in self.interceptors:
            pos = interceptor.get_position()
            if np.linalg.norm(pos) > world_bound or pos[2] < 0:
                return True, False  # Out of bounds
        
        # Check collision between interceptors
        if self.min_interceptor_separation < 10.0:
            return True, False  # Collision
        
        # Truncation condition
        if self.current_step >= self.max_episode_steps:
            return False, True
        
        return False, False
    
    def _get_multi_interceptor_observation(self) -> np.ndarray:
        """Get observation for multi-interceptor scenario."""
        observations = []
        
        # Observations for each interceptor
        for i, interceptor in enumerate(self.interceptors):
            # Individual interceptor state
            interceptor_state = interceptor.get_full_state()
            
            # Relative information to adversary
            interceptor_pos = interceptor.get_position()
            adversary_pos = self.adversary.get_position()
            relative_pos = adversary_pos - interceptor_pos
            relative_vel = self.adversary.get_velocity() - interceptor.get_velocity()
            
            # Relative info to other interceptors
            closest_teammate_dist = np.inf
            if self.n_interceptors > 1:
                for j, other_interceptor in enumerate(self.interceptors):
                    if i != j:
                        other_pos = other_interceptor.get_position()
                        dist = distance(interceptor_pos, other_pos)
                        closest_teammate_dist = min(closest_teammate_dist, dist)
            
            # Fuel level
            fuel_level = self.interceptor_fuels[i]
            
            # Combine individual interceptor observation
            interceptor_obs = np.concatenate([
                interceptor_state / np.array([1000, 1000, 1000, 100, 100, 100, 1, 1, 1, 1, 1, 1, 1]),  # Normalize state
                relative_pos / 1000.0,  # Normalize to km
                relative_vel / 100.0,   # Normalize velocity
                [closest_teammate_dist / 1000.0],  # Normalize distance
                [fuel_level]
            ])
            
            observations.append(interceptor_obs)
        
        # Adversary state
        adversary_state = self.adversary.get_full_state()
        adversary_obs = adversary_state / np.array([1000, 1000, 1000, 100, 100, 100, 1, 1, 1, 1, 1, 1, 1])
        
        # Environment features
        time_normalized = self.episode_time / (self.max_episode_steps * self.time_step)
        avg_fuel = np.mean(self.interceptor_fuels)
        
        env_obs = np.array([
            time_normalized,
            avg_fuel,
            self.coordination_bonus,
            self.min_interceptor_separation / 1000.0,
            len([f for f in self.interceptor_fuels if f > 0.1]),  # Active interceptors
            self.n_interceptors
        ])
        
        # Coordination information (relative positions between interceptors)
        coordination_obs = []
        for i in range(self.n_interceptors):
            pos_i = self.interceptors[i].get_position()
            for j in range(self.n_interceptors):
                if i != j:
                    pos_j = self.interceptors[j].get_position()
                    relative = (pos_j - pos_i) / 1000.0  # Normalize
                    coordination_obs.extend(relative[:3])  # Only position (not full state)
        
        # Pad coordination observations to fixed size
        max_coord_size = self.n_interceptors * 3
        while len(coordination_obs) < max_coord_size:
            coordination_obs.append(0.0)
        
        # Concatenate all observations
        full_observation = np.concatenate([
            np.concatenate(observations),  # All interceptor observations
            adversary_obs,                 # Adversary state
            env_obs,                      # Environment features
            coordination_obs[:max_coord_size]  # Coordination info
        ])
        
        return full_observation.astype(np.float32)
    
    def _get_multi_interceptor_info(self) -> Dict[str, Any]:
        """Get info dictionary for multi-interceptor scenario."""
        # Get distances to adversary
        adversary_pos = self.adversary.get_position()
        interceptor_distances = []
        
        for interceptor in self.interceptors:
            interceptor_pos = interceptor.get_position()
            dist = distance(interceptor_pos, adversary_pos)
            interceptor_distances.append(dist)
        
        info = {
            'interceptor_distances': interceptor_distances,
            'min_intercept_distance': min(interceptor_distances),
            'target_distance': distance(adversary_pos, self.target_position),
            'interceptor_fuels': self.interceptor_fuels.copy(),
            'min_fuel': min(self.interceptor_fuels),
            'episode_time': self.episode_time,
            'current_step': self.current_step,
            'n_active_interceptors': sum(1 for f in self.interceptor_fuels if f > 0.0),
            'coordination_bonus': self.coordination_bonus,
            'min_interceptor_separation': self.min_interceptor_separation,
            'formation_bonus': self._calculate_formation_bonus(),
            'collision_penalties': self.collision_penalties
        }
        
        return info
    
    def get_individual_interceptor_states(self) -> List[Dict[str, Any]]:
        """Get individual state information for each interceptor."""
        states = []
        
        for i, interceptor in enumerate(self.interceptors):
            state = {
                'interceptor_id': i,
                'position': interceptor.get_position(),
                'velocity': interceptor.get_velocity(),
                'orientation': interceptor.get_orientation(),
                'angular_velocity': interceptor.get_angular_velocity(),
                'fuel_remaining': self.interceptor_fuels[i],
                'distance_to_adversary': distance(
                    interceptor.get_position(), 
                    self.adversary.get_position()
                )
            }
            states.append(state)
        
        return states