"""
Radar-based 17-dimensional observation space for Phase 4 RL.

This module provides a consistent 17-dimensional observation vector for
1v1 intercept scenarios using radar-based sensing only.
"""

import numpy as np
from typing import Dict, Any, Tuple


class Radar17DObservation:
    """
    Generates consistent 17-dimensional radar observations for RL training and inference.
    
    Observation components (17 total):
    - [0-2]: Relative position to target (range-normalized) [3D]
    - [3-5]: Relative velocity estimate (normalized) [3D]
    - [6-8]: Interceptor velocity (normalized) [3D]
    - [9-11]: Interceptor orientation (euler angles) [3D]
    - [12]: Interceptor fuel fraction [1D]
    - [13]: Time to intercept estimate (normalized) [1D]
    - [14]: Radar lock quality (0-1) [1D]
    - [15]: Closing rate (normalized) [1D]
    - [16]: Off-axis angle (normalized) [1D]
    """
    
    def __init__(self, max_range: float = 10000.0, max_velocity: float = 1000.0):
        """
        Initialize the radar observation generator.
        
        Args:
            max_range: Maximum radar range for normalization (meters)
            max_velocity: Maximum velocity for normalization (m/s)
        """
        self.max_range = max_range
        self.max_velocity = max_velocity
        self.observation_dim = 17
        self.rng = np.random.default_rng()
        
    def seed(self, seed: int):
        """Seed the random number generator for reproducible noise."""
        self.rng = np.random.default_rng(seed)
        
    def compute_observation(self, 
                          interceptor_state: Dict[str, Any],
                          missile_state: Dict[str, Any],
                          radar_quality: float = 1.0,
                          radar_noise: float = 0.05) -> np.ndarray:
        """
        Compute 17-dimensional radar observation vector.
        
        Args:
            interceptor_state: Dict with keys 'position', 'velocity', 'orientation', 'fuel'
            missile_state: Dict with keys 'position', 'velocity'
            radar_quality: Quality of radar lock (0-1)
            radar_noise: Noise level for radar measurements
            
        Returns:
            17-dimensional observation vector
        """
        obs = np.zeros(17, dtype=np.float32)
        
        # Extract states
        int_pos = np.array(interceptor_state['position'], dtype=np.float32)
        int_vel = np.array(interceptor_state['velocity'], dtype=np.float32)
        int_orient = np.array(interceptor_state['orientation'], dtype=np.float32)  # quaternion
        int_fuel = interceptor_state.get('fuel', 100.0)
        
        mis_pos = np.array(missile_state['position'], dtype=np.float32)
        mis_vel = np.array(missile_state['velocity'], dtype=np.float32)
        
        # Compute relative position (with radar noise)
        rel_pos = mis_pos - int_pos
        rel_pos += self.rng.normal(0, radar_noise * np.linalg.norm(rel_pos), 3)
        range_to_target = np.linalg.norm(rel_pos)
        
        # Normalize relative position [0-2]
        obs[0:3] = np.clip(rel_pos / self.max_range, -1.0, 1.0)
        
        # Compute relative velocity (with radar doppler)
        rel_vel = mis_vel - int_vel
        rel_vel += self.rng.normal(0, radar_noise * np.linalg.norm(rel_vel), 3)
        
        # Normalize relative velocity [3-5]
        obs[3:6] = np.clip(rel_vel / self.max_velocity, -1.0, 1.0)
        
        # Interceptor velocity (body knowledge) [6-8]
        obs[6:9] = np.clip(int_vel / self.max_velocity, -1.0, 1.0)
        
        # Interceptor orientation (convert quaternion to euler angles) [9-11]
        euler = self._quaternion_to_euler(int_orient)
        obs[9:12] = euler / np.pi  # Normalize to [-1, 1]
        
        # Fuel fraction [12]
        obs[12] = np.clip(int_fuel / 100.0, 0.0, 1.0)
        
        # Time to intercept estimate [13]
        closing_speed = -np.dot(rel_pos, rel_vel) / (range_to_target + 1e-6)
        if closing_speed > 0:
            time_to_intercept = range_to_target / closing_speed
            obs[13] = np.clip(1.0 - time_to_intercept / 100.0, -1.0, 1.0)
        else:
            obs[13] = -1.0  # Not closing
        
        # Radar lock quality [14]
        obs[14] = radar_quality
        
        # Closing rate (normalized) [15]
        obs[15] = np.clip(closing_speed / self.max_velocity, -1.0, 1.0)
        
        # Off-axis angle (how aligned interceptor is to target) [16]
        if range_to_target > 1e-6:
            int_forward = self._get_forward_vector(int_orient)
            to_target = rel_pos / range_to_target
            alignment = np.dot(int_forward, to_target)
            obs[16] = alignment  # Already in [-1, 1]
        else:
            obs[16] = 1.0  # Perfect alignment if at same position
        
        return obs
    
    def _quaternion_to_euler(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion [w,x,y,z] to euler angles [roll, pitch, yaw]."""
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw], dtype=np.float32)
    
    def _get_forward_vector(self, q: np.ndarray) -> np.ndarray:
        """Get forward direction vector from quaternion."""
        w, x, y, z = q
        forward = np.array([
            2 * (x * z + w * y),
            2 * (y * z - w * x),
            1 - 2 * (x * x + y * y)
        ], dtype=np.float32)
        return forward / (np.linalg.norm(forward) + 1e-6)
    
    def get_observation_space_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the bounds for the observation space."""
        return np.full(17, -1.0, dtype=np.float32), np.full(17, 1.0, dtype=np.float32)