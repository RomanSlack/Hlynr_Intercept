"""
17-dimensional radar observation system for 1v1 missile interception.

This module provides the Radar17DObservation class that computes realistic 
radar-based observations for training RL models on missile interception tasks.
"""

import numpy as np
from typing import Dict, Any, Optional


class Radar17DObservation:
    """
    Computes 17-dimensional radar observations for 1v1 missile interception scenarios.
    
    The 17 dimensions are:
    [0-2]: Relative position (missile - interceptor) normalized by max_range
    [3-5]: Relative velocity normalized by max_velocity  
    [6-8]: Interceptor velocity normalized by max_velocity
    [9-11]: Interceptor orientation (quaternion components, simplified)
    [12]: Fuel remaining (0-1)
    [13]: Time to intercept estimate (0-1) 
    [14]: Radar quality (0-1)
    [15]: Closing rate normalized (-1 to 1)
    [16]: Target alignment factor (-1 to 1)
    """
    
    def __init__(self, max_range: float = 10000.0, max_velocity: float = 1000.0):
        """
        Initialize the radar observation system.
        
        Args:
            max_range: Maximum detection range for normalization
            max_velocity: Maximum velocity for normalization
        """
        self.max_range = max_range
        self.max_velocity = max_velocity
        self._rng = np.random.default_rng()
        
    def seed(self, seed: Optional[int] = None) -> None:
        """Seed the random number generator for consistent noise."""
        self._rng = np.random.default_rng(seed)
        
    def compute_observation(self, 
                          interceptor_state: Dict[str, Any],
                          missile_state: Dict[str, Any], 
                          radar_quality: float = 1.0,
                          radar_noise: float = 0.0,
                          dt: float = 0.1) -> np.ndarray:
        """
        Compute 17D radar observation from current states.
        
        Args:
            interceptor_state: Dict with 'position', 'velocity', 'orientation', 'fuel'
            missile_state: Dict with 'position', 'velocity'
            radar_quality: Quality factor 0-1 (affects noise)
            radar_noise: Additional noise level 0-1
            dt: Time step for calculations
            
        Returns:
            17D numpy array with normalized observations
        """
        obs = np.zeros(17, dtype=np.float32)
        
        # Extract positions and velocities
        interceptor_pos = np.array(interceptor_state['position'], dtype=np.float64)
        interceptor_vel = np.array(interceptor_state['velocity'], dtype=np.float64)
        missile_pos = np.array(missile_state['position'], dtype=np.float64)
        missile_vel = np.array(missile_state['velocity'], dtype=np.float64)
        
        # [0-2]: Relative position (normalized)
        rel_pos = missile_pos - interceptor_pos
        obs[0:3] = np.clip(rel_pos / self.max_range, -1.0, 1.0)
        
        # [3-5]: Relative velocity (normalized)
        rel_vel = missile_vel - interceptor_vel
        obs[3:6] = np.clip(rel_vel / self.max_velocity, -1.0, 1.0)
        
        # [6-8]: Interceptor velocity (normalized)
        obs[6:9] = np.clip(interceptor_vel / self.max_velocity, -1.0, 1.0)
        
        # [9-11]: Interceptor orientation (simplified from quaternion)
        if 'orientation' in interceptor_state:
            orientation = interceptor_state['orientation']
            if isinstance(orientation, (list, tuple, np.ndarray)) and len(orientation) >= 3:
                obs[9:12] = np.array(orientation[:3], dtype=np.float32)
            else:
                obs[9:12] = [0.0, 0.0, 0.0]  # Default orientation
        else:
            obs[9:12] = [0.0, 0.0, 0.0]
            
        # [12]: Fuel remaining (0-1)
        if 'fuel' in interceptor_state:
            obs[12] = np.clip(interceptor_state['fuel'], 0.0, 1.0)
        else:
            obs[12] = 1.0  # Assume full fuel if not specified
            
        # [13]: Time to intercept estimate (0-1)
        distance = np.linalg.norm(rel_pos)
        if distance > 0:
            # Simple time to intercept based on current closing rate
            closing_rate = -np.dot(rel_pos, rel_vel) / distance
            if closing_rate > 0:
                time_to_intercept = distance / closing_rate
                obs[13] = np.clip(time_to_intercept / 60.0, 0.0, 1.0)  # Normalize to 60 seconds
            else:
                obs[13] = 1.0  # Maximum time if not closing
        else:
            obs[13] = 0.0  # Already intercepted
            
        # [14]: Radar quality (0-1)
        obs[14] = np.clip(radar_quality, 0.0, 1.0)
        
        # [15]: Closing rate normalized (-1 to 1)
        if distance > 0:
            closing_rate = -np.dot(rel_pos, rel_vel) / distance
            obs[15] = np.clip(closing_rate / self.max_velocity, -1.0, 1.0)
        else:
            obs[15] = 0.0
            
        # [16]: Target alignment factor (-1 to 1)
        # How well aligned is the interceptor velocity with the intercept vector
        if np.linalg.norm(interceptor_vel) > 0 and distance > 0:
            vel_unit = interceptor_vel / np.linalg.norm(interceptor_vel)
            intercept_unit = rel_pos / distance
            alignment = np.dot(vel_unit, intercept_unit)
            obs[16] = np.clip(alignment, -1.0, 1.0)
        else:
            obs[16] = 0.0
            
        # Add realistic radar noise based on quality and additional noise
        total_noise_factor = max(1.0 - radar_quality, radar_noise)
        if total_noise_factor > 0:
            noise_scale = total_noise_factor * 0.1  # Up to 10% noise
            noise = self._rng.normal(0, noise_scale, size=17)
            obs += noise
            obs = np.clip(obs, -1.0, 1.0)  # Keep within bounds
            
        return obs.astype(np.float32)