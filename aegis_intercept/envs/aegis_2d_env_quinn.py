"""Quinn's custom 2D Missile Intercept Environment"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from .aegis_2d_env import Aegis2DInterceptEnv


class Aegis2DInterceptEnvQuinn(Aegis2DInterceptEnv):
    """
    Quinn's customized version of the 2D missile intercept environment.
    
    Key differences from base environment:
    - Observation space bounds: [-100, 100] for all elements
    - Action space bounds: [-1, 1] for both actions
    - Enhanced reward shaping for improved training convergence
    """
    
    def __init__(
        self,
        world_size: float = 100.0,
        max_steps: int = 500,
        dt: float = 0.1,
        intercept_threshold: float = 2.0,
        miss_threshold: float = 5.0,
        max_velocity: float = 10.0,
        render_mode: Optional[str] = None
    ):
        super().__init__(
            world_size=world_size,
            max_steps=max_steps,
            dt=dt,
            intercept_threshold=intercept_threshold,
            miss_threshold=miss_threshold,
            max_velocity=max_velocity,
            render_mode=render_mode
        )
        
        # Override observation space with Quinn's specification
        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(8,),
            dtype=np.float32
        )
        
        # Override action space with Quinn's specification
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Quinn's enhanced step function with improved reward shaping.
        """
        # Store previous positions for reward calculation
        prev_interceptor_pos = self.interceptor_pos.copy()
        prev_missile_pos = self.missile_pos.copy()
        
        # Apply action to interceptor velocity
        action = np.clip(action, -1.0, 1.0)
        velocity_change = action * 3.0  # Quinn's scaling factor
        
        self.interceptor_vel += velocity_change
        # Clamp interceptor velocity
        vel_magnitude = np.linalg.norm(self.interceptor_vel)
        if vel_magnitude > self.max_velocity:
            self.interceptor_vel = self.interceptor_vel / vel_magnitude * self.max_velocity
        
        # Update positions
        self.interceptor_pos += self.interceptor_vel * self.dt
        self.missile_pos += self.missile_vel * self.dt
        
        # Calculate distances
        intercept_distance = np.linalg.norm(self.interceptor_pos - self.missile_pos)
        miss_distance = np.linalg.norm(self.missile_pos - self.defended_point)
        
        # Check termination conditions
        intercepted = intercept_distance < self.intercept_threshold
        missile_hit = miss_distance < self.miss_threshold
        
        # Check bounds
        out_of_bounds = (
            np.abs(self.interceptor_pos).max() > self.world_size or
            np.abs(self.missile_pos).max() > self.world_size
        )
        
        self.step_count += 1
        max_steps_reached = self.step_count >= self.max_steps
        
        # Quinn's enhanced reward calculation
        reward = 0.0
        terminated = False
        
        if intercepted:
            # Success bonus with time efficiency reward
            time_bonus = max(0, (self.max_steps - self.step_count) / self.max_steps * 0.5)
            reward = 5.0 + time_bonus
            terminated = True
        elif missile_hit:
            # Failure penalty
            reward = -1.0
            terminated = True
        elif out_of_bounds:
            # Out of bounds penalty
            reward = -0.5
            terminated = True
        elif max_steps_reached:
            # Timeout penalty (less severe than other failures)
            reward = -0.5
            terminated = True
        else:
            # Step-based reward shaping
            reward = -0.01  # Small time penalty
            
            # Distance-based reward shaping
            prev_distance = np.linalg.norm(prev_interceptor_pos - prev_missile_pos)
            curr_distance = intercept_distance
            if curr_distance < prev_distance:
                reward += 0.001
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation with Quinn's normalization.
        """
        obs = np.concatenate([
            self.interceptor_pos,
            self.interceptor_vel,
            self.missile_pos,
            self.missile_vel
        ]).astype(np.float32)
        
        # Ensure observation is within specified bounds
        obs = np.clip(obs, -100.0, 100.0)
        
        return obs