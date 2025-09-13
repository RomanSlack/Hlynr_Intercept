"""
Gymnasium environment for missile interception with 6DOF physics.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
from core import Radar17DObservation, SafetyClamp, SafetyLimits, euler_to_quaternion


class InterceptEnvironment(gym.Env):
    """Clean missile interception environment with realistic physics."""
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Default configuration
        self.config = config or {}
        self.dt = self.config.get('dt', 0.01)  # 100Hz simulation
        self.max_steps = self.config.get('max_steps', 1000)
        self.max_range = self.config.get('max_range', 10000.0)
        self.max_velocity = self.config.get('max_velocity', 1000.0)
        
        # Spawn configuration
        self.missile_spawn_range = self.config.get('missile_spawn', {
            'position': [[-500, -500, 200], [500, 500, 500]],
            'velocity': [[50, 50, -20], [150, 150, -50]]
        })
        self.interceptor_spawn_range = self.config.get('interceptor_spawn', {
            'position': [[400, 400, 50], [600, 600, 200]],
            'velocity': [[0, 0, 0], [50, 50, 20]]
        })
        self.target_position = np.array(self.config.get('target_position', [900, 900, 5]), dtype=np.float32)
        
        # Physics parameters
        self.gravity = np.array([0, 0, -9.81], dtype=np.float32)
        self.drag_coefficient = 0.3
        self.air_density = 1.225  # kg/m^3 at sea level
        
        # Wind configuration
        wind_config = self.config.get('wind', {})
        self.base_wind = np.array(wind_config.get('velocity', [5.0, 0.0, 0.0]), dtype=np.float32)
        self.wind_variability = wind_config.get('variability', 0.1)
        
        # Radar configuration
        self.radar_noise = self.config.get('radar_noise', 0.05)
        self.radar_quality = 1.0
        
        # Components
        self.observation_generator = Radar17DObservation(self.max_range, self.max_velocity)
        self.safety_clamp = SafetyClamp(SafetyLimits())
        
        # Spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(17,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        
        # State variables
        self.missile_state = None
        self.interceptor_state = None
        self.current_wind = None
        self.steps = 0
        self.total_fuel_used = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            self.observation_generator.seed(seed)
        
        # Initialize missile state
        mis_pos_min, mis_pos_max = self.missile_spawn_range['position']
        mis_vel_min, mis_vel_max = self.missile_spawn_range['velocity']
        
        self.missile_state = {
            'position': np.random.uniform(mis_pos_min, mis_pos_max).astype(np.float32),
            'velocity': np.random.uniform(mis_vel_min, mis_vel_max).astype(np.float32),
            'orientation': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'angular_velocity': np.zeros(3, dtype=np.float32),
            'active': True
        }
        
        # Initialize interceptor state
        int_pos_min, int_pos_max = self.interceptor_spawn_range['position']
        int_vel_min, int_vel_max = self.interceptor_spawn_range['velocity']
        
        self.interceptor_state = {
            'position': np.random.uniform(int_pos_min, int_pos_max).astype(np.float32),
            'velocity': np.random.uniform(int_vel_min, int_vel_max).astype(np.float32),
            'orientation': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            'angular_velocity': np.zeros(3, dtype=np.float32),
            'fuel': 100.0,
            'active': True
        }
        
        # Initialize wind
        self.current_wind = self.base_wind.copy()
        
        # Reset counters
        self.steps = 0
        self.total_fuel_used = 0
        
        # Compute initial observation
        obs = self.observation_generator.compute(
            self.interceptor_state, self.missile_state,
            self.radar_quality, self.radar_noise
        )
        
        info = {
            'missile_pos': self.missile_state['position'].copy(),
            'interceptor_pos': self.interceptor_state['position'].copy(),
            'distance': np.linalg.norm(
                self.missile_state['position'] - self.interceptor_state['position']
            )
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        self.steps += 1
        
        # Apply safety constraints
        clamped_action, clamp_info = self.safety_clamp.apply(
            action, self.interceptor_state['fuel']
        )
        
        # Update interceptor physics
        if self.interceptor_state['active']:
            self._update_interceptor(clamped_action)
        
        # Update missile physics
        if self.missile_state['active']:
            self._update_missile()
        
        # Update wind
        self._update_wind()
        
        # Check interception
        distance = np.linalg.norm(
            self.missile_state['position'] - self.interceptor_state['position']
        )
        intercepted = distance < 10.0  # 10m interception radius
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        if intercepted:
            terminated = True
        elif self.missile_state['position'][2] < 0:  # Missile hit ground
            terminated = True
        elif self.interceptor_state['position'][2] < 0:  # Interceptor crashed
            terminated = True
        elif self.steps >= self.max_steps:
            truncated = True
        
        # Calculate reward
        reward = self._calculate_reward(distance, intercepted, terminated)
        
        # Generate observation
        obs = self.observation_generator.compute(
            self.interceptor_state, self.missile_state,
            self.radar_quality, self.radar_noise
        )
        
        # Info dict
        info = {
            'distance': distance,
            'intercepted': intercepted,
            'fuel_remaining': self.interceptor_state['fuel'],
            'fuel_used': self.total_fuel_used,
            'clamped': clamp_info['clamped'],
            'missile_pos': self.missile_state['position'].copy(),
            'interceptor_pos': self.interceptor_state['position'].copy(),
            'steps': self.steps
        }
        
        return obs, reward, terminated, truncated, info
    
    def _update_interceptor(self, action: np.ndarray):
        """Update interceptor state with 6DOF physics."""
        # Extract thrust and angular commands from action
        thrust_cmd = action[0:3] * 500.0  # Scale to N
        angular_cmd = action[3:6] * 2.0   # Scale to rad/s
        
        # Calculate fuel consumption
        thrust_mag = np.linalg.norm(thrust_cmd)
        fuel_consumed = (thrust_mag / 500.0) * self.safety_clamp.limits.fuel_depletion_rate * self.dt
        self.interceptor_state['fuel'] -= fuel_consumed
        self.total_fuel_used += fuel_consumed
        
        if self.interceptor_state['fuel'] <= 0:
            self.interceptor_state['fuel'] = 0
            thrust_cmd *= 0
        
        # Calculate forces
        mass = 500.0  # kg
        thrust_accel = thrust_cmd / mass
        
        # Drag force
        vel = self.interceptor_state['velocity']
        vel_air = vel - self.current_wind
        drag_accel = -0.5 * self.drag_coefficient * self.air_density * \
                     np.linalg.norm(vel_air) * vel_air / mass
        
        # Total acceleration
        total_accel = thrust_accel + drag_accel + self.gravity
        
        # Update velocity and position
        self.interceptor_state['velocity'] += total_accel * self.dt
        self.interceptor_state['position'] += self.interceptor_state['velocity'] * self.dt
        
        # Update angular velocity and orientation
        self.interceptor_state['angular_velocity'] = angular_cmd
        
        # Simple quaternion integration (small angle approximation)
        w = self.interceptor_state['angular_velocity']
        angle = np.linalg.norm(w) * self.dt
        if angle > 1e-6:
            axis = w / np.linalg.norm(w)
            dq = np.array([
                np.cos(angle/2),
                axis[0] * np.sin(angle/2),
                axis[1] * np.sin(angle/2),
                axis[2] * np.sin(angle/2)
            ])
            self.interceptor_state['orientation'] = self._quaternion_multiply(
                dq, self.interceptor_state['orientation']
            )
            # Normalize quaternion
            self.interceptor_state['orientation'] /= np.linalg.norm(
                self.interceptor_state['orientation']
            )
    
    def _update_missile(self):
        """Update missile state with simple ballistic trajectory."""
        # Simple ballistic missile - minimal maneuvering
        mass = 1000.0  # kg
        
        # Drag force
        vel = self.missile_state['velocity']
        vel_air = vel - self.current_wind
        drag_accel = -0.5 * self.drag_coefficient * self.air_density * \
                     np.linalg.norm(vel_air) * vel_air / mass
        
        # Add slight evasion (if configured)
        evasion = np.zeros(3)
        if self.config.get('missile_evasion', False):
            evasion = np.random.randn(3) * 2.0  # Small random acceleration
        
        # Total acceleration
        total_accel = drag_accel + self.gravity + evasion
        
        # Update velocity and position
        self.missile_state['velocity'] += total_accel * self.dt
        self.missile_state['position'] += self.missile_state['velocity'] * self.dt
    
    def _update_wind(self):
        """Update wind with variability."""
        if self.wind_variability > 0:
            wind_change = np.random.randn(3) * self.wind_variability
            self.current_wind = 0.95 * self.current_wind + 0.05 * (self.base_wind + wind_change)
    
    def _calculate_reward(self, distance: float, intercepted: bool, terminated: bool) -> float:
        """Calculate step reward."""
        reward = 0.0
        
        if intercepted:
            # Successful interception
            reward = 100.0
            # Bonus for fuel efficiency
            reward += self.interceptor_state['fuel'] * 0.5
            # Bonus for quick interception
            reward += (self.max_steps - self.steps) * 0.1
        elif terminated:
            # Failed interception
            if self.missile_state['position'][2] < 0:
                # Missile hit target
                reward = -100.0
            elif self.interceptor_state['position'][2] < 0:
                # Interceptor crashed
                reward = -50.0
        else:
            # Shaping reward
            # Encourage closing distance
            reward = -distance * 0.001
            # Small penalty for time
            reward -= 0.01
            # Small penalty for fuel use
            reward -= self.total_fuel_used * 0.001
        
        return reward
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dtype=np.float32)
    
    def render(self):
        """Render environment (placeholder for future implementation)."""
        pass