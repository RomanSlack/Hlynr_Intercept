"""3D Missile Intercept Environment for AegisIntercept"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from ..utils.physics3d import distance, linear_drag
from ..rendering.viewer3d import Viewer3D

class Aegis3DInterceptEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        world_size: float = 150.0,  # Reduced from 200 for faster episodes
        max_steps: int = 300,  # Reduced from 500 for faster episodes
        dt: float = 0.05,
        intercept_threshold: float = 3.0,  # Slightly larger for easier intercepts
        miss_threshold: float = 8.0,  # Larger target area
        max_velocity: float = 15.0,
        max_accel: float = 5.0,
        drag_coefficient: float = 0.1,
        missile_speed: float = 12.0,  # Faster missile
        evasion_freq: int = 10,
        evasion_magnitude: float = 5.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.world_size = world_size
        self.max_steps = max_steps
        self.dt = dt
        self.intercept_threshold = intercept_threshold
        self.miss_threshold = miss_threshold
        self.max_velocity = max_velocity
        self.max_accel = max_accel
        self.drag_coefficient = drag_coefficient
        self.missile_speed = missile_speed
        self.evasion_freq = evasion_freq
        self.evasion_magnitude = evasion_magnitude
        self.render_mode = render_mode

        # State: [interceptor_pos(3), interceptor_vel(3), missile_pos(3), missile_vel(3), time_remaining]
        self.observation_space = spaces.Box(
            low=-world_size,
            high=world_size,
            shape=(13,),
            dtype=np.float32
        )

        # Action: [accel_x, accel_y, accel_z]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        self.interceptor_pos = np.zeros(3, dtype=np.float32)
        self.interceptor_vel = np.zeros(3, dtype=np.float32)
        self.missile_pos = np.zeros(3, dtype=np.float32)
        self.missile_vel = np.zeros(3, dtype=np.float32)
        self.target_pos = np.zeros(3, dtype=np.float32)

        self.step_count = 0
        self.viewer = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.step_count = 0

        # Target position (what we're defending) - on ground plane
        self.target_pos = np.array([
            self.np_random.uniform(-self.world_size * 0.1, self.world_size * 0.1),
            self.np_random.uniform(-self.world_size * 0.1, self.world_size * 0.1),
            0.0  # Ground level
        ], dtype=np.float32)
        
        # Interceptor starts at ground level near target (launch site)
        interceptor_offset = self.np_random.uniform(5, 15)  # 5-15 units from target
        interceptor_angle = self.np_random.uniform(0, 2 * np.pi)
        self.interceptor_pos = self.target_pos + np.array([
            interceptor_offset * np.cos(interceptor_angle),
            interceptor_offset * np.sin(interceptor_angle),
            0.0  # Start at ground level
        ], dtype=np.float32)
        # Start with small upward velocity (launch)
        self.interceptor_vel = np.array([0.0, 0.0, 2.0], dtype=np.float32)

        # Missile starts at medium distance for reasonable episode length
        missile_distance = self.np_random.uniform(self.world_size * 0.5, self.world_size * 0.7)
        missile_angle = self.np_random.uniform(0, 2 * np.pi)
        self.missile_pos = self.target_pos + np.array([
            missile_distance * np.cos(missile_angle),
            missile_distance * np.sin(missile_angle),
            self.np_random.uniform(20, 60)
        ], dtype=np.float32)
        
        # Missile heads toward target
        direction = (self.target_pos - self.missile_pos) / np.linalg.norm(self.target_pos - self.missile_pos)
        self.missile_vel = direction * self.missile_speed

        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, -1.0, 1.0)
        thrust = action * self.max_accel

        drag = linear_drag(self.interceptor_vel, self.drag_coefficient)
        self.interceptor_vel += (thrust + drag) * self.dt
        vel_mag = np.linalg.norm(self.interceptor_vel)
        if vel_mag > self.max_velocity:
            self.interceptor_vel = self.interceptor_vel / vel_mag * self.max_velocity
        self.interceptor_pos += self.interceptor_vel * self.dt

        if self.step_count % self.evasion_freq == 0:
            evasion_offset = np.random.randn(3)
            evasion_offset[2] = 0 # No vertical evasion
            evasion_offset = evasion_offset / np.linalg.norm(evasion_offset) * self.evasion_magnitude
            self.missile_pos += evasion_offset

        direction = (self.target_pos - self.missile_pos) / np.linalg.norm(self.target_pos - self.missile_pos)
        self.missile_vel = direction * self.missile_speed
        self.missile_pos += self.missile_vel * self.dt

        intercept_dist = distance(self.interceptor_pos, self.missile_pos)
        miss_dist = distance(self.missile_pos, self.target_pos)

        intercepted = intercept_dist < self.intercept_threshold
        missile_hit = miss_dist < self.miss_threshold
        out_of_bounds = np.abs(self.interceptor_pos).max() > self.world_size
        max_steps_reached = self.step_count >= self.max_steps

        terminated = intercepted or missile_hit or out_of_bounds or max_steps_reached
        reward = 0.0
        if intercepted:
            reward = 1.0
        elif missile_hit:
            reward = -1.0
        elif out_of_bounds or max_steps_reached:
            reward = -0.5
        else:
            reward = -0.01
            prev_dist = distance(self.interceptor_pos - self.interceptor_vel * self.dt, self.missile_pos - self.missile_vel * self.dt)
            if intercept_dist < prev_dist:
                reward += 0.001

        self.step_count += 1
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self) -> np.ndarray:
        time_remaining = (self.max_steps - self.step_count) / self.max_steps
        return np.concatenate([
            self.interceptor_pos,
            self.interceptor_vel,
            self.missile_pos,
            self.missile_vel,
            [time_remaining]
        ]).astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "interceptor_pos": self.interceptor_pos.copy(),
            "missile_pos": self.missile_pos.copy(),
            "distance_to_missile": distance(self.interceptor_pos, self.missile_pos),
        }

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = Viewer3D(self.world_size)
            self.viewer.render(self.interceptor_pos, self.missile_pos, self.target_pos)

    def close(self):
        if self.viewer:
            self.viewer.close()
