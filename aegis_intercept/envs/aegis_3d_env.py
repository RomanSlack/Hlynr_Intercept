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
        world_size: float = 300.0,  # New coordinate system 0-600
        max_steps: int = 300,  # Reasonable time for intercept attempts
        dt: float = 0.05,
        intercept_threshold: float = 8.0,  # Reasonable intercept distance
        miss_threshold: float = 3.0,  # Tighter target protection
        max_velocity: float = 20.0,  # Faster interceptor
        max_accel: float = 8.0,  # More responsive
        drag_coefficient: float = 0.05,  # Less drag
        missile_speed: float = 18.0,  # Much faster missile
        evasion_freq: int = 15,
        evasion_magnitude: float = 3.0,
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
            low=0.0,  # New coordinate system starts at 0
            high=world_size * 2,  # Goes from 0 to 600
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

        # Target position (what we're defending) - at ground level in center
        self.target_pos = np.array([
            self.world_size + self.np_random.uniform(-20, 20),  # Center around 300
            self.world_size + self.np_random.uniform(-20, 20),  # Center around 300
            0.0  # Ground level (z=0)
        ], dtype=np.float32)
        
        # Interceptor starts at ground level near target (launch site)
        interceptor_offset = self.np_random.uniform(10, 30)
        interceptor_angle = self.np_random.uniform(0, 2 * np.pi)
        self.interceptor_pos = self.target_pos + np.array([
            interceptor_offset * np.cos(interceptor_angle),
            interceptor_offset * np.sin(interceptor_angle),
            0.0  # Start at ground level
        ], dtype=np.float32)
        # Start with strong upward launch velocity and prevent ground collision
        self.interceptor_vel = np.array([0.0, 0.0, 8.0], dtype=np.float32)

        # Missile starts high in the sky (z=600) and comes down
        missile_x = self.np_random.uniform(50, self.world_size * 2 - 50)
        missile_y = self.np_random.uniform(50, self.world_size * 2 - 50)
        self.missile_pos = np.array([
            missile_x,
            missile_y,
            self.world_size * 2  # Start at top (z=600)
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
        
        # Update position
        new_pos = self.interceptor_pos + self.interceptor_vel * self.dt
        
        # Prevent going through ground - bounce or stop at ground level
        if new_pos[2] < 0:
            new_pos[2] = 0.0  # Keep at ground level
            self.interceptor_vel[2] = max(0.0, self.interceptor_vel[2])  # Stop downward velocity
        
        self.interceptor_pos = new_pos

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
        # Check bounds for 0-600 coordinate system (no ground penetration allowed)
        above_ceiling = self.interceptor_pos[2] > self.world_size * 2
        outside_horizontal = (self.interceptor_pos[0] < 0) or (self.interceptor_pos[0] > self.world_size * 2) or (self.interceptor_pos[1] < 0) or (self.interceptor_pos[1] > self.world_size * 2)
        out_of_bounds = above_ceiling or outside_horizontal  # Remove ground check since we prevent it
        max_steps_reached = self.step_count >= self.max_steps

        terminated = intercepted or missile_hit or out_of_bounds or max_steps_reached
        reward = 0.0
        if intercepted:
            reward = 1.0
        elif missile_hit:
            reward = -1.0
        elif out_of_bounds:
            reward = -0.5
        elif max_steps_reached:
            reward = -0.3  # Different penalty for timeout
        else:
            reward = -0.01
            prev_dist = distance(self.interceptor_pos - self.interceptor_vel * self.dt, self.missile_pos - self.missile_vel * self.dt)
            if intercept_dist < prev_dist:
                reward += 0.001

        self.step_count += 1
        
        # Debug: Print termination reason occasionally  
        if terminated and self.step_count % 100 == 0:  # Much less spam
            if intercepted:
                print(f"[ENV DEBUG] INTERCEPTED! Reward: {reward}")
            elif missile_hit:
                print(f"[ENV DEBUG] MISSILE HIT TARGET! Reward: {reward}")  
            elif out_of_bounds:
                if above_ceiling:
                    print(f"[ENV DEBUG] OUT OF BOUNDS (ABOVE CEILING)")
                elif outside_horizontal:
                    print(f"[ENV DEBUG] OUT OF BOUNDS (OUTSIDE HORIZONTAL)")
            elif max_steps_reached:
                print(f"[ENV DEBUG] TIMEOUT after {self.max_steps} steps")
                
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

    def render(self, intercepted: bool = False):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = Viewer3D(self.world_size)
            self.viewer.render(self.interceptor_pos, self.missile_pos, self.target_pos, intercepted)

    def close(self):
        if self.viewer:
            self.viewer.close()
