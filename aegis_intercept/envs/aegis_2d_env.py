"""2D Missile Intercept Environment for AegisIntercept"""

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from ..utils.maths import distance, clamp


class Aegis2DInterceptEnv(gym.Env):
    """
    2D continuous environment for missile interception.
    
    The interceptor must intercept an incoming missile before it reaches
    the defended point at the origin.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
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
        super().__init__()
        
        self.world_size = world_size
        self.max_steps = max_steps
        self.dt = dt
        self.intercept_threshold = intercept_threshold
        self.miss_threshold = miss_threshold
        self.max_velocity = max_velocity
        self.render_mode = render_mode
        
        # State: [interceptor_x, interceptor_y, interceptor_vx, interceptor_vy,
        #         missile_x, missile_y, missile_vx, missile_vy]
        self.observation_space = spaces.Box(
            low=-world_size,
            high=world_size,
            shape=(8,),
            dtype=np.float32
        )
        
        # Action: [delta_vx, delta_vy] - velocity change commands
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.interceptor_pos = np.zeros(2, dtype=np.float32)
        self.interceptor_vel = np.zeros(2, dtype=np.float32)
        self.missile_pos = np.zeros(2, dtype=np.float32)
        self.missile_vel = np.zeros(2, dtype=np.float32)
        self.defended_point = np.zeros(2, dtype=np.float32)
        
        self.step_count = 0
        
        # Rendering
        self.screen = None
        self.clock = None
        self.screen_size = 800
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Reset step counter
        self.step_count = 0
        
        # Initialize interceptor at random position on left side
        self.interceptor_pos = np.array([
            self.np_random.uniform(-self.world_size * 0.8, -self.world_size * 0.4),
            self.np_random.uniform(-self.world_size * 0.3, self.world_size * 0.3)
        ], dtype=np.float32)
        self.interceptor_vel = np.zeros(2, dtype=np.float32)
        
        # Initialize missile at random position on right side, heading toward origin
        missile_start = np.array([
            self.np_random.uniform(self.world_size * 0.4, self.world_size * 0.8),
            self.np_random.uniform(-self.world_size * 0.3, self.world_size * 0.3)
        ], dtype=np.float32)
        
        self.missile_pos = missile_start
        # Missile velocity points toward defended point with some noise
        direction = -missile_start / np.linalg.norm(missile_start)
        noise = self.np_random.normal(0, 0.1, 2)
        self.missile_vel = (direction + noise) * self.np_random.uniform(3.0, 7.0)
        
        # Defended point at origin
        self.defended_point = np.zeros(2, dtype=np.float32)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Apply action to interceptor velocity
        action = np.clip(action, -1.0, 1.0)
        velocity_change = action * 2.0  # Scale action to reasonable velocity change
        
        self.interceptor_vel += velocity_change
        # Clamp interceptor velocity
        vel_magnitude = np.linalg.norm(self.interceptor_vel)
        if vel_magnitude > self.max_velocity:
            self.interceptor_vel = self.interceptor_vel / vel_magnitude * self.max_velocity
        
        # Update positions
        self.interceptor_pos += self.interceptor_vel * self.dt
        self.missile_pos += self.missile_vel * self.dt
        
        # Check for intercept
        intercept_distance = distance(self.interceptor_pos, self.missile_pos)
        intercepted = intercept_distance < self.intercept_threshold
        
        # Check if missile hit defended point
        miss_distance = distance(self.missile_pos, self.defended_point)
        missile_hit = miss_distance < self.miss_threshold
        
        # Check bounds
        out_of_bounds = (
            np.abs(self.interceptor_pos).max() > self.world_size or
            np.abs(self.missile_pos).max() > self.world_size
        )
        
        self.step_count += 1
        max_steps_reached = self.step_count >= self.max_steps
        
        # Calculate reward
        reward = 0.0
        terminated = False
        
        if intercepted:
            reward = 1.0
            terminated = True
        elif missile_hit:
            reward = -1.0
            terminated = True
        elif out_of_bounds or max_steps_reached:
            reward = -0.5
            terminated = True
        else:
            # Small negative reward for each step to encourage efficiency
            reward = -0.01
            # Small positive reward for getting closer to missile
            prev_distance = distance(self.interceptor_pos - self.interceptor_vel * self.dt, self.missile_pos - self.missile_vel * self.dt)
            curr_distance = distance(self.interceptor_pos, self.missile_pos)
            if curr_distance < prev_distance:
                reward += 0.001
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        return np.concatenate([
            self.interceptor_pos,
            self.interceptor_vel,
            self.missile_pos,
            self.missile_vel
        ]).astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        return {
            "interceptor_pos": self.interceptor_pos.copy(),
            "missile_pos": self.missile_pos.copy(),
            "defended_point": self.defended_point.copy(),
            "distance_to_missile": distance(self.interceptor_pos, self.missile_pos),
            "distance_to_defended": distance(self.missile_pos, self.defended_point),
            "step_count": self.step_count
        }
    
    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        
    def _render_human(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("AegisIntercept 2D")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        # Clear screen
        self.screen.fill((0, 0, 0))
        
        # Convert world coordinates to screen coordinates
        def world_to_screen(pos):
            x = int((pos[0] + self.world_size) / (2 * self.world_size) * self.screen_size)
            y = int((pos[1] + self.world_size) / (2 * self.world_size) * self.screen_size)
            return (x, y)
        
        # Draw defended point (origin) as green circle
        defended_screen = world_to_screen(self.defended_point)
        pygame.draw.circle(self.screen, (0, 255, 0), defended_screen, 10)
        
        # Draw missile as red circle
        missile_screen = world_to_screen(self.missile_pos)
        pygame.draw.circle(self.screen, (255, 0, 0), missile_screen, 8)
        
        # Draw missile velocity vector
        missile_vel_end = self.missile_pos + self.missile_vel * 2
        missile_vel_screen = world_to_screen(missile_vel_end)
        pygame.draw.line(self.screen, (255, 100, 100), missile_screen, missile_vel_screen, 2)
        
        # Draw interceptor as blue circle
        interceptor_screen = world_to_screen(self.interceptor_pos)
        pygame.draw.circle(self.screen, (0, 0, 255), interceptor_screen, 8)
        
        # Draw interceptor velocity vector
        interceptor_vel_end = self.interceptor_pos + self.interceptor_vel * 2
        interceptor_vel_screen = world_to_screen(interceptor_vel_end)
        pygame.draw.line(self.screen, (100, 100, 255), interceptor_screen, interceptor_vel_screen, 2)
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()