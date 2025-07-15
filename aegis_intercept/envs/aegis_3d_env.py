"""
3D Missile Intercept Environment for AegisIntercept Phase 2

This environment extends the 2D implementation to full 3D space with:
- 3D physics including gravity
- Smart adversary missile with evasion behavior  
- Checkpointing system for training resume
- 3D visualization with matplotlib or headless mode
- Configurable episode length via max_episode_steps parameter

Usage:
    python train_ppo_phase2.py --headless --checkpoint-interval 10000
    
To visualize a trained model:
    python -c "from aegis_intercept.envs import Aegis3DEnv; env = Aegis3DEnv(render_mode='human'); ..."
    
To extend episode length:
    env = Aegis3DEnv(render_mode="human", max_episode_steps=500)
"""

import os
import pickle
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import warnings

from ..utils.maths import distance, normalize_vector, clamp


class Aegis3DEnv(gym.Env):
    """
    3D continuous environment for missile interception.
    
    The agent missile must intercept an adversary missile before it reaches
    the defended target point. The adversary performs evasive maneuvers when
    the agent approaches within a threshold distance.
    
    State space: 12D [agent_xyz, agent_vel_xyz, adversary_xyz, adversary_vel_xyz]
    Action space: 3D [thrust_x, thrust_y, thrust_z] normalized to [-1, 1]
    
    Args:
        max_episode_steps (int): Maximum number of steps per episode before timeout.
            Default is 300. This allows configuring longer episodes to give the
            agent more time to reach and intercept the adversary.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        world_size: float = 100.0,
        max_steps: int = 1000,
        dt: float = 0.1,
        intercept_threshold: float = 5.0,
        target_threshold: float = 5.0,
        evasion_threshold: float = 15.0,
        max_thrust: float = 20.0,
        max_velocity: float = 15.0,
        gravity: float = -9.81,
        render_mode: Optional[str] = None,
        checkpoint_interval: int = 10000,
        max_episode_steps: int = 600,
        hover_thrust: float = 30
    ):
        super().__init__()
        
        self.world_size = world_size
        self.max_steps = max_steps  # Keep for backward compatibility
        self.max_episode_steps = max_episode_steps  # New configurable episode timeout
        self.dt = dt
        self.intercept_threshold = intercept_threshold
        self.target_threshold = target_threshold
        self.evasion_threshold = evasion_threshold
        self.max_thrust = max_thrust
        self.max_velocity = max_velocity
        self.gravity = np.array([0.0, 0.0, gravity])
        self.render_mode = render_mode or os.environ.get("AEGIS_RENDER_MODE", "human")
        self.checkpoint_interval = checkpoint_interval
        self.hover_thrust = hover_thrust
        
        # State: [agent_xyz, agent_vel_xyz, adversary_xyz, adversary_vel_xyz]
        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(12,),
            dtype=np.float32
        )
        
        # Action: [thrust_x, thrust_y, thrust_z] normalized
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.agent_pos = np.zeros(3, dtype=np.float32)
        self.agent_vel = np.zeros(3, dtype=np.float32)
        self.adversary_pos = np.zeros(3, dtype=np.float32)
        self.adversary_vel = np.zeros(3, dtype=np.float32)
        self.target_pos = np.zeros(3, dtype=np.float32)
        
        self.step_count = 0
        self.total_steps = 0
        self._elapsed_steps = 0  # Track steps within current episode
        
        # Rendering
        self.fig = None
        self.ax = None
        self.agent_trajectory = []
        self.adversary_trajectory = []
        
        # Checkpointing
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Reset step counters
        self.step_count = 0
        self._elapsed_steps = 0
        
        # Initialize agent at random position in left hemisphere
        self.agent_pos = np.array([
            self.np_random.uniform(-self.world_size * 0.4, -self.world_size * 0.2),
            self.np_random.uniform(-self.world_size * 0.2, self.world_size * 0.2),
            self.np_random.uniform(self.world_size * 0.01, self.world_size * 0.05)
        ], dtype=np.float32)
        self.agent_vel = np.zeros(3, dtype=np.float32)
        
        # Initialize adversary at random position in right hemisphere
        adversary_start = np.array([
            self.np_random.uniform(self.world_size * 0.6, self.world_size * 0.9),
            self.np_random.uniform(-self.world_size * 0.2, self.world_size * 0.3),
            self.np_random.uniform(self.world_size * 0.5, self.world_size * 0.8)
        ], dtype=np.float32)
        
        self.adversary_pos = adversary_start
        
        # Target at origin with some random offset
        self.target_pos = np.array([
            self.np_random.uniform(-5.0, 5.0),
            self.np_random.uniform(-5.0, 5.0),
            self.np_random.uniform(0.0, 10.0)
        ], dtype=np.float32)
        
        # Adversary velocity points toward target with noise
        direction = normalize_vector(self.target_pos - adversary_start)
        noise = self.np_random.normal(0, 0.05, 3)
        self.adversary_vel = (direction + noise) * self.np_random.uniform(4.0, 8.0)
        
        # Reset trajectories for rendering
        self.agent_trajectory = [self.agent_pos.copy()]
        self.adversary_trajectory = [self.adversary_pos.copy()]
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Apply action as thrust to agent
        action = np.clip(action, -1.0, 1.0)
        thrust = action * self.max_thrust
        thrust[2] += self.hover_thrust
        
        # Update agent physics with thrust and gravity
        acceleration = thrust + self.gravity
        self.agent_vel += acceleration * self.dt
        
        # Clamp agent velocity
        vel_magnitude = np.linalg.norm(self.agent_vel)
        if vel_magnitude > self.max_velocity:
            self.agent_vel = self.agent_vel / vel_magnitude * self.max_velocity
        
        # Update agent position
        self.agent_pos += self.agent_vel * self.dt
        
        # Update adversary with smart evasion behavior
        self._update_adversary()
        
        # Store trajectories for rendering
        self.agent_trajectory.append(self.agent_pos.copy())
        self.adversary_trajectory.append(self.adversary_pos.copy())
        
        # Calculate distances
        agent_adversary_dist = distance(self.agent_pos, self.adversary_pos)
        adversary_target_dist = distance(self.adversary_pos, self.target_pos)
        
        # Check termination conditions
        intercepted = agent_adversary_dist < self.intercept_threshold
        target_hit = adversary_target_dist < self.target_threshold
        
        # Check bounds
        out_of_bounds = (
            np.abs(self.agent_pos).max() > self.world_size or
            np.abs(self.adversary_pos).max() > self.world_size or
            self.agent_pos[2] < 0 or  # Agent crashed into ground
            self.adversary_pos[2] < 0  # Adversary crashed into ground
        )
        
        self.step_count += 1
        self.total_steps += 1
        self._elapsed_steps += 1  # Increment episode step counter
        
        # Check for episode timeout using new configurable parameter
        episode_timeout = self._elapsed_steps >= self.max_episode_steps
        max_steps_reached = self.step_count >= self.max_steps  # Keep for backward compatibility
        
        # Calculate reward with enhanced 3D shaping
        reward = self._calculate_reward(
            intercepted, target_hit, out_of_bounds, 
            agent_adversary_dist, adversary_target_dist
        )
        
        terminated = intercepted or target_hit or out_of_bounds or episode_timeout or max_steps_reached
        
        # Checkpointing
        if self.total_steps % self.checkpoint_interval == 0:
            self._save_checkpoint()
        
        observation = self._get_observation()
        info = self._get_info()
        return observation, reward, terminated, False, info
    
    def _update_adversary(self):
        """Update adversary missile with evasion behavior."""
        # Base movement toward target
        direction_to_target = normalize_vector(self.target_pos - self.adversary_pos)
        
        # Check if agent is close enough to trigger evasion
        agent_distance = distance(self.agent_pos, self.adversary_pos)
        
        if agent_distance < self.evasion_threshold:
            # Evasion behavior: move perpendicular to agent approach vector
            agent_direction = normalize_vector(self.agent_pos - self.adversary_pos)
            
            # Create perpendicular vector in XY plane
            perpendicular = np.array([-agent_direction[1], agent_direction[0], 0])
            if np.linalg.norm(perpendicular) < 0.1:  # Handle edge case
                perpendicular = np.array([1, 0, 0])
            perpendicular = normalize_vector(perpendicular)
            
            # Blend evasion with target approach based on distance
            evasion_strength = max(0, 1.0 - agent_distance / self.evasion_threshold)
            desired_direction = (
                (1 - evasion_strength) * direction_to_target +
                evasion_strength * perpendicular
            )
            desired_direction = normalize_vector(desired_direction)
        else:
            desired_direction = direction_to_target
        
        # Update adversary velocity with some inertia
        target_velocity = desired_direction * 6.0
        self.adversary_vel = 0.8 * self.adversary_vel + 0.2 * target_velocity
        
        # Apply gravity to adversary
        self.adversary_vel += self.gravity * self.dt
        
        # Update adversary position
        self.adversary_pos += self.adversary_vel * self.dt
    
    def _calculate_reward(
        self, intercepted: bool, target_hit: bool, out_of_bounds: bool,
        agent_adversary_dist: float, adversary_target_dist: float
    ) -> float:
        """Calculate reward with enhanced 3D shaping."""
        reward = 0.0
        
        if intercepted:
            # Success bonus with time efficiency reward (use new episode steps for calculation)
            time_bonus = max(0, (self.max_episode_steps - self._elapsed_steps) / self.max_episode_steps * 2.0)
            reward = 10.0 + time_bonus
        elif target_hit:
            # Major failure penalty
            reward = -4.0
        elif out_of_bounds:
            # Out of bounds penalty
            reward = -3.0
        else:
            # Step-based reward shaping
            reward = -0.02  # Time penalty
            
            # Distance-based reward shaping
            # Reward getting closer to adversary
            if len(self.agent_trajectory) > 1:
                prev_agent_pos = self.agent_trajectory[-2]
                prev_distance = distance(prev_agent_pos, self.adversary_pos)
                if agent_adversary_dist < prev_distance:
                    reward += 0.05
            
            # Penalty if adversary gets closer to target
            if len(self.adversary_trajectory) > 1:
                prev_adversary_pos = self.adversary_trajectory[-2]
                prev_target_dist = distance(prev_adversary_pos, self.target_pos)
                if adversary_target_dist < prev_target_dist:
                    reward -= 0.005
            
            # Height maintenance reward (prevent ground crash)
            if self.agent_pos[2] > 5.0:
                reward += 0.001
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation with proper normalization."""
        obs = np.concatenate([
            self.agent_pos,
            self.agent_vel,
            self.adversary_pos,
            self.adversary_vel
        ]).astype(np.float32)
        
        # Ensure observation is within specified bounds
        obs = np.clip(obs, -100.0, 100.0)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        return {
            "agent_pos": self.agent_pos.copy(),
            "adversary_pos": self.adversary_pos.copy(),
            "target_pos": self.target_pos.copy(),
            "distance_to_adversary": distance(self.agent_pos, self.adversary_pos),
            "distance_adversary_to_target": distance(self.adversary_pos, self.target_pos),
            "step_count": self.step_count,
            "total_steps": self.total_steps,
            "elapsed_steps": self._elapsed_steps
        }
    
    def _save_checkpoint(self):
        """Save environment state for training resume."""
        checkpoint_data = {
            "total_steps": self.total_steps,
            "np_random_state": self.np_random.bit_generator.state,
            "agent_pos": self.agent_pos.copy(),
            "agent_vel": self.agent_vel.copy(),
            "adversary_pos": self.adversary_pos.copy(),
            "adversary_vel": self.adversary_vel.copy(),
            "target_pos": self.target_pos.copy(),
            "step_count": self.step_count,
            "_elapsed_steps": self._elapsed_steps
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"env_checkpoint_{self.total_steps}.pkl")
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load environment state from checkpoint."""
        if not os.path.exists(checkpoint_path):
            warnings.warn(f"Checkpoint file {checkpoint_path} not found")
            return
        
        with open(checkpoint_path, "rb") as f:
            checkpoint_data = pickle.load(f)
        
        self.total_steps = checkpoint_data["total_steps"]
        self.np_random.bit_generator.state = checkpoint_data["np_random_state"]
        self.agent_pos = checkpoint_data["agent_pos"]
        self.agent_vel = checkpoint_data["agent_vel"]
        self.adversary_pos = checkpoint_data["adversary_pos"]
        self.adversary_vel = checkpoint_data["adversary_vel"]
        self.target_pos = checkpoint_data["target_pos"]
        self.step_count = checkpoint_data["step_count"]
        # Handle backward compatibility for _elapsed_steps
        self._elapsed_steps = checkpoint_data.get("_elapsed_steps", 0)
    
    def render(self):
        """Render the environment in 3D."""
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Render using matplotlib 3D."""
        if os.environ.get("AEGIS_HEADLESS", "false").lower() == "true":
            return
        
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            plt.ion()
        
        self.ax.clear()
        
        # Set limits and labels
        self.ax.set_xlim([-self.world_size, self.world_size])
        self.ax.set_ylim([-self.world_size, self.world_size])
        self.ax.set_zlim([0, self.world_size])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Aegis 3D Missile Intercept')
        
        # Draw target as green sphere
        self.ax.scatter(*self.target_pos, color='green', s=200, marker='o', label='Target')
        
        # Draw agent as blue point
        self.ax.scatter(*self.agent_pos, color='blue', s=100, marker='^', label='Agent')
        
        # Draw adversary as red point
        self.ax.scatter(*self.adversary_pos, color='red', s=100, marker='v', label='Adversary')
        
        # Draw trajectories
        if len(self.agent_trajectory) > 1:
            agent_traj = np.array(self.agent_trajectory)
            self.ax.plot(agent_traj[:, 0], agent_traj[:, 1], agent_traj[:, 2], 
                        'b-', alpha=0.6, linewidth=1, label='Agent Trail')
        
        if len(self.adversary_trajectory) > 1:
            adversary_traj = np.array(self.adversary_trajectory)
            self.ax.plot(adversary_traj[:, 0], adversary_traj[:, 1], adversary_traj[:, 2], 
                        'r-', alpha=0.6, linewidth=1, label='Adversary Trail')
        
        # Draw velocity vectors
        self.ax.quiver(*self.agent_pos, *self.agent_vel, color='blue', alpha=0.8, length=2)
        self.ax.quiver(*self.adversary_pos, *self.adversary_vel, color='red', alpha=0.8, length=2)
        
        self.ax.legend()
        plt.pause(0.01)
    
    def _render_rgb_array(self):
        """Render as RGB array for recording."""
        # Implementation for headless rendering would go here
        # For now, return empty array
        return np.zeros((400, 400, 3), dtype=np.uint8)
    
    def close(self):
        """Close rendering."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None