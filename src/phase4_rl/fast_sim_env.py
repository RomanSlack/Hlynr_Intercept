"""
Fast simulation environment wrapper for Phase 4 RL training.

This wrapper provides a headless, fast-running version of RadarEnv optimized
for training with no visual rendering and radar-only observations.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import gymnasium as gym

try:
    from .radar_env import RadarEnv
    from .config import get_config
    from .scenarios import get_scenario_loader
    from .episode_logger import EpisodeLogger
    from .radar_observations import Radar17DObservation
except ImportError:
    # Fallback for direct execution
    from radar_env import RadarEnv
    from config import get_config
    from scenarios import get_scenario_loader
    from episode_logger import EpisodeLogger
    from radar_observations import Radar17DObservation


class FastSimEnv(gym.Wrapper):
    """
    Fast simulation wrapper around RadarEnv for headless training.
    
    This wrapper:
    - Forces render_mode="none" for maximum speed
    - Ensures radar-only observations (no omniscient state)
    - Optimizes for training performance
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 scenario_name: Optional[str] = None,
                 num_missiles: Optional[int] = None,
                 num_interceptors: Optional[int] = None,
                 enable_episode_logging: bool = False,
                 episode_log_dir: Optional[str] = None):
        """
        Initialize FastSimEnv wrapper.
        
        Args:
            config: Environment configuration dictionary
            scenario_name: Name of scenario to load
            num_missiles: Number of missiles (overrides config)
            num_interceptors: Number of interceptors (overrides config)
            enable_episode_logging: Whether to enable episode logging for Unity replay
            episode_log_dir: Directory for episode logs (default: "runs")
        """
        # Create RadarEnv with forced headless rendering
        env = RadarEnv(
            config=config,
            scenario_name=scenario_name,
            num_missiles=num_missiles,
            num_interceptors=num_interceptors,
            render_mode=None  # Force headless mode
        )
        
        super().__init__(env)
        
        # Store configuration for reference
        self.config = self.env.config
        self.scenario_name = scenario_name
        
        # Initialize 17D radar observation system
        self.radar_observer = Radar17DObservation(
            max_range=10000.0,
            max_velocity=1000.0
        )
        
        # Update observation space to exactly 17 dimensions
        self._setup_radar_observation_space()
        
        # Initialize episode logger if enabled
        self.enable_episode_logging = enable_episode_logging
        self.episode_logger = None
        if enable_episode_logging:
            self.episode_logger = EpisodeLogger(
                output_dir=episode_log_dir or "runs",
                coord_frame="ENU_RH",  # Using right-handed East-North-Up
                dt_nominal=0.01,  # 100 Hz sampling
                enable_logging=True
            )
        
        # Episode tracking
        self.current_episode_time = 0.0
        self.episode_step_count = 0
    
    def _setup_radar_observation_space(self):
        """Setup observation space for exactly 17-dimensional radar observations."""
        # Fixed 17-dimensional observation space for 1v1 scenarios
        from gymnasium import spaces
        
        # Verify we're in 1v1 scenario
        if self.env.num_missiles != 1 or self.env.num_interceptors != 1:
            raise ValueError(
                f"Radar17D observation only supports 1v1 scenarios. "
                f"Got {self.env.num_missiles} missiles and {self.env.num_interceptors} interceptors."
            )
        
        # Set observation space to exactly 17 dimensions
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(17,),
            dtype=np.float32
        )
        
        # Action space remains the same
        self.action_space = self.env.action_space
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment and ensure radar-only observations.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            Tuple of (radar_observation, info)
        """
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Reset episode tracking
        self.current_episode_time = 0.0
        self.episode_step_count = 0
        
        # Start episode logging if enabled
        if self.episode_logger:
            # Extract entity configurations from the environment
            interceptor_config = {
                "mass": 10.0,  # Default mass
                "max_torque": [8000, 8000, 2000],
                "sensor_fov": 30.0,
                "max_thrust": 1000.0
            }
            
            threat_config = {
                "type": "ballistic",
                "mass": 100.0,
                "aim_point": [0, 0, 0]
            }
            
            # Begin new episode
            self.episode_logger.begin_episode(
                seed=seed,
                scenario_name=self.scenario_name,
                interceptor_config=interceptor_config,
                threat_config=threat_config,
                notes=f"FastSimEnv training episode"
            )
            
            # Log initial state
            self._log_current_state()
        
        # Verify observation is radar-only
        radar_obs = self._ensure_radar_only_observation(obs)
        
        return radar_obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute environment step with radar-only observations.
        
        Args:
            action: Action array for all interceptors
            
        Returns:
            Tuple of (radar_observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update episode tracking
        self.episode_step_count += 1
        # Use dt from config or default to 0.1
        dt = getattr(self.env, 'dt', 0.1)
        self.current_episode_time += dt
        
        # Log step if enabled
        if self.episode_logger:
            self._log_current_state(action)
            
            # If episode ended, log summary
            if terminated or truncated:
                self._log_episode_end(info)
        
        # Ensure observation is radar-only
        radar_obs = self._ensure_radar_only_observation(obs)
        
        return radar_obs, reward, terminated, truncated, info
    
    def _ensure_radar_only_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Generate consistent 17-dimensional radar observation.
        
        Uses the Radar17DObservation system to create a consistent observation
        vector from the current environment state.
        
        Args:
            obs: Raw observation from RadarEnv (unused, we query state directly)
            
        Returns:
            17-dimensional radar observation vector
        """
        # Get interceptor state
        interceptor_state = {
            'position': self.env.interceptor_positions[0],
            'velocity': self.env.interceptor_velocities[0],
            'orientation': self.env.interceptor_orientations[0],
            'fuel': self.env.interceptor_fuel[0] if hasattr(self.env, 'interceptor_fuel') else 100.0
        }
        
        # Get missile state
        missile_state = {
            'position': self.env.missile_positions[0],
            'velocity': self.env.missile_velocities[0]
        }
        
        # Get radar quality (simulated based on range)
        rel_pos = missile_state['position'] - interceptor_state['position']
        range_to_target = np.linalg.norm(rel_pos)
        radar_quality = np.clip(1.0 - range_to_target / self.radar_observer.max_range, 0.1, 1.0)
        
        # Get radar noise from config
        radar_noise = self.env.radar_noise if hasattr(self.env, 'radar_noise') else 0.05
        
        # Compute 17D observation
        radar_obs = self.radar_observer.compute_observation(
            interceptor_state=interceptor_state,
            missile_state=missile_state,
            radar_quality=radar_quality,
            radar_noise=radar_noise
        )
        
        return radar_obs
    
    def render(self, mode: Optional[str] = None):
        """
        Override render to prevent accidental rendering in fast sim mode.
        
        Args:
            mode: Render mode (ignored in fast sim)
        """
        # Fast sim mode doesn't render - silently ignore render calls
        pass
    
    def get_radar_config(self) -> Dict[str, Any]:
        """
        Get radar configuration for this environment.
        
        Returns:
            Radar configuration dictionary
        """
        return self.env.radar_config.copy()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for fast simulation.
        
        Returns:
            Performance statistics dictionary
        """
        return {
            'render_mode': None,
            'radar_only': self._radar_only,
            'num_missiles': self.env.num_missiles,
            'num_interceptors': self.env.num_interceptors,
            'observation_dim': self.env.obs_dim,
            'action_dim': self.env.action_dim
        }
    
    def _log_current_state(self, action: Optional[np.ndarray] = None):
        """Log current state to episode logger."""
        if not self.episode_logger:
            return
        
        # Try to access entity positions from RadarEnv
        # These might be stored as numpy arrays
        missile_positions = getattr(self.env, 'missile_positions', None)
        missile_velocities = getattr(self.env, 'missile_velocities', None)
        interceptor_positions = getattr(self.env, 'interceptor_positions', None)
        interceptor_velocities = getattr(self.env, 'interceptor_velocities', None)
        interceptor_fuel = getattr(self.env, 'interceptor_fuel', None)
        interceptor_active = getattr(self.env, 'interceptor_active', None)
        missile_active = getattr(self.env, 'missile_active', None)
        
        # Build state dictionaries
        interceptor_state = {}
        threat_state = {}
        
        # Log first interceptor (if data available)
        if interceptor_positions is not None and len(interceptor_positions) > 0:
            pos = interceptor_positions[0] if hasattr(interceptor_positions[0], '__len__') else [0.0, 0.0, 0.0]
            vel = interceptor_velocities[0] if interceptor_velocities is not None and len(interceptor_velocities) > 0 else [0.0, 0.0, 0.0]
            
            # Handle both 2D and 3D positions
            if len(pos) == 2:
                position_3d = [float(pos[0]), float(pos[1]), 0.0]
            else:
                position_3d = [float(pos[0]), float(pos[1]), float(pos[2])]
            
            if len(vel) == 2:
                velocity_3d = [float(vel[0]), float(vel[1]), 0.0]
            else:
                velocity_3d = [float(vel[0]), float(vel[1]), float(vel[2])]
            
            # Get orientation data
            interceptor_orientations = getattr(self.env, 'interceptor_orientations', None)
            interceptor_angular_velocities = getattr(self.env, 'interceptor_angular_velocities', None)
            
            quaternion = [1.0, 0.0, 0.0, 0.0]  # Default identity
            angular_velocity = [0.0, 0.0, 0.0]  # Default zero
            
            if interceptor_orientations is not None and len(interceptor_orientations) > 0:
                q = interceptor_orientations[0]
                quaternion = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
            
            if interceptor_angular_velocities is not None and len(interceptor_angular_velocities) > 0:
                w = interceptor_angular_velocities[0]
                angular_velocity = [float(w[0]), float(w[1]), float(w[2])]
            
            interceptor_state = {
                "position": position_3d,
                "quaternion": quaternion,
                "velocity": velocity_3d,
                "angular_velocity": angular_velocity,
                "fuel": float(interceptor_fuel[0]) if interceptor_fuel is not None and len(interceptor_fuel) > 0 else 100.0,
                "status": "active" if interceptor_active is None or (len(interceptor_active) > 0 and interceptor_active[0]) else "destroyed"
            }
            
            if action is not None and len(action) >= 6:
                # Add action for first interceptor
                interceptor_state["action"] = action[:6].tolist()
        
        # Log first missile (if data available)
        if missile_positions is not None and len(missile_positions) > 0:
            pos = missile_positions[0] if hasattr(missile_positions[0], '__len__') else [0.0, 0.0, 0.0]
            vel = missile_velocities[0] if missile_velocities is not None and len(missile_velocities) > 0 else [0.0, 0.0, 0.0]
            
            # Handle both 2D and 3D positions
            if len(pos) == 2:
                position_3d = [float(pos[0]), float(pos[1]), 0.0]
            else:
                position_3d = [float(pos[0]), float(pos[1]), float(pos[2])]
            
            if len(vel) == 2:
                velocity_3d = [float(vel[0]), float(vel[1]), 0.0]
            else:
                velocity_3d = [float(vel[0]), float(vel[1]), float(vel[2])]
            
            # Get missile orientation data
            missile_orientations = getattr(self.env, 'missile_orientations', None)
            missile_angular_velocities = getattr(self.env, 'missile_angular_velocities', None)
            
            quaternion = [1.0, 0.0, 0.0, 0.0]  # Default identity
            angular_velocity = [0.0, 0.0, 0.0]  # Default zero
            
            if missile_orientations is not None and len(missile_orientations) > 0:
                q = missile_orientations[0]
                quaternion = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
            
            if missile_angular_velocities is not None and len(missile_angular_velocities) > 0:
                w = missile_angular_velocities[0]
                angular_velocity = [float(w[0]), float(w[1]), float(w[2])]
            
            threat_state = {
                "position": position_3d,
                "quaternion": quaternion,
                "velocity": velocity_3d,
                "angular_velocity": angular_velocity,
                "status": "active" if missile_active is None or (len(missile_active) > 0 and missile_active[0]) else "destroyed"
            }
        
        # Log the timestep
        self.episode_logger.log_step(
            t=self.current_episode_time,
            interceptor_state=interceptor_state,
            threat_state=threat_state,
            events=None  # Could add events like "lock_acquired" later
        )
    
    def _log_episode_end(self, info: Dict[str, Any]):
        """Log episode end summary."""
        if not self.episode_logger:
            return
        
        # Determine outcome from info
        outcome = "timeout"  # Default
        miss_distance = None
        
        if "episode" in info:
            episode_info = info["episode"]
            if "success" in episode_info:
                outcome = "hit" if episode_info["success"] else "miss"
            if "miss_distance" in episode_info:
                miss_distance = episode_info["miss_distance"]
        
        # End the episode
        self.episode_logger.end_episode(
            outcome=outcome,
            final_time=self.current_episode_time,
            miss_distance=miss_distance,
            impact_time=self.current_episode_time if outcome == "hit" else None,
            notes=f"Episode ended after {self.episode_step_count} steps"
        )


def make_fast_sim_env(scenario_name: str = "easy", 
                     config_path: Optional[str] = None,
                     enable_episode_logging: bool = False,
                     episode_log_dir: Optional[str] = None,
                     **kwargs) -> FastSimEnv:
    """
    Factory function to create FastSimEnv with scenario configuration.
    
    Args:
        scenario_name: Name of scenario to load
        config_path: Path to configuration file
        enable_episode_logging: Whether to enable episode logging
        episode_log_dir: Directory for episode logs
        **kwargs: Additional arguments for FastSimEnv
        
    Returns:
        Configured FastSimEnv instance
    """
    # Load configuration and scenario
    config_loader = get_config(config_path)
    scenario_loader = get_scenario_loader()
    
    # Create environment configuration for scenario
    env_config = scenario_loader.create_environment_config(
        scenario_name, config_loader._config
    )
    
    # Create fast sim environment
    env = FastSimEnv(
        config=env_config,
        scenario_name=scenario_name,
        enable_episode_logging=enable_episode_logging,
        episode_log_dir=episode_log_dir,
        **kwargs
    )
    
    return env


if __name__ == "__main__":
    # Example usage and testing
    print("Testing FastSimEnv...")
    
    # Create environment
    env = make_fast_sim_env("easy")
    
    print(f"Environment created with configuration:")
    stats = env.get_performance_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test basic functionality
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step completed - reward: {reward:.3f}, terminated: {terminated}")
    
    env.close()
    print("FastSimEnv test completed successfully!")