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
except ImportError:
    # Fallback for direct execution
    from radar_env import RadarEnv
    from config import get_config
    from scenarios import get_scenario_loader


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
                 num_interceptors: Optional[int] = None):
        """
        Initialize FastSimEnv wrapper.
        
        Args:
            config: Environment configuration dictionary
            scenario_name: Name of scenario to load
            num_missiles: Number of missiles (overrides config)
            num_interceptors: Number of interceptors (overrides config)
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
        
        # Flag to ensure we're radar-only
        self._radar_only = True
        
        # Update observation space for radar-only observations
        self._setup_radar_observation_space()
    
    def _setup_radar_observation_space(self):
        """Setup observation space for radar-only observations."""
        # Calculate expected radar-only observation size
        num_missiles = self.env.num_missiles
        num_interceptors = self.env.num_interceptors
        
        # Components in radar-only observation:
        # 1. Interceptor essential state: 4 * num_interceptors (fuel, status, target_id, distance)
        # 2. Ground radar returns: 4 * num_missiles (detected_x, detected_y, confidence, range)
        # 3. Onboard radar returns: 3 * num_interceptors (detections, bearing, range)
        # 4. Environmental data: 6 (wind, atmosphere, etc.)
        
        interceptor_essential_dim = 4 * num_interceptors
        ground_radar_dim = 4 * num_missiles
        onboard_radar_dim = 3 * num_interceptors
        env_data_dim = 6
        
        radar_obs_dim = interceptor_essential_dim + ground_radar_dim + onboard_radar_dim + env_data_dim
        
        # Update observation space to match radar-only observations
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(radar_obs_dim,),
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
        
        # Ensure observation is radar-only
        radar_obs = self._ensure_radar_only_observation(obs)
        
        return radar_obs, reward, terminated, truncated, info
    
    def _ensure_radar_only_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Ensure observation contains only radar-based information.
        
        This method filters the raw observation from RadarEnv to extract only
        radar-based components and exclude omniscient information like perfect
        missile/interceptor positions and velocities.
        
        Args:
            obs: Raw observation from RadarEnv
            
        Returns:
            Radar-only observation containing only radar returns and essential state
        """
        # Calculate component dimensions based on environment configuration
        num_missiles = self.env.num_missiles
        num_interceptors = self.env.num_interceptors
        
        # RadarEnv observation structure (from _get_observation):
        # 1. Missile data: 6 * num_missiles (pos_x, pos_y, vel_x, vel_y, target_x, target_y)
        # 2. Interceptor data: 8 * num_interceptors (pos_x, pos_y, vel_x, vel_y, fuel, status, target_id, distance)
        # 3. Ground radar returns: 4 * num_missiles (detected_x, detected_y, confidence, range)
        # 4. Onboard radar returns: 3 * num_interceptors (detections, bearing, range)
        # 5. Environmental data: 6 (wind, atmosphere, etc.)
        # 6. Relative positioning: 3 * num_missiles * num_interceptors (distance, bearing, rel_velocity)
        
        missile_data_dim = 6 * num_missiles
        interceptor_data_dim = 8 * num_interceptors
        ground_radar_dim = 4 * num_missiles
        onboard_radar_dim = 3 * num_interceptors
        env_data_dim = 6
        relative_pos_dim = 3 * num_missiles * num_interceptors
        
        # Calculate start indices for each component
        start_idx = 0
        
        # Skip omniscient missile data (direct positions/velocities)
        start_idx += missile_data_dim
        
        # Extract limited interceptor state (fuel, status) but skip perfect positions/velocities
        interceptor_start = start_idx
        interceptor_essential = []
        for i in range(num_interceptors):
            interceptor_offset = interceptor_start + i * 8
            # Extract only fuel (idx 4), status (idx 5), target_id (idx 6), distance (idx 7)
            # Skip perfect position (idx 0,1) and velocity (idx 2,3)
            if interceptor_offset + 7 < len(obs):
                interceptor_essential.extend([
                    obs[interceptor_offset + 4],  # fuel
                    obs[interceptor_offset + 5],  # status
                    obs[interceptor_offset + 6],  # target_id
                    obs[interceptor_offset + 7]   # distance to target
                ])
        
        start_idx += interceptor_data_dim
        
        # Extract ground radar returns (this is proper radar data with noise)
        ground_radar_start = start_idx
        ground_radar_end = ground_radar_start + ground_radar_dim
        ground_radar_data = obs[ground_radar_start:ground_radar_end] if ground_radar_end <= len(obs) else []
        
        start_idx += ground_radar_dim
        
        # Extract onboard radar returns (this is proper radar data)
        onboard_radar_start = start_idx
        onboard_radar_end = onboard_radar_start + onboard_radar_dim
        onboard_radar_data = obs[onboard_radar_start:onboard_radar_end] if onboard_radar_end <= len(obs) else []
        
        start_idx += onboard_radar_dim
        
        # Extract environmental data (acceptable as it's external conditions)
        env_data_start = start_idx
        env_data_end = env_data_start + env_data_dim
        env_data = obs[env_data_start:env_data_end] if env_data_end <= len(obs) else []
        
        # Skip relative positioning as it's derived from perfect position data
        
        # Construct radar-only observation
        radar_obs_components = []
        
        # Add interceptor essential state (non-positional)
        if interceptor_essential:
            radar_obs_components.extend(interceptor_essential)
        
        # Add ground radar returns
        if len(ground_radar_data) > 0:
            radar_obs_components.extend(ground_radar_data)
        
        # Add onboard radar returns
        if len(onboard_radar_data) > 0:
            radar_obs_components.extend(onboard_radar_data)
        
        # Add environmental data
        if len(env_data) > 0:
            radar_obs_components.extend(env_data)
        
        # Convert to numpy array and ensure consistent size
        if radar_obs_components:
            radar_obs = np.array(radar_obs_components, dtype=np.float32)
        else:
            # Fallback to minimal observation if extraction failed
            radar_obs = np.zeros(10, dtype=np.float32)
        
        # Ensure observation is not empty and has reasonable size
        if len(radar_obs) < 5:
            # Pad with zeros if too small
            radar_obs = np.pad(radar_obs, (0, 5 - len(radar_obs)))
        
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


def make_fast_sim_env(scenario_name: str = "easy", 
                     config_path: Optional[str] = None,
                     **kwargs) -> FastSimEnv:
    """
    Factory function to create FastSimEnv with scenario configuration.
    
    Args:
        scenario_name: Name of scenario to load
        config_path: Path to configuration file
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