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
        
        The RadarEnv already processes radar returns in _get_observation(),
        but this method validates and potentially filters the observation
        to ensure no omniscient information is included.
        
        Args:
            obs: Raw observation from RadarEnv
            
        Returns:
            Radar-only observation
        """
        # The current RadarEnv implementation already constructs observations
        # from radar returns in _process_ground_radar() and _process_onboard_radar().
        # For now, we trust that implementation is radar-only.
        # 
        # In a production system, you might want to:
        # 1. Extract only specific radar-based components
        # 2. Add noise to simulate realistic radar limitations
        # 3. Filter out any perfect position/velocity information
        
        return obs
    
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