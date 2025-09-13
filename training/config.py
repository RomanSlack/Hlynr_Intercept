"""
Shared configuration management for Phase 4 RL training and inference.

This module provides centralized configuration loading and management for both
Roman's sandbox environment and the RL training pipeline. Supports YAML files
and programmatic configuration access.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Central configuration loader for Phase 4 RL components."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config if config is not None else self._get_default_config()
        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration values."""
        return {
            # Environment configuration
            "environment": {
                "num_missiles": 1,
                "num_interceptors": 1,
                "observation_dim": 34,
                "action_dim": 6,
                "max_episode_steps": 1000,
                "render_mode": "human"
            },
            
            # Radar configuration
            "radar": {
                "range": 1000.0,
                "noise_level": 0.05,
                "update_rate": 10,
                "ground_radar_positions": [[0, 0], [500, 500]],
                "onboard_radar_range": 200.0
            },
            
            # Spawn configuration
            "spawn": {
                "missile_spawn_area": [[-100, -100], [100, 100]],
                "interceptor_spawn_area": [[400, 400], [600, 600]],
                "target_area": [[800, 800], [1000, 1000]]
            },
            
            # Wind and environmental conditions
            "environment_conditions": {
                "wind_speed": 5.0,
                "wind_direction": 0.0,
                "wind_variability": 0.1,
                "atmospheric_density": 1.0
            },
            
            # Training configuration
            "training": {
                "algorithm": "PPO",
                "total_timesteps": 1000000,
                "checkpoint_interval": 10000,
                "n_envs": 8,
                "learning_rate": 3e-4,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5
            },
            
            # Checkpointing and logging
            "checkpointing": {
                "checkpoint_dir": "checkpoints",
                "log_dir": "logs",
                "tensorboard_dir": "logs/tensorboard",
                "save_replay_buffer": False,
                "save_vec_normalize": True
            },
            
            # Inference configuration  
            "inference": {
                "num_episodes": 100,
                "render": False,
                "real_time": False,
                "export_diagnostics": True,
                "video_recording": False
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "environment.num_missiles")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "environment.num_missiles")
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            new_config: Dictionary of new configuration values
        """
        self._deep_update(self._config, new_config)
    
    def _deep_update(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Recursively update nested dictionaries."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            path: Path to save file. If None, uses original config path.
        """
        save_path = Path(path) if path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        return {
            "num_missiles": self.get("environment.num_missiles"),
            "num_interceptors": self.get("environment.num_interceptors"),
            "observation_dim": self.get("environment.observation_dim"),
            "action_dim": self.get("environment.action_dim"),
            "max_episode_steps": self.get("environment.max_episode_steps"),
            "radar_config": self._config.get("radar", {}),
            "spawn_config": self._config.get("spawn", {}),
            "environment_conditions": self._config.get("environment_conditions", {})
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration."""
        return self._config.get("training", {})
    
    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference-specific configuration."""
        return self._config.get("inference", {})
    
    def __repr__(self) -> str:
        return f"ConfigLoader(config_path={self.config_path})"


# Global configuration instance
_global_config = None

def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Get global configuration instance.
    
    Args:
        config_path: Path to configuration file. Only used for first call.
        
    Returns:
        Global ConfigLoader instance
    """
    global _global_config
    if _global_config is None:
        _global_config = ConfigLoader(config_path)
    return _global_config

def reset_config() -> None:
    """Reset global configuration instance (mainly for testing)."""
    global _global_config
    _global_config = None