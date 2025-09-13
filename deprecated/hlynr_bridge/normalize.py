"""
VecNormalize management with versioned statistics tracking.

This module provides deterministic observation normalization with versioned statistics
to ensure reproducible inference across different sessions.
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize, VecEnv
from dataclasses import dataclass
import json
import hashlib
import torch

logger = logging.getLogger(__name__)


@dataclass
class VecNormalizeInfo:
    """Information about VecNormalize statistics."""
    stats_id: str
    obs_version: str
    creation_time: float
    source_checkpoint: str
    obs_mean: List[float]
    obs_var: List[float]
    obs_count: int
    description: str = ""


class VecNormalizeManager:
    """
    Manager for versioned VecNormalize statistics.
    
    Handles loading, saving, and tracking of normalization statistics
    with deterministic IDs for reproducible inference.
    """
    
    def __init__(self, stats_dir: str = "vecnorm_stats"):
        """
        Initialize VecNormalize manager.
        
        Args:
            stats_dir: Directory to store vecnorm statistics
        """
        self.stats_dir = Path(stats_dir)
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        
        # Currently loaded normalization
        self.current_vecnorm: Optional[VecNormalize] = None
        self.current_info: Optional[VecNormalizeInfo] = None
        
        # Statistics registry
        self.registry_path = self.stats_dir / "registry.json"
        self.registry = self._load_registry()
        
        logger.info(f"VecNormalize manager initialized with stats_dir: {self.stats_dir}")
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load the statistics registry."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save the statistics registry."""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def _generate_stats_id(self, checkpoint_path: str, obs_version: str) -> str:
        """
        Generate deterministic statistics ID based on checkpoint and obs version.
        
        Args:
            checkpoint_path: Path to the source checkpoint
            obs_version: Observation version identifier
            
        Returns:
            Unique statistics ID
        """
        # Use checkpoint filename and obs_version to generate ID
        checkpoint_name = Path(checkpoint_path).stem
        content = f"{checkpoint_name}_{obs_version}"
        hash_digest = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"vecnorm_{checkpoint_name}_{obs_version}_{hash_digest}"
    
    def register_vecnormalize(self, 
                            vecnorm: VecNormalize,
                            checkpoint_path: str,
                            obs_version: str,
                            stats_id: Optional[str] = None,
                            description: str = "") -> str:
        """
        Register and save VecNormalize statistics.
        
        Args:
            vecnorm: VecNormalize instance to register
            checkpoint_path: Source checkpoint path
            obs_version: Observation version
            stats_id: Optional custom stats ID
            description: Optional description
            
        Returns:
            Generated or provided stats ID
        """
        if stats_id is None:
            stats_id = self._generate_stats_id(checkpoint_path, obs_version)
        
        # Extract statistics
        obs_mean = vecnorm.obs_rms.mean.tolist() if vecnorm.obs_rms else []
        obs_var = vecnorm.obs_rms.var.tolist() if vecnorm.obs_rms else []
        obs_count = int(vecnorm.obs_rms.count) if vecnorm.obs_rms else 0
        
        # Create info object
        info = VecNormalizeInfo(
            stats_id=stats_id,
            obs_version=obs_version,
            creation_time=time.time(),
            source_checkpoint=str(checkpoint_path),
            obs_mean=obs_mean,
            obs_var=obs_var,
            obs_count=obs_count,
            description=description
        )
        
        # Save VecNormalize object
        vecnorm_path = self.stats_dir / f"{stats_id}.pkl"
        vecnorm.save(str(vecnorm_path))
        
        # Save info to registry
        self.registry[stats_id] = {
            'stats_id': stats_id,
            'obs_version': obs_version,
            'creation_time': info.creation_time,
            'source_checkpoint': info.source_checkpoint,
            'obs_mean': obs_mean,
            'obs_var': obs_var,
            'obs_count': obs_count,
            'description': description,
            'file_path': str(vecnorm_path)
        }
        
        self._save_registry()
        
        logger.info(f"Registered VecNormalize stats: {stats_id} (obs_version: {obs_version})")
        return stats_id
    
    def load_vecnormalize(self, 
                         stats_id: str, 
                         env: VecEnv,
                         obs_version: Optional[str] = None) -> Tuple[VecNormalize, VecNormalizeInfo]:
        """
        Load VecNormalize statistics by ID.
        
        Args:
            stats_id: Statistics ID to load
            env: Vectorized environment to wrap
            obs_version: Optional observation version check
            
        Returns:
            (VecNormalize instance, info object)
        """
        if stats_id not in self.registry:
            raise ValueError(f"VecNormalize stats ID not found: {stats_id}")
        
        registry_entry = self.registry[stats_id]
        
        # Check observation version compatibility
        if obs_version and registry_entry['obs_version'] != obs_version:
            raise ValueError(
                f"Observation version mismatch: requested {obs_version}, "
                f"but stats {stats_id} is for {registry_entry['obs_version']}"
            )
        
        # Load VecNormalize
        vecnorm_path = Path(registry_entry['file_path'])
        if not vecnorm_path.exists():
            raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_path}")
        
        vecnorm = VecNormalize.load(str(vecnorm_path), env)
        
        # Set to evaluation mode (frozen statistics)
        vecnorm.training = False
        vecnorm.norm_reward = False
        
        # Create info object
        info = VecNormalizeInfo(
            stats_id=registry_entry['stats_id'],
            obs_version=registry_entry['obs_version'],
            creation_time=registry_entry['creation_time'],
            source_checkpoint=registry_entry['source_checkpoint'],
            obs_mean=registry_entry['obs_mean'],
            obs_var=registry_entry['obs_var'],
            obs_count=registry_entry['obs_count'],
            description=registry_entry.get('description', '')
        )
        
        # Store as current
        self.current_vecnorm = vecnorm
        self.current_info = info
        
        logger.info(f"Loaded VecNormalize stats: {stats_id} (count: {info.obs_count})")
        return vecnorm, info
    
    def get_current_info(self) -> Optional[VecNormalizeInfo]:
        """Get information about currently loaded VecNormalize."""
        return self.current_info
    
    def list_available_stats(self, obs_version: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available VecNormalize statistics.
        
        Args:
            obs_version: Optional filter by observation version
            
        Returns:
            List of available statistics info
        """
        stats_list = []
        
        for stats_id, entry in self.registry.items():
            if obs_version is None or entry['obs_version'] == obs_version:
                stats_list.append({
                    'stats_id': stats_id,
                    'obs_version': entry['obs_version'],
                    'creation_time': entry['creation_time'],
                    'source_checkpoint': entry['source_checkpoint'],
                    'description': entry.get('description', ''),
                    'obs_count': entry['obs_count']
                })
        
        return sorted(stats_list, key=lambda x: x['creation_time'], reverse=True)
    
    def compute_clip_fractions(self, 
                              observations: np.ndarray,
                              epsilon: float = 1e-8) -> Dict[str, float]:
        """
        Compute observation clipping fractions for diagnostics.
        
        Args:
            observations: Raw observations to analyze
            epsilon: Small value to avoid division by zero
            
        Returns:
            Dictionary with low and high clip fractions
        """
        if self.current_vecnorm is None or self.current_vecnorm.obs_rms is None:
            return {'low': 0.0, 'high': 0.0}
        
        # Get normalization parameters
        mean = self.current_vecnorm.obs_rms.mean
        var = self.current_vecnorm.obs_rms.var
        std = np.sqrt(var + epsilon)
        
        # Normalize observations
        normalized_obs = (observations - mean) / std
        
        # Apply clipping (VecNormalize default is ±10)
        clip_range = self.current_vecnorm.clip_obs
        
        # Count clipped values
        total_elements = normalized_obs.size
        low_clipped = np.sum(normalized_obs < -clip_range)
        high_clipped = np.sum(normalized_obs > clip_range)
        
        return {
            'low': float(low_clipped) / total_elements,
            'high': float(high_clipped) / total_elements
        }
    
    def validate_observation_shape(self, observations: np.ndarray) -> bool:
        """
        Validate that observation shape matches loaded VecNormalize.
        
        Args:
            observations: Observations to validate
            
        Returns:
            True if shape is valid
        """
        if self.current_vecnorm is None:
            return True  # No normalization loaded
        
        expected_shape = self.current_vecnorm.observation_space.shape
        if observations.shape[-len(expected_shape):] != expected_shape:
            logger.error(
                f"Observation shape mismatch: got {observations.shape}, "
                f"expected {expected_shape}"
            )
            return False
        
        return True
    
    def create_from_legacy_file(self, 
                               legacy_path: str,
                               env: VecEnv,
                               obs_version: str,
                               stats_id: Optional[str] = None,
                               description: str = "Imported from legacy file") -> str:
        """
        Create versioned stats from legacy vec_normalize.pkl file.
        
        Args:
            legacy_path: Path to legacy vec_normalize.pkl
            env: Environment to wrap
            obs_version: Observation version for these stats
            stats_id: Optional custom stats ID
            description: Description for the imported stats
            
        Returns:
            Generated stats ID
        """
        legacy_path = Path(legacy_path)
        if not legacy_path.exists():
            raise FileNotFoundError(f"Legacy VecNormalize file not found: {legacy_path}")
        
        # Load legacy VecNormalize
        vecnorm = VecNormalize.load(str(legacy_path), env)
        
        # Generate stats ID if not provided
        if stats_id is None:
            checkpoint_dir = legacy_path.parent
            checkpoint_name = checkpoint_dir.name
            stats_id = self._generate_stats_id(str(checkpoint_dir), obs_version)
        
        # Register the VecNormalize
        return self.register_vecnormalize(
            vecnorm=vecnorm,
            checkpoint_path=str(legacy_path.parent),
            obs_version=obs_version,
            stats_id=stats_id,
            description=description
        )
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary of the VecNormalize registry."""
        return {
            'total_stats': len(self.registry),
            'obs_versions': list(set(entry['obs_version'] for entry in self.registry.values())),
            'current_loaded': self.current_info.stats_id if self.current_info else None,
            'registry_path': str(self.registry_path),
            'stats_dir': str(self.stats_dir)
        }


# Global manager instance
_vecnorm_manager = None


def get_vecnorm_manager(stats_dir: str = "vecnorm_stats") -> VecNormalizeManager:
    """Get global VecNormalize manager instance."""
    global _vecnorm_manager
    
    if _vecnorm_manager is None:
        _vecnorm_manager = VecNormalizeManager(stats_dir)
    
    return _vecnorm_manager


def load_vecnormalize_by_id(stats_id: str, 
                           env: VecEnv,
                           obs_version: Optional[str] = None) -> Tuple[VecNormalize, VecNormalizeInfo]:
    """Load VecNormalize by statistics ID."""
    manager = get_vecnorm_manager()
    return manager.load_vecnormalize(stats_id, env, obs_version)


def load_vecnorm(stats_id: str, env: VecEnv = None) -> VecNormalize:
    """
    BLOCKER: Simple load function for VecNormalize by ID.
    
    Loads frozen VecNormalize stats by ID in eval-only mode for deterministic inference.
    
    Args:
        stats_id: Statistics ID to load
        env: Optional vectorized environment (if None, uses dummy env)
        
    Returns:
        VecNormalize instance in eval-only mode
    """
    manager = get_vecnorm_manager()
    
    # Create dummy env if not provided for the load
    if env is None:
        from stable_baselines3.common.vec_env import DummyVecEnv
        from unittest.mock import MagicMock
        
        def make_dummy_env():
            mock_env = MagicMock()
            mock_env.observation_space.shape = (34,)  # Default obs space
            mock_env.action_space.shape = (6,)
            return mock_env
        
        env = DummyVecEnv([make_dummy_env])
    
    vec_normalize, _ = manager.load_vecnormalize(stats_id, env)
    
    # Ensure deterministic eval-only mode
    vec_normalize.training = False
    vec_normalize.norm_reward = False
    
    logger.info(f"Loaded VecNormalize {stats_id} in eval-only mode")
    return vec_normalize


def set_deterministic_inference_mode():
    """
    BLOCKER: Set deterministic mode for policy inference.
    
    Ensures torch uses single thread and deterministic operations for inference.
    """
    torch.set_num_threads(1)
    torch.set_deterministic_debug_mode(1)  # Enable deterministic mode checking
    logger.info("Set deterministic inference mode: single thread, deterministic debug on")


def register_vecnormalize_from_checkpoint(checkpoint_dir: str,
                                        env: VecEnv,
                                        obs_version: str,
                                        stats_id: Optional[str] = None) -> str:
    """Register VecNormalize from checkpoint directory."""
    manager = get_vecnorm_manager()
    
    # Look for vec_normalize.pkl or vec_normalize_final.pkl
    checkpoint_path = Path(checkpoint_dir)
    vecnorm_files = [
        checkpoint_path / "vec_normalize_final.pkl",
        checkpoint_path / "vec_normalize.pkl"
    ]
    
    vecnorm_path = None
    for path in vecnorm_files:
        if path.exists():
            vecnorm_path = path
            break
    
    if vecnorm_path is None:
        raise FileNotFoundError(f"No VecNormalize file found in {checkpoint_dir}")
    
    return manager.create_from_legacy_file(
        legacy_path=str(vecnorm_path),
        env=env,
        obs_version=obs_version,
        stats_id=stats_id,
        description=f"From checkpoint: {checkpoint_path.name}"
    )