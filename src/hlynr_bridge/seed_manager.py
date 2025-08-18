"""
CRITICAL: Comprehensive seed management for deterministic inference.

This module handles all aspects of deterministic seed management per PRP requirements:
- SEED environment variable handling
- Random, numpy, torch seed setting
- Single-thread enforcement for PyTorch
- Seed validation and storage
"""

import os
import random
import logging
from typing import Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)


class SeedManager:
    """
    Manages deterministic seeding for reproducible inference.
    
    Ensures all random number generators are seeded consistently
    and PyTorch operates in single-thread deterministic mode.
    """
    
    def __init__(self):
        self._current_seed: Optional[int] = None
        self._is_deterministic_mode_set = False
    
    def set_deterministic_seeds(self, seed: Optional[int] = None) -> int:
        """
        Set deterministic seeds for all random number generators.
        
        Args:
            seed: Seed value (if None, loads from SEED env var or uses default)
            
        Returns:
            The seed value that was set
        """
        # Get seed from parameter, environment, or default
        if seed is None:
            seed = self._get_seed_from_env()
        
        # Validate seed
        if not isinstance(seed, int) or seed < 0:
            raise ValueError(f"Seed must be a non-negative integer, got: {seed}")
        
        # Set all seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Set PyTorch to deterministic single-thread mode
        torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Enable deterministic debug mode for development
        try:
            torch.set_deterministic_debug_mode(1)
        except AttributeError:
            # Older PyTorch versions don't have this
            pass
        
        self._current_seed = seed
        self._is_deterministic_mode_set = True
        
        logger.info(f"Set deterministic seeds: {seed}")
        logger.info("  - random.seed set")
        logger.info("  - numpy.random.seed set") 
        logger.info("  - torch.manual_seed set")
        logger.info("  - torch.set_num_threads(1)")
        logger.info("  - torch deterministic mode enabled")
        
        return seed
    
    def _get_seed_from_env(self) -> int:
        """Get seed from SEED environment variable or use default."""
        seed_str = os.getenv('SEED', '42')  # Default seed is 42
        
        try:
            seed = int(seed_str)
            if seed < 0:
                raise ValueError("Seed must be non-negative")
            return seed
        except ValueError as e:
            logger.warning(f"Invalid SEED environment variable '{seed_str}': {e}")
            logger.warning("Using default seed: 42")
            return 42
    
    def get_current_seed(self) -> Optional[int]:
        """Get the currently set seed."""
        return self._current_seed
    
    def is_deterministic_mode_active(self) -> bool:
        """Check if deterministic mode has been activated."""
        return self._is_deterministic_mode_set
    
    def validate_deterministic_setup(self) -> bool:
        """
        Validate that deterministic setup is correctly configured.
        
        Returns:
            True if all deterministic settings are correct
        """
        checks = []
        
        # Check if seed is set
        seed_ok = self._current_seed is not None
        checks.append(("Seed set", seed_ok))
        
        # Check PyTorch thread count
        torch_threads_ok = torch.get_num_threads() == 1
        checks.append(("PyTorch single thread", torch_threads_ok))
        
        # Check deterministic mode flag
        det_mode_ok = self._is_deterministic_mode_set
        checks.append(("Deterministic mode set", det_mode_ok))
        
        # Check CUDNN settings (if CUDA available)
        if torch.cuda.is_available():
            cudnn_det_ok = torch.backends.cudnn.deterministic
            cudnn_bench_ok = not torch.backends.cudnn.benchmark
            checks.append(("CUDNN deterministic", cudnn_det_ok))
            checks.append(("CUDNN benchmark disabled", cudnn_bench_ok))
        
        # Log validation results
        all_ok = all(result for _, result in checks)
        
        if all_ok:
            logger.info("✅ Deterministic setup validation: PASS")
        else:
            logger.warning("⚠️ Deterministic setup validation: FAIL")
        
        for check_name, result in checks:
            status = "✅" if result else "❌"
            logger.info(f"  {status} {check_name}")
        
        return all_ok


# Global seed manager instance
_seed_manager = None


def get_seed_manager() -> SeedManager:
    """Get global seed manager instance."""
    global _seed_manager
    
    if _seed_manager is None:
        _seed_manager = SeedManager()
    
    return _seed_manager


def set_deterministic_seeds(seed: Optional[int] = None) -> int:
    """
    CRITICAL: Set deterministic seeds for all random number generators.
    
    This is the main function to call for setting up deterministic inference.
    
    Args:
        seed: Seed value (if None, loads from SEED env var)
        
    Returns:
        The seed value that was set
    """
    manager = get_seed_manager()
    return manager.set_deterministic_seeds(seed)


def get_current_seed() -> Optional[int]:
    """Get the currently active seed."""
    manager = get_seed_manager()
    return manager.get_current_seed()


def validate_deterministic_setup() -> bool:
    """Validate that deterministic setup is correctly configured."""
    manager = get_seed_manager()
    return manager.validate_deterministic_setup()


def enforce_single_thread_inference():
    """
    Enforce single-thread mode for deterministic inference.
    
    This should be called before each policy inference to ensure determinism.
    """
    torch.set_num_threads(1)


# Environment variable validation
def validate_seed_env_var() -> bool:
    """Validate SEED environment variable if present."""
    seed_str = os.getenv('SEED')
    
    if seed_str is None:
        logger.info("SEED environment variable not set, will use default")
        return True
    
    try:
        seed = int(seed_str)
        if seed < 0:
            logger.error(f"SEED environment variable must be non-negative, got: {seed}")
            return False
        
        logger.info(f"SEED environment variable validated: {seed}")
        return True
        
    except ValueError:
        logger.error(f"SEED environment variable must be an integer, got: '{seed_str}'")
        return False


if __name__ == "__main__":
    # Test seed management
    print("Seed Management Test")
    print("=" * 40)
    
    # Test environment variable validation
    valid = validate_seed_env_var()
    print(f"Environment validation: {'PASS' if valid else 'FAIL'}")
    
    # Test seed setting
    seed = set_deterministic_seeds()
    print(f"Set seed: {seed}")
    
    # Test validation
    setup_valid = validate_deterministic_setup()
    print(f"Setup validation: {'PASS' if setup_valid else 'FAIL'}")
    
    # Test reproducibility
    print("\\nTesting reproducibility:")
    
    # Generate some random numbers
    random_val_1 = random.random()
    numpy_val_1 = np.random.random()
    torch_val_1 = torch.rand(1).item()
    
    print(f"First run - Random: {random_val_1:.6f}, NumPy: {numpy_val_1:.6f}, Torch: {torch_val_1:.6f}")
    
    # Reset with same seed
    set_deterministic_seeds(seed)
    
    random_val_2 = random.random()
    numpy_val_2 = np.random.random()
    torch_val_2 = torch.rand(1).item()
    
    print(f"Second run - Random: {random_val_2:.6f}, NumPy: {numpy_val_2:.6f}, Torch: {torch_val_2:.6f}")
    
    # Check reproducibility
    reproducible = (
        abs(random_val_1 - random_val_2) < 1e-10 and
        abs(numpy_val_1 - numpy_val_2) < 1e-10 and 
        abs(torch_val_1 - torch_val_2) < 1e-10
    )
    
    print(f"Reproducibility test: {'PASS' if reproducible else 'FAIL'}")
    
    print("\\nSeed management test completed!")