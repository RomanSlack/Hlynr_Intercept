"""
Main HRL environment wrapper factory.

Phase 1 stub: Creates HRL-wrapped environment with no-op specialists.
"""
import gymnasium as gym
from typing import Optional, Dict, Any

from .manager import HierarchicalManager
from .wrappers import HRLActionWrapper


def make_hrl_env(
    base_env: Optional[gym.Env] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> gym.Env:
    """
    Create HRL-enabled environment (stub).

    Args:
        base_env: Base InterceptEnvironment (or None to create from config)
        cfg: Configuration dictionary

    Returns:
        Wrapped environment with HRL manager
    """
    if base_env is None:
        raise ValueError("base_env must be provided (stub implementation)")

    # Extract HRL config
    if cfg is None:
        cfg = {}
    hrl_cfg = cfg.get('hrl', {})

    # Create HRL manager with stub specialists
    decision_interval = hrl_cfg.get('decision_interval_steps', 100)
    manager = HierarchicalManager(
        action_dim=6,  # 3D thrust + 3D angular
        decision_interval=decision_interval,
        enable_forced_transitions=True,
    )

    # Wrap environment
    wrapped_env = HRLActionWrapper(
        env=base_env,
        manager=manager,
        return_hrl_info=True,
    )

    return wrapped_env
