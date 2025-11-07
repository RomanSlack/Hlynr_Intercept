"""
Main HRL environment wrapper factory.

This module provides factory functions to create HRL-enabled environments with
proper configuration, specialist loading, and reward integration.
"""
import gymnasium as gym
from typing import Optional, Dict, Any, Union
from pathlib import Path

from .manager import HierarchicalManager
from .wrappers import HRLActionWrapper
from .selector_policy import SelectorPolicy
from .specialist_policies import SearchSpecialist, TrackSpecialist, TerminalSpecialist
from .option_definitions import Option, load_thresholds_from_config


def make_hrl_env(
    base_env: Optional[gym.Env] = None,
    cfg: Optional[Dict[str, Any]] = None,
    selector_path: Optional[str] = None,
    specialist_paths: Optional[Dict[Option, str]] = None,
    mode: str = 'training',
) -> gym.Env:
    """
    Create HRL-enabled environment with configurable specialists and selector.

    Args:
        base_env: Base InterceptEnvironment instance (required)
        cfg: Configuration dictionary with HRL settings
        selector_path: Path to trained selector model (None for rule-based or training)
        specialist_paths: Dict mapping Option to specialist model paths
        mode: Operating mode - 'training', 'inference', or 'pretrain'
            - 'training': Train selector with frozen specialists
            - 'inference': Evaluate trained HRL policy
            - 'pretrain': Train individual specialists (single specialist active)

    Returns:
        Wrapped environment with HRL manager

    Examples:
        # Training selector with frozen specialists:
        >>> env = make_hrl_env(
        ...     base_env=InterceptEnvironment(config),
        ...     cfg=config,
        ...     specialist_paths={
        ...         Option.SEARCH: 'checkpoints/hrl/specialists/search/model',
        ...         Option.TRACK: 'checkpoints/hrl/specialists/track/model',
        ...         Option.TERMINAL: 'checkpoints/hrl/specialists/terminal/model',
        ...     },
        ...     mode='training'
        ... )

        # Inference with trained HRL policy:
        >>> env = make_hrl_env(
        ...     base_env=InterceptEnvironment(config),
        ...     cfg=config,
        ...     selector_path='checkpoints/hrl/selector/model',
        ...     specialist_paths={...},
        ...     mode='inference'
        ... )

        # Pre-training search specialist:
        >>> env = make_hrl_env(
        ...     base_env=InterceptEnvironment(config),
        ...     cfg=config,
        ...     mode='pretrain'
        ... )
    """
    if base_env is None:
        raise ValueError("base_env must be provided")

    # Extract HRL config
    if cfg is None:
        cfg = {}
    hrl_cfg = cfg.get('hrl', {})

    # Get configuration parameters
    decision_interval = hrl_cfg.get('decision_interval_steps', 100)
    enable_forced = hrl_cfg.get('enable_forced_transitions', True)

    # Load forced transition thresholds from config
    if 'hrl' in hrl_cfg and 'forced_transitions' in hrl_cfg:
        # Use custom thresholds from config dict
        thresholds = hrl_cfg['forced_transitions']
    else:
        # Use default thresholds
        from .option_definitions import FORCED_TRANSITION_THRESHOLDS
        thresholds = FORCED_TRANSITION_THRESHOLDS

    # Create selector policy
    if selector_path is not None:
        selector_mode = 'model'
    else:
        selector_mode = 'rules'  # Default to rule-based

    selector = SelectorPolicy(
        obs_dim=7,  # Abstract state dimension
        n_options=3,  # SEARCH, TRACK, TERMINAL
        model_path=selector_path,
        mode=selector_mode,
    )

    # Create specialist policies
    specialists = {}
    if specialist_paths:
        # Load trained specialists from disk (with fallback to stubs)
        for option, path in specialist_paths.items():
            try:
                if option == Option.SEARCH:
                    specialists[option] = SearchSpecialist(model_path=path)
                elif option == Option.TRACK:
                    specialists[option] = TrackSpecialist(model_path=path)
                elif option == Option.TERMINAL:
                    specialists[option] = TerminalSpecialist(model_path=path)
            except FileNotFoundError:
                # Fall back to stub if model file not found
                if option == Option.SEARCH:
                    specialists[option] = SearchSpecialist()
                elif option == Option.TRACK:
                    specialists[option] = TrackSpecialist()
                elif option == Option.TERMINAL:
                    specialists[option] = TerminalSpecialist()

    # Ensure all three specialists exist (fill in missing ones with stubs)
    if Option.SEARCH not in specialists:
        specialists[Option.SEARCH] = SearchSpecialist()
    if Option.TRACK not in specialists:
        specialists[Option.TRACK] = TrackSpecialist()
    if Option.TERMINAL not in specialists:
        specialists[Option.TERMINAL] = TerminalSpecialist()

    # Create HRL manager
    manager = HierarchicalManager(
        selector=selector,
        specialists=specialists,
        action_dim=6,  # 3D thrust + 3D angular
        decision_interval=decision_interval,
        enable_forced_transitions=enable_forced,
        enable_hysteresis=True,
        enable_min_dwell=True,
    )

    # Wrap environment
    wrapped_env = HRLActionWrapper(
        env=base_env,
        manager=manager,
        return_hrl_info=True,
    )

    return wrapped_env


def create_hrl_env(
    base_env: Optional[gym.Env] = None,
    config_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> gym.Env:
    """
    Convenience function to create HRL environment from config file.

    This is a simpler interface for common use cases where configuration is loaded from YAML.

    Args:
        base_env: Base InterceptEnvironment instance (required)
        config_path: Path to YAML config file (optional, uses base_env's config if None)
        **kwargs: Additional arguments passed to make_hrl_env()

    Returns:
        Wrapped environment with HRL manager

    Examples:
        # Simple usage with existing environment:
        >>> from environment import InterceptEnvironment
        >>> base_env = InterceptEnvironment(config)
        >>> hrl_env = create_hrl_env(base_env=base_env)

        # Load config from file:
        >>> hrl_env = create_hrl_env(
        ...     base_env=base_env,
        ...     config_path='configs/hrl/hrl_base.yaml'
        ... )
    """
    if base_env is None:
        raise ValueError("base_env must be provided")

    # Load config if path provided
    cfg = None
    if config_path is not None:
        import yaml
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
    elif hasattr(base_env, 'config'):
        # Use environment's config if available
        cfg = base_env.config

    return make_hrl_env(base_env=base_env, cfg=cfg, **kwargs)
