"""
Hierarchical Reinforcement Learning (HRL) module for Hlynr Intercept.

This module provides a 2-level hierarchy for missile interception:
- High-level Selector: Chooses options (SEARCH, TRACK, TERMINAL)
- Low-level Specialists: Execute continuous actions for each option

Phases 2-3 Complete:
- Phase 2: Core HRL modules (option manager, policies, observation abstraction)
- Phase 3: Reward decomposition, wrappers, hierarchical environment factory
"""

__version__ = "0.3.0"

# Public API exports
from .option_definitions import Option, OPTION_METADATA, FORCED_TRANSITION_THRESHOLDS
from .observation_abstraction import abstract_observation, extract_env_state_for_transitions
from .option_manager import OptionManager
from .specialist_policies import SpecialistPolicy, SearchSpecialist, TrackSpecialist, TerminalSpecialist
from .selector_policy import SelectorPolicy
from .manager import HierarchicalManager, HRLState
from .wrappers import HRLActionWrapper
from .hierarchical_env import make_hrl_env, create_hrl_env
from .reward_decomposition import (
    compute_strategic_reward,
    compute_tactical_reward,
    compute_search_reward,
    compute_track_reward,
    compute_terminal_reward,
    compute_combined_reward,
)

__all__ = [
    "Option",
    "OPTION_METADATA",
    "FORCED_TRANSITION_THRESHOLDS",
    "abstract_observation",
    "extract_env_state_for_transitions",
    "OptionManager",
    "SpecialistPolicy",
    "SearchSpecialist",
    "TrackSpecialist",
    "TerminalSpecialist",
    "SelectorPolicy",
    "HierarchicalManager",
    "HRLState",
    "HRLActionWrapper",
    "make_hrl_env",
    "create_hrl_env",
    "compute_strategic_reward",
    "compute_tactical_reward",
    "compute_search_reward",
    "compute_track_reward",
    "compute_terminal_reward",
    "compute_combined_reward",
]
