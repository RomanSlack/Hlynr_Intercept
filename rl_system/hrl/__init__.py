"""
Hierarchical Reinforcement Learning (HRL) module for Hlynr Intercept.

This module provides a 2-level hierarchy for missile interception:
- High-level Selector: Chooses options (SEARCH, TRACK, TERMINAL)
- Low-level Specialists: Execute continuous actions for each option

Phase 1: Stub implementation (scaffolding only, no training logic yet)
"""

__version__ = "0.1.0"

# Public API exports
from .option_definitions import Option, OPTION_METADATA, FORCED_TRANSITION_THRESHOLDS
from .observation_abstraction import abstract_observation, extract_env_state_for_transitions
from .option_manager import OptionManager
from .specialist_policies import SpecialistPolicy, SearchSpecialist, TrackSpecialist, TerminalSpecialist
from .selector_policy import SelectorPolicy
from .manager import HierarchicalManager, HRLState
from .wrappers import HRLActionWrapper
from .hierarchical_env import make_hrl_env

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
]
