"""
Scenario template system for Phase 4 RL training.

This module provides scenario management for different difficulty levels
and training configurations.
"""

from .scenarios import ScenarioLoader, get_scenario_loader, reset_scenario_loader

__all__ = ['ScenarioLoader', 'get_scenario_loader', 'reset_scenario_loader']