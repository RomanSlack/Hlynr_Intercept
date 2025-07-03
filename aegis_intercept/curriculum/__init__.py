"""
Curriculum Learning Module for AegisIntercept Phase 3

This module provides curriculum learning capabilities for progressive
training from simple 3DOF scenarios to complex 6DOF intercept missions.
"""

from .curriculum_manager import (
    CurriculumManager,
    CurriculumPhase,
    AdvancementCriteria,
    PhaseConfig,
    ScenarioConfig,
    PerformanceMetrics,
    create_curriculum_manager,
    setup_curriculum_directories
)

__all__ = [
    'CurriculumManager',
    'CurriculumPhase', 
    'AdvancementCriteria',
    'PhaseConfig',
    'ScenarioConfig',
    'PerformanceMetrics',
    'create_curriculum_manager',
    'setup_curriculum_directories'
]