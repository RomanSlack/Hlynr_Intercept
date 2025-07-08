"""
Curriculum learning system for AegisIntercept Phase 3.

This package provides automatic curriculum progression based on training performance,
with configurable difficulty tiers and scenario generation.
"""

from .curriculum_manager import CurriculumManager
from .scenario_generator import ScenarioGenerator

__all__ = ["CurriculumManager", "ScenarioGenerator"]