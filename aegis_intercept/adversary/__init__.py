"""
Advanced adversary system for AegisIntercept Phase 3.

This package provides intelligent adversary behaviors including evasion maneuvers,
guidance systems, and adaptive tactics.
"""

from .adversary_controller import AdversaryController
from .evasion_behaviors import EvasionBehaviors
from .enhanced_adversary import EnhancedAdversary

__all__ = ["AdversaryController", "EvasionBehaviors", "EnhancedAdversary"]