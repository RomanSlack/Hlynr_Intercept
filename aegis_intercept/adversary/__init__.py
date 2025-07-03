"""
Adversary Module for AegisIntercept Phase 3

This module provides enhanced adversary behavior for 6DOF missile simulations,
including sophisticated evasive maneuvers, threat assessment, and adaptive AI.
"""

from .enhanced_adversary import (
    EnhancedAdversary,
    AdversaryConfig,
    AdversaryBehaviorModel,
    EvasionPattern,
    ThreatLevel,
    GuidanceMode,
    ThreatAssessment,
    EvasionConfig,
    create_default_adversary_config,
    create_enhanced_adversary
)

__all__ = [
    'EnhancedAdversary',
    'AdversaryConfig',
    'AdversaryBehaviorModel',
    'EvasionPattern',
    'ThreatLevel',
    'GuidanceMode',
    'ThreatAssessment',
    'EvasionConfig',
    'create_default_adversary_config',
    'create_enhanced_adversary'
]