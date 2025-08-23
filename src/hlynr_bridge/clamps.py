"""
Safety clamping system for control outputs.

This module implements hard safety limits on control commands to ensure
the system operates within safe bounds, with comprehensive logging.
"""

import logging
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from .schemas import RateCommand, ActionCommand, SafetyInfo

logger = logging.getLogger(__name__)


@dataclass
class SafetyLimits:
    """Safety limits configuration."""
    rate_max_radps: float = 5.0       # Maximum angular rate (rad/s) - CANONICAL CLAMP REASONS
    rate_min_radps: float = -5.0      # Minimum angular rate (rad/s)
    thrust_max: float = 1.0           # Maximum thrust command
    thrust_min: float = 0.0           # Minimum thrust command
    aux_max: float = 1.0              # Maximum auxiliary command
    aux_min: float = -1.0             # Minimum auxiliary command


class SafetyClampSystem:
    """Safety clamping system implementing canonical clamp reasons."""
    
    def __init__(self, safety_limits: Optional[SafetyLimits] = None):
        self.limits = safety_limits or SafetyLimits()
        self.total_commands = 0
        self.total_clamped = 0
        
        # Per-axis clamp counters for canonical reasons
        self.rate_limit_pitch = 0
        self.rate_limit_yaw = 0 
        self.rate_limit_roll = 0
        self.thrust_floor = 0
        self.thrust_ceiling = 0
        
        logger.info(f"Safety clamp system initialized with limits: {self.limits}")
    
    def apply_safety_clamps(self, action: ActionCommand) -> Tuple[ActionCommand, SafetyInfo]:
        """Apply safety clamps with canonical clamp reasons."""
        self.total_commands += 1
        
        # Create copy for clamping
        clamped_action = ActionCommand(
            rate_cmd_radps=RateCommand(
                pitch=action.rate_cmd_radps.pitch,
                yaw=action.rate_cmd_radps.yaw,
                roll=action.rate_cmd_radps.roll
            ),
            thrust_cmd=action.thrust_cmd,
            aux=action.aux.copy() if action.aux else []
        )
        
        clamp_reasons = []
        any_clamped = False
        
        # Clamp angular rates with canonical reasons
        if abs(clamped_action.rate_cmd_radps.pitch) > self.limits.rate_max_radps:
            clamped_action.rate_cmd_radps.pitch = np.clip(
                clamped_action.rate_cmd_radps.pitch, 
                self.limits.rate_min_radps, 
                self.limits.rate_max_radps
            )
            clamp_reasons.append("rate_limit_pitch")
            self.rate_limit_pitch += 1
            any_clamped = True
        
        if abs(clamped_action.rate_cmd_radps.yaw) > self.limits.rate_max_radps:
            clamped_action.rate_cmd_radps.yaw = np.clip(
                clamped_action.rate_cmd_radps.yaw,
                self.limits.rate_min_radps,
                self.limits.rate_max_radps
            )
            clamp_reasons.append("rate_limit_yaw")
            self.rate_limit_yaw += 1
            any_clamped = True
        
        if abs(clamped_action.rate_cmd_radps.roll) > self.limits.rate_max_radps:
            clamped_action.rate_cmd_radps.roll = np.clip(
                clamped_action.rate_cmd_radps.roll,
                self.limits.rate_min_radps,
                self.limits.rate_max_radps
            )
            clamp_reasons.append("rate_limit_roll")
            self.rate_limit_roll += 1
            any_clamped = True
        
        # Clamp thrust with canonical reasons
        if clamped_action.thrust_cmd < self.limits.thrust_min:
            clamped_action.thrust_cmd = self.limits.thrust_min
            clamp_reasons.append("thrust_floor")
            self.thrust_floor += 1
            any_clamped = True
        
        if clamped_action.thrust_cmd > self.limits.thrust_max:
            clamped_action.thrust_cmd = self.limits.thrust_max
            clamp_reasons.append("thrust_ceiling")
            self.thrust_ceiling += 1
            any_clamped = True
        
        if any_clamped:
            self.total_clamped += 1
        
        # Create safety info with canonical clamp reason
        clamp_reason = "|".join(clamp_reasons) if clamp_reasons else None
        safety_info = SafetyInfo(
            clamped=any_clamped,
            clamp_reason=clamp_reason
        )
        
        if any_clamped:
            logger.warning(f"Safety clamps applied: {clamp_reason}")
        
        return clamped_action, safety_info


# Global safety clamp system
_safety_clamp_system = None


def get_safety_clamp_system(safety_limits: Optional[SafetyLimits] = None) -> SafetyClampSystem:
    """Get global safety clamp system instance."""
    global _safety_clamp_system
    
    if _safety_clamp_system is None:
        _safety_clamp_system = SafetyClampSystem(safety_limits)
    
    return _safety_clamp_system