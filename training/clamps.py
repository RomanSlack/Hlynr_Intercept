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
from hlynr_bridge.schemas import RateCommand, ActionCommand, SafetyInfo

logger = logging.getLogger(__name__)


@dataclass
class SafetyLimits:
    """Safety limits configuration."""
    rate_max_radps: float = 10.0      # Maximum angular rate (rad/s)
    rate_min_radps: float = -10.0     # Minimum angular rate (rad/s)
    thrust_max: float = 1.0           # Maximum thrust command
    thrust_min: float = 0.0           # Minimum thrust command
    aux_max: float = 1.0              # Maximum auxiliary command
    aux_min: float = -1.0             # Minimum auxiliary command


@dataclass
class ClampResult:
    """Result of safety clamping operation."""
    clamped: bool
    original_action: ActionCommand
    clamped_action: ActionCommand
    clamp_reason: Optional[str]
    clamp_details: Dict[str, Any]
    timestamp: float


class SafetyClampSystem:
    """
    Safety clamping system for control outputs.
    
    Implements hard limits on control commands with comprehensive logging
    and statistics tracking for diagnostics.
    """
    
    def __init__(self, safety_limits: Optional[SafetyLimits] = None):
        """
        Initialize safety clamp system.
        
        Args:
            safety_limits: Safety limits configuration
        """
        self.limits = safety_limits or SafetyLimits()
        
        # Statistics tracking
        self.total_commands = 0
        self.total_clamped = 0
        self.clamp_history: List[ClampResult] = []
        self.max_history = 1000  # Keep last 1000 clamp events
        
        # Per-axis clamp counters
        self.pitch_clamps = 0
        self.yaw_clamps = 0
        self.roll_clamps = 0
        self.thrust_clamps = 0
        self.aux_clamps = 0
        
        logger.info(f"Safety clamp system initialized with limits: {self.limits}")
    
    def apply_safety_clamps(self, action: ActionCommand) -> Tuple[ActionCommand, SafetyInfo]:
        """
        Apply safety clamps to action command.
        
        Args:
            action: Original action command
            
        Returns:
            (Clamped action command, Safety info)
        """
        start_time = time.time()
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
        
        # Track what was clamped
        clamp_details = {}
        clamp_reasons = []
        any_clamped = False
        
        # Clamp angular rates
        rate_clamped, rate_details = self._clamp_rates(clamped_action.rate_cmd_radps)
        if rate_clamped:
            any_clamped = True
            clamp_details.update(rate_details)
            clamp_reasons.append("angular_rates")
        
        # Clamp thrust
        thrust_clamped, thrust_details = self._clamp_thrust(clamped_action)
        if thrust_clamped:
            any_clamped = True
            clamp_details.update(thrust_details)
            clamp_reasons.append("thrust")
        
        # Clamp auxiliary commands
        aux_clamped, aux_details = self._clamp_aux(clamped_action)
        if aux_clamped:
            any_clamped = True
            clamp_details.update(aux_details)
            clamp_reasons.append("auxiliary")
        
        # Create safety info
        clamp_reason = None
        if any_clamped:
            self.total_clamped += 1
            clamp_reason = f"Clamped: {', '.join(clamp_reasons)}"
        
        safety_info = SafetyInfo(
            clamped=any_clamped,
            clamp_reason=clamp_reason
        )
        
        # Log and store clamp result
        clamp_result = ClampResult(
            clamped=any_clamped,
            original_action=action,
            clamped_action=clamped_action,
            clamp_reason=clamp_reason,
            clamp_details=clamp_details,
            timestamp=start_time
        )
        
        self._record_clamp_result(clamp_result)
        
        # Log if clamped
        if any_clamped:
            logger.warning(f"Safety clamps applied: {clamp_reason}")
            logger.debug(f"Clamp details: {clamp_details}")
        
        return clamped_action, safety_info
    
    def _clamp_rates(self, rate_cmd: RateCommand) -> Tuple[bool, Dict[str, Any]]:
        """
        Clamp angular rate commands.
        
        Args:
            rate_cmd: Rate command to clamp (modified in place)
            
        Returns:
            (was_clamped, clamp_details)
        """
        clamped = False
        details = {}
        
        # Clamp pitch
        original_pitch = rate_cmd.pitch
        rate_cmd.pitch = np.clip(rate_cmd.pitch, self.limits.rate_min_radps, self.limits.rate_max_radps)
        if rate_cmd.pitch != original_pitch:
            clamped = True
            self.pitch_clamps += 1
            details['pitch'] = {
                'original': original_pitch,
                'clamped': rate_cmd.pitch,
                'limit_violated': 'max' if original_pitch > self.limits.rate_max_radps else 'min'
            }
        
        # Clamp yaw
        original_yaw = rate_cmd.yaw
        rate_cmd.yaw = np.clip(rate_cmd.yaw, self.limits.rate_min_radps, self.limits.rate_max_radps)
        if rate_cmd.yaw != original_yaw:
            clamped = True
            self.yaw_clamps += 1
            details['yaw'] = {
                'original': original_yaw,
                'clamped': rate_cmd.yaw,
                'limit_violated': 'max' if original_yaw > self.limits.rate_max_radps else 'min'
            }
        
        # Clamp roll
        original_roll = rate_cmd.roll
        rate_cmd.roll = np.clip(rate_cmd.roll, self.limits.rate_min_radps, self.limits.rate_max_radps)
        if rate_cmd.roll != original_roll:
            clamped = True
            self.roll_clamps += 1
            details['roll'] = {
                'original': original_roll,
                'clamped': rate_cmd.roll,
                'limit_violated': 'max' if original_roll > self.limits.rate_max_radps else 'min'
            }
        
        return clamped, details
    
    def _clamp_thrust(self, action: ActionCommand) -> Tuple[bool, Dict[str, Any]]:
        """
        Clamp thrust command.
        
        Args:
            action: Action command to clamp (modified in place)
            
        Returns:
            (was_clamped, clamp_details)
        """
        original_thrust = action.thrust_cmd
        action.thrust_cmd = np.clip(action.thrust_cmd, self.limits.thrust_min, self.limits.thrust_max)
        
        if action.thrust_cmd != original_thrust:
            self.thrust_clamps += 1
            details = {
                'thrust': {
                    'original': original_thrust,
                    'clamped': action.thrust_cmd,
                    'limit_violated': 'max' if original_thrust > self.limits.thrust_max else 'min'
                }
            }
            return True, details
        
        return False, {}
    
    def _clamp_aux(self, action: ActionCommand) -> Tuple[bool, Dict[str, Any]]:
        """
        Clamp auxiliary commands.
        
        Args:
            action: Action command to clamp (modified in place)
            
        Returns:
            (was_clamped, clamp_details)
        """
        if not action.aux:
            return False, {}
        
        clamped = False
        details = {}
        
        for i, aux_val in enumerate(action.aux):
            original_val = aux_val
            clamped_val = np.clip(aux_val, self.limits.aux_min, self.limits.aux_max)
            action.aux[i] = clamped_val
            
            if clamped_val != original_val:
                clamped = True
                self.aux_clamps += 1
                details[f'aux_{i}'] = {
                    'original': original_val,
                    'clamped': clamped_val,
                    'limit_violated': 'max' if original_val > self.limits.aux_max else 'min'
                }
        
        return clamped, details
    
    def _record_clamp_result(self, result: ClampResult):
        """Record clamp result for statistics."""
        self.clamp_history.append(result)
        
        # Trim history if too long
        if len(self.clamp_history) > self.max_history:
            self.clamp_history = self.clamp_history[-self.max_history:]
    
    def get_clamp_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive clamp statistics.
        
        Returns:
            Dictionary with clamp statistics
        """
        clamp_rate = self.total_clamped / max(self.total_commands, 1)
        
        # Recent clamp rate (last 100 commands)
        recent_history = self.clamp_history[-100:]
        recent_clamped = sum(1 for r in recent_history if r.clamped)
        recent_rate = recent_clamped / max(len(recent_history), 1)
        
        return {
            'total_commands': self.total_commands,
            'total_clamped': self.total_clamped,
            'overall_clamp_rate': clamp_rate,
            'recent_clamp_rate': recent_rate,
            'clamps_by_axis': {
                'pitch': self.pitch_clamps,
                'yaw': self.yaw_clamps,
                'roll': self.roll_clamps,
                'thrust': self.thrust_clamps,
                'aux': self.aux_clamps
            },
            'safety_limits': {
                'rate_max_radps': self.limits.rate_max_radps,
                'rate_min_radps': self.limits.rate_min_radps,
                'thrust_max': self.limits.thrust_max,
                'thrust_min': self.limits.thrust_min,
                'aux_max': self.limits.aux_max,
                'aux_min': self.limits.aux_min
            }
        }
    
    def get_recent_clamps(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent clamp events.
        
        Args:
            count: Number of recent clamps to return
            
        Returns:
            List of recent clamp events
        """
        recent_clamps = [r for r in self.clamp_history if r.clamped][-count:]
        
        return [{
            'timestamp': r.timestamp,
            'clamp_reason': r.clamp_reason,
            'clamp_details': r.clamp_details,
            'original_action': {
                'rate_cmd_radps': {
                    'pitch': r.original_action.rate_cmd_radps.pitch,
                    'yaw': r.original_action.rate_cmd_radps.yaw,
                    'roll': r.original_action.rate_cmd_radps.roll
                },
                'thrust_cmd': r.original_action.thrust_cmd,
                'aux': r.original_action.aux
            },
            'clamped_action': {
                'rate_cmd_radps': {
                    'pitch': r.clamped_action.rate_cmd_radps.pitch,
                    'yaw': r.clamped_action.rate_cmd_radps.yaw,
                    'roll': r.clamped_action.rate_cmd_radps.roll
                },
                'thrust_cmd': r.clamped_action.thrust_cmd,
                'aux': r.clamped_action.aux
            }
        } for r in recent_clamps]
    
    def reset_statistics(self):
        """Reset clamp statistics."""
        self.total_commands = 0
        self.total_clamped = 0
        self.clamp_history.clear()
        self.pitch_clamps = 0
        self.yaw_clamps = 0
        self.roll_clamps = 0
        self.thrust_clamps = 0
        self.aux_clamps = 0
        
        logger.info("Safety clamp statistics reset")
    
    def update_limits(self, new_limits: SafetyLimits):
        """
        Update safety limits.
        
        Args:
            new_limits: New safety limits
        """
        old_limits = self.limits
        self.limits = new_limits
        
        logger.info(f"Safety limits updated from {old_limits} to {new_limits}")
    
    def validate_action(self, action: ActionCommand) -> List[str]:
        """
        Validate action without clamping and return list of violations.
        
        Args:
            action: Action to validate
            
        Returns:
            List of violation descriptions
        """
        violations = []
        
        # Check angular rates
        if action.rate_cmd_radps.pitch > self.limits.rate_max_radps:
            violations.append(f"Pitch rate {action.rate_cmd_radps.pitch:.3f} > {self.limits.rate_max_radps}")
        if action.rate_cmd_radps.pitch < self.limits.rate_min_radps:
            violations.append(f"Pitch rate {action.rate_cmd_radps.pitch:.3f} < {self.limits.rate_min_radps}")
        
        if action.rate_cmd_radps.yaw > self.limits.rate_max_radps:
            violations.append(f"Yaw rate {action.rate_cmd_radps.yaw:.3f} > {self.limits.rate_max_radps}")
        if action.rate_cmd_radps.yaw < self.limits.rate_min_radps:
            violations.append(f"Yaw rate {action.rate_cmd_radps.yaw:.3f} < {self.limits.rate_min_radps}")
        
        if action.rate_cmd_radps.roll > self.limits.rate_max_radps:
            violations.append(f"Roll rate {action.rate_cmd_radps.roll:.3f} > {self.limits.rate_max_radps}")
        if action.rate_cmd_radps.roll < self.limits.rate_min_radps:
            violations.append(f"Roll rate {action.rate_cmd_radps.roll:.3f} < {self.limits.rate_min_radps}")
        
        # Check thrust
        if action.thrust_cmd > self.limits.thrust_max:
            violations.append(f"Thrust {action.thrust_cmd:.3f} > {self.limits.thrust_max}")
        if action.thrust_cmd < self.limits.thrust_min:
            violations.append(f"Thrust {action.thrust_cmd:.3f} < {self.limits.thrust_min}")
        
        # Check auxiliary commands
        for i, aux_val in enumerate(action.aux):
            if aux_val > self.limits.aux_max:
                violations.append(f"Aux[{i}] {aux_val:.3f} > {self.limits.aux_max}")
            if aux_val < self.limits.aux_min:
                violations.append(f"Aux[{i}] {aux_val:.3f} < {self.limits.aux_min}")
        
        return violations


# Global safety clamp system
_safety_clamp_system = None


def get_safety_clamp_system(safety_limits: Optional[SafetyLimits] = None) -> SafetyClampSystem:
    """Get global safety clamp system instance."""
    global _safety_clamp_system
    
    if _safety_clamp_system is None:
        _safety_clamp_system = SafetyClampSystem(safety_limits)
    
    return _safety_clamp_system


def apply_safety_clamps(action: ActionCommand, 
                       safety_limits: Optional[SafetyLimits] = None) -> Tuple[ActionCommand, SafetyInfo]:
    """Apply safety clamps to action command using global system."""
    clamp_system = get_safety_clamp_system(safety_limits)
    return clamp_system.apply_safety_clamps(action)


# Example usage and testing
if __name__ == "__main__":
    from hlynr_bridge.schemas import RateCommand, ActionCommand
    
    # Test safety clamp system
    safety_system = SafetyClampSystem()
    
    print("Safety Clamp System Test")
    print("=" * 40)
    
    # Test cases
    test_actions = [
        # Normal action (should not be clamped)
        ActionCommand(
            rate_cmd_radps=RateCommand(pitch=0.5, yaw=-0.2, roll=0.1),
            thrust_cmd=0.8,
            aux=[0.0, 0.0]
        ),
        # Excessive rates (should be clamped)
        ActionCommand(
            rate_cmd_radps=RateCommand(pitch=15.0, yaw=-12.0, roll=20.0),
            thrust_cmd=0.8,
            aux=[0.0, 0.0]
        ),
        # Invalid thrust (should be clamped)
        ActionCommand(
            rate_cmd_radps=RateCommand(pitch=0.5, yaw=-0.2, roll=0.1),
            thrust_cmd=1.5,
            aux=[0.0, 0.0]
        ),
        # Negative thrust (should be clamped)
        ActionCommand(
            rate_cmd_radps=RateCommand(pitch=0.5, yaw=-0.2, roll=0.1),
            thrust_cmd=-0.1,
            aux=[0.0, 0.0]
        )
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\nTest case {i+1}:")
        print(f"Original action: {action}")
        
        # Validate without clamping
        violations = safety_system.validate_action(action)
        if violations:
            print(f"Violations: {violations}")
        
        # Apply clamps
        clamped_action, safety_info = safety_system.apply_safety_clamps(action)
        print(f"Clamped action: {clamped_action}")
        print(f"Safety info: {safety_info}")
    
    # Show statistics
    print(f"\nSafety Statistics:")
    stats = safety_system.get_clamp_statistics()
    print(f"Total commands: {stats['total_commands']}")
    print(f"Total clamped: {stats['total_clamped']}")
    print(f"Clamp rate: {stats['overall_clamp_rate']:.2%}")
    print(f"Clamps by axis: {stats['clamps_by_axis']}")
    
    # Show recent clamps
    recent = safety_system.get_recent_clamps(3)
    print(f"\nRecent clamps: {len(recent)}")
    for clamp in recent:
        print(f"  - {clamp['clamp_reason']} at {clamp['timestamp']}")