"""
Enhanced Adversary System for AegisIntercept Phase 3.

This module provides an enhanced adversary system that combines the existing
adversary controller and evasion behaviors with additional intelligence,
adaptive behavior patterns, and scenario-specific configurations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pyquaternion import Quaternion
import json

from .adversary_controller import AdversaryController
from .evasion_behaviors import EvasionBehaviors
from ..utils.maths import normalize_vector, clamp, distance


class EnhancedAdversary:
    """
    Enhanced adversary system with adaptive and intelligent behavior.
    
    This class extends the basic adversary controller with:
    - Adaptive behavior based on interceptor proximity and capabilities
    - Multi-stage evasion patterns
    - Scenario-specific configurations
    - Learning from previous encounters
    - Enhanced threat assessment
    """
    
    def __init__(self, 
                 base_controller: AdversaryController,
                 intelligence_level: float = 0.5,
                 adaptability: float = 0.3,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced adversary system.
        
        Args:
            base_controller: Base adversary controller
            intelligence_level: Intelligence level (0-1)
            adaptability: Adaptability factor (0-1)
            config: Configuration dictionary
        """
        self.base_controller = base_controller
        self.intelligence_level = clamp(intelligence_level, 0.0, 1.0)
        self.adaptability = clamp(adaptability, 0.0, 1.0)
        
        # Default configuration
        self.config = {
            'threat_assessment_range': 5000.0,    # Range for threat assessment (m)
            'learning_rate': 0.1,                 # Learning rate for adaptation
            'behavior_switching_threshold': 0.7,  # Threshold for behavior switching
            'energy_management': True,            # Enable energy management
            'cooperative_behavior': False,        # Multi-adversary coordination
            'countermeasure_probability': 0.0,    # Probability of countermeasures
            'sensor_spoofing_capability': 0.0,    # Sensor spoofing capability
            'adaptive_timing': True,              # Adaptive timing for maneuvers
            'pattern_recognition': True,          # Pattern recognition capability
            'threat_prediction': True,            # Threat prediction capability
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Initialize enhanced systems
        self.threat_assessor = ThreatAssessment(self.intelligence_level)
        self.behavior_planner = BehaviorPlanner(self.intelligence_level, self.adaptability)
        self.energy_manager = EnergyManager()
        
        # State tracking
        self.encounter_history = []
        self.current_threats = []
        self.behavior_state = 'cruise'
        self.last_maneuver_time = 0.0
        self.energy_reserves = 1.0
        
        # Performance metrics
        self.evasion_success_rate = 0.0
        self.adaptation_history = []
    
    def update(self, 
               current_state: Dict[str, Any],
               interceptor_states: List[Dict[str, Any]],
               target_state: Dict[str, Any],
               current_time: float,
               dt: float) -> Dict[str, Any]:
        """
        Update enhanced adversary system.
        
        Args:
            current_state: Current adversary state
            interceptor_states: List of interceptor states
            target_state: Target state
            current_time: Current simulation time
            dt: Time step
            
        Returns:
            Enhanced control commands
        """
        # Assess current threats
        threat_assessment = self.threat_assessor.assess_threats(
            current_state, interceptor_states, current_time
        )
        
        # Plan behavior based on threat assessment
        behavior_plan = self.behavior_planner.plan_behavior(
            current_state, threat_assessment, self.behavior_state, current_time
        )
        
        # Manage energy reserves
        if self.config['energy_management']:
            energy_constraints = self.energy_manager.get_constraints(
                self.energy_reserves, threat_assessment
            )
            behavior_plan = self._apply_energy_constraints(behavior_plan, energy_constraints)
        
        # Generate base control commands
        base_commands = self.base_controller.update(
            current_state, interceptor_states, target_state, current_time, dt
        )
        
        # Enhance commands with intelligent behavior
        enhanced_commands = self._enhance_commands(
            base_commands, behavior_plan, threat_assessment, current_time
        )
        
        # Update internal state
        self._update_internal_state(
            current_state, threat_assessment, behavior_plan, current_time
        )
        
        return enhanced_commands
    
    def _enhance_commands(self, 
                         base_commands: Dict[str, Any],
                         behavior_plan: Dict[str, Any],
                         threat_assessment: Dict[str, Any],
                         current_time: float) -> Dict[str, Any]:
        """Enhance base commands with intelligent behavior."""
        enhanced_commands = base_commands.copy()
        
        # Apply behavior modifications
        if behavior_plan['behavior_type'] == 'aggressive_evasion':
            enhanced_commands = self._apply_aggressive_evasion(
                enhanced_commands, behavior_plan, threat_assessment
            )
        elif behavior_plan['behavior_type'] == 'energy_conserving':
            enhanced_commands = self._apply_energy_conserving(
                enhanced_commands, behavior_plan
            )
        elif behavior_plan['behavior_type'] == 'deceptive_maneuver':
            enhanced_commands = self._apply_deceptive_maneuver(
                enhanced_commands, behavior_plan, current_time
            )
        elif behavior_plan['behavior_type'] == 'cooperative_evasion':
            enhanced_commands = self._apply_cooperative_evasion(
                enhanced_commands, behavior_plan
            )
        
        # Apply sensor spoofing if configured
        if self.config['sensor_spoofing_capability'] > 0:
            enhanced_commands = self._apply_sensor_spoofing(
                enhanced_commands, threat_assessment
            )
        
        # Apply countermeasures if configured
        if self.config['countermeasure_probability'] > 0:
            enhanced_commands = self._apply_countermeasures(
                enhanced_commands, threat_assessment, current_time
            )
        
        return enhanced_commands
    
    def _apply_aggressive_evasion(self, 
                                 commands: Dict[str, Any],
                                 behavior_plan: Dict[str, Any],
                                 threat_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Apply aggressive evasion modifications."""
        # Increase evasion aggressiveness
        if 'evasion_factor' in commands:
            commands['evasion_factor'] *= 1.5
        
        # Increase maneuver frequency
        if 'maneuver_frequency' in commands:
            commands['maneuver_frequency'] *= 1.3
        
        # Prioritize high-g maneuvers
        if 'max_acceleration' in commands:
            commands['max_acceleration'] *= 1.2
        
        return commands
    
    def _apply_energy_conserving(self, 
                                commands: Dict[str, Any],
                                behavior_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Apply energy conserving modifications."""
        # Reduce thrust usage
        if 'thrust_factor' in commands:
            commands['thrust_factor'] *= 0.8
        
        # Prefer gradual maneuvers
        if 'maneuver_smoothness' in commands:
            commands['maneuver_smoothness'] *= 1.5
        
        return commands
    
    def _apply_deceptive_maneuver(self, 
                                 commands: Dict[str, Any],
                                 behavior_plan: Dict[str, Any],
                                 current_time: float) -> Dict[str, Any]:
        """Apply deceptive maneuver modifications."""
        # Add feint maneuvers
        if current_time - self.last_maneuver_time > 2.0:
            commands['feint_maneuver'] = True
            self.last_maneuver_time = current_time
        
        # Add trajectory prediction errors
        if 'trajectory_noise' in commands:
            commands['trajectory_noise'] *= 1.5
        
        return commands
    
    def _apply_cooperative_evasion(self, 
                                  commands: Dict[str, Any],
                                  behavior_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Apply cooperative evasion modifications."""
        # This would be implemented for multi-adversary scenarios
        # For now, just return unmodified commands
        return commands
    
    def _apply_sensor_spoofing(self, 
                              commands: Dict[str, Any],
                              threat_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Apply sensor spoofing capabilities."""
        spoofing_strength = self.config['sensor_spoofing_capability']
        
        if spoofing_strength > 0 and threat_assessment['threat_level'] > 0.5:
            # Add sensor spoofing commands
            commands['sensor_spoofing'] = {
                'strength': spoofing_strength,
                'type': 'radar_jamming',
                'duration': 1.0
            }
        
        return commands
    
    def _apply_countermeasures(self, 
                              commands: Dict[str, Any],
                              threat_assessment: Dict[str, Any],
                              current_time: float) -> Dict[str, Any]:
        """Apply countermeasure capabilities."""
        if np.random.random() < self.config['countermeasure_probability']:
            if threat_assessment['threat_level'] > 0.7:
                commands['deploy_countermeasures'] = True
        
        return commands
    
    def _apply_energy_constraints(self, 
                                 behavior_plan: Dict[str, Any],
                                 energy_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Apply energy management constraints."""
        if energy_constraints['low_energy']:
            behavior_plan['behavior_type'] = 'energy_conserving'
            behavior_plan['aggressiveness'] *= 0.5
        
        return behavior_plan
    
    def _update_internal_state(self, 
                              current_state: Dict[str, Any],
                              threat_assessment: Dict[str, Any],
                              behavior_plan: Dict[str, Any],
                              current_time: float) -> None:
        """Update internal state tracking."""
        # Update behavior state
        self.behavior_state = behavior_plan['behavior_type']
        
        # Update energy reserves
        energy_usage = behavior_plan.get('energy_usage', 0.01)
        self.energy_reserves = max(0.0, self.energy_reserves - energy_usage)
        
        # Update threat history
        self.current_threats = threat_assessment['threats']
        
        # Update encounter history
        encounter = {
            'time': current_time,
            'threat_level': threat_assessment['threat_level'],
            'behavior': behavior_plan['behavior_type'],
            'energy_level': self.energy_reserves
        }
        self.encounter_history.append(encounter)
        
        # Limit history size
        if len(self.encounter_history) > 100:
            self.encounter_history.pop(0)
    
    def adapt_to_scenario(self, scenario_config: Dict[str, Any]) -> None:
        """Adapt behavior to specific scenario parameters."""
        # Update intelligence level based on scenario
        if 'adversary_intelligence' in scenario_config:
            self.intelligence_level = scenario_config['adversary_intelligence']
        
        # Update configuration based on scenario
        if 'adversary_config' in scenario_config:
            self.config.update(scenario_config['adversary_config'])
        
        # Reinitialize systems if needed
        self.threat_assessor.update_intelligence(self.intelligence_level)
        self.behavior_planner.update_parameters(self.intelligence_level, self.adaptability)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the enhanced adversary."""
        return {
            'evasion_success_rate': self.evasion_success_rate,
            'average_threat_level': np.mean([e['threat_level'] for e in self.encounter_history]),
            'energy_efficiency': self.energy_reserves,
            'adaptation_count': len(self.adaptation_history),
            'behavior_distribution': self._get_behavior_distribution()
        }
    
    def _get_behavior_distribution(self) -> Dict[str, float]:
        """Get distribution of behavior types used."""
        if not self.encounter_history:
            return {}
        
        behaviors = [e['behavior'] for e in self.encounter_history]
        unique_behaviors = set(behaviors)
        distribution = {}
        
        for behavior in unique_behaviors:
            distribution[behavior] = behaviors.count(behavior) / len(behaviors)
        
        return distribution
    
    def reset(self) -> None:
        """Reset the enhanced adversary system."""
        self.encounter_history = []
        self.current_threats = []
        self.behavior_state = 'cruise'
        self.last_maneuver_time = 0.0
        self.energy_reserves = 1.0
        self.evasion_success_rate = 0.0
        self.adaptation_history = []
        
        # Reset subsystems
        self.threat_assessor.reset()
        self.behavior_planner.reset()
        self.energy_manager.reset()


class ThreatAssessment:
    """Threat assessment system for enhanced adversary."""
    
    def __init__(self, intelligence_level: float):
        self.intelligence_level = intelligence_level
        self.threat_memory = []
    
    def assess_threats(self, 
                      current_state: Dict[str, Any],
                      interceptor_states: List[Dict[str, Any]],
                      current_time: float) -> Dict[str, Any]:
        """Assess current threat level and characteristics."""
        threats = []
        overall_threat = 0.0
        
        for interceptor in interceptor_states:
            threat_level = self._assess_single_threat(current_state, interceptor)
            threats.append({
                'interceptor_id': interceptor.get('id', 'unknown'),
                'threat_level': threat_level,
                'distance': distance(current_state['position'], interceptor['position']),
                'closing_rate': self._calculate_closing_rate(current_state, interceptor)
            })
            overall_threat = max(overall_threat, threat_level)
        
        return {
            'threats': threats,
            'threat_level': overall_threat,
            'primary_threat': max(threats, key=lambda x: x['threat_level']) if threats else None,
            'threat_direction': self._get_threat_direction(current_state, threats)
        }
    
    def _assess_single_threat(self, 
                             current_state: Dict[str, Any],
                             interceptor_state: Dict[str, Any]) -> float:
        """Assess threat level from single interceptor."""
        # Calculate distance factor
        dist = distance(current_state['position'], interceptor_state['position'])
        distance_factor = max(0.0, 1.0 - dist / 10000.0)  # Normalize to 10km
        
        # Calculate closing rate factor
        closing_rate = self._calculate_closing_rate(current_state, interceptor_state)
        closing_factor = max(0.0, min(1.0, closing_rate / 500.0))  # Normalize to 500 m/s
        
        # Combine factors
        threat_level = distance_factor * 0.6 + closing_factor * 0.4
        
        return clamp(threat_level, 0.0, 1.0)
    
    def _calculate_closing_rate(self, 
                               current_state: Dict[str, Any],
                               interceptor_state: Dict[str, Any]) -> float:
        """Calculate closing rate between adversary and interceptor."""
        relative_position = interceptor_state['position'] - current_state['position']
        relative_velocity = interceptor_state['velocity'] - current_state['velocity']
        
        if np.linalg.norm(relative_position) < 1e-6:
            return 0.0
        
        # Project relative velocity onto position vector
        closing_rate = -np.dot(relative_velocity, relative_position) / np.linalg.norm(relative_position)
        
        return max(0.0, closing_rate)
    
    def _get_threat_direction(self, 
                             current_state: Dict[str, Any],
                             threats: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Get primary threat direction."""
        if not threats:
            return None
        
        primary_threat = max(threats, key=lambda x: x['threat_level'])
        # This would need interceptor positions to calculate direction
        # For now, return a placeholder
        return np.array([1.0, 0.0, 0.0])
    
    def update_intelligence(self, intelligence_level: float) -> None:
        """Update intelligence level."""
        self.intelligence_level = intelligence_level
    
    def reset(self) -> None:
        """Reset threat assessment."""
        self.threat_memory = []


class BehaviorPlanner:
    """Behavior planning system for enhanced adversary."""
    
    def __init__(self, intelligence_level: float, adaptability: float):
        self.intelligence_level = intelligence_level
        self.adaptability = adaptability
        self.behavior_history = []
    
    def plan_behavior(self, 
                     current_state: Dict[str, Any],
                     threat_assessment: Dict[str, Any],
                     current_behavior: str,
                     current_time: float) -> Dict[str, Any]:
        """Plan behavior based on current situation."""
        threat_level = threat_assessment['threat_level']
        
        # Determine behavior type
        if threat_level > 0.8:
            behavior_type = 'aggressive_evasion'
            aggressiveness = 0.9
        elif threat_level > 0.5:
            behavior_type = 'deceptive_maneuver'
            aggressiveness = 0.6
        elif threat_level > 0.2:
            behavior_type = 'energy_conserving'
            aggressiveness = 0.3
        else:
            behavior_type = 'cruise'
            aggressiveness = 0.1
        
        # Adjust based on intelligence level
        aggressiveness *= self.intelligence_level
        
        return {
            'behavior_type': behavior_type,
            'aggressiveness': aggressiveness,
            'energy_usage': aggressiveness * 0.02,
            'duration': 1.0 + 2.0 * self.intelligence_level
        }
    
    def update_parameters(self, intelligence_level: float, adaptability: float) -> None:
        """Update planning parameters."""
        self.intelligence_level = intelligence_level
        self.adaptability = adaptability
    
    def reset(self) -> None:
        """Reset behavior planner."""
        self.behavior_history = []


class EnergyManager:
    """Energy management system for enhanced adversary."""
    
    def __init__(self):
        self.energy_history = []
    
    def get_constraints(self, 
                       current_energy: float,
                       threat_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Get energy management constraints."""
        constraints = {
            'low_energy': current_energy < 0.3,
            'critical_energy': current_energy < 0.1,
            'max_energy_usage': 0.1 if current_energy < 0.3 else 0.2
        }
        
        # Adjust based on threat level
        if threat_assessment['threat_level'] > 0.8:
            constraints['max_energy_usage'] *= 2.0  # Allow more energy usage in high threat
        
        return constraints
    
    def reset(self) -> None:
        """Reset energy manager."""
        self.energy_history = []