"""
Evasion behaviors for AegisIntercept Phase 3 adversary.

This module provides various evasion maneuvers including jinks, spirals,
barrel rolls, and complex multi-stage evasion patterns.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pyquaternion import Quaternion
from ..utils.maths import normalize_vector, clamp


class EvasionBehaviors:
    """
    Advanced evasion behaviors for adversary missiles.
    
    This class implements various evasion maneuvers that can be combined
    and adapted based on the tactical situation. All behaviors are designed
    to be state-based (no look-ahead) to maintain computational efficiency.
    """
    
    def __init__(self, difficulty_level: float = 0.5):
        """
        Initialize evasion behaviors.
        
        Args:
            difficulty_level: Evasion aggressiveness (0-1 scale)
        """
        self.difficulty_level = clamp(difficulty_level, 0.0, 1.0)
        
        # Evasion parameters that scale with difficulty
        self.base_params = {
            'jink_frequency': 0.5,      # Hz
            'jink_amplitude': 50.0,     # m/s
            'spiral_rate': 0.2,         # rad/s
            'spiral_radius': 100.0,     # m
            'barrel_roll_rate': 0.1,    # rad/s
            'evasion_threshold': 300.0, # m
            'reaction_time': 0.5,       # s
            'energy_conservation': 0.8  # 0-1 scale
        }
        
        # Scale parameters by difficulty
        self.params = {}
        for key, value in self.base_params.items():
            if key in ['jink_frequency', 'spiral_rate', 'barrel_roll_rate']:
                self.params[key] = value * (0.5 + 0.5 * self.difficulty_level)
            elif key == 'evasion_threshold':
                self.params[key] = value * (0.8 + 0.4 * self.difficulty_level)
            elif key == 'reaction_time':
                self.params[key] = value * (1.5 - 0.5 * self.difficulty_level)
            else:
                self.params[key] = value
        
        # Evasion state
        self.evasion_state = {
            'active': False,
            'pattern': None,
            'start_time': 0.0,
            'phase': 0.0,
            'last_threat_position': None,
            'evasion_history': []
        }
        
        # Pattern-specific states
        self.pattern_states = {
            'jink': {'direction': 1, 'last_jink_time': 0.0},
            'spiral': {'axis': np.array([0, 0, 1]), 'center': None},
            'barrel_roll': {'axis': np.array([1, 0, 0]), 'phase': 0.0},
            'weave': {'amplitude': 0.0, 'frequency': 0.0},
            'split_s': {'phase': 0.0, 'target_altitude': 0.0}
        }
    
    def update_evasion(self, 
                      current_position: np.ndarray,
                      current_velocity: np.ndarray,
                      target_position: np.ndarray,
                      threat_position: np.ndarray,
                      time: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update evasion behavior and return desired velocity and acceleration.
        
        Args:
            current_position: Current position [x, y, z]
            current_velocity: Current velocity [vx, vy, vz]
            target_position: Target (defended) position
            threat_position: Interceptor position
            time: Current simulation time
            
        Returns:
            Tuple of (desired_velocity, evasion_acceleration)
        """
        # Calculate threat distance
        threat_distance = np.linalg.norm(threat_position - current_position)
        
        # Determine if evasion should be active
        should_evade = threat_distance < self.params['evasion_threshold']
        
        if should_evade and not self.evasion_state['active']:
            # Start evasion
            self._start_evasion(current_position, current_velocity, 
                              threat_position, time)
        elif not should_evade and self.evasion_state['active']:
            # Stop evasion
            self._stop_evasion()
        
        # Calculate base navigation toward target
        target_direction = normalize_vector(target_position - current_position)
        base_speed = np.linalg.norm(current_velocity)
        desired_velocity = target_direction * base_speed
        
        # Add evasion component if active
        if self.evasion_state['active']:
            evasion_velocity = self._calculate_evasion_velocity(
                current_position, current_velocity, threat_position, time
            )
            
            # Blend base navigation with evasion
            evasion_weight = self.difficulty_level
            desired_velocity = (1.0 - evasion_weight) * desired_velocity + \
                             evasion_weight * evasion_velocity
        
        # Calculate acceleration to achieve desired velocity
        velocity_error = desired_velocity - current_velocity
        evasion_acceleration = velocity_error * 2.0  # Proportional gain
        
        return desired_velocity, evasion_acceleration
    
    def _start_evasion(self, position: np.ndarray, velocity: np.ndarray,
                      threat_position: np.ndarray, time: float):
        """Start evasion behavior."""
        # Select evasion pattern based on difficulty and situation
        pattern = self._select_evasion_pattern(position, velocity, threat_position)
        
        self.evasion_state.update({
            'active': True,
            'pattern': pattern,
            'start_time': time,
            'phase': 0.0,
            'last_threat_position': threat_position.copy()
        })
        
        # Initialize pattern-specific state
        self._initialize_pattern_state(pattern, position, velocity, threat_position)
    
    def _stop_evasion(self):
        """Stop evasion behavior."""
        self.evasion_state.update({
            'active': False,
            'pattern': None,
            'phase': 0.0
        })
    
    def _select_evasion_pattern(self, position: np.ndarray, velocity: np.ndarray,
                               threat_position: np.ndarray) -> str:
        """Select appropriate evasion pattern based on situation."""
        # Available patterns based on difficulty level
        available_patterns = ['jink']
        
        if self.difficulty_level > 0.3:
            available_patterns.extend(['spiral', 'weave'])
        
        if self.difficulty_level > 0.6:
            available_patterns.extend(['barrel_roll'])
        
        if self.difficulty_level > 0.8:
            available_patterns.extend(['split_s'])
        
        # Simple selection based on threat geometry
        threat_vector = threat_position - position
        velocity_direction = normalize_vector(velocity)
        threat_direction = normalize_vector(threat_vector)
        
        # Check if threat is approaching from behind
        approach_angle = np.arccos(np.clip(np.dot(velocity_direction, threat_direction), -1, 1))
        
        if approach_angle > np.pi * 0.7:  # Threat from behind
            if 'barrel_roll' in available_patterns:
                return 'barrel_roll'
            elif 'spiral' in available_patterns:
                return 'spiral'
        elif approach_angle < np.pi * 0.3:  # Head-on approach
            if 'split_s' in available_patterns:
                return 'split_s'
            elif 'weave' in available_patterns:
                return 'weave'
        
        # Default to jink
        return np.random.choice(available_patterns)
    
    def _initialize_pattern_state(self, pattern: str, position: np.ndarray,
                                velocity: np.ndarray, threat_position: np.ndarray):
        """Initialize pattern-specific state variables."""
        if pattern == 'jink':
            self.pattern_states['jink']['direction'] = np.random.choice([-1, 1])
            self.pattern_states['jink']['last_jink_time'] = 0.0
        
        elif pattern == 'spiral':
            # Set spiral axis perpendicular to velocity
            velocity_norm = normalize_vector(velocity)
            # Create perpendicular vector
            if abs(velocity_norm[2]) < 0.9:
                spiral_axis = np.cross(velocity_norm, [0, 0, 1])
            else:
                spiral_axis = np.cross(velocity_norm, [1, 0, 0])
            
            self.pattern_states['spiral']['axis'] = normalize_vector(spiral_axis)
            self.pattern_states['spiral']['center'] = position.copy()
        
        elif pattern == 'barrel_roll':
            # Set roll axis along velocity direction
            self.pattern_states['barrel_roll']['axis'] = normalize_vector(velocity)
            self.pattern_states['barrel_roll']['phase'] = 0.0
        
        elif pattern == 'weave':
            self.pattern_states['weave']['amplitude'] = self.params['jink_amplitude']
            self.pattern_states['weave']['frequency'] = self.params['jink_frequency'] * 0.7
        
        elif pattern == 'split_s':
            self.pattern_states['split_s']['phase'] = 0.0
            self.pattern_states['split_s']['target_altitude'] = position[2] - 200.0
    
    def _calculate_evasion_velocity(self, position: np.ndarray, velocity: np.ndarray,
                                  threat_position: np.ndarray, time: float) -> np.ndarray:
        """Calculate evasion velocity based on active pattern."""
        pattern = self.evasion_state['pattern']
        elapsed_time = time - self.evasion_state['start_time']
        
        if pattern == 'jink':
            return self._calculate_jink_velocity(position, velocity, threat_position, elapsed_time)
        elif pattern == 'spiral':
            return self._calculate_spiral_velocity(position, velocity, elapsed_time)
        elif pattern == 'barrel_roll':
            return self._calculate_barrel_roll_velocity(position, velocity, elapsed_time)
        elif pattern == 'weave':
            return self._calculate_weave_velocity(position, velocity, elapsed_time)
        elif pattern == 'split_s':
            return self._calculate_split_s_velocity(position, velocity, elapsed_time)
        else:
            return velocity
    
    def _calculate_jink_velocity(self, position: np.ndarray, velocity: np.ndarray,
                               threat_position: np.ndarray, elapsed_time: float) -> np.ndarray:
        """Calculate jinking evasion velocity."""
        # Jink perpendicular to threat line
        threat_vector = threat_position - position
        threat_direction = normalize_vector(threat_vector)
        
        # Find perpendicular direction
        velocity_direction = normalize_vector(velocity)
        jink_axis = np.cross(threat_direction, velocity_direction)
        if np.linalg.norm(jink_axis) < 0.1:
            # Fallback if vectors are parallel
            jink_axis = np.cross(threat_direction, [0, 0, 1])
        jink_axis = normalize_vector(jink_axis)
        
        # Jink with varying frequency
        jink_freq = self.params['jink_frequency']
        jink_amplitude = self.params['jink_amplitude']
        
        # Random jink timing
        if elapsed_time - self.pattern_states['jink']['last_jink_time'] > (1.0 / jink_freq):
            self.pattern_states['jink']['direction'] *= -1
            self.pattern_states['jink']['last_jink_time'] = elapsed_time
        
        # Calculate jink velocity
        jink_velocity = (jink_axis * self.pattern_states['jink']['direction'] * 
                        jink_amplitude * np.sin(2 * np.pi * jink_freq * elapsed_time))
        
        return velocity + jink_velocity
    
    def _calculate_spiral_velocity(self, position: np.ndarray, velocity: np.ndarray,
                                 elapsed_time: float) -> np.ndarray:
        """Calculate spiral evasion velocity."""
        spiral_rate = self.params['spiral_rate']
        spiral_radius = self.params['spiral_radius']
        
        # Current velocity magnitude
        speed = np.linalg.norm(velocity)
        
        # Spiral around the original velocity direction
        velocity_direction = normalize_vector(velocity)
        spiral_axis = self.pattern_states['spiral']['axis']
        
        # Create rotation matrix for spiral
        angle = spiral_rate * elapsed_time
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # Rodrigues' rotation formula
        cross_product = np.cross(spiral_axis, velocity_direction)
        dot_product = np.dot(spiral_axis, velocity_direction)
        
        rotated_direction = (velocity_direction * cos_angle + 
                           cross_product * sin_angle + 
                           spiral_axis * dot_product * (1 - cos_angle))
        
        # Add centripetal component
        centripetal = spiral_axis * spiral_radius * spiral_rate * np.sin(angle)
        
        return speed * rotated_direction + centripetal
    
    def _calculate_barrel_roll_velocity(self, position: np.ndarray, velocity: np.ndarray,
                                      elapsed_time: float) -> np.ndarray:
        """Calculate barrel roll evasion velocity."""
        roll_rate = self.params['barrel_roll_rate']
        roll_radius = 50.0  # Barrel roll radius
        
        # Roll around velocity vector
        velocity_direction = normalize_vector(velocity)
        speed = np.linalg.norm(velocity)
        
        # Create perpendicular vectors for roll
        if abs(velocity_direction[2]) < 0.9:
            perp1 = normalize_vector(np.cross(velocity_direction, [0, 0, 1]))
        else:
            perp1 = normalize_vector(np.cross(velocity_direction, [1, 0, 0]))
        
        perp2 = np.cross(velocity_direction, perp1)
        
        # Calculate roll offset
        roll_angle = roll_rate * elapsed_time
        roll_offset = (perp1 * np.cos(roll_angle) + perp2 * np.sin(roll_angle)) * roll_radius
        
        # Barrel roll velocity includes forward motion plus roll
        roll_velocity = roll_rate * roll_radius * (-perp1 * np.sin(roll_angle) + 
                                                  perp2 * np.cos(roll_angle))
        
        return speed * velocity_direction + roll_velocity
    
    def _calculate_weave_velocity(self, position: np.ndarray, velocity: np.ndarray,
                                elapsed_time: float) -> np.ndarray:
        """Calculate weaving evasion velocity."""
        weave_amplitude = self.pattern_states['weave']['amplitude']
        weave_frequency = self.pattern_states['weave']['frequency']
        
        # Weave perpendicular to velocity
        velocity_direction = normalize_vector(velocity)
        speed = np.linalg.norm(velocity)
        
        # Create weave axes
        if abs(velocity_direction[2]) < 0.9:
            weave_axis1 = normalize_vector(np.cross(velocity_direction, [0, 0, 1]))
        else:
            weave_axis1 = normalize_vector(np.cross(velocity_direction, [1, 0, 0]))
        
        weave_axis2 = np.cross(velocity_direction, weave_axis1)
        
        # Calculate weave components
        weave_phase1 = 2 * np.pi * weave_frequency * elapsed_time
        weave_phase2 = weave_phase1 + np.pi/2  # 90 degree phase shift
        
        weave_velocity = (weave_axis1 * weave_amplitude * np.sin(weave_phase1) +
                         weave_axis2 * weave_amplitude * 0.5 * np.sin(weave_phase2))
        
        return speed * velocity_direction + weave_velocity
    
    def _calculate_split_s_velocity(self, position: np.ndarray, velocity: np.ndarray,
                                  elapsed_time: float) -> np.ndarray:
        """Calculate split-S evasion velocity."""
        # Split-S: dive while turning
        speed = np.linalg.norm(velocity)
        velocity_direction = normalize_vector(velocity)
        
        # Calculate turn rate and dive rate
        turn_rate = 0.3  # rad/s
        dive_rate = 100.0  # m/s downward
        
        # Turn component
        turn_angle = turn_rate * elapsed_time
        if abs(velocity_direction[2]) < 0.9:
            turn_axis = normalize_vector(np.cross(velocity_direction, [0, 0, 1]))
        else:
            turn_axis = normalize_vector(np.cross(velocity_direction, [1, 0, 0]))
        
        # Rodrigues' rotation for turn
        cos_turn = np.cos(turn_angle)
        sin_turn = np.sin(turn_angle)
        
        turned_direction = (velocity_direction * cos_turn + 
                          np.cross(turn_axis, velocity_direction) * sin_turn +
                          turn_axis * np.dot(turn_axis, velocity_direction) * (1 - cos_turn))
        
        # Add dive component
        dive_velocity = np.array([0, 0, -dive_rate])
        
        return speed * turned_direction + dive_velocity
    
    def get_evasion_status(self) -> Dict[str, Any]:
        """Get current evasion status information."""
        return {
            'active': self.evasion_state['active'],
            'pattern': self.evasion_state['pattern'],
            'difficulty_level': self.difficulty_level,
            'parameters': self.params.copy(),
            'pattern_states': self.pattern_states.copy()
        }
    
    def set_difficulty_level(self, level: float):
        """Update difficulty level and recalculate parameters."""
        self.difficulty_level = clamp(level, 0.0, 1.0)
        
        # Recalculate parameters
        for key, value in self.base_params.items():
            if key in ['jink_frequency', 'spiral_rate', 'barrel_roll_rate']:
                self.params[key] = value * (0.5 + 0.5 * self.difficulty_level)
            elif key == 'evasion_threshold':
                self.params[key] = value * (0.8 + 0.4 * self.difficulty_level)
            elif key == 'reaction_time':
                self.params[key] = value * (1.5 - 0.5 * self.difficulty_level)
            else:
                self.params[key] = value