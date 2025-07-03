"""
Enhanced Adversary System for AegisIntercept Phase 3

This module provides advanced 6DOF adversary (missile) behavior with
sophisticated evasive maneuvers, threat assessment, and configurable
complexity levels. The adversary system simulates realistic missile
behavior including barrel rolls, spirals, jinking, and adaptive responses.

Features:
- 6DOF evasive maneuvers (barrel rolls, spirals, jinking)
- Threat assessment and reactive behaviors
- Parameter-driven complexity from JSON configs
- Adaptive AI that responds to interceptor behavior
- Realistic missile guidance and control systems
- Performance-based difficulty scaling

Author: Coder Agent
Date: Phase 3 Implementation
"""

import numpy as np
import json
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path

from ..utils.physics6dof import (
    RigidBody6DOF, VehicleType, QuaternionUtils, intercept_geometry_6dof,
    distance_6dof, AtmosphericModel
)


class EvasionPattern(Enum):
    """Types of evasive maneuvers"""
    NONE = "none"
    BARREL_ROLL = "barrel_roll"
    SPIRAL = "spiral"
    WEAVE = "weave"
    JINK = "jink"
    CORKSCREW = "corkscrew"
    SPLIT_S = "split_s"
    IMMELMANN = "immelmann"
    RANDOM_WALK = "random_walk"
    DEFENSIVE_SPIRAL = "defensive_spiral"
    CHAFF_DEPLOYMENT = "chaff_deployment"


class ThreatLevel(Enum):
    """Threat assessment levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class GuidanceMode(Enum):
    """Missile guidance modes"""
    BALLISTIC = "ballistic"           # Simple ballistic trajectory
    PROPORTIONAL_NAVIGATION = "proportional_navigation"  # Basic PN guidance
    AUGMENTED_PN = "augmented_PN"     # Augmented proportional navigation
    OPTIMAL_GUIDANCE = "optimal_guidance"  # Optimal intercept guidance
    TERRAIN_FOLLOWING = "terrain_following"  # Low-altitude terrain following


@dataclass
class ThreatAssessment:
    """Threat assessment data structure"""
    overall_threat_level: ThreatLevel
    distance_threat: float        # 0-1, closer = higher threat
    closing_velocity_threat: float  # 0-1, faster closing = higher
    intercept_probability: float  # 0-1, probability of successful intercept
    time_to_intercept: float      # Estimated time to intercept (seconds)
    escape_possibility: float     # 0-1, chance of successful evasion
    
    # Threat components
    kinematic_threat: float       # Based on relative motion
    geometric_threat: float       # Based on intercept geometry
    performance_threat: float     # Based on interceptor capability
    
    # Recommended response
    recommended_evasion: EvasionPattern
    recommended_intensity: float  # 0-1, how aggressive the response should be


@dataclass
class EvasionConfig:
    """Configuration for evasive maneuvers"""
    pattern: EvasionPattern
    intensity: float = 1.0        # 0-1, maneuver intensity
    duration: float = 5.0         # Duration in seconds
    frequency: float = 2.0        # Oscillation frequency (Hz)
    amplitude: float = 1.0        # Amplitude multiplier
    randomness: float = 0.1       # 0-1, amount of randomness
    
    # Pattern-specific parameters
    roll_rate: float = 180.0      # degrees/second for barrel roll
    spiral_radius: float = 50.0   # meters for spiral maneuvers
    jink_period: float = 1.0      # seconds for jinking
    weave_amplitude: float = 30.0 # meters for weaving


@dataclass
class AdversaryConfig:
    """Complete adversary configuration"""
    # Basic parameters
    aggressiveness: float = 1.0   # Overall aggressiveness (0-2)
    intelligence: float = 1.0     # AI intelligence level (0-2)
    skill_level: float = 1.0      # Pilot skill level (0-2)
    
    # Guidance system
    guidance_mode: GuidanceMode = GuidanceMode.PROPORTIONAL_NAVIGATION
    navigation_gain: float = 3.0  # PN guidance gain
    guidance_accuracy: float = 0.95  # 0-1, guidance system accuracy
    
    # Threat response
    threat_detection_range: float = 500.0  # meters
    reaction_time: float = 0.5    # seconds delay for threat response
    threat_threshold: float = 0.3  # Threat level to trigger evasion
    
    # Evasion capabilities
    max_g_force: float = 15.0     # Maximum G-force capability
    max_roll_rate: float = 360.0  # degrees/second
    energy_management: float = 1.0  # Energy conservation factor
    
    # Available evasion patterns
    available_patterns: List[EvasionPattern] = field(default_factory=lambda: [
        EvasionPattern.BARREL_ROLL,
        EvasionPattern.SPIRAL,
        EvasionPattern.WEAVE,
        EvasionPattern.JINK
    ])
    
    # Adaptation parameters
    learning_rate: float = 0.1    # How quickly to adapt to threats
    memory_length: int = 100      # Steps to remember for adaptation
    
    @classmethod
    def from_json(cls, config_path: str) -> 'AdversaryConfig':
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        # Convert pattern strings to enums
        if 'available_patterns' in data:
            data['available_patterns'] = [EvasionPattern(p) for p in data['available_patterns']]
        if 'guidance_mode' in data:
            data['guidance_mode'] = GuidanceMode(data['guidance_mode'])
        
        return cls(**data)


class AdversaryBehaviorModel:
    """Models adversary behavior patterns and learning"""
    
    def __init__(self, config: AdversaryConfig):
        self.config = config
        
        # Behavior history
        self.threat_history: List[ThreatAssessment] = []
        self.evasion_history: List[EvasionConfig] = []
        self.performance_history: List[float] = []
        
        # Current state
        self.current_evasion: Optional[EvasionConfig] = None
        self.evasion_start_time: float = 0.0
        self.last_threat_assessment: Optional[ThreatAssessment] = None
        
        # Adaptation parameters
        self.pattern_success_rates: Dict[EvasionPattern, float] = {
            pattern: 0.5 for pattern in self.config.available_patterns
        }
        self.pattern_usage_count: Dict[EvasionPattern, int] = {
            pattern: 0 for pattern in self.config.available_patterns
        }
        
        # Internal state
        self.energy_level: float = 1.0  # Current energy level (0-1)
        self.stress_level: float = 0.0  # Current stress level (0-1)
        
    def assess_threat(self, 
                     missile_state: np.ndarray,
                     interceptor_state: np.ndarray,
                     target_position: np.ndarray,
                     dt: float) -> ThreatAssessment:
        """Assess current threat level and recommend response"""
        
        # Extract positions and velocities
        missile_pos = missile_state[0:3]
        missile_vel = missile_state[3:6]
        interceptor_pos = interceptor_state[0:3]
        interceptor_vel = interceptor_state[3:6]
        
        # Calculate basic metrics
        distance = distance_6dof(missile_pos, interceptor_pos)
        intercept_geometry = intercept_geometry_6dof(
            missile_pos, missile_vel, interceptor_pos, interceptor_vel
        )
        
        # Distance threat (closer = higher)
        distance_threat = max(0, 1.0 - distance / self.config.threat_detection_range)
        
        # Closing velocity threat
        closing_velocity = intercept_geometry['closing_velocity']
        closing_velocity_threat = min(1.0, max(0, closing_velocity / 100.0))
        
        # Time to intercept
        time_to_intercept = intercept_geometry['time_to_closest_approach']
        if time_to_intercept <= 0 or time_to_intercept > 100:
            time_to_intercept = float('inf')
        
        # Kinematic threat assessment
        relative_velocity = np.linalg.norm(interceptor_vel - missile_vel)
        kinematic_threat = min(1.0, relative_velocity / 200.0)
        
        # Geometric threat (based on intercept angles)
        aspect_angle = intercept_geometry['aspect_angle']
        geometric_threat = 1.0 - abs(aspect_angle) / np.pi  # Head-on is more threatening
        
        # Performance threat (based on interceptor capability)
        interceptor_speed = np.linalg.norm(interceptor_vel)
        missile_speed = np.linalg.norm(missile_vel)
        performance_threat = min(1.0, interceptor_speed / max(missile_speed, 1.0))
        
        # Overall threat calculation
        threat_components = [distance_threat, closing_velocity_threat, kinematic_threat, 
                           geometric_threat, performance_threat]
        overall_threat_value = np.mean(threat_components) * self.config.intelligence
        
        # Determine threat level
        if overall_threat_value < 0.2:
            threat_level = ThreatLevel.NONE
        elif overall_threat_value < 0.4:
            threat_level = ThreatLevel.LOW
        elif overall_threat_value < 0.6:
            threat_level = ThreatLevel.MEDIUM
        elif overall_threat_value < 0.8:
            threat_level = ThreatLevel.HIGH
        else:
            threat_level = ThreatLevel.CRITICAL
        
        # Intercept probability estimation
        if time_to_intercept == float('inf'):
            intercept_probability = 0.0
        else:
            # Simple model based on geometry and capabilities
            intercept_probability = min(1.0, (closing_velocity_threat + geometric_threat) / 2.0)
        
        # Escape possibility (inverse of intercept probability with skill factor)
        escape_possibility = (1.0 - intercept_probability) * self.config.skill_level
        
        # Recommend evasion pattern
        recommended_evasion, recommended_intensity = self._recommend_evasion(
            threat_level, overall_threat_value, time_to_intercept
        )
        
        # Create threat assessment
        assessment = ThreatAssessment(
            overall_threat_level=threat_level,
            distance_threat=distance_threat,
            closing_velocity_threat=closing_velocity_threat,
            intercept_probability=intercept_probability,
            time_to_intercept=time_to_intercept,
            escape_possibility=escape_possibility,
            kinematic_threat=kinematic_threat,
            geometric_threat=geometric_threat,
            performance_threat=performance_threat,
            recommended_evasion=recommended_evasion,
            recommended_intensity=recommended_intensity
        )
        
        # Update history
        self.threat_history.append(assessment)
        if len(self.threat_history) > self.config.memory_length:
            self.threat_history = self.threat_history[-self.config.memory_length:]
        
        self.last_threat_assessment = assessment
        return assessment
    
    def _recommend_evasion(self, 
                          threat_level: ThreatLevel, 
                          threat_value: float,
                          time_to_intercept: float) -> Tuple[EvasionPattern, float]:
        """Recommend evasion pattern based on threat assessment"""
        
        if threat_level == ThreatLevel.NONE:
            return EvasionPattern.NONE, 0.0
        
        # Consider energy and stress levels
        energy_factor = self.energy_level * self.config.energy_management
        stress_factor = 1.0 + self.stress_level * 0.5
        
        # Base intensity
        intensity = threat_value * self.config.aggressiveness * stress_factor
        intensity = min(1.0, intensity * energy_factor)
        
        # Pattern selection based on threat characteristics and success history
        if threat_level == ThreatLevel.CRITICAL or time_to_intercept < 5.0:
            # Emergency maneuvers
            candidates = [EvasionPattern.JINK, EvasionPattern.BARREL_ROLL, EvasionPattern.RANDOM_WALK]
        elif threat_level == ThreatLevel.HIGH:
            # Aggressive evasion
            candidates = [EvasionPattern.SPIRAL, EvasionPattern.CORKSCREW, EvasionPattern.BARREL_ROLL]
        elif threat_level == ThreatLevel.MEDIUM:
            # Moderate evasion
            candidates = [EvasionPattern.WEAVE, EvasionPattern.SPIRAL, EvasionPattern.DEFENSIVE_SPIRAL]
        else:  # LOW threat
            # Light evasion
            candidates = [EvasionPattern.WEAVE, EvasionPattern.RANDOM_WALK]
        
        # Filter by available patterns
        available_candidates = [p for p in candidates if p in self.config.available_patterns]
        if not available_candidates:
            available_candidates = self.config.available_patterns
        
        # Select pattern based on success rates
        if self.config.intelligence > 1.0:  # Intelligent adversary learns
            best_pattern = max(available_candidates, 
                             key=lambda p: self.pattern_success_rates.get(p, 0.5))
        else:
            # Random or simple selection
            best_pattern = np.random.choice(available_candidates)
        
        return best_pattern, intensity
    
    def update_performance(self, success: bool, pattern_used: EvasionPattern):
        """Update performance history and pattern success rates"""
        self.performance_history.append(1.0 if success else 0.0)
        
        if len(self.performance_history) > self.config.memory_length:
            self.performance_history = self.performance_history[-self.config.memory_length:]
        
        # Update pattern success rates
        if pattern_used in self.pattern_success_rates:
            current_rate = self.pattern_success_rates[pattern_used]
            new_rate = current_rate + self.config.learning_rate * (
                (1.0 if success else 0.0) - current_rate
            )
            self.pattern_success_rates[pattern_used] = np.clip(new_rate, 0.0, 1.0)
            self.pattern_usage_count[pattern_used] += 1
    
    def update_stress_and_energy(self, threat_level: ThreatLevel, dt: float):
        """Update stress and energy levels"""
        # Stress increases with threat
        stress_increase = threat_level.value * 0.1 * dt
        self.stress_level = min(1.0, self.stress_level + stress_increase)
        
        # Stress naturally decreases over time
        stress_decay = 0.05 * dt
        self.stress_level = max(0.0, self.stress_level - stress_decay)
        
        # Energy decreases with evasive maneuvers
        if self.current_evasion and self.current_evasion.pattern != EvasionPattern.NONE:
            energy_consumption = self.current_evasion.intensity * 0.02 * dt
            self.energy_level = max(0.1, self.energy_level - energy_consumption)
        
        # Energy naturally recovers
        energy_recovery = 0.01 * dt
        self.energy_level = min(1.0, self.energy_level + energy_recovery)


class EnhancedAdversary:
    """Enhanced adversary system with 6DOF evasive behaviors"""
    
    def __init__(self, 
                 missile_6dof: RigidBody6DOF,
                 config: Optional[AdversaryConfig] = None,
                 target_position: Optional[np.ndarray] = None):
        """
        Initialize enhanced adversary system
        
        Args:
            missile_6dof: 6DOF missile physics object
            config: Adversary configuration
            target_position: Target position vector
        """
        self.missile_6dof = missile_6dof
        self.config = config or AdversaryConfig()
        self.target_position = target_position or np.zeros(3)
        
        # Behavior model
        self.behavior_model = AdversaryBehaviorModel(self.config)
        
        # Guidance system
        self.guidance_system = self._initialize_guidance_system()
        
        # Current maneuver state
        self.current_maneuver_time = 0.0
        self.maneuver_phase = 0.0
        self.base_guidance_force = np.zeros(3)
        self.evasion_force = np.zeros(3)
        self.evasion_torque = np.zeros(3)
        
        # Threat tracking
        self.last_threat_update = 0.0
        self.threat_reaction_delay = self.config.reaction_time
        
        print(f"Enhanced Adversary initialized: guidance={self.config.guidance_mode.value}, "
              f"skill_level={self.config.skill_level}")
    
    def _initialize_guidance_system(self) -> Dict[str, Any]:
        """Initialize guidance system parameters"""
        return {
            'mode': self.config.guidance_mode,
            'navigation_gain': self.config.navigation_gain,
            'previous_los_rate': np.zeros(3),
            'integrated_los_error': np.zeros(3),
            'guidance_commands': np.zeros(3)
        }
    
    def update(self, 
               interceptor_state: np.ndarray,
               dt: float,
               simulation_time: float,
               wind_velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update adversary behavior and return control commands
        
        Args:
            interceptor_state: 13D interceptor state vector
            dt: Time step
            simulation_time: Current simulation time
            wind_velocity: Wind velocity vector
            
        Returns:
            Tuple of (thrust_force, control_torque) in body frame
        """
        # Get current missile state
        missile_state = self.missile_6dof.get_state_vector()
        
        # Assess threat periodically
        if simulation_time - self.last_threat_update > 0.1:  # Update every 100ms
            threat = self.behavior_model.assess_threat(
                missile_state, interceptor_state, self.target_position, dt
            )
            self.last_threat_update = simulation_time
            
            # Update stress and energy
            self.behavior_model.update_stress_and_energy(threat.overall_threat_level, dt)
            
            # Check if new evasive maneuver is needed
            if (threat.overall_threat_level.value >= self.config.threat_threshold * 5 and
                (self.behavior_model.current_evasion is None or 
                 simulation_time - self.behavior_model.evasion_start_time > 
                 self.behavior_model.current_evasion.duration)):
                
                self._start_evasive_maneuver(threat, simulation_time)
        
        # Calculate base guidance toward target
        guidance_force = self._calculate_guidance_force(missile_state, dt)
        
        # Calculate evasive forces and torques
        if self.behavior_model.current_evasion:
            evasion_force, evasion_torque = self._calculate_evasive_maneuver(
                missile_state, simulation_time, dt
            )
        else:
            evasion_force, evasion_torque = np.zeros(3), np.zeros(3)
        
        # Combine guidance and evasion
        total_force = guidance_force + evasion_force
        total_torque = evasion_torque + self._calculate_stability_torque(missile_state)
        
        # Apply limitations based on missile capabilities
        total_force = self._limit_control_forces(total_force)
        total_torque = self._limit_control_torques(total_torque)
        
        return total_force, total_torque
    
    def _calculate_guidance_force(self, missile_state: np.ndarray, dt: float) -> np.ndarray:
        """Calculate guidance force toward target"""
        missile_pos = missile_state[0:3]
        missile_vel = missile_state[3:6]
        
        # Vector to target
        to_target = self.target_position - missile_pos
        target_distance = np.linalg.norm(to_target)
        
        if target_distance < 1e-6:
            return np.zeros(3)
        
        to_target_unit = to_target / target_distance
        
        if self.config.guidance_mode == GuidanceMode.BALLISTIC:
            # Simple ballistic guidance
            desired_velocity = to_target_unit * 40.0  # Desired speed
            velocity_error = desired_velocity - missile_vel
            guidance_force = velocity_error * 500.0  # Proportional control
            
        elif self.config.guidance_mode == GuidanceMode.PROPORTIONAL_NAVIGATION:
            # Proportional Navigation guidance
            line_of_sight = to_target_unit
            
            # Calculate line-of-sight rate
            if hasattr(self, 'previous_los') and dt > 0:
                los_rate = (line_of_sight - self.previous_los) / dt
            else:
                los_rate = np.zeros(3)
            
            self.previous_los = line_of_sight.copy()
            
            # PN command
            closing_velocity = -np.dot(missile_vel, line_of_sight)
            pn_acceleration = self.config.navigation_gain * closing_velocity * los_rate
            
            guidance_force = pn_acceleration * self.missile_6dof.aero_props.mass
            
        elif self.config.guidance_mode == GuidanceMode.AUGMENTED_PN:
            # Augmented PN with gravity compensation
            line_of_sight = to_target_unit
            
            if hasattr(self, 'previous_los') and dt > 0:
                los_rate = (line_of_sight - self.previous_los) / dt
            else:
                los_rate = np.zeros(3)
            
            self.previous_los = line_of_sight.copy()
            
            closing_velocity = -np.dot(missile_vel, line_of_sight)
            pn_acceleration = self.config.navigation_gain * closing_velocity * los_rate
            
            # Add gravity compensation
            gravity_compensation = np.array([0, 0, 9.81])
            
            guidance_force = (pn_acceleration + gravity_compensation) * self.missile_6dof.aero_props.mass
            
        else:
            # Default to simple proportional guidance
            velocity_error = to_target_unit * 35.0 - missile_vel
            guidance_force = velocity_error * 300.0
        
        # Apply guidance accuracy
        if self.config.guidance_accuracy < 1.0:
            noise_factor = 1.0 - self.config.guidance_accuracy
            noise = np.random.normal(0, noise_factor * 0.1, 3)
            guidance_force = guidance_force * (1.0 + noise)
        
        return guidance_force
    
    def _start_evasive_maneuver(self, threat: ThreatAssessment, simulation_time: float):
        """Start a new evasive maneuver"""
        pattern = threat.recommended_evasion
        intensity = threat.recommended_intensity
        
        # Create evasion configuration
        evasion_config = EvasionConfig(
            pattern=pattern,
            intensity=intensity * self.config.aggressiveness,
            duration=np.random.uniform(3.0, 8.0),  # Variable duration
            frequency=np.random.uniform(1.0, 4.0),  # Variable frequency
            amplitude=intensity * np.random.uniform(0.8, 1.2),
            randomness=min(0.3, intensity * 0.5)
        )
        
        # Pattern-specific adjustments
        if pattern == EvasionPattern.BARREL_ROLL:
            evasion_config.roll_rate = intensity * self.config.max_roll_rate
            evasion_config.duration = np.random.uniform(2.0, 4.0)
        elif pattern == EvasionPattern.SPIRAL:
            evasion_config.spiral_radius = 30.0 + intensity * 50.0
            evasion_config.frequency = 0.5 + intensity * 1.5
        elif pattern == EvasionPattern.JINK:
            evasion_config.jink_period = 0.5 + (1.0 - intensity) * 1.0
            evasion_config.amplitude = intensity * 2.0
        
        # Set current evasion
        self.behavior_model.current_evasion = evasion_config
        self.behavior_model.evasion_start_time = simulation_time
        self.current_maneuver_time = 0.0
        
        print(f"Starting evasive maneuver: {pattern.value}, intensity={intensity:.2f}, duration={evasion_config.duration:.1f}s")
    
    def _calculate_evasive_maneuver(self, 
                                   missile_state: np.ndarray, 
                                   simulation_time: float,
                                   dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate evasive maneuver forces and torques"""
        evasion = self.behavior_model.current_evasion
        if not evasion:
            return np.zeros(3), np.zeros(3)
        
        # Update maneuver time
        self.current_maneuver_time += dt
        
        # Check if maneuver is complete
        if self.current_maneuver_time > evasion.duration:
            self.behavior_model.current_evasion = None
            return np.zeros(3), np.zeros(3)
        
        # Calculate phase for oscillatory patterns
        phase = 2 * np.pi * evasion.frequency * self.current_maneuver_time
        
        # Generate pattern-specific commands
        if evasion.pattern == EvasionPattern.BARREL_ROLL:
            return self._barrel_roll_maneuver(evasion, phase, dt)
        elif evasion.pattern == EvasionPattern.SPIRAL:
            return self._spiral_maneuver(evasion, phase, missile_state)
        elif evasion.pattern == EvasionPattern.WEAVE:
            return self._weave_maneuver(evasion, phase)
        elif evasion.pattern == EvasionPattern.JINK:
            return self._jink_maneuver(evasion, simulation_time)
        elif evasion.pattern == EvasionPattern.CORKSCREW:
            return self._corkscrew_maneuver(evasion, phase)
        elif evasion.pattern == EvasionPattern.RANDOM_WALK:
            return self._random_walk_maneuver(evasion, dt)
        elif evasion.pattern == EvasionPattern.DEFENSIVE_SPIRAL:
            return self._defensive_spiral_maneuver(evasion, phase, missile_state)
        else:
            return np.zeros(3), np.zeros(3)
    
    def _barrel_roll_maneuver(self, evasion: EvasionConfig, phase: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate barrel roll maneuver"""
        roll_rate = np.radians(evasion.roll_rate) * evasion.intensity
        
        # Lateral force for barrel roll
        lateral_force = evasion.amplitude * 800.0 * np.array([
            0,
            np.sin(phase) * evasion.intensity,
            np.cos(phase) * evasion.intensity
        ])
        
        # Roll torque
        roll_torque = np.array([roll_rate * 100.0, 0, 0])
        
        # Add randomness
        if evasion.randomness > 0:
            noise = np.random.normal(0, evasion.randomness * 0.1, 3)
            lateral_force += noise * 200.0
        
        return lateral_force, roll_torque
    
    def _spiral_maneuver(self, evasion: EvasionConfig, phase: float, missile_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate spiral maneuver"""
        spiral_force = evasion.amplitude * 600.0 * np.array([
            0,  # No longitudinal force change
            evasion.spiral_radius * np.sin(phase) / 50.0,
            evasion.spiral_radius * np.cos(phase) / 50.0
        ])
        
        # Torque for coordinated turn
        spiral_torque = np.array([
            np.sin(phase) * 50.0 * evasion.intensity,
            np.cos(phase) * 30.0 * evasion.intensity,
            -np.sin(phase) * 40.0 * evasion.intensity
        ])
        
        return spiral_force, spiral_torque
    
    def _weave_maneuver(self, evasion: EvasionConfig, phase: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate weaving maneuver"""
        weave_force = evasion.amplitude * 400.0 * np.array([
            0,
            evasion.weave_amplitude * np.sin(phase) / 30.0,
            0.3 * evasion.weave_amplitude * np.sin(phase * 2) / 30.0
        ])
        
        weave_torque = np.array([
            0,
            np.sin(phase) * 20.0 * evasion.intensity,
            np.cos(phase) * 15.0 * evasion.intensity
        ])
        
        return weave_force, weave_torque
    
    def _jink_maneuver(self, evasion: EvasionConfig, simulation_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate jinking maneuver"""
        # Rapid direction changes
        jink_time = simulation_time % evasion.jink_period
        direction = 1.0 if jink_time < evasion.jink_period / 2 else -1.0
        
        jink_force = direction * evasion.amplitude * 1000.0 * np.array([
            0,
            np.random.choice([-1, 1]) * evasion.intensity,
            np.random.choice([-1, 1]) * evasion.intensity * 0.7
        ])
        
        jink_torque = direction * np.array([
            np.random.uniform(-100, 100) * evasion.intensity,
            np.random.uniform(-50, 50) * evasion.intensity,
            np.random.uniform(-80, 80) * evasion.intensity
        ])
        
        return jink_force, jink_torque
    
    def _corkscrew_maneuver(self, evasion: EvasionConfig, phase: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate corkscrew maneuver"""
        # Combined spiral and roll
        corkscrew_force = evasion.amplitude * 500.0 * np.array([
            0,
            np.sin(phase) * evasion.intensity,
            np.cos(phase) * evasion.intensity
        ])
        
        corkscrew_torque = np.array([
            np.sin(phase * 2) * 80.0 * evasion.intensity,  # Roll
            np.cos(phase) * 40.0 * evasion.intensity,      # Pitch
            np.sin(phase) * 60.0 * evasion.intensity       # Yaw
        ])
        
        return corkscrew_force, corkscrew_torque
    
    def _random_walk_maneuver(self, evasion: EvasionConfig, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random walk maneuver"""
        # Brownian motion with bias
        random_force = np.random.normal(0, evasion.amplitude * 300.0, 3)
        random_force[0] *= 0.3  # Reduce longitudinal randomness
        
        random_torque = np.random.normal(0, evasion.intensity * 30.0, 3)
        
        return random_force, random_torque
    
    def _defensive_spiral_maneuver(self, evasion: EvasionConfig, phase: float, missile_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate defensive spiral (climbing spiral)"""
        # Spiral with climbing component
        defensive_force = evasion.amplitude * 700.0 * np.array([
            -0.2 * evasion.intensity,  # Slight deceleration
            np.sin(phase) * evasion.intensity,
            0.5 + 0.3 * np.cos(phase) * evasion.intensity  # Climbing bias
        ])
        
        defensive_torque = np.array([
            np.sin(phase) * 60.0 * evasion.intensity,
            0.2 * evasion.intensity * 30.0,  # Slight pitch up
            -np.sin(phase) * 50.0 * evasion.intensity
        ])
        
        return defensive_force, defensive_torque
    
    def _calculate_stability_torque(self, missile_state: np.ndarray) -> np.ndarray:
        """Calculate stability torque to maintain controlled flight"""
        angular_velocity = missile_state[10:13]
        
        # Damping torque to prevent excessive rotation
        damping_torque = -angular_velocity * 10.0 * self.config.skill_level
        
        return damping_torque
    
    def _limit_control_forces(self, force: np.ndarray) -> np.ndarray:
        """Limit control forces based on missile capabilities"""
        max_force = self.config.max_g_force * self.missile_6dof.aero_props.mass * 9.81
        
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > max_force:
            force = force / force_magnitude * max_force
        
        return force
    
    def _limit_control_torques(self, torque: np.ndarray) -> np.ndarray:
        """Limit control torques based on missile capabilities"""
        max_torque = np.array([
            np.radians(self.config.max_roll_rate) * 50.0,  # Roll
            np.radians(self.config.max_roll_rate) * 30.0,  # Pitch
            np.radians(self.config.max_roll_rate) * 40.0   # Yaw
        ])
        
        limited_torque = np.clip(torque, -max_torque, max_torque)
        return limited_torque
    
    def get_behavior_stats(self) -> Dict[str, Any]:
        """Get current behavior statistics"""
        return {
            'current_evasion': self.behavior_model.current_evasion.pattern.value if self.behavior_model.current_evasion else 'none',
            'threat_level': self.behavior_model.last_threat_assessment.overall_threat_level.value if self.behavior_model.last_threat_assessment else 0,
            'energy_level': self.behavior_model.energy_level,
            'stress_level': self.behavior_model.stress_level,
            'pattern_success_rates': {p.value: rate for p, rate in self.behavior_model.pattern_success_rates.items()},
            'pattern_usage_count': {p.value: count for p, count in self.behavior_model.pattern_usage_count.items()}
        }
    
    def update_performance_feedback(self, mission_outcome: str):
        """Update adversary performance based on mission outcome"""
        success = mission_outcome in ['missile_hit_target', 'intercept_failed']
        
        if self.behavior_model.current_evasion:
            pattern_used = self.behavior_model.current_evasion.pattern
        else:
            pattern_used = EvasionPattern.NONE
        
        self.behavior_model.update_performance(success, pattern_used)
        
        print(f"Adversary performance updated: {mission_outcome}, pattern={pattern_used.value}, success={success}")
    
    def load_config(self, config_path: str):
        """Load adversary configuration from file"""
        self.config = AdversaryConfig.from_json(config_path)
        self.behavior_model = AdversaryBehaviorModel(self.config)
        print(f"Loaded adversary configuration from {config_path}")
    
    def save_config(self, config_path: str):
        """Save current adversary configuration to file"""
        config_dict = {
            'aggressiveness': self.config.aggressiveness,
            'intelligence': self.config.intelligence,
            'skill_level': self.config.skill_level,
            'guidance_mode': self.config.guidance_mode.value,
            'navigation_gain': self.config.navigation_gain,
            'guidance_accuracy': self.config.guidance_accuracy,
            'threat_detection_range': self.config.threat_detection_range,
            'reaction_time': self.config.reaction_time,
            'threat_threshold': self.config.threat_threshold,
            'max_g_force': self.config.max_g_force,
            'max_roll_rate': self.config.max_roll_rate,
            'energy_management': self.config.energy_management,
            'available_patterns': [p.value for p in self.config.available_patterns],
            'learning_rate': self.config.learning_rate,
            'memory_length': self.config.memory_length
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Saved adversary configuration to {config_path}")


# Utility functions
def create_default_adversary_config(difficulty_level: str = "medium") -> AdversaryConfig:
    """Create default adversary configuration for different difficulty levels"""
    
    if difficulty_level == "easy":
        return AdversaryConfig(
            aggressiveness=0.5,
            intelligence=0.5,
            skill_level=0.5,
            guidance_mode=GuidanceMode.BALLISTIC,
            max_g_force=8.0,
            available_patterns=[EvasionPattern.WEAVE, EvasionPattern.RANDOM_WALK],
            learning_rate=0.05
        )
    elif difficulty_level == "medium":
        return AdversaryConfig(
            aggressiveness=1.0,
            intelligence=1.0,
            skill_level=1.0,
            guidance_mode=GuidanceMode.PROPORTIONAL_NAVIGATION,
            max_g_force=12.0,
            available_patterns=[EvasionPattern.BARREL_ROLL, EvasionPattern.SPIRAL, 
                              EvasionPattern.WEAVE, EvasionPattern.JINK],
            learning_rate=0.1
        )
    elif difficulty_level == "hard":
        return AdversaryConfig(
            aggressiveness=1.5,
            intelligence=1.5,
            skill_level=1.5,
            guidance_mode=GuidanceMode.AUGMENTED_PN,
            max_g_force=18.0,
            available_patterns=[EvasionPattern.BARREL_ROLL, EvasionPattern.SPIRAL,
                              EvasionPattern.CORKSCREW, EvasionPattern.JINK,
                              EvasionPattern.DEFENSIVE_SPIRAL],
            learning_rate=0.15
        )
    else:  # expert
        return AdversaryConfig(
            aggressiveness=2.0,
            intelligence=2.0,
            skill_level=2.0,
            guidance_mode=GuidanceMode.OPTIMAL_GUIDANCE,
            max_g_force=25.0,
            available_patterns=list(EvasionPattern)[1:],  # All except NONE
            learning_rate=0.2
        )


def create_enhanced_adversary(missile_6dof: RigidBody6DOF,
                            target_position: np.ndarray,
                            difficulty_level: str = "medium") -> EnhancedAdversary:
    """Create an enhanced adversary with specified difficulty"""
    config = create_default_adversary_config(difficulty_level)
    return EnhancedAdversary(missile_6dof, config, target_position)