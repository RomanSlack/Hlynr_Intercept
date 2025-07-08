"""
Adversary Controller for AegisIntercept Phase 3.

This module provides intelligent adversary control combining proportional
navigation guidance with advanced evasion behaviors.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from pyquaternion import Quaternion

from .evasion_behaviors import EvasionBehaviors
from ..physics import RigidBody6DOF, DragModel, WindField
from ..utils.maths import normalize_vector, clamp, distance


class AdversaryController:
    """
    Intelligent adversary controller with guidance and evasion.
    
    This class combines proportional navigation guidance toward the target
    with sophisticated evasion behaviors when threatened by an interceptor.
    It operates using only current state information (no look-ahead) for
    computational efficiency.
    """
    
    def __init__(self,
                 rigid_body: RigidBody6DOF,
                 drag_model: DragModel,
                 difficulty_level: float = 0.5,
                 max_thrust: float = 3000.0,
                 guidance_gain: float = 3.0):
        """
        Initialize adversary controller.
        
        Args:
            rigid_body: 6-DOF rigid body physics model
            drag_model: Aerodynamic drag model
            difficulty_level: Evasion aggressiveness (0-1)
            max_thrust: Maximum thrust capability (N)
            guidance_gain: Proportional navigation gain
        """
        self.rigid_body = rigid_body
        self.drag_model = drag_model
        self.difficulty_level = clamp(difficulty_level, 0.0, 1.0)
        self.max_thrust = max_thrust
        self.guidance_gain = guidance_gain
        
        # Initialize evasion behaviors
        self.evasion = EvasionBehaviors(difficulty_level)
        
        # Controller state
        self.controller_state = {
            'guidance_mode': 'proportional_nav',
            'target_position': np.zeros(3),
            'threat_detected': False,
            'threat_position': np.zeros(3),
            'last_los_rate': np.zeros(3),
            'closing_velocity': 0.0,
            'time_to_impact': np.inf
        }
        
        # Performance parameters
        self.performance_params = {
            'thrust_efficiency': 0.9,
            'response_lag': 0.1,  # Control response delay (s)
            'guidance_noise': 0.05,  # Guidance noise level
            'sensor_noise': 0.02,   # Sensor noise level
            'max_g_limit': 20.0     # Maximum G-loading limit
        }
        
        # Scale performance by difficulty
        self.performance_params['guidance_noise'] *= (1.0 - 0.5 * difficulty_level)
        self.performance_params['response_lag'] *= (1.5 - 0.5 * difficulty_level)
        self.performance_params['max_g_limit'] *= (0.8 + 0.4 * difficulty_level)
    
    def update_control(self,
                      target_position: np.ndarray,
                      threat_position: Optional[np.ndarray] = None,
                      time: float = 0.0,
                      wind_field: Optional[WindField] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update adversary control and return force and torque commands.
        
        Args:
            target_position: Position of defended target
            threat_position: Position of interceptor (if detected)
            time: Current simulation time
            wind_field: Wind field model for environmental effects
            
        Returns:
            Tuple of (force_command, torque_command) in world frame
        """
        # Update controller state
        self.controller_state['target_position'] = target_position.copy()
        
        if threat_position is not None:
            self.controller_state['threat_detected'] = True
            self.controller_state['threat_position'] = threat_position.copy()
        else:
            self.controller_state['threat_detected'] = False
        
        # Get current state
        position = self.rigid_body.get_position()
        velocity = self.rigid_body.get_velocity()
        
        # Calculate guidance command
        guidance_acceleration = self._calculate_guidance_acceleration(
            position, velocity, target_position, time
        )
        
        # Add evasion if threat detected
        if self.controller_state['threat_detected']:
            evasion_acceleration = self._calculate_evasion_acceleration(
                position, velocity, target_position, threat_position, time
            )
            
            # Blend guidance and evasion
            evasion_weight = self.difficulty_level
            total_acceleration = ((1.0 - evasion_weight) * guidance_acceleration + 
                                evasion_weight * evasion_acceleration)
        else:
            total_acceleration = guidance_acceleration
        
        # Apply G-limit
        total_acceleration = self._apply_g_limit(total_acceleration)
        
        # Convert acceleration to force
        force_command = self._acceleration_to_force(
            total_acceleration, position, velocity, wind_field
        )
        
        # Calculate torque for attitude control
        torque_command = self._calculate_attitude_torque(
            total_acceleration, velocity
        )
        
        return force_command, torque_command
    
    def _calculate_guidance_acceleration(self,
                                       position: np.ndarray,
                                       velocity: np.ndarray,
                                       target_position: np.ndarray,
                                       time: float) -> np.ndarray:
        """Calculate proportional navigation guidance acceleration."""
        # Range and range rate to target
        range_vector = target_position - position
        range_magnitude = np.linalg.norm(range_vector)
        
        if range_magnitude < 1.0:
            return np.zeros(3)  # Already at target
        
        range_unit = range_vector / range_magnitude
        range_rate = np.dot(velocity, range_unit)
        
        # Line-of-sight rate
        los_rate = np.cross(range_vector, velocity) / (range_magnitude ** 2)
        
        # Smooth LOS rate (simple low-pass filter)
        alpha = 0.7
        self.controller_state['last_los_rate'] = (
            alpha * self.controller_state['last_los_rate'] + 
            (1 - alpha) * los_rate
        )
        
        # Proportional navigation law
        closing_velocity = -range_rate
        self.controller_state['closing_velocity'] = closing_velocity
        
        if closing_velocity > 0:
            self.controller_state['time_to_impact'] = range_magnitude / closing_velocity
        else:
            self.controller_state['time_to_impact'] = np.inf
        
        # Acceleration command (perpendicular to LOS)
        navigation_gain = self.guidance_gain
        acceleration_command = navigation_gain * closing_velocity * self.controller_state['last_los_rate']
        
        # Add guidance noise
        noise_level = self.performance_params['guidance_noise']
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, 3)
            acceleration_command += noise * np.linalg.norm(acceleration_command)
        
        return acceleration_command
    
    def _calculate_evasion_acceleration(self,
                                      position: np.ndarray,
                                      velocity: np.ndarray,
                                      target_position: np.ndarray,
                                      threat_position: np.ndarray,
                                      time: float) -> np.ndarray:
        """Calculate evasion acceleration using evasion behaviors."""
        # Get evasion velocity from behavior system
        desired_velocity, evasion_acceleration = self.evasion.update_evasion(
            position, velocity, target_position, threat_position, time
        )
        
        return evasion_acceleration
    
    def _apply_g_limit(self, acceleration: np.ndarray) -> np.ndarray:
        """Apply G-loading limits to acceleration command."""
        max_acceleration = self.performance_params['max_g_limit'] * 9.80665  # Convert G to m/s²
        
        acceleration_magnitude = np.linalg.norm(acceleration)
        if acceleration_magnitude > max_acceleration:
            return acceleration * (max_acceleration / acceleration_magnitude)
        
        return acceleration
    
    def _acceleration_to_force(self,
                             acceleration: np.ndarray,
                             position: np.ndarray,
                             velocity: np.ndarray,
                             wind_field: Optional[WindField] = None) -> np.ndarray:
        """Convert acceleration command to force command."""
        # Required force for acceleration
        mass = self.rigid_body.mass
        required_force = mass * acceleration
        
        # Account for drag and wind
        altitude = position[2]
        drag_force = self.drag_model.get_drag_force(velocity, altitude)
        
        wind_force = np.zeros(3)
        if wind_field is not None:
            wind_force = wind_field.get_wind_force(
                velocity, position, 0.0, self.drag_model
            )
        
        # Gravity compensation
        gravity_force = mass * np.array([0, 0, -9.80665])
        
        # Total force needed
        total_force = required_force - drag_force - wind_force - gravity_force
        
        # Limit to maximum thrust
        force_magnitude = np.linalg.norm(total_force)
        if force_magnitude > self.max_thrust:
            total_force = total_force * (self.max_thrust / force_magnitude)
        
        # Apply efficiency factor
        efficiency = self.performance_params['thrust_efficiency']
        total_force *= efficiency
        
        return total_force
    
    def _calculate_attitude_torque(self,
                                 acceleration: np.ndarray,
                                 velocity: np.ndarray) -> np.ndarray:
        """Calculate torque for attitude control."""
        # Simple attitude control to align with acceleration
        if np.linalg.norm(acceleration) < 1.0:
            return np.zeros(3)
        
        # Desired attitude: align body x-axis with acceleration
        desired_forward = normalize_vector(acceleration)
        
        # Current attitude
        current_attitude = self.rigid_body.get_attitude_matrix()
        current_forward = current_attitude[:, 0]  # Body x-axis in world frame
        
        # Calculate rotation needed
        cross_product = np.cross(current_forward, desired_forward)
        sin_angle = np.linalg.norm(cross_product)
        
        if sin_angle < 1e-6:
            return np.zeros(3)  # Already aligned
        
        # Rotation axis
        rotation_axis = cross_product / sin_angle
        
        # Convert to body frame
        rotation_axis_body = self.rigid_body.world_to_body(rotation_axis)
        
        # Proportional control
        torque_gain = 1000.0  # N⋅m per radian
        max_torque = 200.0    # Maximum torque
        
        torque_magnitude = min(torque_gain * sin_angle, max_torque)
        torque_command = rotation_axis_body * torque_magnitude
        
        return torque_command
    
    def get_guidance_info(self) -> Dict[str, Any]:
        """Get current guidance information for analysis."""
        position = self.rigid_body.get_position()
        velocity = self.rigid_body.get_velocity()
        
        target_range = distance(position, self.controller_state['target_position'])
        
        info = {
            'guidance_mode': self.controller_state['guidance_mode'],
            'target_range': target_range,
            'closing_velocity': self.controller_state['closing_velocity'],
            'time_to_impact': self.controller_state['time_to_impact'],
            'threat_detected': self.controller_state['threat_detected'],
            'evasion_status': self.evasion.get_evasion_status(),
            'speed': np.linalg.norm(velocity),
            'altitude': position[2]
        }
        
        if self.controller_state['threat_detected']:
            threat_range = distance(position, self.controller_state['threat_position'])
            info['threat_range'] = threat_range
        
        return info
    
    def set_difficulty_level(self, level: float):
        """Update difficulty level for both controller and evasion."""
        self.difficulty_level = clamp(level, 0.0, 1.0)
        self.evasion.set_difficulty_level(level)
        
        # Update performance parameters
        base_params = {
            'guidance_noise': 0.05,
            'response_lag': 0.1,
            'max_g_limit': 20.0
        }
        
        self.performance_params['guidance_noise'] = base_params['guidance_noise'] * (1.0 - 0.5 * level)
        self.performance_params['response_lag'] = base_params['response_lag'] * (1.5 - 0.5 * level)
        self.performance_params['max_g_limit'] = base_params['max_g_limit'] * (0.8 + 0.4 * level)
    
    def set_guidance_mode(self, mode: str):
        """Set guidance mode."""
        valid_modes = ['proportional_nav', 'pure_pursuit', 'lead_pursuit']
        if mode in valid_modes:
            self.controller_state['guidance_mode'] = mode
    
    def reset_controller(self):
        """Reset controller state."""
        self.controller_state.update({
            'target_position': np.zeros(3),
            'threat_detected': False,
            'threat_position': np.zeros(3),
            'last_los_rate': np.zeros(3),
            'closing_velocity': 0.0,
            'time_to_impact': np.inf
        })
        
        # Reset evasion behaviors
        self.evasion.evasion_state.update({
            'active': False,
            'pattern': None,
            'phase': 0.0
        })