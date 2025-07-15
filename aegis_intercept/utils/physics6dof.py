"""
6-DOF Rigid Body Physics for AegisIntercept Phase 3.

This module provides 6-DOF rigid body dynamics with quaternion-based
attitude representation for realistic missile simulation.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from pyquaternion import Quaternion


class RigidBody6DOF:
    """
    6-DOF rigid body dynamics with quaternion attitude representation.
    
    This class provides complete 6-DOF rigid body simulation including:
    - Position and velocity dynamics
    - Quaternion-based attitude representation
    - Angular velocity and acceleration
    - Mass and inertia properties
    - External force and torque application
    """
    
    def __init__(self, 
                 mass: float = 50.0,
                 inertia: Optional[np.ndarray] = None,
                 position: Optional[np.ndarray] = None,
                 velocity: Optional[np.ndarray] = None,
                 quaternion: Optional[Quaternion] = None,
                 angular_velocity: Optional[np.ndarray] = None,
                 gravity: Optional[np.ndarray] = None):
        """
        Initialize 6-DOF rigid body.
        
        Args:
            mass: Mass of the body (kg)
            inertia: 3x3 inertia matrix (kg⋅m²)
            position: Initial position [x, y, z] (m)
            velocity: Initial velocity [vx, vy, vz] (m/s)
            quaternion: Initial quaternion orientation
            angular_velocity: Initial angular velocity [wx, wy, wz] (rad/s)
            gravity: Gravity vector [gx, gy, gz] (m/s²) or scalar (Earth-like)
        """
        self.mass = mass
        
        # Default inertia matrix for missile-like object
        if inertia is None:
            self.inertia = np.diag([0.1, 1.0, 1.0])  # [Ixx, Iyy, Izz]
        else:
            self.inertia = inertia.copy()
        
        # Calculate inverse inertia matrix
        self.inertia_inv = np.linalg.inv(self.inertia)
        
        # Initialize state
        self.position = position.copy() if position is not None else np.zeros(3)
        self.velocity = velocity.copy() if velocity is not None else np.zeros(3)
        self.quaternion = quaternion if quaternion is not None else Quaternion(1, 0, 0, 0)
        self.angular_velocity = angular_velocity.copy() if angular_velocity is not None else np.zeros(3)
        
        # Initialize forces and torques
        self.applied_force = np.zeros(3)
        self.applied_torque = np.zeros(3)
        
        # Initialize gravity vector
        if gravity is None:
            self.gravity = np.array([0, 0, -9.81])  # Default Earth gravity (m/s²)
        elif np.isscalar(gravity):
            self.gravity = np.array([0, 0, -float(gravity)])  # Scalar -> Earth-like gravity
        else:
            gravity_array = np.asarray(gravity)
            if gravity_array.shape != (3,):
                raise ValueError("Gravity must be a scalar or 3-element array")
            self.gravity = gravity_array.copy()  # Vector gravity
    
    def apply_force(self, force: np.ndarray, point: Optional[np.ndarray] = None) -> None:
        """
        Apply external force to the body.
        
        Args:
            force: Force vector [fx, fy, fz] in N
            point: Point of application relative to center of mass (m)
        """
        self.applied_force += force
        
        # If force is applied at a point, calculate resulting torque
        if point is not None:
            torque = np.cross(point, force)
            self.applied_torque += torque
    
    def apply_torque(self, torque: np.ndarray) -> None:
        """
        Apply external torque to the body.
        
        Args:
            torque: Torque vector [tx, ty, tz] in N⋅m
        """
        self.applied_torque += torque
    
    def update(self, dt: float) -> None:
        """
        Update rigid body state using numerical integration with numerical stability.
        
        Args:
            dt: Time step (s)
        """
        # Numerical stability limits
        MAX_LINEAR_VELOCITY = 500.0    # m/s
        MAX_ANGULAR_VELOCITY = 50.0    # rad/s
        MAX_POSITION = 100000.0        # m
        MAX_ACCELERATION = 1000.0      # m/s²
        MAX_ANGULAR_ACCELERATION = 500.0  # rad/s²
        MAX_FORCE = 100000.0          # N
        MAX_TORQUE = 10000.0          # N⋅m
        
        # Sanitize input forces and torques
        self.applied_force = self._sanitize_vector(self.applied_force, MAX_FORCE)
        self.applied_torque = self._sanitize_vector(self.applied_torque, MAX_TORQUE)
        
        # Calculate linear acceleration
        total_force = self.applied_force + self.mass * self.gravity
        total_force = self._sanitize_vector(total_force, MAX_FORCE)
        acceleration = total_force / self.mass
        acceleration = self._sanitize_vector(acceleration, MAX_ACCELERATION)
        
        # Update linear motion with stability checks
        self.velocity += acceleration * dt
        self.velocity = self._sanitize_vector(self.velocity, MAX_LINEAR_VELOCITY)
        
        self.position += self.velocity * dt
        self.position = self._sanitize_vector(self.position, MAX_POSITION)
        
        # Calculate angular acceleration with stability safeguards
        # Limit angular velocity before calculations
        self.angular_velocity = self._sanitize_vector(self.angular_velocity, MAX_ANGULAR_VELOCITY)
        
        # Calculate angular momentum with overflow protection
        angular_momentum = self.inertia @ self.angular_velocity
        angular_momentum = self._sanitize_vector(angular_momentum, MAX_TORQUE * 10)
        
        # Calculate gyroscopic torque with numerical stability
        gyroscopic_torque = self._safe_cross_product(self.angular_velocity, angular_momentum)
        gyroscopic_torque = self._sanitize_vector(gyroscopic_torque, MAX_TORQUE)
        
        # Calculate angular acceleration
        net_torque = self.applied_torque - gyroscopic_torque
        net_torque = self._sanitize_vector(net_torque, MAX_TORQUE)
        angular_acceleration = self.inertia_inv @ net_torque
        angular_acceleration = self._sanitize_vector(angular_acceleration, MAX_ANGULAR_ACCELERATION)
        
        # Update angular motion
        self.angular_velocity += angular_acceleration * dt
        self.angular_velocity = self._sanitize_vector(self.angular_velocity, MAX_ANGULAR_VELOCITY)
        
        # Update quaternion with stability checks
        self._update_quaternion_stable(dt)
        
        # Reset applied forces and torques
        self.applied_force = np.zeros(3)
        self.applied_torque = np.zeros(3)
    
    def _sanitize_vector(self, vector: np.ndarray, max_magnitude: float) -> np.ndarray:
        """Sanitize a vector to prevent numerical overflow."""
        # Replace NaN and Inf values
        vector = np.nan_to_num(vector, nan=0.0, posinf=max_magnitude, neginf=-max_magnitude)
        
        # Clamp magnitude if too large
        magnitude = np.linalg.norm(vector)
        if magnitude > max_magnitude:
            if magnitude > 0:
                vector = vector * (max_magnitude / magnitude)
            else:
                vector = np.zeros_like(vector)
        
        return vector
    
    def _safe_cross_product(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cross product with overflow protection."""
        # Check for extreme values before cross product
        a_mag = np.linalg.norm(a)
        b_mag = np.linalg.norm(b)
        
        if a_mag > 1e6 or b_mag > 1e6:
            # If vectors are too large, use normalized versions
            a_safe = a / max(a_mag, 1.0) if a_mag > 0 else np.zeros_like(a)
            b_safe = b / max(b_mag, 1.0) if b_mag > 0 else np.zeros_like(b)
            result = np.cross(a_safe, b_safe)
            # Scale back appropriately
            scale = min(a_mag * b_mag, 1e6)
            return result * scale
        else:
            # Normal cross product
            result = np.cross(a, b)
            # Sanitize result
            return self._sanitize_vector(result, 1e6)
    
    def _update_quaternion_stable(self, dt: float) -> None:
        """Update quaternion with numerical stability."""
        # Limit angular velocity magnitude for quaternion integration
        omega_mag = np.linalg.norm(self.angular_velocity)
        if omega_mag > 10.0:  # Reduce limit for quaternion stability
            self.angular_velocity = self.angular_velocity * (10.0 / omega_mag)
        
        # Use stable quaternion integration
        if omega_mag > 1e-12:
            # Quaternion integration: q̇ = 0.5 * q * ω
            omega_quaternion = Quaternion(0, *self.angular_velocity)
            quaternion_rate = 0.5 * self.quaternion * omega_quaternion
            self.quaternion += quaternion_rate * dt
            
            # Normalize and validate quaternion
            self.quaternion = self.quaternion.normalised
            
            # Check for quaternion validity
            if not np.isfinite(self.quaternion.norm) or self.quaternion.norm < 1e-6:
                # Reset to identity quaternion if invalid
                self.quaternion = Quaternion(1, 0, 0, 0)
        
        # Ensure quaternion remains normalized
        if abs(self.quaternion.norm - 1.0) > 1e-6:
            self.quaternion = self.quaternion.normalised
    
    def get_rotation_matrix(self) -> np.ndarray:
        """Get rotation matrix from quaternion."""
        return self.quaternion.rotation_matrix
    
    def get_euler_angles(self) -> Tuple[float, float, float]:
        """Get Euler angles (roll, pitch, yaw) from quaternion."""
        return self.quaternion.yaw_pitch_roll
    
    def get_body_velocity(self) -> np.ndarray:
        """Get velocity in body frame."""
        return self.quaternion.inverse.rotate(self.velocity)
    
    def get_body_angular_velocity(self) -> np.ndarray:
        """Get angular velocity in body frame."""
        return self.angular_velocity.copy()
    
    def transform_vector_to_body(self, vector: np.ndarray) -> np.ndarray:
        """Transform vector from world frame to body frame."""
        return self.quaternion.inverse.rotate(vector)
    
    def transform_vector_to_world(self, vector: np.ndarray) -> np.ndarray:
        """Transform vector from body frame to world frame."""
        return self.quaternion.rotate(vector)
    
    def get_kinetic_energy(self) -> float:
        """Calculate total kinetic energy."""
        translational_ke = 0.5 * self.mass * np.dot(self.velocity, self.velocity)
        rotational_ke = 0.5 * np.dot(self.angular_velocity, 
                                    self.inertia @ self.angular_velocity)
        return translational_ke + rotational_ke
    
    def get_position(self) -> np.ndarray:
        """Get current position."""
        return self.position.copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity."""
        return self.velocity.copy()
    
    def get_full_state(self) -> np.ndarray:
        """Get full state vector [position, velocity, quaternion, angular_velocity]."""
        return self.get_state_vector()
    
    def body_to_world(self, vector: np.ndarray) -> np.ndarray:
        """Transform vector from body frame to world frame."""
        return self.transform_vector_to_world(vector)
    
    def apply_force_torque(self, force: np.ndarray, torque: np.ndarray, dt: float) -> None:
        """Apply force and torque to the body and update physics."""
        self.apply_force(force)
        self.apply_torque(torque)
        self.update(dt)
    
    def get_state_vector(self) -> np.ndarray:
        """Get complete state vector."""
        return np.concatenate([
            self.position,
            self.velocity,
            self.quaternion.q,  # [w, x, y, z]
            self.angular_velocity
        ])
    
    def set_state_vector(self, state: np.ndarray) -> None:
        """Set state from complete state vector."""
        self.position = state[0:3].copy()
        self.velocity = state[3:6].copy()
        self.quaternion = Quaternion(state[6], state[7], state[8], state[9])
        self.angular_velocity = state[10:13].copy()
    
    def reset(self, 
              position: Optional[np.ndarray] = None,
              velocity: Optional[np.ndarray] = None,
              quaternion: Optional[Quaternion] = None,
              angular_velocity: Optional[np.ndarray] = None) -> None:
        """Reset rigid body to initial state."""
        if position is not None:
            self.position = position.copy()
        else:
            self.position = np.zeros(3)
        
        if velocity is not None:
            self.velocity = velocity.copy()
        else:
            self.velocity = np.zeros(3)
        
        if quaternion is not None:
            self.quaternion = quaternion
        else:
            self.quaternion = Quaternion(1, 0, 0, 0)
        
        if angular_velocity is not None:
            self.angular_velocity = angular_velocity.copy()
        else:
            self.angular_velocity = np.zeros(3)
        
        self.applied_force = np.zeros(3)
        self.applied_torque = np.zeros(3)
    
    def copy(self) -> 'RigidBody6DOF':
        """Create a copy of the rigid body."""
        return RigidBody6DOF(
            mass=self.mass,
            inertia=self.inertia.copy(),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            quaternion=Quaternion(self.quaternion),
            angular_velocity=self.angular_velocity.copy(),
            gravity=self.gravity.copy()
        )