"""
6-DOF Rigid Body Dynamics for AegisIntercept Phase 3.

This module provides a complete implementation of 6-DOF rigid body dynamics
using quaternion representation for attitude. Includes realistic physics
for missile simulation with proper integration schemes.
"""

import numpy as np
from typing import Tuple, Optional
from pyquaternion import Quaternion


class RigidBody6DOF:
    """
    6-DOF rigid body dynamics with quaternion attitude representation.
    
    This class simulates the motion of a rigid body in 3D space, including
    both translational and rotational dynamics. It uses quaternions to avoid
    gimbal lock issues and provides stable numerical integration.
    
    State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
    - Position: [x, y, z] in meters
    - Velocity: [vx, vy, vz] in m/s
    - Orientation: [qw, qx, qy, qz] quaternion (w, x, y, z)
    - Angular velocity: [wx, wy, wz] in rad/s (body frame)
    """
    
    def __init__(self, 
                 mass: float = 100.0,
                 inertia: Optional[np.ndarray] = None,
                 gravity: float = 9.80665):
        """
        Initialize 6-DOF rigid body.
        
        Args:
            mass: Mass in kg
            inertia: 3x3 inertia tensor in kg⋅m². If None, uses spherical approximation
            gravity: Gravitational acceleration in m/s²
        """
        self.mass = mass
        self.gravity = gravity
        
        # Default inertia tensor (spherical approximation)
        if inertia is None:
            I = 0.4 * mass * (0.5 ** 2)  # Sphere with radius 0.5m
            self.inertia = np.diag([I, I, I])
        else:
            self.inertia = np.array(inertia)
        
        self.inv_inertia = np.linalg.inv(self.inertia)
        
        # State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
        self.state = np.zeros(13)
        self.state[6] = 1.0  # Initialize quaternion to identity (w=1)
        
        # Constants
        self.EARTH_GRAVITY = np.array([0.0, 0.0, -gravity])
    
    def reset(self, 
              position: np.ndarray = None,
              velocity: np.ndarray = None,
              orientation: Quaternion = None,
              angular_velocity: np.ndarray = None):
        """
        Reset the rigid body to initial state.
        
        Args:
            position: Initial position [x, y, z] in meters
            velocity: Initial velocity [vx, vy, vz] in m/s
            orientation: Initial orientation as quaternion
            angular_velocity: Initial angular velocity [wx, wy, wz] in rad/s
        """
        self.state.fill(0.0)
        
        if position is not None:
            self.state[0:3] = position
        
        if velocity is not None:
            self.state[3:6] = velocity
        
        if orientation is not None:
            self.state[6:10] = [orientation.w, orientation.x, orientation.y, orientation.z]
        else:
            self.state[6] = 1.0  # Identity quaternion
        
        if angular_velocity is not None:
            self.state[10:13] = angular_velocity
    
    def get_position(self) -> np.ndarray:
        """Get current position [x, y, z]."""
        return self.state[0:3].copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity [vx, vy, vz]."""
        return self.state[3:6].copy()
    
    def get_orientation(self) -> Quaternion:
        """Get current orientation as quaternion."""
        return Quaternion(w=self.state[6], x=self.state[7], 
                         y=self.state[8], z=self.state[9])
    
    def get_angular_velocity(self) -> np.ndarray:
        """Get current angular velocity [wx, wy, wz] in body frame."""
        return self.state[10:13].copy()
    
    def get_attitude_matrix(self) -> np.ndarray:
        """Get rotation matrix from body to world frame."""
        q = self.get_orientation()
        return q.rotation_matrix
    
    def body_to_world(self, vector: np.ndarray) -> np.ndarray:
        """Transform vector from body frame to world frame."""
        R = self.get_attitude_matrix()
        return R @ vector
    
    def world_to_body(self, vector: np.ndarray) -> np.ndarray:
        """Transform vector from world frame to body frame."""
        R = self.get_attitude_matrix()
        return R.T @ vector
    
    def apply_force_torque(self, 
                          force: np.ndarray, 
                          torque: np.ndarray, 
                          dt: float) -> None:
        """
        Apply force and torque to the rigid body and integrate dynamics.
        
        Args:
            force: Applied force [fx, fy, fz] in world frame (N)
            torque: Applied torque [tx, ty, tz] in body frame (N⋅m)
            dt: Time step in seconds
        """
        # Current state
        pos = self.get_position()
        vel = self.get_velocity()
        quat = self.get_orientation()
        omega = self.get_angular_velocity()
        
        # Total force = applied force + gravity
        total_force = force + self.mass * self.EARTH_GRAVITY
        
        # Translational dynamics: F = ma
        accel = total_force / self.mass
        
        # Rotational dynamics: τ = Iω̇ + ω × Iω (Euler's equation)
        I_omega = self.inertia @ omega
        omega_cross_I_omega = np.cross(omega, I_omega)
        alpha = self.inv_inertia @ (torque - omega_cross_I_omega)
        
        # Apply numerical stability limits (increased for missile dynamics)
        MAX_ACCEL = 2000.0  # 2000 m/s² (200g) - increased for missile capability
        MAX_ALPHA = 200.0   # 200 rad/s² angular acceleration - doubled
        MAX_VEL = 1500.0    # 1500 m/s maximum velocity - increased
        MAX_OMEGA = 100.0   # 100 rad/s maximum angular velocity - doubled
        
        accel = np.clip(accel, -MAX_ACCEL, MAX_ACCEL)
        alpha = np.clip(alpha, -MAX_ALPHA, MAX_ALPHA)
        
        # Integrate translational motion
        new_pos = pos + vel * dt + 0.5 * accel * dt**2
        new_vel = vel + accel * dt
        new_vel = np.clip(new_vel, -MAX_VEL, MAX_VEL)
        
        # Integrate rotational motion
        new_omega = omega + alpha * dt
        new_omega = np.clip(new_omega, -MAX_OMEGA, MAX_OMEGA)
        
        # Quaternion integration using angular velocity
        # q̇ = 0.5 * q * ω_quaternion
        omega_quat = Quaternion(w=0, x=new_omega[0], y=new_omega[1], z=new_omega[2])
        q_dot = 0.5 * quat * omega_quat
        
        new_quat = quat + q_dot * dt
        new_quat = new_quat.normalised  # Normalize to prevent drift
        
        # Validate quaternion is not corrupted
        if abs(new_quat.norm - 1.0) > 0.1:  # Quaternion severely corrupted
            new_quat = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)  # Reset to identity
        
        # Update state
        self.state[0:3] = new_pos
        self.state[3:6] = new_vel
        self.state[6:10] = [new_quat.w, new_quat.x, new_quat.y, new_quat.z]
        self.state[10:13] = new_omega
    
    def get_full_state(self) -> np.ndarray:
        """Get complete state vector."""
        return self.state.copy()
    
    def set_full_state(self, state: np.ndarray) -> None:
        """Set complete state vector."""
        self.state = state.copy()
        # Normalize quaternion to prevent drift
        quat = Quaternion(w=state[6], x=state[7], y=state[8], z=state[9])
        quat = quat.normalised
        self.state[6:10] = [quat.w, quat.x, quat.y, quat.z]
    
    def get_kinetic_energy(self) -> Tuple[float, float]:
        """
        Get kinetic energy components.
        
        Returns:
            Tuple of (translational_KE, rotational_KE) in Joules
        """
        vel = self.get_velocity()
        omega = self.get_angular_velocity()
        
        trans_ke = 0.5 * self.mass * np.dot(vel, vel)
        rot_ke = 0.5 * np.dot(omega, self.inertia @ omega)
        
        return trans_ke, rot_ke
    
    def get_momentum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get momentum components.
        
        Returns:
            Tuple of (linear_momentum, angular_momentum) in world frame
        """
        vel = self.get_velocity()
        omega = self.get_angular_velocity()
        
        linear_momentum = self.mass * vel
        
        # Angular momentum in world frame: L = R * I * ω
        R = self.get_attitude_matrix()
        angular_momentum = R @ (self.inertia @ omega)
        
        return linear_momentum, angular_momentum