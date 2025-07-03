"""
6DOF Physics Engine for Advanced Missile Intercept System

This module provides a complete 6 Degrees of Freedom (6DOF) physics simulation
for realistic missile and interceptor dynamics. It includes:
- Quaternion-based orientation representation
- Aerodynamic force and torque calculations
- Atmospheric effects and wind modeling
- Realistic drag coefficients and physics constants
- Integration methods for full 6DOF state evolution

Author: Coder Agent
Date: Phase 3 Implementation
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import math


class VehicleType(Enum):
    """Vehicle type enumeration for different aerodynamic properties"""
    INTERCEPTOR = "interceptor"
    MISSILE = "missile"
    TARGET = "target"


@dataclass
class PhysicsConstants:
    """Physical constants for realistic simulation"""
    # Atmospheric properties (at sea level)
    AIR_DENSITY: float = 1.225  # kg/m³
    SPEED_OF_SOUND: float = 343.0  # m/s
    GRAVITY: float = 9.81  # m/s²
    
    # Integration parameters
    INTEGRATION_SUBSTEPS: int = 4  # RK4 substeps for stability
    MIN_TIMESTEP: float = 1e-6  # Minimum timestep for stability
    MAX_TIMESTEP: float = 0.1  # Maximum timestep
    
    # Numerical stability
    QUATERNION_NORMALIZE_THRESHOLD: float = 1e-6
    ANGULAR_VELOCITY_LIMIT: float = 50.0  # rad/s
    VELOCITY_LIMIT: float = 2000.0  # m/s (Mach 6)


@dataclass
class AerodynamicProperties:
    """Aerodynamic properties for different vehicle types"""
    # Drag coefficients
    cd_0: float  # Zero-lift drag coefficient
    cd_alpha: float  # Induced drag coefficient
    cd_mach: float  # Mach-dependent drag coefficient
    
    # Lift coefficients
    cl_alpha: float  # Lift coefficient per angle of attack
    cl_delta: float  # Lift coefficient per control surface deflection
    
    # Moment coefficients
    cm_alpha: float  # Pitching moment coefficient per angle of attack
    cm_delta: float  # Pitching moment coefficient per control surface
    cm_q: float  # Pitch damping coefficient
    
    # Reference dimensions
    reference_area: float  # m²
    reference_length: float  # m (for moments)
    mass: float  # kg
    
    # Inertia tensor (kg⋅m²)
    inertia_xx: float
    inertia_yy: float
    inertia_zz: float
    inertia_xy: float = 0.0
    inertia_xz: float = 0.0
    inertia_yz: float = 0.0
    
    @property
    def inertia_tensor(self) -> np.ndarray:
        """Get the inertia tensor as a 3x3 matrix"""
        return np.array([
            [self.inertia_xx, -self.inertia_xy, -self.inertia_xz],
            [-self.inertia_xy, self.inertia_yy, -self.inertia_yz],
            [-self.inertia_xz, -self.inertia_yz, self.inertia_zz]
        ])


# Predefined aerodynamic properties for different vehicle types
AERODYNAMIC_PROPERTIES = {
    VehicleType.INTERCEPTOR: AerodynamicProperties(
        cd_0=0.15, cd_alpha=0.35, cd_mach=0.1,
        cl_alpha=3.5, cl_delta=0.6,
        cm_alpha=-0.8, cm_delta=-1.2, cm_q=-12.0,
        reference_area=0.05, reference_length=2.5, mass=150.0,
        inertia_xx=25.0, inertia_yy=300.0, inertia_zz=310.0
    ),
    VehicleType.MISSILE: AerodynamicProperties(
        cd_0=0.12, cd_alpha=0.28, cd_mach=0.08,
        cl_alpha=2.8, cl_delta=0.5,
        cm_alpha=-0.6, cm_delta=-1.0, cm_q=-8.0,
        reference_area=0.08, reference_length=3.0, mass=200.0,
        inertia_xx=35.0, inertia_yy=450.0, inertia_zz=465.0
    )
}


class QuaternionUtils:
    """Utility functions for quaternion operations"""
    
    @staticmethod
    def normalize(q: np.ndarray) -> np.ndarray:
        """Normalize a quaternion to unit length"""
        norm = np.linalg.norm(q)
        if norm < PhysicsConstants.QUATERNION_NORMALIZE_THRESHOLD:
            return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        return q / norm
    
    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions: q1 * q2"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def conjugate(q: np.ndarray) -> np.ndarray:
        """Get the conjugate of a quaternion"""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    @staticmethod
    def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix"""
        q = QuaternionUtils.normalize(q)
        w, x, y, z = q
        
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])
    
    @staticmethod
    def from_angular_velocity(omega: np.ndarray, dt: float) -> np.ndarray:
        """Convert angular velocity to quaternion rotation over time dt"""
        angle = np.linalg.norm(omega) * dt
        if angle < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0])
        
        axis = omega / np.linalg.norm(omega)
        half_angle = angle * 0.5
        sin_half = np.sin(half_angle)
        
        return np.array([
            np.cos(half_angle),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half
        ])
    
    @staticmethod
    def to_euler(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        q = QuaternionUtils.normalize(q)
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])


class AtmosphericModel:
    """Atmospheric model for altitude-dependent properties"""
    
    @staticmethod
    def get_air_density(altitude: float) -> float:
        """Get air density at given altitude (m) using exponential model"""
        # Exponential atmosphere model
        h_scale = 8500.0  # Scale height in meters
        rho_0 = PhysicsConstants.AIR_DENSITY
        return rho_0 * np.exp(-altitude / h_scale)
    
    @staticmethod
    def get_speed_of_sound(altitude: float) -> float:
        """Get speed of sound at given altitude"""
        # Simplified temperature model
        T_0 = 288.15  # Sea level temperature (K)
        lapse_rate = 0.0065  # K/m
        T = T_0 - lapse_rate * altitude
        T = max(T, 200.0)  # Minimum temperature limit
        
        return np.sqrt(1.4 * 287.0 * T)  # Speed of sound formula
    
    @staticmethod
    def get_wind_velocity(position: np.ndarray, time: float) -> np.ndarray:
        """Get wind velocity at given position and time"""
        # Simple wind model with turbulence
        x, y, z = position
        
        # Base wind (varies with altitude)
        base_wind = np.array([
            5.0 * (1.0 + z / 1000.0),  # Eastward wind increases with altitude
            2.0 * np.sin(z / 500.0),   # Northward wind varies with altitude
            0.0                        # No vertical wind component
        ])
        
        # Turbulence (random but spatially correlated)
        turbulence_scale = 0.1 * z / 1000.0  # Increases with altitude
        turbulence = turbulence_scale * np.array([
            np.sin(x / 100.0 + time) * np.cos(y / 150.0),
            np.cos(x / 120.0 - time) * np.sin(y / 100.0),
            0.5 * np.sin(x / 200.0) * np.cos(y / 200.0) * np.sin(time * 2.0)
        ])
        
        return base_wind + turbulence


class RigidBody6DOF:
    """
    Complete 6DOF rigid body dynamics simulation
    
    State vector (29 dimensions total for two vehicles + environment):
    - Position: [x, y, z] (3D)
    - Velocity: [vx, vy, vz] (3D) 
    - Orientation: [qw, qx, qy, qz] (quaternion, 4D)
    - Angular velocity: [wx, wy, wz] (3D)
    Total per vehicle: 13 dimensions
    Environment state: [time, wind_x, wind_y, wind_z] (3D)
    Total system state: 13 + 13 + 3 = 29 dimensions
    """
    
    def __init__(self, 
                 vehicle_type: VehicleType,
                 initial_position: np.ndarray = None,
                 initial_velocity: np.ndarray = None,
                 initial_orientation: np.ndarray = None,
                 initial_angular_velocity: np.ndarray = None):
        """
        Initialize 6DOF rigid body
        
        Args:
            vehicle_type: Type of vehicle (affects aerodynamic properties)
            initial_position: Initial position [x, y, z] in meters
            initial_velocity: Initial velocity [vx, vy, vz] in m/s
            initial_orientation: Initial orientation quaternion [w, x, y, z]
            initial_angular_velocity: Initial angular velocity [wx, wy, wz] in rad/s
        """
        self.vehicle_type = vehicle_type
        self.aero_props = AERODYNAMIC_PROPERTIES[vehicle_type]
        self.constants = PhysicsConstants()
        
        # Initialize state
        self.position = initial_position if initial_position is not None else np.zeros(3)
        self.velocity = initial_velocity if initial_velocity is not None else np.zeros(3)
        self.orientation = initial_orientation if initial_orientation is not None else np.array([1.0, 0.0, 0.0, 0.0])
        self.angular_velocity = initial_angular_velocity if initial_angular_velocity is not None else np.zeros(3)
        
        # Normalize orientation quaternion
        self.orientation = QuaternionUtils.normalize(self.orientation)
        
        # Control inputs
        self.thrust_force = np.zeros(3)  # Body frame thrust force
        self.control_torque = np.zeros(3)  # Body frame control torque
        
        # Aerodynamic state
        self.angle_of_attack = 0.0
        self.sideslip_angle = 0.0
        self.mach_number = 0.0
        
        # Inverse inertia tensor for efficiency
        self.inv_inertia = np.linalg.inv(self.aero_props.inertia_tensor)
    
    def get_state_vector(self) -> np.ndarray:
        """Get complete 13D state vector for this vehicle"""
        return np.concatenate([
            self.position,           # 3D position
            self.velocity,           # 3D velocity
            self.orientation,        # 4D quaternion
            self.angular_velocity    # 3D angular velocity
        ])
    
    def set_state_vector(self, state: np.ndarray) -> None:
        """Set state from 13D state vector"""
        self.position = state[0:3].copy()
        self.velocity = state[3:6].copy()
        self.orientation = QuaternionUtils.normalize(state[6:10].copy())
        self.angular_velocity = state[10:13].copy()
    
    def set_control_inputs(self, thrust: np.ndarray, torque: np.ndarray) -> None:
        """Set control inputs (thrust force and torque in body frame)"""
        self.thrust_force = thrust.copy()
        self.control_torque = torque.copy()
    
    def compute_aerodynamic_angles(self, air_velocity: np.ndarray) -> Tuple[float, float]:
        """
        Compute angle of attack and sideslip angle from air-relative velocity
        
        Args:
            air_velocity: Air-relative velocity in body frame
            
        Returns:
            Tuple of (angle_of_attack, sideslip_angle) in radians
        """
        if np.linalg.norm(air_velocity) < 1e-3:
            return 0.0, 0.0
        
        # Angle of attack: angle between velocity and x-y plane
        alpha = np.arctan2(air_velocity[2], air_velocity[0])
        
        # Sideslip angle: angle between velocity projection and x-axis
        beta = np.arctan2(air_velocity[1], np.sqrt(air_velocity[0]**2 + air_velocity[2]**2))
        
        return alpha, beta
    
    def compute_aerodynamic_forces_torques(self, air_velocity: np.ndarray, 
                                         air_density: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute aerodynamic forces and torques in body frame
        
        Args:
            air_velocity: Air-relative velocity in body frame
            air_density: Air density at current altitude
            
        Returns:
            Tuple of (forces, torques) in body frame
        """
        speed = np.linalg.norm(air_velocity)
        if speed < 1e-3:
            return np.zeros(3), np.zeros(3)
        
        # Compute aerodynamic angles
        alpha, beta = self.compute_aerodynamic_angles(air_velocity)
        self.angle_of_attack = alpha
        self.sideslip_angle = beta
        
        # Mach number
        sound_speed = AtmosphericModel.get_speed_of_sound(self.position[2])
        self.mach_number = speed / sound_speed
        
        # Dynamic pressure
        q = 0.5 * air_density * speed**2
        
        # Drag coefficient (function of angle of attack and Mach number)
        cd = (self.aero_props.cd_0 + 
              self.aero_props.cd_alpha * alpha**2 +
              self.aero_props.cd_mach * max(0, self.mach_number - 1.0)**2)
        
        # Lift coefficient (linear in angle of attack for small angles)
        cl = self.aero_props.cl_alpha * alpha
        
        # Side force coefficient (function of sideslip angle)
        cy = self.aero_props.cl_alpha * beta  # Assume similar to lift
        
        # Forces in wind frame
        drag = -cd * q * self.aero_props.reference_area
        lift = cl * q * self.aero_props.reference_area
        side_force = cy * q * self.aero_props.reference_area
        
        # Convert to body frame (wind frame aligned with velocity)
        velocity_unit = air_velocity / speed
        # Drag opposes velocity
        drag_force = drag * velocity_unit
        
        # Lift perpendicular to velocity in x-z plane
        lift_direction = np.array([-velocity_unit[2], 0, velocity_unit[0]])
        if np.linalg.norm(lift_direction) > 1e-6:
            lift_direction = lift_direction / np.linalg.norm(lift_direction)
        lift_force = lift * lift_direction
        
        # Side force in y direction
        side_force_vector = np.array([0, side_force, 0])
        
        total_force = drag_force + lift_force + side_force_vector
        
        # Aerodynamic moments
        cm = (self.aero_props.cm_alpha * alpha +
              self.aero_props.cm_q * self.angular_velocity[1] * 
              self.aero_props.reference_length / (2 * speed))
        
        # Pitching moment
        pitch_moment = cm * q * self.aero_props.reference_area * self.aero_props.reference_length
        
        # Roll and yaw moments (simplified)
        roll_moment = -0.1 * beta * q * self.aero_props.reference_area * self.aero_props.reference_length
        yaw_moment = 0.05 * beta * q * self.aero_props.reference_area * self.aero_props.reference_length
        
        total_torque = np.array([roll_moment, pitch_moment, yaw_moment])
        
        return total_force, total_torque
    
    def compute_forces_torques(self, time: float, wind_velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute all forces and torques acting on the vehicle
        
        Args:
            time: Current simulation time
            wind_velocity: Wind velocity in world frame
            
        Returns:
            Tuple of (total_forces, total_torques) in body frame
        """
        # Get rotation matrix from body to world frame
        R_body_to_world = QuaternionUtils.to_rotation_matrix(self.orientation)
        R_world_to_body = R_body_to_world.T
        
        # Air-relative velocity in world frame
        air_velocity_world = self.velocity - wind_velocity
        
        # Transform to body frame
        air_velocity_body = R_world_to_body @ air_velocity_world
        
        # Air density at current altitude
        air_density = AtmosphericModel.get_air_density(self.position[2])
        
        # Compute aerodynamic forces and torques
        aero_forces, aero_torques = self.compute_aerodynamic_forces_torques(
            air_velocity_body, air_density)
        
        # Total forces in body frame
        total_forces = self.thrust_force + aero_forces
        
        # Total torques in body frame
        total_torques = self.control_torque + aero_torques
        
        return total_forces, total_torques
    
    def step(self, dt: float, time: float, wind_velocity: np.ndarray) -> None:
        """
        Integrate 6DOF dynamics one time step using RK4
        
        Args:
            dt: Time step size
            time: Current simulation time
            wind_velocity: Wind velocity in world frame
        """
        # Limit time step for stability
        dt = np.clip(dt, self.constants.MIN_TIMESTEP, self.constants.MAX_TIMESTEP)
        
        # Use RK4 integration for better accuracy
        substep_dt = dt / self.constants.INTEGRATION_SUBSTEPS
        
        for _ in range(self.constants.INTEGRATION_SUBSTEPS):
            self._rk4_step(substep_dt, time, wind_velocity)
            time += substep_dt
        
        # Normalize quaternion to prevent drift
        self.orientation = QuaternionUtils.normalize(self.orientation)
        
        # Apply velocity and angular velocity limits for stability
        vel_mag = np.linalg.norm(self.velocity)
        if vel_mag > self.constants.VELOCITY_LIMIT:
            self.velocity = self.velocity / vel_mag * self.constants.VELOCITY_LIMIT
        
        omega_mag = np.linalg.norm(self.angular_velocity)
        if omega_mag > self.constants.ANGULAR_VELOCITY_LIMIT:
            self.angular_velocity = (self.angular_velocity / omega_mag * 
                                   self.constants.ANGULAR_VELOCITY_LIMIT)
    
    def _rk4_step(self, dt: float, time: float, wind_velocity: np.ndarray) -> None:
        """Single RK4 integration step"""
        # Save initial state
        pos0 = self.position.copy()
        vel0 = self.velocity.copy()
        quat0 = self.orientation.copy()
        omega0 = self.angular_velocity.copy()
        
        # K1
        k1_pos, k1_vel, k1_quat, k1_omega = self._compute_derivatives(time, wind_velocity)
        
        # K2
        self.position = pos0 + 0.5 * dt * k1_pos
        self.velocity = vel0 + 0.5 * dt * k1_vel
        self.orientation = QuaternionUtils.normalize(quat0 + 0.5 * dt * k1_quat)
        self.angular_velocity = omega0 + 0.5 * dt * k1_omega
        
        k2_pos, k2_vel, k2_quat, k2_omega = self._compute_derivatives(time + 0.5*dt, wind_velocity)
        
        # K3
        self.position = pos0 + 0.5 * dt * k2_pos
        self.velocity = vel0 + 0.5 * dt * k2_vel
        self.orientation = QuaternionUtils.normalize(quat0 + 0.5 * dt * k2_quat)
        self.angular_velocity = omega0 + 0.5 * dt * k2_omega
        
        k3_pos, k3_vel, k3_quat, k3_omega = self._compute_derivatives(time + 0.5*dt, wind_velocity)
        
        # K4
        self.position = pos0 + dt * k3_pos
        self.velocity = vel0 + dt * k3_vel
        self.orientation = QuaternionUtils.normalize(quat0 + dt * k3_quat)
        self.angular_velocity = omega0 + dt * k3_omega
        
        k4_pos, k4_vel, k4_quat, k4_omega = self._compute_derivatives(time + dt, wind_velocity)
        
        # Final update
        self.position = pos0 + (dt/6.0) * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)
        self.velocity = vel0 + (dt/6.0) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)
        self.orientation = QuaternionUtils.normalize(quat0 + (dt/6.0) * (k1_quat + 2*k2_quat + 2*k3_quat + k4_quat))
        self.angular_velocity = omega0 + (dt/6.0) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
    
    def _compute_derivatives(self, time: float, wind_velocity: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute state derivatives for integration"""
        # Position derivative is velocity
        pos_dot = self.velocity.copy()
        
        # Compute forces and torques
        forces_body, torques_body = self.compute_forces_torques(time, wind_velocity)
        
        # Transform forces to world frame
        R_body_to_world = QuaternionUtils.to_rotation_matrix(self.orientation)
        forces_world = R_body_to_world @ forces_body
        
        # Add gravity
        gravity_force = np.array([0, 0, -self.aero_props.mass * self.constants.GRAVITY])
        forces_world += gravity_force
        
        # Velocity derivative from Newton's second law
        vel_dot = forces_world / self.aero_props.mass
        
        # Quaternion derivative from angular velocity
        omega_quat = np.array([0, self.angular_velocity[0], self.angular_velocity[1], self.angular_velocity[2]])
        quat_dot = 0.5 * QuaternionUtils.multiply(self.orientation, omega_quat)
        
        # Angular velocity derivative from Euler's equation
        I_omega = self.aero_props.inertia_tensor @ self.angular_velocity
        omega_cross_I_omega = np.cross(self.angular_velocity, I_omega)
        omega_dot = self.inv_inertia @ (torques_body - omega_cross_I_omega)
        
        return pos_dot, vel_dot, quat_dot, omega_dot
    
    def get_aerodynamic_info(self) -> Dict[str, Any]:
        """Get current aerodynamic state information"""
        return {
            'angle_of_attack': self.angle_of_attack,
            'sideslip_angle': self.sideslip_angle,
            'mach_number': self.mach_number,
            'dynamic_pressure': 0.5 * AtmosphericModel.get_air_density(self.position[2]) * np.linalg.norm(self.velocity)**2,
            'air_density': AtmosphericModel.get_air_density(self.position[2]),
            'speed_of_sound': AtmosphericModel.get_speed_of_sound(self.position[2])
        }


def distance_6dof(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Calculate Euclidean distance between two 3D points"""
    return np.linalg.norm(pos1 - pos2)


def relative_velocity_6dof(vel1: np.ndarray, vel2: np.ndarray) -> np.ndarray:
    """Calculate relative velocity vector"""
    return vel1 - vel2


def intercept_geometry_6dof(pos1: np.ndarray, vel1: np.ndarray, 
                           pos2: np.ndarray, vel2: np.ndarray) -> Dict[str, float]:
    """
    Compute intercept geometry parameters for two vehicles
    
    Returns:
        Dictionary with geometry parameters including:
        - closing_velocity: Rate of distance reduction
        - time_to_closest_approach: Time to closest approach
        - aspect_angle: Angle between velocity vectors
        - approach_angle: Angle of approach
    """
    relative_pos = pos2 - pos1
    relative_vel = vel2 - vel1
    
    distance = np.linalg.norm(relative_pos)
    relative_speed = np.linalg.norm(relative_vel)
    
    if distance < 1e-6 or relative_speed < 1e-6:
        return {
            'closing_velocity': 0.0,
            'time_to_closest_approach': float('inf'),
            'aspect_angle': 0.0,
            'approach_angle': 0.0
        }
    
    # Closing velocity (negative if separating)
    closing_velocity = -np.dot(relative_pos, relative_vel) / distance
    
    # Time to closest approach
    if abs(closing_velocity) > 1e-6:
        time_to_closest = distance / closing_velocity
    else:
        time_to_closest = float('inf')
    
    # Aspect angle (angle between velocity vectors)
    vel1_mag = np.linalg.norm(vel1)
    vel2_mag = np.linalg.norm(vel2)
    if vel1_mag > 1e-6 and vel2_mag > 1e-6:
        cos_aspect = np.dot(vel1, vel2) / (vel1_mag * vel2_mag)
        aspect_angle = np.arccos(np.clip(cos_aspect, -1.0, 1.0))
    else:
        aspect_angle = 0.0
    
    # Approach angle (angle between relative position and relative velocity)
    if relative_speed > 1e-6:
        cos_approach = np.dot(relative_pos, relative_vel) / (distance * relative_speed)
        approach_angle = np.arccos(np.clip(cos_approach, -1.0, 1.0))
    else:
        approach_angle = 0.0
    
    return {
        'closing_velocity': closing_velocity,
        'time_to_closest_approach': time_to_closest,
        'aspect_angle': aspect_angle,
        'approach_angle': approach_angle
    }