import numpy as np
from typing import Tuple


class Physics6DOF:
    """6-degree-of-freedom physics simulation with atmosphere, gravity, and wind"""
    
    def __init__(self, dt: float = 1/60):
        self.dt = dt
        
        # Earth parameters
        self.g = 9.81  # m/s^2
        self.earth_radius = 6.371e6  # meters
        
        # Atmosphere model (simplified)
        self.air_density_sea_level = 1.225  # kg/m^3
        self.scale_height = 8400  # meters
        
        # Wind parameters
        self.wind_velocity = np.array([0.0, 0.0, 0.0])  # m/s [x, y, z]
        
    def set_wind(self, wind_vel: np.ndarray):
        """Set wind velocity vector [x, y, z] in m/s"""
        self.wind_velocity = np.array(wind_vel)
        
    def get_air_density(self, altitude: float) -> float:
        """Calculate air density at given altitude using exponential atmosphere"""
        if altitude < 0:
            altitude = 0
        return self.air_density_sea_level * np.exp(-altitude / self.scale_height)
        
    def get_gravity(self, position: np.ndarray) -> np.ndarray:
        """Calculate gravity vector at position [x, y, z]"""
        altitude = position[2]  # z is up
        r = self.earth_radius + altitude
        g_magnitude = self.g * (self.earth_radius / r) ** 2
        return np.array([0.0, 0.0, -g_magnitude])
        
    def calculate_drag_force(self, velocity: np.ndarray, position: np.ndarray, 
                           drag_coefficient: float, reference_area: float, 
                           mass: float) -> np.ndarray:
        """Calculate aerodynamic drag force"""
        # Relative velocity (subtract wind)
        rel_velocity = velocity - self.wind_velocity
        speed = np.linalg.norm(rel_velocity)
        
        if speed < 1e-6:
            return np.zeros(3)
            
        air_density = self.get_air_density(position[2])
        drag_magnitude = 0.5 * air_density * speed**2 * drag_coefficient * reference_area
        
        # Drag opposes relative motion
        drag_direction = -rel_velocity / speed
        return drag_magnitude * drag_direction
        
    def integrate_6dof(self, position: np.ndarray, velocity: np.ndarray,
                      orientation: np.ndarray, angular_velocity: np.ndarray,
                      forces: np.ndarray, moments: np.ndarray, 
                      mass: float, inertia_tensor: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Integrate 6-DOF equations of motion
        
        Args:
            position: [x, y, z] in meters
            velocity: [vx, vy, vz] in m/s  
            orientation: [roll, pitch, yaw] in radians
            angular_velocity: [wx, wy, wz] in rad/s
            forces: [fx, fy, fz] total forces in N
            moments: [mx, my, mz] total moments in N*m
            mass: mass in kg
            inertia_tensor: 3x3 inertia tensor in kg*m^2
            
        Returns:
            Updated position, velocity, orientation, angular_velocity
        """
        # Linear motion
        acceleration = forces / mass
        new_velocity = velocity + acceleration * self.dt
        new_position = position + velocity * self.dt + 0.5 * acceleration * self.dt**2
        
        # Angular motion (simplified - assuming diagonal inertia tensor)
        if inertia_tensor.ndim == 1:
            # Diagonal elements only
            angular_accel = moments / inertia_tensor
        else:
            # Full 3x3 tensor
            angular_accel = np.linalg.solve(inertia_tensor, moments)
            
        new_angular_velocity = angular_velocity + angular_accel * self.dt
        new_orientation = orientation + angular_velocity * self.dt + 0.5 * angular_accel * self.dt**2
        
        # Wrap angles to [-pi, pi]
        new_orientation = np.arctan2(np.sin(new_orientation), np.cos(new_orientation))
        
        return new_position, new_velocity, new_orientation, new_angular_velocity
        
    def rotation_matrix_from_euler(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to rotation matrix (ZYX convention)"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)  
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ])
        return R
        
    def body_to_world_frame(self, vector_body: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Transform vector from body frame to world frame"""
        R = self.rotation_matrix_from_euler(orientation[0], orientation[1], orientation[2])
        return R @ vector_body
        
    def world_to_body_frame(self, vector_world: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Transform vector from world frame to body frame"""  
        R = self.rotation_matrix_from_euler(orientation[0], orientation[1], orientation[2])
        return R.T @ vector_world