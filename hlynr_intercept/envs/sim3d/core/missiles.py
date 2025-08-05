import numpy as np
from typing import Optional, Dict, Any
from .physics import Physics6DOF


class Missile:
    """Missile entity with 6-DOF dynamics, fuel system, and thrust control"""
    
    def __init__(self, 
                 missile_type: str,
                 position: np.ndarray,
                 velocity: np.ndarray,
                 orientation: np.ndarray,
                 physics: Physics6DOF):
        
        self.type = missile_type
        self.physics = physics
        
        # State vectors
        self.position = np.array(position, dtype=float)  # [x, y, z] meters
        self.velocity = np.array(velocity, dtype=float)  # [vx, vy, vz] m/s
        self.orientation = np.array(orientation, dtype=float)  # [roll, pitch, yaw] radians
        self.angular_velocity = np.zeros(3)  # [wx, wy, wz] rad/s
        
        # Physical properties (typical medium-range missile)
        if missile_type == "attacker":
            self.mass_dry = 150.0  # kg
            self.fuel_mass = 50.0  # kg  
            self.max_thrust = 8000.0  # N
            self.burn_time = 20.0  # seconds
            self.drag_coefficient = 0.3
            self.reference_area = 0.05  # m^2
            self.length = 4.0  # meters
        elif missile_type == "interceptor":
            self.mass_dry = 100.0  # kg
            self.fuel_mass = 30.0  # kg
            self.max_thrust = 6000.0  # N  
            self.burn_time = 15.0  # seconds
            self.drag_coefficient = 0.25
            self.reference_area = 0.03  # m^2
            self.length = 3.0  # meters
        else:
            raise ValueError(f"Unknown missile type: {missile_type}")
            
        # Current state
        self.fuel_remaining = self.fuel_mass
        self.thrust_fraction = 0.0  # 0.0 to 1.0
        self.active = True
        self.stage_fired = False
        
        # Inertia tensor (simplified as diagonal)
        self.inertia_tensor = self._calculate_inertia()
        
        # Control surfaces effectiveness (simplified)
        self.control_authority = 5000.0  # N*m per radian of deflection
        self.max_control_deflection = np.radians(15)  # radians
        
        # Current control inputs
        self.control_pitch = 0.0  # -1 to 1
        self.control_yaw = 0.0    # -1 to 1  
        self.control_roll = 0.0   # -1 to 1
        
    def _calculate_inertia(self) -> np.ndarray:
        """Calculate moment of inertia tensor (simplified cylindrical missile)"""
        total_mass = self.mass_dry + self.fuel_remaining
        radius = np.sqrt(self.reference_area / np.pi)
        
        # Cylinder inertia
        I_xx = 0.5 * total_mass * radius**2  # Roll
        I_yy = I_zz = (1/12) * total_mass * (3 * radius**2 + self.length**2)  # Pitch/Yaw
        
        return np.array([I_xx, I_yy, I_zz])
        
    def get_mass(self) -> float:
        """Get current total mass"""
        return self.mass_dry + self.fuel_remaining
        
    def set_thrust_fraction(self, fraction: float):
        """Set thrust as fraction of maximum (0.0 to 1.0)"""
        self.thrust_fraction = np.clip(fraction, 0.0, 1.0)
        
    def set_control_inputs(self, pitch: float, yaw: float, roll: float):
        """Set control surface deflections (-1.0 to 1.0)"""
        self.control_pitch = np.clip(pitch, -1.0, 1.0)
        self.control_yaw = np.clip(yaw, -1.0, 1.0) 
        self.control_roll = np.clip(roll, -1.0, 1.0)
        
    def stage_fire(self):
        """Fire/separate stage (for multi-stage missiles)"""
        self.stage_fired = True
        
    def update(self, dt: float):
        """Update missile state by one time step"""
        if not self.active:
            return
            
        # Fuel consumption
        if self.thrust_fraction > 0 and self.fuel_remaining > 0:
            fuel_burn_rate = self.fuel_mass / self.burn_time
            fuel_consumed = fuel_burn_rate * self.thrust_fraction * dt
            self.fuel_remaining = max(0, self.fuel_remaining - fuel_consumed)
            
        # Current thrust
        if self.fuel_remaining > 0:
            current_thrust = self.thrust_fraction * self.max_thrust
        else:
            current_thrust = 0.0
            self.thrust_fraction = 0.0
            
        # Forces in body frame
        thrust_force_body = np.array([0, 0, current_thrust])  # Thrust along body z-axis
        
        # Transform thrust to world frame
        thrust_force_world = self.physics.body_to_world_frame(thrust_force_body, self.orientation)
        
        # Aerodynamic drag
        drag_force = self.physics.calculate_drag_force(
            self.velocity, self.position, self.drag_coefficient, 
            self.reference_area, self.get_mass()
        )
        
        # Gravity
        gravity_force = self.physics.get_gravity(self.position) * self.get_mass()
        
        # Total forces
        total_forces = thrust_force_world + drag_force + gravity_force
        
        # Moments from control surfaces
        moments = np.array([
            self.control_roll * self.control_authority * self.max_control_deflection,
            self.control_pitch * self.control_authority * self.max_control_deflection,
            self.control_yaw * self.control_authority * self.max_control_deflection
        ])
        
        # Aerodynamic damping (simplified)
        speed = np.linalg.norm(self.velocity)
        if speed > 1.0:
            air_density = self.physics.get_air_density(self.position[2])
            damping_factor = 0.1 * air_density * speed * self.reference_area * self.length
            moments -= damping_factor * self.angular_velocity
            
        # Update inertia tensor
        self.inertia_tensor = self._calculate_inertia()
        
        # Integrate 6-DOF motion
        self.position, self.velocity, self.orientation, self.angular_velocity = \
            self.physics.integrate_6dof(
                self.position, self.velocity, self.orientation, self.angular_velocity,
                total_forces, moments, self.get_mass(), self.inertia_tensor
            )
            
        # Check for ground collision
        if self.position[2] <= 0:
            print(f"{self.type} missile hit ground at position {self.position}, velocity {self.velocity}")
            self.active = False
            self.position[2] = 0
            self.velocity = np.zeros(3)
            self.angular_velocity = np.zeros(3)
            
    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete missile state as dictionary"""
        return {
            "type": self.type,
            "position": self.position.copy(),
            "velocity": self.velocity.copy(), 
            "orientation": self.orientation.copy(),
            "angular_velocity": self.angular_velocity.copy(),
            "mass": self.get_mass(),
            "fuel_remaining": self.fuel_remaining,
            "thrust_fraction": self.thrust_fraction,
            "active": self.active,
            "stage_fired": self.stage_fired,
            "control_inputs": [self.control_pitch, self.control_yaw, self.control_roll]
        }
        
    def get_speed(self) -> float:
        """Get current speed in m/s"""
        return np.linalg.norm(self.velocity)
        
    def get_altitude(self) -> float:
        """Get current altitude in meters"""
        return self.position[2]
        
    def get_range(self) -> float:
        """Get horizontal range from origin"""
        return np.linalg.norm(self.position[:2])