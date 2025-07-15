"""
3D Physics models for AegisIntercept - Atmospheric and Wind modeling.

This module combines atmospheric modeling (based on International Standard
Atmosphere) and wind field modeling for realistic 3D physics simulation.
Originally split between atmosphere.py and wind.py, now unified for better
organization and Phase 3 structure.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import json


class AtmosphericModel:
    """
    International Standard Atmosphere (ISA) model for atmospheric properties.
    
    This class provides atmospheric density, pressure, and temperature as
    functions of altitude. It uses the ISA standard which is widely accepted
    for aerospace applications.
    
    Valid altitude range: 0 to 47,000 meters (troposphere and lower stratosphere)
    """
    
    # ISA Constants
    R = 287.0528  # Specific gas constant for dry air (J/(kg⋅K))
    g0 = 9.80665  # Standard gravity (m/s²)
    M = 0.0289644  # Molar mass of dry air (kg/mol)
    
    # Sea level standard conditions
    P0 = 101325.0  # Standard pressure (Pa)
    T0 = 288.15    # Standard temperature (K)
    rho0 = 1.225   # Standard density (kg/m³)
    
    # Atmospheric layers (altitude in m, lapse rate in K/m)
    LAYERS = [
        (0,     -0.0065),   # Troposphere
        (11000,  0.0),      # Tropopause
        (20000,  0.001),    # Lower stratosphere
        (32000,  0.0028),   # Upper stratosphere
        (47000,  0.0),      # Stratopause
    ]
    
    def __init__(self):
        """Initialize atmospheric model."""
        self._precompute_layer_data()
    
    def _precompute_layer_data(self):
        """Precompute temperature and pressure at layer boundaries."""
        self.layer_data = []
        
        T = self.T0
        P = self.P0
        
        for i, (h, lapse_rate) in enumerate(self.LAYERS):
            self.layer_data.append({
                'altitude': h,
                'temperature': T,
                'pressure': P,
                'lapse_rate': lapse_rate
            })
            
            # Calculate temperature and pressure at next layer boundary
            if i < len(self.LAYERS) - 1:
                next_h = self.LAYERS[i + 1][0]
                dh = next_h - h
                
                if abs(lapse_rate) > 1e-10:  # Non-isothermal layer
                    T_next = T + lapse_rate * dh
                    P_next = P * (T_next / T) ** (-self.g0 / (lapse_rate * self.R))
                else:  # Isothermal layer
                    T_next = T
                    P_next = P * np.exp(-self.g0 * dh / (self.R * T))
                
                T = T_next
                P = P_next
    
    def get_properties(self, altitude: float) -> Tuple[float, float, float]:
        """
        Get atmospheric properties at given altitude.
        
        Args:
            altitude: Altitude in meters above sea level
            
        Returns:
            Tuple of (density, pressure, temperature) in SI units
        """
        # Clamp altitude to valid range
        altitude = max(0, min(altitude, 47000))
        
        # Find the appropriate atmospheric layer
        layer_idx = 0
        for i, data in enumerate(self.layer_data[:-1]):
            if altitude >= data['altitude']:
                layer_idx = i
            else:
                break
        
        layer = self.layer_data[layer_idx]
        h0 = layer['altitude']
        T0 = layer['temperature']
        P0 = layer['pressure']
        L = layer['lapse_rate']
        
        dh = altitude - h0
        
        # Calculate temperature and pressure at altitude
        if abs(L) > 1e-10:  # Non-isothermal layer
            T = T0 + L * dh
            P = P0 * (T / T0) ** (-self.g0 / (L * self.R))
        else:  # Isothermal layer
            T = T0
            P = P0 * np.exp(-self.g0 * dh / (self.R * T0))
        
        # Calculate density using ideal gas law
        rho = P / (self.R * T)
        
        return rho, P, T
    
    def get_density(self, altitude: float) -> float:
        """Get atmospheric density at given altitude."""
        rho, _, _ = self.get_properties(altitude)
        return rho
    
    def get_pressure(self, altitude: float) -> float:
        """Get atmospheric pressure at given altitude."""
        _, P, _ = self.get_properties(altitude)
        return P
    
    def get_temperature(self, altitude: float) -> float:
        """Get atmospheric temperature at given altitude."""
        _, _, T = self.get_properties(altitude)
        return T
    
    def get_speed_of_sound(self, altitude: float) -> float:
        """
        Get speed of sound at given altitude.
        
        Args:
            altitude: Altitude in meters
            
        Returns:
            Speed of sound in m/s
        """
        _, _, T = self.get_properties(altitude)
        gamma = 1.4  # Specific heat ratio for air
        return np.sqrt(gamma * self.R * T)
    
    def get_mach_number(self, velocity: float, altitude: float) -> float:
        """
        Get Mach number for given velocity and altitude.
        
        Args:
            velocity: Velocity magnitude in m/s
            altitude: Altitude in meters
            
        Returns:
            Mach number (dimensionless)
        """
        a = self.get_speed_of_sound(altitude)
        return velocity / a


class DragModel:
    """
    Aerodynamic drag model for missile simulation.
    
    This class provides drag force calculations based on atmospheric conditions,
    object geometry, and flight regime. It includes both subsonic and supersonic
    drag characteristics.
    """
    
    def __init__(self, 
                 reference_area: float = 0.05,
                 cd_subsonic: float = 0.5,
                 cd_supersonic: float = 0.8,
                 cd_transonic_peak: float = 1.2):
        """
        Initialize drag model.
        
        Args:
            reference_area: Reference area for drag calculation (m²)
            cd_subsonic: Drag coefficient for subsonic flight
            cd_supersonic: Drag coefficient for supersonic flight
            cd_transonic_peak: Peak drag coefficient in transonic regime
        """
        self.reference_area = reference_area
        self.cd_subsonic = cd_subsonic
        self.cd_supersonic = cd_supersonic
        self.cd_transonic_peak = cd_transonic_peak
        
        self.atmosphere = AtmosphericModel()
    
    def get_drag_coefficient(self, mach: float) -> float:
        """
        Get drag coefficient as function of Mach number.
        
        Args:
            mach: Mach number
            
        Returns:
            Drag coefficient (dimensionless)
        """
        if mach < 0.8:
            # Subsonic regime
            return self.cd_subsonic
        elif mach < 1.2:
            # Transonic regime - smooth transition with peak
            t = (mach - 0.8) / 0.4
            # Smooth curve with peak at mach = 1.0
            peak_factor = 1.0 - 4.0 * (t - 0.5)**2
            cd_peak = self.cd_transonic_peak * peak_factor
            cd_base = self.cd_subsonic + t * (self.cd_supersonic - self.cd_subsonic)
            return max(cd_base, cd_peak)
        else:
            # Supersonic regime
            return self.cd_supersonic
    
    def get_drag_force(self, 
                      velocity: np.ndarray, 
                      altitude: float) -> np.ndarray:
        """
        Calculate drag force vector with numerical stability.
        
        Args:
            velocity: Velocity vector [vx, vy, vz] in m/s
            altitude: Altitude in meters
            
        Returns:
            Drag force vector [fx, fy, fz] in N (opposite to velocity)
        """
        speed = np.linalg.norm(velocity)
        
        if speed < 1e-6:
            return np.zeros(3)
        
        # Clamp speed to prevent extreme calculations
        speed = min(speed, 1000.0)  # Limit to Mach 3 at sea level
        
        # Get atmospheric properties
        rho = self.atmosphere.get_density(altitude)
        mach = self.atmosphere.get_mach_number(speed, altitude)
        
        # Calculate drag coefficient
        cd = self.get_drag_coefficient(mach)
        
        # Drag force magnitude: F = 0.5 * ρ * v² * Cd * A
        drag_magnitude = 0.5 * rho * speed**2 * cd * self.reference_area
        
        # Apply maximum drag force limit for numerical stability
        MAX_DRAG = 10000.0  # 10kN maximum drag force
        drag_magnitude = min(drag_magnitude, MAX_DRAG)
        
        # Drag force direction (opposite to velocity)
        drag_direction = -velocity / speed
        
        return drag_magnitude * drag_direction


class WindField:
    """
    Configurable wind field model for atmospheric simulation.
    
    This class provides wind velocity as a function of position and time,
    including both steady wind components and turbulent gusts. It supports
    various wind profiles and can be configured for different scenarios.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize wind field model.
        
        Args:
            config: Configuration dictionary with wind parameters
        """
        # Default configuration
        self.config = {
            'steady_wind': [0.0, 0.0, 0.0],      # Steady wind vector (m/s)
            'wind_profile': 'uniform',            # 'uniform', 'logarithmic', 'power_law'
            'reference_height': 10.0,             # Reference height for wind profile (m)
            'reference_speed': 10.0,              # Reference wind speed (m/s)
            'turbulence_intensity': 0.1,          # Turbulence intensity (0-1)
            'turbulence_scale': 100.0,            # Turbulence length scale (m)
            'gust_enabled': True,                 # Enable wind gusts
            'gust_amplitude': 5.0,                # Gust amplitude (m/s)
            'gust_frequency': 0.1,                # Gust frequency (Hz)
            'wind_shear_enabled': False,          # Enable wind shear
            'shear_gradient': 0.01,               # Wind shear gradient (1/s)
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Initialize turbulence state
        self.turbulence_state = {
            'seed': 42,
            'time_offset': 0.0,
            'spatial_offset': np.zeros(3)
        }
        
        # Set random seed for reproducible turbulence
        np.random.seed(self.turbulence_state['seed'])
    
    def get_wind_velocity(self, 
                         position: np.ndarray, 
                         time: float) -> np.ndarray:
        """
        Get wind velocity at given position and time.
        
        Args:
            position: Position vector [x, y, z] in meters
            time: Current time in seconds
            
        Returns:
            Wind velocity vector [vx, vy, vz] in m/s
        """
        # Start with steady wind
        wind = np.array(self.config['steady_wind'])
        
        # Add wind profile effects
        wind += self._get_profile_wind(position)
        
        # Add turbulence
        if self.config['turbulence_intensity'] > 0:
            wind += self._get_turbulent_wind(position, time)
        
        # Add gusts
        if self.config['gust_enabled']:
            wind += self._get_gust_wind(position, time)
        
        # Add wind shear
        if self.config['wind_shear_enabled']:
            wind += self._get_shear_wind(position, time)
        
        return wind
    
    def _get_profile_wind(self, position: np.ndarray) -> np.ndarray:
        """Get wind based on altitude profile."""
        altitude = position[2]
        ref_height = self.config['reference_height']
        ref_speed = self.config['reference_speed']
        
        if self.config['wind_profile'] == 'uniform':
            # Uniform wind profile
            scale_factor = 1.0
        elif self.config['wind_profile'] == 'logarithmic':
            # Logarithmic wind profile (typical for atmospheric boundary layer)
            if altitude <= 0:
                scale_factor = 0.0
            else:
                scale_factor = np.log(altitude / 0.01) / np.log(ref_height / 0.01)
        elif self.config['wind_profile'] == 'power_law':
            # Power law wind profile
            if altitude <= 0:
                scale_factor = 0.0
            else:
                alpha = 0.15  # Power law exponent
                scale_factor = (altitude / ref_height) ** alpha
        else:
            scale_factor = 1.0
        
        # Apply scale factor to reference wind speed
        wind_magnitude = ref_speed * scale_factor
        
        # Assume wind is primarily horizontal (x-direction)
        return np.array([wind_magnitude, 0.0, 0.0])
    
    def _get_turbulent_wind(self, 
                           position: np.ndarray, 
                           time: float) -> np.ndarray:
        """Get turbulent wind component using simplified turbulence model."""
        intensity = self.config['turbulence_intensity']
        scale = self.config['turbulence_scale']
        
        # Use position and time to generate pseudo-random turbulence
        # This creates spatially and temporally correlated turbulence
        x, y, z = position
        t = time + self.turbulence_state['time_offset']
        
        # Generate turbulent components using sinusoidal functions
        # This is a simplified model - real turbulence would use more complex methods
        freq_x = 2 * np.pi / scale
        freq_y = 2 * np.pi / (scale * 1.2)
        freq_z = 2 * np.pi / (scale * 0.8)
        freq_t = 2 * np.pi * 0.5  # Temporal frequency
        
        # Turbulent velocity components
        turb_x = intensity * np.sin(freq_x * x + freq_t * t) * np.cos(freq_y * y)
        turb_y = intensity * np.sin(freq_y * y + freq_t * t * 1.1) * np.cos(freq_z * z)
        turb_z = intensity * np.sin(freq_z * z + freq_t * t * 0.9) * np.cos(freq_x * x)
        
        # Add some randomness
        phase_x = np.sin(0.1 * x + 0.2 * t)
        phase_y = np.sin(0.15 * y + 0.18 * t)
        phase_z = np.sin(0.12 * z + 0.22 * t)
        
        turb_x += 0.3 * intensity * phase_x
        turb_y += 0.3 * intensity * phase_y
        turb_z += 0.3 * intensity * phase_z
        
        return np.array([turb_x, turb_y, turb_z])
    
    def _get_gust_wind(self, 
                      position: np.ndarray, 
                      time: float) -> np.ndarray:
        """Get wind gust component."""
        amplitude = self.config['gust_amplitude']
        frequency = self.config['gust_frequency']
        
        # Generate gusts using low-frequency oscillations
        gust_phase = 2 * np.pi * frequency * time
        
        # Multiple gust components with different phases
        gust_x = amplitude * np.sin(gust_phase) * np.exp(-0.5 * np.sin(gust_phase * 0.3)**2)
        gust_y = amplitude * 0.3 * np.sin(gust_phase * 1.2 + np.pi/4)
        gust_z = amplitude * 0.2 * np.sin(gust_phase * 0.8 + np.pi/2)
        
        return np.array([gust_x, gust_y, gust_z])
    
    def _get_shear_wind(self, 
                       position: np.ndarray, 
                       time: float) -> np.ndarray:
        """Get wind shear component."""
        gradient = self.config['shear_gradient']
        
        # Simple vertical wind shear
        altitude = position[2]
        shear_x = gradient * altitude
        shear_y = 0.0
        shear_z = 0.0
        
        return np.array([shear_x, shear_y, shear_z])
    
    def get_wind_force(self, 
                      velocity: np.ndarray, 
                      position: np.ndarray, 
                      time: float,
                      drag_model) -> np.ndarray:
        """
        Calculate wind force on object.
        
        Args:
            velocity: Object velocity vector in m/s
            position: Object position vector in m
            time: Current time in seconds
            drag_model: Drag model instance
            
        Returns:
            Wind force vector in N
        """
        # Get wind velocity at object position
        wind_velocity = self.get_wind_velocity(position, time)
        
        # Calculate relative velocity (object velocity relative to air)
        relative_velocity = velocity - wind_velocity
        
        # Calculate drag force using relative velocity
        altitude = position[2]
        wind_force = drag_model.get_drag_force(relative_velocity, altitude)
        
        return wind_force
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Update wind field configuration."""
        self.config.update(config)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current wind field configuration."""
        return self.config.copy()
    
    def save_config(self, filename: str) -> None:
        """Save wind field configuration to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_config(self, filename: str) -> None:
        """Load wind field configuration from JSON file."""
        with open(filename, 'r') as f:
            config = json.load(f)
        self.set_config(config)
    
    def reset_turbulence(self, seed: Optional[int] = None) -> None:
        """Reset turbulence state with optional new seed."""
        if seed is not None:
            self.turbulence_state['seed'] = seed
        
        self.turbulence_state['time_offset'] = 0.0
        self.turbulence_state['spatial_offset'] = np.zeros(3)
        np.random.seed(self.turbulence_state['seed'])


# Predefined wind configurations for common scenarios
WIND_CONFIGS = {
    'calm': {
        'steady_wind': [0.0, 0.0, 0.0],
        'turbulence_intensity': 0.0,
        'gust_enabled': False,
        'wind_shear_enabled': False
    },
    
    'light_breeze': {
        'steady_wind': [5.0, 0.0, 0.0],
        'turbulence_intensity': 0.05,
        'gust_amplitude': 2.0,
        'wind_profile': 'logarithmic'
    },
    
    'moderate_wind': {
        'steady_wind': [15.0, 2.0, 0.0],
        'turbulence_intensity': 0.15,
        'gust_amplitude': 8.0,
        'wind_profile': 'power_law',
        'wind_shear_enabled': True
    },
    
    'strong_wind': {
        'steady_wind': [25.0, 5.0, 0.0],
        'turbulence_intensity': 0.25,
        'gust_amplitude': 15.0,
        'gust_frequency': 0.2,
        'wind_profile': 'power_law',
        'wind_shear_enabled': True,
        'shear_gradient': 0.02
    },
    
    'crosswind': {
        'steady_wind': [5.0, 20.0, 0.0],
        'turbulence_intensity': 0.1,
        'gust_amplitude': 10.0,
        'wind_profile': 'logarithmic'
    }
}