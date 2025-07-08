"""
Wind field modeling for AegisIntercept Phase 3.

This module provides configurable wind field models including steady wind,
turbulence, and gust effects for realistic atmospheric simulation.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import json


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