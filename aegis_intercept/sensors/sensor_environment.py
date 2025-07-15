"""
Sensor environment simulation for AegisIntercept Phase 3.

This module provides environmental effects on sensor systems including
atmospheric attenuation, interference, and multi-path effects.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math


class SensorEnvironment:
    """
    Environmental effects simulation for sensor systems.
    
    This class models environmental factors that affect sensor performance:
    - Atmospheric attenuation
    - Weather effects (rain, fog, clouds)
    - Electronic interference
    - Multi-path effects
    - Clutter and false alarms
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize sensor environment.
        
        Args:
            config: Configuration dictionary with environment parameters
        """
        # Default environment configuration
        self.config = {
            'weather_condition': 'clear',     # 'clear', 'rain', 'fog', 'storm'
            'precipitation_rate': 0.0,        # Precipitation rate (mm/hr)
            'visibility': 10000.0,            # Visibility (m)
            'humidity': 0.5,                  # Relative humidity (0-1)
            'temperature': 288.15,            # Temperature (K)
            'atmospheric_pressure': 101325.0, # Atmospheric pressure (Pa)
            'electromagnetic_interference': 0.0, # EMI level (0-1)
            'clutter_density': 0.1,           # Clutter density (0-1)
            'multipath_factor': 0.05,         # Multi-path interference factor
            'false_alarm_rate': 0.01,         # False alarm rate (per second)
            'ducting_present': False,         # Atmospheric ducting effect
            'solar_activity': 0.0,            # Solar activity level (0-1)
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Initialize environment state
        self.clutter_objects = []
        self.interference_sources = []
        self.false_alarms = []
        
        # Generate static clutter
        self._generate_clutter()
    
    def apply_atmospheric_effects(self, 
                                 measurement: Dict[str, Any],
                                 sensor_position: np.ndarray) -> Dict[str, Any]:
        """
        Apply atmospheric effects to a sensor measurement.
        
        Args:
            measurement: Original measurement dictionary
            sensor_position: Position of the sensor
            
        Returns:
            Modified measurement with atmospheric effects applied
        """
        modified_measurement = measurement.copy()
        
        # Calculate range to target
        target_position = measurement['position']
        range_to_target = np.linalg.norm(target_position - sensor_position)
        
        # Apply atmospheric attenuation
        attenuation_factor = self._calculate_atmospheric_attenuation(range_to_target)
        
        # Modify measurement quality based on attenuation
        if 'quality' in modified_measurement:
            modified_measurement['quality'] *= attenuation_factor
        
        # Apply weather effects
        weather_factor = self._calculate_weather_effects(range_to_target)
        if 'quality' in modified_measurement:
            modified_measurement['quality'] *= weather_factor
        
        # Add measurement noise based on conditions
        noise_factor = self._calculate_environmental_noise()
        if 'uncertainty' in modified_measurement:
            modified_measurement['uncertainty'] *= (1 + noise_factor)
        
        # Apply multi-path effects
        if self.config['multipath_factor'] > 0:
            modified_measurement = self._apply_multipath_effects(modified_measurement)
        
        return modified_measurement
    
    def _calculate_atmospheric_attenuation(self, range_m: float) -> float:
        """Calculate atmospheric attenuation factor."""
        # Base attenuation from humidity and temperature
        base_attenuation = 0.0001  # Base attenuation per meter
        
        # Humidity effect (water vapor absorption)
        humidity_factor = 1 + 2 * self.config['humidity']
        
        # Temperature effect
        temp_factor = 1 + abs(self.config['temperature'] - 288.15) / 100.0
        
        # Calculate total attenuation
        total_attenuation = base_attenuation * humidity_factor * temp_factor
        
        # Exponential decay with range
        attenuation_factor = np.exp(-total_attenuation * range_m)
        
        return max(attenuation_factor, 0.1)  # Minimum 10% transmission
    
    def _calculate_weather_effects(self, range_m: float) -> float:
        """Calculate weather-related attenuation."""
        weather_factor = 1.0
        
        if self.config['weather_condition'] == 'rain':
            # Rain attenuation (frequency dependent, simplified)
            rain_rate = self.config['precipitation_rate']  # mm/hr
            attenuation_db_per_km = 0.01 * rain_rate**1.2  # Simplified model
            attenuation_factor = 10**(-attenuation_db_per_km * range_m / 1000.0 / 10.0)
            weather_factor *= attenuation_factor
        
        elif self.config['weather_condition'] == 'fog':
            # Fog attenuation based on visibility
            visibility = self.config['visibility']
            if visibility < 1000:  # Dense fog
                attenuation_factor = visibility / 1000.0
                weather_factor *= attenuation_factor
        
        elif self.config['weather_condition'] == 'storm':
            # Storm effects (combined rain and electrical interference)
            storm_factor = 0.5 + 0.3 * np.random.random()
            weather_factor *= storm_factor
        
        return max(weather_factor, 0.05)  # Minimum 5% transmission
    
    def _calculate_environmental_noise(self) -> float:
        """Calculate environmental noise factor."""
        noise_factor = 0.0
        
        # Electromagnetic interference
        emi_level = self.config['electromagnetic_interference']
        noise_factor += emi_level * 0.5
        
        # Solar activity effects
        solar_level = self.config['solar_activity']
        noise_factor += solar_level * 0.3
        
        # Weather-related noise
        if self.config['weather_condition'] == 'storm':
            noise_factor += 0.2  # Electrical storm interference
        
        return noise_factor
    
    def _apply_multipath_effects(self, measurement: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multi-path interference effects."""
        multipath_factor = self.config['multipath_factor']
        
        # Add position error due to multi-path
        position_error = np.random.normal(0, multipath_factor * 10, 3)  # 10m per 0.1 factor
        measurement['position'] += position_error
        
        # Increase uncertainty
        if 'uncertainty' in measurement:
            multipath_uncertainty = np.eye(3) * (multipath_factor * 100)**2
            measurement['uncertainty'] += multipath_uncertainty
        
        return measurement
    
    def generate_clutter_detections(self, 
                                   sensor_position: np.ndarray,
                                   sensor_range: float,
                                   current_time: float) -> List[Dict[str, Any]]:
        """
        Generate false detections due to clutter.
        
        Args:
            sensor_position: Position of the sensor
            sensor_range: Maximum sensor range
            current_time: Current simulation time
            
        Returns:
            List of false detection dictionaries
        """
        clutter_detections = []
        
        # Generate clutter based on density
        num_clutter = int(self.config['clutter_density'] * 10)
        
        for _ in range(num_clutter):
            # Random position within sensor range
            angle = np.random.uniform(0, 2 * np.pi)
            elevation = np.random.uniform(-np.pi/6, np.pi/6)  # +/- 30 degrees
            range_m = np.random.uniform(100, sensor_range)
            
            # Convert to Cartesian coordinates
            x = range_m * np.cos(elevation) * np.cos(angle)
            y = range_m * np.cos(elevation) * np.sin(angle)
            z = range_m * np.sin(elevation)
            
            clutter_position = sensor_position + np.array([x, y, z])
            
            # Create clutter detection
            clutter_detection = {
                'position': clutter_position,
                'timestamp': current_time,
                'sensor_type': 'radar',
                'quality': 0.3,  # Low quality for clutter
                'is_clutter': True,
                'uncertainty': np.eye(3) * 100**2  # High uncertainty
            }
            
            clutter_detections.append(clutter_detection)
        
        return clutter_detections
    
    def generate_false_alarms(self, 
                             sensor_position: np.ndarray,
                             sensor_range: float,
                             current_time: float,
                             dt: float) -> List[Dict[str, Any]]:
        """
        Generate false alarm detections.
        
        Args:
            sensor_position: Position of the sensor
            sensor_range: Maximum sensor range
            current_time: Current simulation time
            dt: Time step
            
        Returns:
            List of false alarm dictionaries
        """
        false_alarms = []
        
        # Calculate probability of false alarm in this time step
        false_alarm_prob = self.config['false_alarm_rate'] * dt
        
        if np.random.random() < false_alarm_prob:
            # Generate false alarm at random position
            angle = np.random.uniform(0, 2 * np.pi)
            elevation = np.random.uniform(-np.pi/6, np.pi/6)
            range_m = np.random.uniform(1000, sensor_range)
            
            x = range_m * np.cos(elevation) * np.cos(angle)
            y = range_m * np.cos(elevation) * np.sin(angle)
            z = range_m * np.sin(elevation)
            
            false_alarm_position = sensor_position + np.array([x, y, z])
            
            false_alarm = {
                'position': false_alarm_position,
                'timestamp': current_time,
                'sensor_type': 'radar',
                'quality': 0.5,  # Medium quality to make it believable
                'is_false_alarm': True,
                'uncertainty': np.eye(3) * 50**2
            }
            
            false_alarms.append(false_alarm)
        
        return false_alarms
    
    def _generate_clutter(self) -> None:
        """Generate static clutter objects."""
        # This could be extended to include terrain, buildings, etc.
        # For now, just placeholder
        pass
    
    def update_weather(self, new_weather_config: Dict[str, Any]) -> None:
        """Update weather conditions."""
        weather_params = {
            'weather_condition', 'precipitation_rate', 'visibility',
            'humidity', 'temperature', 'atmospheric_pressure'
        }
        
        for param in weather_params:
            if param in new_weather_config:
                self.config[param] = new_weather_config[param]
    
    def set_electromagnetic_interference(self, emi_level: float) -> None:
        """Set electromagnetic interference level."""
        self.config['electromagnetic_interference'] = np.clip(emi_level, 0.0, 1.0)
    
    def set_solar_activity(self, solar_level: float) -> None:
        """Set solar activity level."""
        self.config['solar_activity'] = np.clip(solar_level, 0.0, 1.0)
    
    def get_environment_status(self) -> Dict[str, Any]:
        """Get current environment status."""
        return {
            'weather': self.config['weather_condition'],
            'visibility': self.config['visibility'],
            'precipitation': self.config['precipitation_rate'],
            'emi_level': self.config['electromagnetic_interference'],
            'solar_activity': self.config['solar_activity'],
            'atmospheric_conditions': {
                'temperature': self.config['temperature'],
                'humidity': self.config['humidity'],
                'pressure': self.config['atmospheric_pressure']
            }
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get current environment configuration."""
        return self.config.copy()
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Update environment configuration."""
        self.config.update(config)
    
    def reset(self) -> None:
        """Reset environment state."""
        self.clutter_objects = []
        self.interference_sources = []
        self.false_alarms = []
        self._generate_clutter()


# Predefined environment configurations
ENVIRONMENT_CONFIGS = {
    'clear_day': {
        'weather_condition': 'clear',
        'visibility': 20000.0,
        'humidity': 0.4,
        'electromagnetic_interference': 0.0,
        'false_alarm_rate': 0.001
    },
    
    'rainy_day': {
        'weather_condition': 'rain',
        'precipitation_rate': 5.0,
        'visibility': 5000.0,
        'humidity': 0.8,
        'electromagnetic_interference': 0.1,
        'false_alarm_rate': 0.01
    },
    
    'foggy_conditions': {
        'weather_condition': 'fog',
        'visibility': 500.0,
        'humidity': 0.95,
        'electromagnetic_interference': 0.05,
        'false_alarm_rate': 0.02
    },
    
    'storm_conditions': {
        'weather_condition': 'storm',
        'precipitation_rate': 20.0,
        'visibility': 1000.0,
        'humidity': 0.9,
        'electromagnetic_interference': 0.3,
        'false_alarm_rate': 0.05,
        'solar_activity': 0.2
    },
    
    'high_interference': {
        'weather_condition': 'clear',
        'electromagnetic_interference': 0.5,
        'solar_activity': 0.3,
        'false_alarm_rate': 0.03
    }
}