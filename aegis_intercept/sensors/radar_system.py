"""
Radar system simulation for AegisIntercept Phase 3.

This module provides realistic radar sensor modeling including range,
bearing, and elevation measurements with appropriate noise characteristics
and detection limitations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math


class RadarSystem:
    """
    Realistic radar system simulation for missile interception.
    
    This class models a radar system with realistic characteristics including:
    - Range-dependent noise
    - Detection probability based on target RCS and range
    - Beam width limitations
    - Update rate constraints
    - Atmospheric attenuation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize radar system.
        
        Args:
            config: Configuration dictionary with radar parameters
        """
        # Default radar configuration
        self.config = {
            'max_range': 50000.0,           # Maximum detection range (m)
            'min_range': 100.0,             # Minimum detection range (m)
            'azimuth_fov': 120.0,           # Azimuth field of view (degrees)
            'elevation_fov': 60.0,          # Elevation field of view (degrees)
            'range_resolution': 10.0,       # Range resolution (m)
            'angle_resolution': 0.5,        # Angular resolution (degrees)
            'update_rate': 10.0,            # Update rate (Hz)
            'noise_factor': 0.1,            # Noise factor for measurements
            'detection_threshold': 0.7,     # Detection probability threshold
            'atmospheric_loss': 0.0001,     # Atmospheric loss factor (1/m)
            'beam_width': 2.0,              # Beam width (degrees)
            'transmit_power': 1000000.0,    # Transmit power (W)
            'antenna_gain': 40.0,           # Antenna gain (dB)
            'system_temp': 300.0,           # System temperature (K)
            'bandwidth': 1000000.0,         # Bandwidth (Hz)
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Initialize radar state
        self.last_update_time = 0.0
        self.tracked_targets = {}
        self.detection_history = []
        
        # Convert some parameters to radians
        self.azimuth_fov_rad = math.radians(self.config['azimuth_fov'])
        self.elevation_fov_rad = math.radians(self.config['elevation_fov'])
        self.angle_resolution_rad = math.radians(self.config['angle_resolution'])
        self.beam_width_rad = math.radians(self.config['beam_width'])
    
    def detect_target(self, 
                     target_position: np.ndarray,
                     target_rcs: float = 1.0,
                     current_time: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Attempt to detect a target at given position.
        
        Args:
            target_position: Target position [x, y, z] in meters
            target_rcs: Target radar cross section (m²)
            current_time: Current simulation time (s)
            
        Returns:
            Detection dictionary if target is detected, None otherwise
        """
        # Check if enough time has passed for next update
        if current_time - self.last_update_time < 1.0 / self.config['update_rate']:
            return None
        
        # Convert position to spherical coordinates
        range_m, azimuth_rad, elevation_rad = self._cartesian_to_spherical(target_position)
        
        # Check if target is within radar coverage
        if not self._is_in_coverage(range_m, azimuth_rad, elevation_rad):
            return None
        
        # Calculate detection probability
        detection_prob = self._calculate_detection_probability(range_m, target_rcs)
        
        # Random detection based on probability
        if np.random.random() > detection_prob:
            return None
        
        # Add measurement noise
        noisy_range = self._add_range_noise(range_m)
        noisy_azimuth = self._add_angle_noise(azimuth_rad)
        noisy_elevation = self._add_angle_noise(elevation_rad)
        
        # Update last update time
        self.last_update_time = current_time
        
        # Create detection record
        detection = {
            'timestamp': current_time,
            'range': noisy_range,
            'azimuth': noisy_azimuth,
            'elevation': noisy_elevation,
            'rcs': target_rcs,
            'detection_probability': detection_prob,
            'snr': self._calculate_snr(range_m, target_rcs)
        }
        
        # Store in detection history
        self.detection_history.append(detection)
        
        return detection
    
    def _cartesian_to_spherical(self, position: np.ndarray) -> Tuple[float, float, float]:
        """Convert Cartesian coordinates to spherical (range, azimuth, elevation)."""
        x, y, z = position
        
        range_m = np.sqrt(x**2 + y**2 + z**2)
        azimuth_rad = np.arctan2(y, x)
        elevation_rad = np.arctan2(z, np.sqrt(x**2 + y**2))
        
        return range_m, azimuth_rad, elevation_rad
    
    def _spherical_to_cartesian(self, range_m: float, azimuth_rad: float, elevation_rad: float) -> np.ndarray:
        """Convert spherical coordinates to Cartesian."""
        x = range_m * np.cos(elevation_rad) * np.cos(azimuth_rad)
        y = range_m * np.cos(elevation_rad) * np.sin(azimuth_rad)
        z = range_m * np.sin(elevation_rad)
        
        return np.array([x, y, z])
    
    def _is_in_coverage(self, range_m: float, azimuth_rad: float, elevation_rad: float) -> bool:
        """Check if target is within radar coverage."""
        # Range check
        if range_m < self.config['min_range'] or range_m > self.config['max_range']:
            return False
        
        # Azimuth check
        if abs(azimuth_rad) > self.azimuth_fov_rad / 2:
            return False
        
        # Elevation check
        if abs(elevation_rad) > self.elevation_fov_rad / 2:
            return False
        
        return True
    
    def _calculate_detection_probability(self, range_m: float, target_rcs: float) -> float:
        """Calculate detection probability based on radar equation."""
        # Simplified radar equation: SNR = (P_t * G * σ * λ²) / ((4π)³ * R⁴ * k * T * B * L)
        # Where: P_t = transmit power, G = antenna gain, σ = RCS, λ = wavelength,
        #        R = range, k = Boltzmann constant, T = system temperature, B = bandwidth, L = losses
        
        # Constants
        k_boltzmann = 1.380649e-23  # J/K
        c = 3e8  # Speed of light
        frequency = 10e9  # Assume 10 GHz radar
        wavelength = c / frequency
        
        # Convert antenna gain from dB to linear
        antenna_gain_linear = 10**(self.config['antenna_gain'] / 10)
        
        # Calculate SNR
        numerator = (self.config['transmit_power'] * antenna_gain_linear * target_rcs * wavelength**2)
        denominator = ((4 * np.pi)**3 * range_m**4 * k_boltzmann * 
                      self.config['system_temp'] * self.config['bandwidth'])
        
        # Add atmospheric losses
        atmospheric_loss = np.exp(-self.config['atmospheric_loss'] * range_m)
        
        snr = (numerator / denominator) * atmospheric_loss
        
        # Convert to dB
        snr_db = 10 * np.log10(max(snr, 1e-10))
        
        # Detection probability based on SNR (simplified Swerling model)
        # Assume minimum SNR of 13 dB for reliable detection
        min_snr_db = 13.0
        
        if snr_db < min_snr_db:
            # Exponential decay for low SNR
            detection_prob = np.exp((snr_db - min_snr_db) / 3.0)
        else:
            # High probability for good SNR
            detection_prob = 0.99
        
        return np.clip(detection_prob, 0.0, 1.0)
    
    def _calculate_snr(self, range_m: float, target_rcs: float) -> float:
        """Calculate signal-to-noise ratio."""
        # Simplified SNR calculation
        range_factor = (range_m / 1000.0)**4  # Fourth power range dependence
        snr = (target_rcs * self.config['transmit_power']) / (range_factor * self.config['system_temp'])
        return 10 * np.log10(max(snr, 1e-10))
    
    def _add_range_noise(self, true_range: float) -> float:
        """Add range measurement noise."""
        # Range noise increases with range
        noise_std = self.config['range_resolution'] * self.config['noise_factor'] * (1 + true_range / 10000.0)
        return true_range + np.random.normal(0, noise_std)
    
    def _add_angle_noise(self, true_angle: float) -> float:
        """Add angular measurement noise."""
        # Angular noise is relatively constant
        noise_std = self.angle_resolution_rad * self.config['noise_factor']
        return true_angle + np.random.normal(0, noise_std)
    
    def get_measurement_from_detection(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert detection to measurement format.
        
        Args:
            detection: Detection dictionary from detect_target
            
        Returns:
            Measurement dictionary with position and uncertainty
        """
        # Convert back to Cartesian coordinates
        position = self._spherical_to_cartesian(
            detection['range'],
            detection['azimuth'],
            detection['elevation']
        )
        
        # Calculate measurement uncertainties
        range_uncertainty = self.config['range_resolution'] * self.config['noise_factor']
        angle_uncertainty = self.angle_resolution_rad * self.config['noise_factor']
        
        # Convert angular uncertainty to Cartesian uncertainty
        # This is a simplified approximation
        cross_range_uncertainty = detection['range'] * angle_uncertainty
        
        uncertainty_matrix = np.diag([
            range_uncertainty**2,
            cross_range_uncertainty**2,
            cross_range_uncertainty**2
        ])
        
        return {
            'position': position,
            'uncertainty': uncertainty_matrix,
            'timestamp': detection['timestamp'],
            'sensor_type': 'radar',
            'quality': detection['detection_probability']
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get current radar configuration."""
        return self.config.copy()
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Update radar configuration."""
        self.config.update(config)
        
        # Update derived parameters
        self.azimuth_fov_rad = math.radians(self.config['azimuth_fov'])
        self.elevation_fov_rad = math.radians(self.config['elevation_fov'])
        self.angle_resolution_rad = math.radians(self.config['angle_resolution'])
        self.beam_width_rad = math.radians(self.config['beam_width'])
    
    def reset(self) -> None:
        """Reset radar system state."""
        self.last_update_time = 0.0
        self.tracked_targets = {}
        self.detection_history = []
    
    def get_detection_history(self) -> List[Dict[str, Any]]:
        """Get history of all detections."""
        return self.detection_history.copy()
    
    def clear_detection_history(self) -> None:
        """Clear detection history."""
        self.detection_history = []