"""
Sensor Environment Effects

Implements realistic environmental effects on sensor performance including:
- Atmospheric propagation effects
- Weather impacts
- Electronic warfare
- Ground clutter modeling
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math
import cmath


class WeatherConditions(Enum):
    """Weather condition types"""
    CLEAR = "clear"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    SNOW = "snow"
    FOG = "fog"
    STORM = "storm"


class ClutterType(Enum):
    """Types of radar clutter"""
    GROUND = "ground"
    SEA = "sea"
    WEATHER = "weather"
    CHAFF = "chaff"
    BIRDS = "birds"


@dataclass
class AtmosphericEffects:
    """Atmospheric propagation effects on radar signals"""
    
    def __init__(self):
        """Initialize atmospheric model"""
        # Standard atmosphere parameters
        self.sea_level_density = 1.225  # kg/m³
        self.scale_height = 8400.0  # meters
        self.humidity = 50.0  # percent
        self.temperature = 288.15  # Kelvin at sea level
        
    def calculate_atmospheric_loss(self, frequency_hz: float, range_m: float,
                                 altitude_m: float = 0.0, 
                                 weather: WeatherConditions = WeatherConditions.CLEAR) -> float:
        """
        Calculate one-way atmospheric loss in dB
        
        Args:
            frequency_hz: Radar frequency in Hz
            range_m: Range to target in meters
            altitude_m: Average altitude of propagation path
            weather: Weather conditions
            
        Returns:
            Atmospheric loss in dB
        """
        frequency_ghz = frequency_hz / 1e9
        range_km = range_m / 1000.0
        
        # Base atmospheric absorption (clear air)
        clear_air_loss = self._clear_air_absorption(frequency_ghz, range_km, altitude_m)
        
        # Weather-dependent losses
        weather_loss = self._weather_loss(frequency_ghz, range_km, weather)
        
        # Total loss (two-way path for radar)
        total_loss = 2 * (clear_air_loss + weather_loss)
        
        return total_loss
    
    def _clear_air_absorption(self, freq_ghz: float, range_km: float, 
                            altitude_m: float) -> float:
        """Calculate clear air absorption loss"""
        # Altitude-dependent atmospheric density
        density_ratio = math.exp(-altitude_m / self.scale_height)
        
        # Frequency-dependent absorption coefficients (dB/km)
        if freq_ghz < 1.0:
            absorption_coeff = 0.001 * freq_ghz**2
        elif freq_ghz < 10.0:
            absorption_coeff = 0.01 * freq_ghz
        elif freq_ghz < 35.0:
            absorption_coeff = 0.05 + 0.01 * (freq_ghz - 10.0)
        else:
            # Oxygen and water vapor absorption bands
            absorption_coeff = 0.2 + 0.05 * (freq_ghz - 35.0)
            
        # Scale by atmospheric density
        effective_absorption = absorption_coeff * density_ratio
        
        return effective_absorption * range_km
    
    def _weather_loss(self, freq_ghz: float, range_km: float,
                     weather: WeatherConditions) -> float:
        """Calculate weather-dependent loss"""
        # Weather loss coefficients (dB/km)
        weather_coeffs = {
            WeatherConditions.CLEAR: 0.0,
            WeatherConditions.LIGHT_RAIN: 0.1 * freq_ghz**1.5,
            WeatherConditions.HEAVY_RAIN: 0.5 * freq_ghz**1.5,
            WeatherConditions.SNOW: 0.05 * freq_ghz**1.2,
            WeatherConditions.FOG: 0.02 * freq_ghz**2,
            WeatherConditions.STORM: 1.0 * freq_ghz**1.5
        }
        
        loss_coeff = weather_coeffs.get(weather, 0.0)
        return loss_coeff * range_km
    
    def calculate_multipath_effects(self, radar_height: float, target_height: float,
                                  range_m: float, frequency_hz: float) -> Dict[str, float]:
        """
        Calculate multipath propagation effects
        
        Args:
            radar_height: Radar height above ground
            target_height: Target height above ground
            range_m: Slant range to target
            frequency_hz: Radar frequency
            
        Returns:
            Dictionary with multipath effects
        """
        wavelength = 3e8 / frequency_hz
        
        # Ground reflection coefficient (depends on terrain)
        ground_reflectivity = 0.7  # Typical for land
        
        # Path difference between direct and reflected paths
        path_difference = 2 * radar_height * target_height / range_m
        
        # Phase difference
        phase_diff = 2 * math.pi * path_difference / wavelength
        
        # Multipath factor (can cause fading or enhancement)
        multipath_factor = abs(1 + ground_reflectivity * cmath.exp(1j * phase_diff))
        
        # Convert to dB
        multipath_loss_db = -20 * math.log10(multipath_factor)
        
        return {
            'multipath_loss_db': multipath_loss_db,
            'path_difference_m': path_difference,
            'phase_difference_rad': phase_diff,
            'direct_path_m': range_m,
            'reflected_path_m': range_m + path_difference
        }


@dataclass
class ClutterModel:
    """Ground and weather clutter modeling"""
    
    def __init__(self):
        """Initialize clutter model"""
        # Clutter parameters by type
        self.clutter_params = {
            ClutterType.GROUND: {
                'reflectivity': -40.0,  # dB per m²
                'doppler_spread': 2.0,  # m/s
                'correlation_time': 10.0  # seconds
            },
            ClutterType.SEA: {
                'reflectivity': -35.0,
                'doppler_spread': 5.0,
                'correlation_time': 5.0
            },
            ClutterType.WEATHER: {
                'reflectivity': -30.0,
                'doppler_spread': 10.0,
                'correlation_time': 1.0
            },
            ClutterType.CHAFF: {
                'reflectivity': -20.0,
                'doppler_spread': 15.0,
                'correlation_time': 30.0
            },
            ClutterType.BIRDS: {
                'reflectivity': -50.0,
                'doppler_spread': 8.0,
                'correlation_time': 2.0
            }
        }
    
    def calculate_clutter_power(self, radar_position: np.ndarray, 
                              beam_direction: np.ndarray, beam_width: float,
                              range_m: float, clutter_type: ClutterType) -> float:
        """
        Calculate clutter power in beam
        
        Args:
            radar_position: 3D radar position
            beam_direction: 3D beam pointing direction (unit vector)
            beam_width: Beam width in radians
            range_m: Range to clutter patch
            clutter_type: Type of clutter
            
        Returns:
            Clutter power in dBm
        """
        params = self.clutter_params[clutter_type]
        
        # Clutter patch area illuminated by beam
        patch_area = (beam_width * range_m)**2
        
        # Clutter reflectivity per unit area
        sigma_0 = params['reflectivity']  # dB per m²
        
        # Total clutter RCS
        clutter_rcs_db = sigma_0 + 10 * math.log10(patch_area)
        
        # Convert to linear for power calculation
        clutter_rcs_linear = 10**(clutter_rcs_db / 10)
        
        # Simplified power calculation (would normally use full radar equation)
        # Assume normalized power at 1km = -60 dBm
        reference_power = -60.0  # dBm at 1km
        range_loss = 40 * math.log10(range_m / 1000.0)
        
        clutter_power = reference_power + clutter_rcs_db - range_loss
        
        return clutter_power
    
    def get_clutter_doppler_spectrum(self, clutter_type: ClutterType,
                                   wind_speed: float = 0.0) -> Dict[str, float]:
        """
        Get clutter doppler characteristics
        
        Args:
            clutter_type: Type of clutter
            wind_speed: Wind speed affecting clutter motion
            
        Returns:
            Doppler spectrum parameters
        """
        params = self.clutter_params[clutter_type]
        
        # Base doppler spread
        base_spread = params['doppler_spread']
        
        # Wind effects
        if clutter_type in [ClutterType.GROUND, ClutterType.SEA]:
            wind_spread = wind_speed * 0.5  # Wind affects surface clutter
        else:
            wind_spread = wind_speed * 1.0  # Wind affects volume clutter more
            
        total_spread = base_spread + wind_spread
        
        return {
            'mean_doppler': 0.0,  # Stationary clutter has zero mean doppler
            'doppler_spread': total_spread,
            'correlation_time': params['correlation_time']
        }


@dataclass
class ElectronicWarfare:
    """Electronic warfare effects on radar systems"""
    
    def __init__(self):
        """Initialize electronic warfare model"""
        self.active_jammers = []
        self.jamming_effectiveness = {}
        
    def add_jammer(self, jammer_position: np.ndarray, power_watts: float,
                   frequency_hz: float, jammer_type: str = "noise"):
        """
        Add an active jammer to the environment
        
        Args:
            jammer_position: 3D position of jammer
            power_watts: Jammer power in watts
            frequency_hz: Jammer frequency
            jammer_type: Type of jamming (noise, deception, etc.)
        """
        jammer = {
            'position': jammer_position,
            'power': power_watts,
            'frequency': frequency_hz,
            'type': jammer_type,
            'active': True
        }
        self.active_jammers.append(jammer)
    
    def calculate_jamming_effects(self, radar_position: np.ndarray,
                                radar_frequency: float, radar_bandwidth: float) -> Dict[str, float]:
        """
        Calculate jamming effects on radar performance
        
        Args:
            radar_position: 3D radar position
            radar_frequency: Radar operating frequency
            radar_bandwidth: Radar bandwidth
            
        Returns:
            Dictionary with jamming effects
        """
        total_jamming_power = 0.0
        closest_jammer_range = float('inf')
        
        for jammer in self.active_jammers:
            if not jammer['active']:
                continue
                
            # Calculate range to jammer
            jammer_range = np.linalg.norm(jammer['position'] - radar_position)
            closest_jammer_range = min(closest_jammer_range, jammer_range)
            
            # Frequency overlap factor
            freq_diff = abs(jammer['frequency'] - radar_frequency)
            overlap_factor = max(0, 1.0 - freq_diff / radar_bandwidth)
            
            # Jamming power at radar (simplified)
            # Real calculation would include antenna patterns, polarization, etc.
            path_loss = (4 * math.pi * jammer_range / (3e8 / jammer['frequency']))**2
            jamming_power_at_radar = jammer['power'] / path_loss * overlap_factor
            
            total_jamming_power += jamming_power_at_radar
            
        # Convert to dB
        if total_jamming_power > 0:
            jamming_power_db = 10 * math.log10(total_jamming_power) + 30  # Convert to dBm
        else:
            jamming_power_db = -200.0  # No jamming
            
        # Jamming effectiveness
        if jamming_power_db > -100.0:
            effectiveness = min(1.0, (jamming_power_db + 100.0) / 50.0)
        else:
            effectiveness = 0.0
            
        return {
            'jamming_power_dbm': jamming_power_db,
            'effectiveness': effectiveness,
            'closest_jammer_range': closest_jammer_range,
            'number_of_jammers': len([j for j in self.active_jammers if j['active']])
        }


class SensorEnvironment:
    """Complete sensor environment with all effects"""
    
    def __init__(self, weather: WeatherConditions = WeatherConditions.CLEAR):
        """
        Initialize sensor environment
        
        Args:
            weather: Current weather conditions
        """
        self.weather = weather
        self.atmospheric_effects = AtmosphericEffects()
        self.clutter_model = ClutterModel()
        self.electronic_warfare = ElectronicWarfare()
        
        # Environmental parameters
        self.wind_speed = 0.0  # m/s
        self.temperature = 288.15  # K
        self.humidity = 50.0  # %
        self.visibility = 10000.0  # m
        
        # Time of day effects
        self.time_of_day = "day"  # day, night, dawn, dusk
        
    def update_weather(self, weather: WeatherConditions, wind_speed: float = 0.0):
        """Update weather conditions"""
        self.weather = weather
        self.wind_speed = wind_speed
        
        # Update derived parameters
        if weather == WeatherConditions.FOG:
            self.visibility = 500.0
        elif weather == WeatherConditions.HEAVY_RAIN:
            self.visibility = 2000.0
        elif weather == WeatherConditions.STORM:
            self.visibility = 1000.0
        else:
            self.visibility = 10000.0
            
    def calculate_sensor_degradation(self, radar_pos: np.ndarray, target_pos: np.ndarray,
                                   radar_frequency: float, radar_bandwidth: float) -> Dict[str, Any]:
        """
        Calculate all environmental effects on sensor performance
        
        Args:
            radar_pos: 3D radar position
            target_pos: 3D target position
            radar_frequency: Radar frequency in Hz
            radar_bandwidth: Radar bandwidth in Hz
            
        Returns:
            Dictionary with all environmental effects
        """
        range_m = np.linalg.norm(target_pos - radar_pos)
        avg_altitude = (radar_pos[2] + target_pos[2]) / 2.0
        
        # Atmospheric effects
        atm_loss = self.atmospheric_effects.calculate_atmospheric_loss(
            radar_frequency, range_m, avg_altitude, self.weather
        )
        
        # Multipath effects
        multipath = self.atmospheric_effects.calculate_multipath_effects(
            radar_pos[2], target_pos[2], range_m, radar_frequency
        )
        
        # Electronic warfare effects
        ew_effects = self.electronic_warfare.calculate_jamming_effects(
            radar_pos, radar_frequency, radar_bandwidth
        )
        
        # Total degradation
        total_loss = atm_loss + multipath['multipath_loss_db']
        effective_snr_loss = total_loss + ew_effects['jamming_power_dbm'] / 10.0
        
        return {
            'atmospheric_loss_db': atm_loss,
            'multipath_effects': multipath,
            'electronic_warfare': ew_effects,
            'total_path_loss_db': total_loss,
            'effective_snr_loss_db': effective_snr_loss,
            'weather': self.weather.value,
            'visibility_m': self.visibility,
            'wind_speed_ms': self.wind_speed
        }
    
    def get_clutter_effects(self, radar_pos: np.ndarray, beam_direction: np.ndarray,
                          beam_width: float, range_m: float) -> Dict[str, Any]:
        """
        Get clutter effects for radar beam
        
        Args:
            radar_pos: 3D radar position
            beam_direction: 3D beam direction (unit vector)
            beam_width: Beam width in radians
            range_m: Range to analyze
            
        Returns:
            Dictionary with clutter effects
        """
        # Determine dominant clutter type based on geometry and environment
        elevation_angle = math.asin(beam_direction[2])
        
        if elevation_angle < math.radians(1.0):  # Very low angle
            if radar_pos[2] < 100:  # Low altitude radar
                clutter_type = ClutterType.GROUND
            else:
                clutter_type = ClutterType.SEA  # Assume over water
        elif self.weather in [WeatherConditions.HEAVY_RAIN, WeatherConditions.STORM]:
            clutter_type = ClutterType.WEATHER
        else:
            clutter_type = ClutterType.GROUND  # Default
            
        # Calculate clutter power
        clutter_power = self.clutter_model.calculate_clutter_power(
            radar_pos, beam_direction, beam_width, range_m, clutter_type
        )
        
        # Get doppler characteristics
        doppler_spectrum = self.clutter_model.get_clutter_doppler_spectrum(
            clutter_type, self.wind_speed
        )
        
        return {
            'clutter_type': clutter_type.value,
            'clutter_power_dbm': clutter_power,
            'doppler_spectrum': doppler_spectrum,
            'clutter_to_noise_ratio': clutter_power + 174,  # Assume -174 dBm/Hz noise
        }
    
    def get_environment_summary(self) -> Dict[str, Any]:
        """Get summary of current environment conditions"""
        return {
            'weather': self.weather.value,
            'wind_speed_ms': self.wind_speed,
            'temperature_k': self.temperature,
            'humidity_percent': self.humidity,
            'visibility_m': self.visibility,
            'time_of_day': self.time_of_day,
            'active_jammers': len(self.electronic_warfare.active_jammers)
        }