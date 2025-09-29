"""
Advanced physics models for realistic missile dynamics.

This module implements enhanced physics models including:
- International Standard Atmosphere (ISA) model
- Mach-dependent drag effects
- Enhanced wind modeling
- Domain randomization for robust training
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class AtmosphericConstants:
    """Standard atmosphere constants for ISA model."""
    # Sea level conditions (ISA standard)
    SEA_LEVEL_PRESSURE: float = 101325.0  # Pa
    SEA_LEVEL_TEMPERATURE: float = 288.15  # K
    SEA_LEVEL_DENSITY: float = 1.225  # kg/m³

    # Gas constants
    SPECIFIC_GAS_CONSTANT: float = 287.05  # J/(kg⋅K) for dry air
    GRAVITY: float = 9.80665  # m/s² (standard gravity)
    GAMMA: float = 1.4  # Heat capacity ratio for air

    # Temperature lapse rates
    TROPOSPHERE_LAPSE_RATE: float = 0.0065  # K/m
    TROPOSPHERE_TOP: float = 11000.0  # m
    STRATOSPHERE_TOP: float = 20000.0  # m

    # Stratosphere conditions (isothermal)
    STRATOSPHERE_TEMP: float = 216.65  # K
    STRATOSPHERE_PRESSURE: float = 22632.0  # Pa


class AtmosphericModel:
    """
    International Standard Atmosphere (ISA) model implementation.

    Provides altitude-dependent atmospheric properties including:
    - Air density (for drag calculations)
    - Temperature (for speed of sound)
    - Pressure (for atmospheric effects)
    - Speed of sound (for Mach number calculations)

    Valid for altitudes from sea level to ~50km.
    """

    def __init__(self, constants: Optional[AtmosphericConstants] = None):
        self.constants = constants or AtmosphericConstants()

    def get_temperature(self, altitude: float) -> float:
        """
        Get atmospheric temperature at given altitude.

        Args:
            altitude: Altitude in meters above sea level

        Returns:
            Temperature in Kelvin
        """
        altitude = max(0.0, altitude)  # No negative altitudes

        if altitude <= self.constants.TROPOSPHERE_TOP:
            # Troposphere: Linear temperature decrease
            return (self.constants.SEA_LEVEL_TEMPERATURE -
                   self.constants.TROPOSPHERE_LAPSE_RATE * altitude)
        elif altitude <= self.constants.STRATOSPHERE_TOP:
            # Lower stratosphere: Isothermal
            return self.constants.STRATOSPHERE_TEMP
        else:
            # Above 20km: Exponential decay approximation
            excess_altitude = altitude - self.constants.STRATOSPHERE_TOP
            return self.constants.STRATOSPHERE_TEMP * np.exp(-excess_altitude / 10000.0)

    def get_pressure(self, altitude: float) -> float:
        """
        Get atmospheric pressure at given altitude.

        Args:
            altitude: Altitude in meters above sea level

        Returns:
            Pressure in Pascals
        """
        altitude = max(0.0, altitude)

        if altitude <= self.constants.TROPOSPHERE_TOP:
            # Troposphere: Barometric formula
            temperature = self.get_temperature(altitude)
            temp_ratio = temperature / self.constants.SEA_LEVEL_TEMPERATURE

            # Correct barometric formula exponent
            # exponent = g * M / (R * L) where M = molar mass = 0.0289644 kg/mol, R = 8.314 J/(mol⋅K)
            exponent = self.constants.GRAVITY / (self.constants.SPECIFIC_GAS_CONSTANT * self.constants.TROPOSPHERE_LAPSE_RATE)
            pressure = self.constants.SEA_LEVEL_PRESSURE * (temp_ratio ** exponent)

        elif altitude <= self.constants.STRATOSPHERE_TOP:
            # Lower stratosphere: Exponential decay
            excess_altitude = altitude - self.constants.TROPOSPHERE_TOP
            pressure = self.constants.STRATOSPHERE_PRESSURE * np.exp(
                -self.constants.GRAVITY * excess_altitude /
                (self.constants.SPECIFIC_GAS_CONSTANT * self.constants.STRATOSPHERE_TEMP)
            )
        else:
            # Above 20km: Rapid exponential decay
            excess_altitude = altitude - self.constants.STRATOSPHERE_TOP
            base_pressure = self.get_pressure(self.constants.STRATOSPHERE_TOP)
            pressure = base_pressure * np.exp(-excess_altitude / 6000.0)

        return pressure

    def get_density(self, altitude: float) -> float:
        """
        Get atmospheric density at given altitude.

        Args:
            altitude: Altitude in meters above sea level

        Returns:
            Density in kg/m³
        """
        temperature = self.get_temperature(altitude)
        pressure = self.get_pressure(altitude)

        # Ideal gas law: ρ = P / (R * T)
        density = pressure / (self.constants.SPECIFIC_GAS_CONSTANT * temperature)

        return density

    def get_speed_of_sound(self, altitude: float) -> float:
        """
        Get speed of sound at given altitude.

        Args:
            altitude: Altitude in meters above sea level

        Returns:
            Speed of sound in m/s
        """
        temperature = self.get_temperature(altitude)

        # Speed of sound: a = √(γ * R * T)
        speed_of_sound = np.sqrt(
            self.constants.GAMMA * self.constants.SPECIFIC_GAS_CONSTANT * temperature
        )

        return speed_of_sound

    def get_atmospheric_properties(self, altitude: float) -> Dict[str, float]:
        """
        Get all atmospheric properties at once for efficiency.

        Args:
            altitude: Altitude in meters above sea level

        Returns:
            Dictionary with temperature, pressure, density, and speed_of_sound
        """
        temperature = self.get_temperature(altitude)
        pressure = self.get_pressure(altitude)
        density = pressure / (self.constants.SPECIFIC_GAS_CONSTANT * temperature)
        speed_of_sound = np.sqrt(
            self.constants.GAMMA * self.constants.SPECIFIC_GAS_CONSTANT * temperature
        )

        return {
            'temperature': temperature,
            'pressure': pressure,
            'density': density,
            'speed_of_sound': speed_of_sound,
            'altitude': altitude
        }


class MachDragModel:
    """
    Mach number-dependent drag coefficient model.

    Models the dramatic increase in drag coefficient in the transonic regime
    (Mach 0.8-1.2) and supersonic flight conditions.
    """

    def __init__(self, base_drag_coefficient: float = 0.3):
        self.base_cd = base_drag_coefficient

        # Drag curve parameters
        self.subsonic_mach = 0.8
        self.supersonic_mach = 1.2
        self.transonic_peak_multiplier = 3.0  # Peak drag multiplier in transonic
        self.supersonic_multiplier = 2.5  # Supersonic drag multiplier

    def get_drag_coefficient(self, mach_number: float) -> float:
        """
        Get drag coefficient based on Mach number.

        Args:
            mach_number: Mach number (velocity / speed_of_sound)

        Returns:
            Drag coefficient (dimensionless)
        """
        if mach_number < self.subsonic_mach:
            # Subsonic: Constant drag
            return self.base_cd

        elif mach_number < self.supersonic_mach:
            # Transonic: Linear rise to peak
            mach_fraction = (mach_number - self.subsonic_mach) / (
                self.supersonic_mach - self.subsonic_mach)
            multiplier = 1.0 + (self.transonic_peak_multiplier - 1.0) * mach_fraction
            return self.base_cd * multiplier

        else:
            # Supersonic: Constant high drag
            return self.base_cd * self.supersonic_multiplier

    def get_mach_number(self, velocity: np.ndarray, speed_of_sound: float) -> float:
        """
        Calculate Mach number from velocity vector and speed of sound.

        Args:
            velocity: 3D velocity vector in m/s
            speed_of_sound: Speed of sound at current altitude in m/s

        Returns:
            Mach number (dimensionless)
        """
        velocity_magnitude = np.linalg.norm(velocity)
        return velocity_magnitude / speed_of_sound

    def get_drag_force(self, velocity: np.ndarray, air_density: float,
                      speed_of_sound: float, reference_area: float = 1.0) -> np.ndarray:
        """
        Calculate drag force vector including Mach effects.

        Args:
            velocity: 3D velocity vector in m/s
            air_density: Air density in kg/m³
            speed_of_sound: Speed of sound in m/s
            reference_area: Reference area in m² (default 1.0 for coefficient-based)

        Returns:
            Drag force vector in Newtons
        """
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude < 1e-6:
            return np.zeros(3)

        mach_number = self.get_mach_number(velocity, speed_of_sound)
        drag_coefficient = self.get_drag_coefficient(mach_number)

        # Drag force magnitude: F_d = 0.5 * ρ * V² * C_d * A
        drag_magnitude = (0.5 * air_density * velocity_magnitude**2 *
                         drag_coefficient * reference_area)

        # Drag force opposes velocity direction
        drag_direction = -velocity / velocity_magnitude

        return drag_direction * drag_magnitude


class EnhancedWindModel:
    """
    Advanced wind model with altitude-dependent profiles and turbulence.

    Models realistic atmospheric wind effects including:
    - Boundary layer wind shear
    - Altitude-dependent wind profiles
    - Turbulence intensity
    - Wind gusts
    """

    def __init__(self, base_wind: np.ndarray,
                 boundary_layer_height: float = 1000.0,
                 surface_roughness: float = 0.1):
        """
        Initialize enhanced wind model.

        Args:
            base_wind: Base wind vector at reference height [x, y, z] m/s
            boundary_layer_height: Height of atmospheric boundary layer in meters
            surface_roughness: Surface roughness length in meters
        """
        self.base_wind = np.array(base_wind, dtype=np.float32)
        self.boundary_layer_height = boundary_layer_height
        self.surface_roughness = surface_roughness

        # Turbulence parameters
        self.turbulence_intensity = 0.1  # 10% baseline turbulence
        self.gust_scale = 5.0  # Maximum gust velocity in m/s

        # Random number generator for consistent turbulence
        self.rng = np.random.default_rng()

    def seed(self, seed: int):
        """Set random seed for reproducible wind patterns."""
        self.rng = np.random.default_rng(seed)

    def get_wind_profile_factor(self, altitude: float) -> float:
        """
        Get wind speed multiplier based on altitude profile.

        Uses power law profile in boundary layer, constant above.

        Args:
            altitude: Altitude above ground in meters

        Returns:
            Wind speed multiplier (dimensionless)
        """
        if altitude <= 10.0:  # Below 10m, use reference height
            altitude = 10.0

        if altitude <= self.boundary_layer_height:
            # Power law profile in boundary layer
            # U(z) = U_ref * (z/z_ref)^α where α ≈ 1/7 for neutral conditions
            reference_height = 10.0  # Standard 10m reference
            alpha = 0.143  # 1/7 power law exponent
            return (altitude / reference_height) ** alpha
        else:
            # Above boundary layer: constant wind
            return (self.boundary_layer_height / 10.0) ** 0.143

    def get_turbulence_intensity(self, altitude: float) -> float:
        """
        Get turbulence intensity based on altitude.

        Higher turbulence near surface, decreasing with altitude.

        Args:
            altitude: Altitude above ground in meters

        Returns:
            Turbulence intensity (0-1)
        """
        if altitude <= 10.0:
            return self.turbulence_intensity * 2.0  # High surface turbulence
        elif altitude <= self.boundary_layer_height:
            # Decrease with altitude
            height_factor = 1.0 - (altitude / self.boundary_layer_height) * 0.7
            return self.turbulence_intensity * height_factor
        else:
            # Low turbulence above boundary layer
            return self.turbulence_intensity * 0.3

    def get_wind_vector(self, position: np.ndarray, dt: float) -> np.ndarray:
        """
        Get wind vector at given position including all effects.

        Args:
            position: Position vector [x, y, z] in meters
            dt: Time step for turbulence evolution

        Returns:
            Wind vector [x, y, z] in m/s
        """
        altitude = max(0.0, position[2])

        # Base wind with altitude profile
        profile_factor = self.get_wind_profile_factor(altitude)
        wind = self.base_wind * profile_factor

        # Add turbulence
        turbulence_intensity = self.get_turbulence_intensity(altitude)
        if turbulence_intensity > 0:
            # Correlated turbulence with time evolution
            turbulence = self.rng.normal(0, turbulence_intensity * np.linalg.norm(wind), 3)

            # Low-pass filter for realistic turbulence time scales
            turbulence_scale = 0.1  # Turbulence time scale
            turbulence *= (1.0 - np.exp(-dt / turbulence_scale))

            wind += turbulence

        # Add occasional wind gusts (low probability)
        if self.rng.random() < 0.001:  # 0.1% chance per timestep
            gust_direction = self.rng.normal(0, 1, 3)
            gust_direction /= np.linalg.norm(gust_direction) + 1e-6
            gust_magnitude = self.rng.exponential(self.gust_scale)
            wind += gust_direction * gust_magnitude

        return wind.astype(np.float32)