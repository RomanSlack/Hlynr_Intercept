"""
Atmospheric modeling for AegisIntercept Phase 3.

This module provides realistic atmospheric models based on the International
Standard Atmosphere (ISA) for accurate drag and density calculations.
"""

import numpy as np
from typing import Tuple


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