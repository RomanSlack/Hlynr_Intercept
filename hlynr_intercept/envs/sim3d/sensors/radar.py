import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import math
from dataclasses import dataclass

from ..core.missiles import Missile


@dataclass
class RadarContact:
    """Radar detection contact"""
    bearing: float          # azimuth angle in radians
    elevation: float        # elevation angle in radians  
    range: float           # distance in meters
    doppler_velocity: float # radial velocity in m/s
    rcs: float             # radar cross section in m^2
    snr: float             # signal-to-noise ratio in dB
    timestamp: float       # detection time
    target_id: Optional[int] = None  # Tracked target ID


class GroundRadar:
    """Ground-based radar array with realistic detection characteristics"""
    
    def __init__(self, 
                 position: np.ndarray,
                 max_range: float = 100000,  # 100km
                 beam_width: float = np.radians(2),  # 2 degree beamwidth
                 scan_rate: float = 6.0):    # 6 RPM (10 second scan)
        
        self.position = np.array(position, dtype=float)
        self.max_range = max_range
        self.beam_width = beam_width
        self.scan_rate = scan_rate  # revolutions per minute
        
        # Radar parameters
        self.frequency = 3e9  # 3 GHz (S-band)
        self.peak_power = 1e6  # 1 MW
        self.antenna_gain = 40  # 40 dB
        self.noise_figure = 3   # 3 dB
        self.detection_threshold = 12  # 12 dB SNR
        
        # Scanning state
        self.current_azimuth = 0.0  # radians
        self.current_elevation = np.radians(15)  # Fixed 15 degree elevation
        self.scan_direction = 1  # 1 for clockwise, -1 for counter-clockwise
        
        # Contact tracking
        self.contacts: List[RadarContact] = []
        self.contact_history: List[List[RadarContact]] = []
        self.track_ids: Dict[int, List[RadarContact]] = {}
        self.next_track_id = 1
        
        # Performance characteristics
        self.range_accuracy = 50.0    # meters RMS
        self.bearing_accuracy = np.radians(0.5)  # 0.5 degrees RMS
        self.velocity_accuracy = 5.0  # m/s RMS
        
        # Environmental factors
        self.atmospheric_loss = 0.1   # dB/km
        self.weather_loss = 0.0       # Additional loss from weather
        
    def update_scan(self, dt: float):
        """Update radar scanning position"""
        # Convert scan rate from RPM to rad/s
        angular_velocity = (self.scan_rate / 60) * 2 * math.pi
        
        # Update azimuth
        self.current_azimuth += angular_velocity * dt * self.scan_direction
        
        # Wrap azimuth to [0, 2*pi]
        self.current_azimuth = self.current_azimuth % (2 * math.pi)
        
    def calculate_rcs(self, target: Missile) -> float:
        """Calculate radar cross section of target"""
        # Simplified RCS model based on missile geometry
        if target.type == "attacker":
            base_rcs = 0.5  # m^2 - larger missile
        else:
            base_rcs = 0.2  # m^2 - smaller interceptor
            
        # RCS varies with aspect angle (simplified)
        target_to_radar = self.position - target.position
        target_heading = np.array([
            np.cos(target.orientation[2]),
            np.sin(target.orientation[2]),
            0
        ])
        
        # Dot product gives aspect angle effect
        aspect_factor = abs(np.dot(
            target_to_radar[:2] / np.linalg.norm(target_to_radar[:2]),
            target_heading[:2]
        ))
        
        # Head-on or tail-on gives higher RCS
        rcs_multiplier = 0.5 + 1.5 * aspect_factor
        
        return base_rcs * rcs_multiplier
        
    def calculate_snr(self, target: Missile, rcs: float) -> float:
        """Calculate signal-to-noise ratio for target detection"""
        range_to_target = np.linalg.norm(target.position - self.position)
        
        if range_to_target > self.max_range or range_to_target < 100:
            return -50  # Very low SNR
            
        # Radar equation (simplified)
        # SNR = (Pt * G^2 * λ^2 * σ) / ((4π)^3 * R^4 * kT * B * F)
        
        wavelength = 3e8 / self.frequency  # Speed of light / frequency
        
        # Received power (simplified radar equation)
        numerator = (self.peak_power * 
                    (10**(self.antenna_gain/10))**2 * 
                    wavelength**2 * 
                    rcs)
        denominator = ((4 * math.pi)**3 * 
                      range_to_target**4 * 
                      1.38e-23 * 290 *  # kT at room temp
                      1e6 *              # 1 MHz bandwidth
                      10**(self.noise_figure/10))
                      
        snr_linear = numerator / denominator
        snr_db = 10 * math.log10(max(snr_linear, 1e-10))
        
        # Atmospheric losses
        atm_loss = self.atmospheric_loss * (range_to_target / 1000)
        snr_db -= atm_loss
        
        # Weather losses
        snr_db -= self.weather_loss
        
        return snr_db
        
    def is_in_beam(self, target_position: np.ndarray) -> bool:
        """Check if target is within current radar beam"""
        relative_pos = target_position - self.position
        range_to_target = np.linalg.norm(relative_pos)
        
        if range_to_target < 100:  # Too close
            return False
            
        # Calculate bearing to target
        target_azimuth = math.atan2(relative_pos[1], relative_pos[0])
        if target_azimuth < 0:
            target_azimuth += 2 * math.pi
            
        target_elevation = math.asin(relative_pos[2] / range_to_target)
        
        # Check if within beam
        azimuth_diff = abs(target_azimuth - self.current_azimuth)
        if azimuth_diff > math.pi:
            azimuth_diff = 2 * math.pi - azimuth_diff
            
        elevation_diff = abs(target_elevation - self.current_elevation)
        
        return (azimuth_diff <= self.beam_width / 2 and 
                elevation_diff <= self.beam_width / 2)
                
    def detect_targets(self, missiles: List[Missile], current_time: float) -> List[RadarContact]:
        """Detect targets in current radar beam"""
        detections = []
        
        for missile in missiles:
            if not missile.active:
                continue
                
            # Check if target is in beam
            if not self.is_in_beam(missile.position):
                continue
                
            # Calculate detection parameters
            rcs = self.calculate_rcs(missile)
            snr = self.calculate_snr(missile, rcs)
            
            # Detection threshold check
            if snr < self.detection_threshold:
                continue
                
            # Calculate detection parameters with noise
            relative_pos = missile.position - self.position
            true_range = np.linalg.norm(relative_pos)
            true_bearing = math.atan2(relative_pos[1], relative_pos[0])
            true_elevation = math.asin(relative_pos[2] / true_range)
            
            # Radial velocity (Doppler)
            relative_velocity = missile.velocity
            range_rate = np.dot(relative_velocity, relative_pos) / true_range
            
            # Add measurement noise
            range_noise = np.random.normal(0, self.range_accuracy)
            bearing_noise = np.random.normal(0, self.bearing_accuracy)
            elevation_noise = np.random.normal(0, self.bearing_accuracy)  
            velocity_noise = np.random.normal(0, self.velocity_accuracy)
            
            # Create detection
            contact = RadarContact(
                bearing=true_bearing + bearing_noise,
                elevation=true_elevation + elevation_noise,
                range=true_range + range_noise,
                doppler_velocity=range_rate + velocity_noise,
                rcs=rcs,
                snr=snr,
                timestamp=current_time
            )
            
            detections.append(contact)
            
        return detections
        
    def update(self, missiles: List[Missile], current_time: float, dt: float):
        """Update radar - scan and detect targets"""
        # Update scanning
        self.update_scan(dt)
        
        # Detect targets in current beam
        new_contacts = self.detect_targets(missiles, current_time)
        
        # Store contacts
        self.contacts = new_contacts
        self.contact_history.append(new_contacts.copy())
        
        # Keep limited history
        if len(self.contact_history) > 100:
            self.contact_history.pop(0)
            
    def get_contacts_in_sector(self, azimuth_start: float, azimuth_end: float) -> List[RadarContact]:
        """Get all recent contacts in azimuth sector"""
        sector_contacts = []
        
        # Check recent history
        for contact_list in self.contact_history[-10:]:  # Last 10 scans
            for contact in contact_list:
                bearing = contact.bearing
                if bearing < 0:
                    bearing += 2 * math.pi
                    
                if azimuth_start <= bearing <= azimuth_end:
                    sector_contacts.append(contact)
                    
        return sector_contacts


class InterceptorRadar:
    """On-board interceptor radar with limited range and field of view"""
    
    def __init__(self, missile: Missile):
        self.missile = missile
        
        # Radar characteristics
        self.max_range = 20000    # 20km range
        self.field_of_view = np.radians(60)  # 60 degree FOV
        self.update_rate = 10.0   # 10 Hz
        self.last_update = 0.0
        
        # Performance
        self.range_accuracy = 10.0  # meters
        self.angle_accuracy = np.radians(1.0)  # 1 degree
        self.velocity_accuracy = 2.0  # m/s
        
        # Detection parameters
        self.detection_threshold = 8  # dB SNR
        self.peak_power = 50e3       # 50 kW
        self.antenna_gain = 25       # 25 dB
        
        # Gimbal limits (if gimballed)
        self.max_gimbal_angle = np.radians(45)  # +/- 45 degrees
        self.gimbal_pointing = np.array([0, 0])  # [elevation, azimuth] relative to missile body
        
        # Current contacts
        self.contacts: List[RadarContact] = []
        
    def point_gimbal(self, elevation: float, azimuth: float):
        """Point gimballed antenna (limited by gimbal constraints)"""
        self.gimbal_pointing[0] = np.clip(elevation, -self.max_gimbal_angle, self.max_gimbal_angle)
        self.gimbal_pointing[1] = np.clip(azimuth, -self.max_gimbal_angle, self.max_gimbal_angle)
        
    def get_beam_direction(self) -> np.ndarray:
        """Get current beam direction in world coordinates"""
        # Start with missile's forward direction (body frame Z-axis)
        beam_body = np.array([0, 0, 1])
        
        # Apply gimbal pointing
        # Simplified: just add gimbal angles to missile orientation
        total_elevation = self.missile.orientation[1] + self.gimbal_pointing[0]  # Pitch + gimbal elevation
        total_azimuth = self.missile.orientation[2] + self.gimbal_pointing[1]    # Yaw + gimbal azimuth
        
        # Convert to world frame direction
        beam_world = np.array([
            np.cos(total_elevation) * np.cos(total_azimuth),
            np.cos(total_elevation) * np.sin(total_azimuth),
            np.sin(total_elevation)
        ])
        
        return beam_world
        
    def is_target_visible(self, target_position: np.ndarray) -> bool:
        """Check if target is within radar FOV and range"""
        if not self.missile.active:
            return False
            
        relative_pos = target_position - self.missile.position
        range_to_target = np.linalg.norm(relative_pos)
        
        # Range check
        if range_to_target > self.max_range or range_to_target < 50:
            return False
            
        # Field of view check
        beam_direction = self.get_beam_direction()
        target_direction = relative_pos / range_to_target
        
        # Angle between beam and target
        dot_product = np.dot(beam_direction, target_direction)
        angle = math.acos(np.clip(dot_product, -1, 1))
        
        return angle <= self.field_of_view / 2
        
    def calculate_snr_interceptor(self, target: Missile) -> float:
        """Calculate SNR for interceptor radar"""
        range_to_target = np.linalg.norm(target.position - self.missile.position)
        
        # Simple RCS model
        if target.type == "attacker":
            rcs = 1.0  # m^2
        else:
            rcs = 0.3  # m^2
            
        # Simplified radar equation for small radar
        snr_at_1km = 30  # dB at 1km range
        range_loss = 40 * math.log10(range_to_target / 1000)  # dB
        snr = snr_at_1km - range_loss
        
        return snr
        
    def detect_targets(self, missiles: List[Missile], current_time: float) -> List[RadarContact]:
        """Detect targets with interceptor radar"""
        detections = []
        
        for missile in missiles:
            if not missile.active or missile == self.missile:
                continue
                
            # Visibility check
            if not self.is_target_visible(missile.position):
                continue
                
            # SNR check
            snr = self.calculate_snr_interceptor(missile)
            if snr < self.detection_threshold:
                continue
                
            # Calculate parameters
            relative_pos = missile.position - self.missile.position
            true_range = np.linalg.norm(relative_pos)
            
            # Convert to spherical coordinates relative to missile
            beam_direction = self.get_beam_direction()
            
            # Simplified bearing calculation
            target_direction = relative_pos / true_range
            bearing_angle = math.acos(np.clip(np.dot(beam_direction, target_direction), -1, 1))
            
            # Doppler calculation
            relative_velocity = missile.velocity - self.missile.velocity
            doppler_velocity = np.dot(relative_velocity, relative_pos) / true_range
            
            # Add noise
            range_noise = np.random.normal(0, self.range_accuracy)
            angle_noise = np.random.normal(0, self.angle_accuracy)
            velocity_noise = np.random.normal(0, self.velocity_accuracy)
            
            contact = RadarContact(
                bearing=bearing_angle + angle_noise,
                elevation=0,  # Simplified for interceptor radar
                range=true_range + range_noise,
                doppler_velocity=doppler_velocity + velocity_noise,
                rcs=1.0,
                snr=snr,
                timestamp=current_time
            )
            
            detections.append(contact)
            
        return detections
        
    def update(self, missiles: List[Missile], current_time: float, dt: float):
        """Update interceptor radar"""
        # Check update rate
        if current_time - self.last_update < 1.0 / self.update_rate:
            return
            
        self.last_update = current_time
        
        # Detect targets
        self.contacts = self.detect_targets(missiles, current_time)
        
    def get_strongest_contact(self) -> Optional[RadarContact]:
        """Get contact with highest SNR"""
        if not self.contacts:
            return None
            
        return max(self.contacts, key=lambda c: c.snr)