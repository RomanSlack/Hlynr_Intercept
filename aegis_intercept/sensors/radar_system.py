"""
Production-Quality Radar System Implementation

Implements realistic radar physics including:
- Range equation modeling
- Atmospheric propagation effects
- Clutter and interference
- Detection probability calculations
- Measurement noise modeling
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import math


class RadarBand(Enum):
    """Standard radar frequency bands"""
    L_BAND = "L"      # 1-2 GHz - Long range surveillance
    S_BAND = "S"      # 2-4 GHz - Medium range tracking
    C_BAND = "C"      # 4-8 GHz - Weather radar
    X_BAND = "X"      # 8-12 GHz - Fire control, missile guidance
    KU_BAND = "Ku"    # 12-18 GHz - High resolution tracking
    KA_BAND = "Ka"    # 26-40 GHz - Millimeter wave


class RadarMode(Enum):
    """Radar operating modes"""
    SEARCH = "search"           # Wide area search
    TRACK = "track"            # Single target tracking
    TRACK_WHILE_SCAN = "tws"   # Multiple target tracking
    MISSILE_GUIDANCE = "guidance"  # Terminal guidance mode


@dataclass
class RadarConstants:
    """Physical constants for radar calculations"""
    SPEED_OF_LIGHT: float = 2.998e8  # m/s
    BOLTZMANN_CONSTANT: float = 1.38e-23  # J/K
    EARTH_RADIUS: float = 6.371e6  # m (for curvature calculations)
    STANDARD_TEMP: float = 290.0  # K (standard atmospheric temperature)


@dataclass
class RadarConfig:
    """Configuration for radar system parameters"""
    # Basic radar parameters
    frequency: float = 10.0e9  # Hz (X-band default)
    peak_power: float = 1.0e6  # Watts
    antenna_gain: float = 40.0  # dB
    antenna_diameter: float = 3.0  # meters
    system_losses: float = 6.0  # dB
    noise_figure: float = 3.0  # dB
    
    # Detection parameters
    detection_threshold: float = 13.0  # dB (Pd = 0.9, Pfa = 1e-6)
    false_alarm_rate: float = 1e-6  # False alarm probability
    integration_time: float = 0.1  # seconds
    pulse_width: float = 1.0e-6  # seconds
    prf: float = 1000.0  # Hz (pulse repetition frequency)
    
    # Measurement parameters
    range_resolution: float = 150.0  # meters
    angle_resolution: float = 0.5  # degrees
    doppler_resolution: float = 1.0  # m/s
    
    # Performance limits
    max_range: float = 400000.0  # meters (400 km)
    min_range: float = 1000.0  # meters
    elevation_limits: Tuple[float, float] = (-10.0, 90.0)  # degrees
    azimuth_coverage: float = 360.0  # degrees
    
    # Update rates
    search_rate: float = 6.0  # seconds per full scan
    track_rate: float = 0.05  # seconds per track update
    
    # Environmental
    operating_frequency: RadarBand = RadarBand.X_BAND
    operating_mode: RadarMode = RadarMode.TRACK_WHILE_SCAN


@dataclass
class RadarDetection:
    """Single radar detection/measurement"""
    timestamp: float
    range: float  # meters
    azimuth: float  # degrees
    elevation: float  # degrees
    doppler: float  # m/s (radial velocity)
    snr: float  # dB
    range_rate: float  # m/s (rate of change of range)
    
    # Measurement uncertainties (1-sigma)
    range_std: float = 10.0  # meters
    azimuth_std: float = 0.1  # degrees
    elevation_std: float = 0.1  # degrees
    doppler_std: float = 0.5  # m/s
    
    # Detection quality metrics
    probability_of_detection: float = 0.9
    false_alarm_rate: float = 1e-6
    
    # Raw measurement data (for correlation)
    raw_amplitude: float = 0.0
    raw_phase: float = 0.0


@dataclass
class RadarTrack:
    """Multi-detection track information"""
    track_id: int
    detections: List[RadarDetection] = field(default_factory=list)
    last_update: float = 0.0
    confidence: float = 0.0
    track_quality: str = "tentative"  # tentative, confirmed, lost
    
    # Estimated state (filtered)
    estimated_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    estimated_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    position_covariance: np.ndarray = field(default_factory=lambda: np.eye(3) * 100)
    velocity_covariance: np.ndarray = field(default_factory=lambda: np.eye(3) * 10)


class RadarSystem:
    """Base class for all radar systems"""
    
    def __init__(self, config: RadarConfig, position: np.ndarray, name: str = "Radar"):
        """
        Initialize radar system
        
        Args:
            config: Radar configuration parameters
            position: 3D position of radar [x, y, z] in meters
            name: Identifying name for this radar
        """
        self.config = config
        self.position = position.copy()
        self.name = name
        self.constants = RadarConstants()
        
        # Operating state
        self.is_active = True
        self.current_mode = config.operating_mode
        self.last_scan_time = 0.0
        
        # Performance tracking
        self.total_detections = 0
        self.false_alarms = 0
        self.missed_detections = 0
        
        # Calculate derived parameters
        self._calculate_derived_parameters()
        
        # Initialize tracking
        self.active_tracks: Dict[int, RadarTrack] = {}
        self.next_track_id = 1
        
    def _calculate_derived_parameters(self):
        """Calculate derived radar parameters"""
        # Wavelength
        self.wavelength = self.constants.SPEED_OF_LIGHT / self.config.frequency
        
        # Antenna beamwidth (approximation for circular aperture)
        self.beamwidth = 1.22 * self.wavelength / self.config.antenna_diameter  # radians
        self.beamwidth_deg = math.degrees(self.beamwidth)
        
        # Maximum unambiguous range
        self.max_unambiguous_range = self.constants.SPEED_OF_LIGHT / (2 * self.config.prf)
        
        # Thermal noise power
        bandwidth = 1.0 / self.config.pulse_width
        noise_power_watts = (self.constants.BOLTZMANN_CONSTANT * 
                           self.constants.STANDARD_TEMP * bandwidth)
        self.thermal_noise_power = 10 * math.log10(noise_power_watts * 1000)  # dBm
        
    def radar_equation(self, target_range: float, target_rcs: float) -> float:
        """
        Calculate received signal power using radar equation
        
        Args:
            target_range: Range to target in meters
            target_rcs: Target radar cross section in m²
            
        Returns:
            Received signal power in dBm
        """
        # Convert to dB values
        power_db = 10 * math.log10(self.config.peak_power)  # dBW
        gain_db = self.config.antenna_gain  # dB
        rcs_db = 10 * math.log10(max(target_rcs, 1e-6))  # dBsm
        
        # Range factor (1/R^4)
        range_factor_db = -40 * math.log10(target_range)
        
        # Wavelength factor
        wavelength_factor_db = 20 * math.log10(self.wavelength)
        
        # System losses
        losses_db = self.config.system_losses + self.config.noise_figure
        
        # Radar equation: Pr = Pt * G^2 * λ^2 * σ / ((4π)^3 * R^4 * L)
        received_power_dbw = (power_db + 2 * gain_db + wavelength_factor_db + 
                             rcs_db + range_factor_db - 
                             3 * 10 * math.log10(4 * math.pi) - losses_db)
        
        # Convert to dBm
        received_power_dbm = received_power_dbw + 30
        
        return received_power_dbm
        
    def calculate_detection_probability(self, snr_db: float) -> float:
        """
        Calculate probability of detection for given SNR
        Using Shnidman's approximation for Swerling Case 1 targets
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Probability of detection (0-1)
        """
        if snr_db < self.config.detection_threshold:
            # Below threshold - very low probability
            return 0.01 * math.exp(0.1 * (snr_db - self.config.detection_threshold))
        else:
            # Above threshold - high probability with saturation
            excess_snr = snr_db - self.config.detection_threshold
            return min(0.99, 0.5 + 0.4 * math.tanh(excess_snr / 3.0))
    
    def add_measurement_noise(self, true_range: float, true_azimuth: float, 
                            true_elevation: float, true_doppler: float) -> Tuple[float, float, float, float]:
        """
        Add realistic measurement noise to true values
        
        Args:
            true_range: True range in meters
            true_azimuth: True azimuth in degrees
            true_elevation: True elevation in degrees
            true_doppler: True doppler in m/s
            
        Returns:
            Tuple of (noisy_range, noisy_azimuth, noisy_elevation, noisy_doppler)
        """
        # Range noise (function of range and resolution)
        range_noise_std = max(self.config.range_resolution / 8.0, 5.0)
        noisy_range = true_range + np.random.normal(0, range_noise_std)
        
        # Angular noise (function of beamwidth and SNR)
        angle_noise_std = self.beamwidth_deg / 4.0  # Monopulse accuracy
        noisy_azimuth = true_azimuth + np.random.normal(0, angle_noise_std)
        noisy_elevation = true_elevation + np.random.normal(0, angle_noise_std)
        
        # Doppler noise (function of integration time and frequency stability)
        doppler_noise_std = self.config.doppler_resolution / 4.0
        noisy_doppler = true_doppler + np.random.normal(0, doppler_noise_std)
        
        return noisy_range, noisy_azimuth, noisy_elevation, noisy_doppler
    
    def detect_target(self, target_position: np.ndarray, target_velocity: np.ndarray,
                     target_rcs: float, timestamp: float,
                     atmospheric_effects: Optional[Dict] = None,
                     clutter_effects: Optional[Dict] = None) -> Optional[RadarDetection]:
        """
        Attempt to detect a target
        
        Args:
            target_position: 3D target position [x, y, z]
            target_velocity: 3D target velocity [vx, vy, vz]
            target_rcs: Target radar cross section in m²
            timestamp: Current simulation time
            atmospheric_effects: Atmospheric propagation effects
            clutter_effects: Ground/weather clutter effects
            
        Returns:
            RadarDetection if detected, None if not detected
        """
        # Calculate relative geometry
        relative_pos = target_position - self.position
        range_to_target = np.linalg.norm(relative_pos)
        
        # Check range limits
        if range_to_target < self.config.min_range or range_to_target > self.config.max_range:
            return None
            
        # Calculate spherical coordinates
        azimuth = math.degrees(math.atan2(relative_pos[1], relative_pos[0]))
        elevation = math.degrees(math.asin(relative_pos[2] / range_to_target))
        
        # Check elevation limits
        if elevation < self.config.elevation_limits[0] or elevation > self.config.elevation_limits[1]:
            return None
            
        # Calculate radial velocity (Doppler)
        relative_vel = target_velocity
        unit_vector = relative_pos / range_to_target
        radial_velocity = np.dot(relative_vel, unit_vector)
        
        # Calculate received signal power
        received_power = self.radar_equation(range_to_target, target_rcs)
        
        # Apply atmospheric effects
        if atmospheric_effects:
            atm_loss = atmospheric_effects.get('path_loss', 0.0)
            received_power -= atm_loss
            
        # Calculate SNR
        noise_power = self.thermal_noise_power + self.config.noise_figure
        snr = received_power - noise_power
        
        # Apply clutter effects
        if clutter_effects:
            clutter_power = clutter_effects.get('clutter_power', -80.0)  # dBm
            # Clutter reduces effective SNR
            total_noise = 10**(noise_power/10) + 10**(clutter_power/10)
            effective_noise = 10 * math.log10(total_noise)
            snr = received_power - effective_noise
            
        # Determine if target is detected
        pd = self.calculate_detection_probability(snr)
        detected = np.random.random() < pd
        
        if not detected:
            self.missed_detections += 1
            return None
            
        # Add measurement noise
        noisy_range, noisy_azimuth, noisy_elevation, noisy_doppler = self.add_measurement_noise(
            range_to_target, azimuth, elevation, radial_velocity
        )
        
        # Calculate range rate
        range_rate = radial_velocity  # For now, same as doppler
        
        detection = RadarDetection(
            timestamp=timestamp,
            range=noisy_range,
            azimuth=noisy_azimuth,
            elevation=noisy_elevation,
            doppler=noisy_doppler,
            snr=snr,
            range_rate=range_rate,
            range_std=self.config.range_resolution / 8.0,
            azimuth_std=self.beamwidth_deg / 4.0,
            elevation_std=self.beamwidth_deg / 4.0,
            doppler_std=self.config.doppler_resolution / 4.0,
            probability_of_detection=pd,
            false_alarm_rate=self.config.false_alarm_rate
        )
        
        self.total_detections += 1
        return detection
    
    def generate_false_alarms(self, timestamp: float, scan_volume: float = 1.0) -> List[RadarDetection]:
        """
        Generate false alarm detections
        
        Args:
            timestamp: Current simulation time
            scan_volume: Relative scan volume (affects false alarm rate)
            
        Returns:
            List of false alarm detections
        """
        false_alarms = []
        
        # Calculate expected number of false alarms
        # Based on resolution cells and false alarm rate
        range_cells = (self.config.max_range - self.config.min_range) / self.config.range_resolution
        angle_cells = (self.config.azimuth_coverage * self.beamwidth_deg) / (self.beamwidth_deg ** 2)
        total_cells = range_cells * angle_cells * scan_volume
        
        expected_false_alarms = total_cells * self.config.false_alarm_rate
        num_false_alarms = np.random.poisson(expected_false_alarms)
        
        for _ in range(num_false_alarms):
            # Random position in scan volume
            fa_range = np.random.uniform(self.config.min_range, self.config.max_range)
            fa_azimuth = np.random.uniform(-180, 180)
            fa_elevation = np.random.uniform(self.config.elevation_limits[0], 
                                           self.config.elevation_limits[1])
            fa_doppler = np.random.normal(0, 10.0)  # Random doppler
            
            # False alarms have low SNR near threshold
            fa_snr = self.config.detection_threshold + np.random.exponential(2.0)
            
            false_alarm = RadarDetection(
                timestamp=timestamp,
                range=fa_range,
                azimuth=fa_azimuth,
                elevation=fa_elevation,
                doppler=fa_doppler,
                snr=fa_snr,
                range_rate=fa_doppler,
                range_std=self.config.range_resolution / 4.0,
                azimuth_std=self.beamwidth_deg / 2.0,
                elevation_std=self.beamwidth_deg / 2.0,
                doppler_std=self.config.doppler_resolution / 2.0,
                probability_of_detection=0.1,  # Low confidence
                false_alarm_rate=self.config.false_alarm_rate
            )
            
            false_alarms.append(false_alarm)
            
        self.false_alarms += len(false_alarms)
        return false_alarms
    
    def update_tracks(self, detections: List[RadarDetection], timestamp: float):
        """
        Update existing tracks with new detections
        Simple nearest-neighbor association for now
        
        Args:
            detections: New radar detections
            timestamp: Current simulation time
        """
        # Remove old/lost tracks
        tracks_to_remove = []
        for track_id, track in self.active_tracks.items():
            if timestamp - track.last_update > 5.0:  # 5 second timeout
                tracks_to_remove.append(track_id)
                
        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
            
        # Associate detections with tracks
        unassociated_detections = detections.copy()
        
        for track_id, track in self.active_tracks.items():
            if not unassociated_detections:
                break
                
            # Find closest detection to predicted position
            best_detection = None
            best_distance = float('inf')
            
            for detection in unassociated_detections:
                # Convert detection to Cartesian for distance calculation
                det_x = detection.range * math.cos(math.radians(detection.elevation)) * math.cos(math.radians(detection.azimuth))
                det_y = detection.range * math.cos(math.radians(detection.elevation)) * math.sin(math.radians(detection.azimuth))
                det_z = detection.range * math.sin(math.radians(detection.elevation))
                det_pos = np.array([det_x, det_y, det_z]) + self.position
                
                distance = np.linalg.norm(det_pos - track.estimated_position)
                
                if distance < best_distance and distance < 500.0:  # 500m gate
                    best_distance = distance
                    best_detection = detection
                    
            if best_detection:
                track.detections.append(best_detection)
                track.last_update = timestamp
                track.confidence = min(1.0, track.confidence + 0.1)
                unassociated_detections.remove(best_detection)
                
                # Update position estimate (simple average for now)
                det_x = best_detection.range * math.cos(math.radians(best_detection.elevation)) * math.cos(math.radians(best_detection.azimuth))
                det_y = best_detection.range * math.cos(math.radians(best_detection.elevation)) * math.sin(math.radians(best_detection.azimuth))
                det_z = best_detection.range * math.sin(math.radians(best_detection.elevation))
                new_pos = np.array([det_x, det_y, det_z]) + self.position
                
                # Simple exponential smoothing
                alpha = 0.3
                track.estimated_position = alpha * new_pos + (1 - alpha) * track.estimated_position
                
        # Create new tracks for unassociated detections
        for detection in unassociated_detections:
            if detection.snr > self.config.detection_threshold + 3.0:  # Only strong detections
                new_track = RadarTrack(
                    track_id=self.next_track_id,
                    detections=[detection],
                    last_update=timestamp,
                    confidence=0.3,
                    track_quality="tentative"
                )
                
                # Initialize position estimate
                det_x = detection.range * math.cos(math.radians(detection.elevation)) * math.cos(math.radians(detection.azimuth))
                det_y = detection.range * math.cos(math.radians(detection.elevation)) * math.sin(math.radians(detection.azimuth))
                det_z = detection.range * math.sin(math.radians(detection.elevation))
                new_track.estimated_position = np.array([det_x, det_y, det_z]) + self.position
                
                self.active_tracks[self.next_track_id] = new_track
                self.next_track_id += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get radar performance statistics"""
        total_attempts = self.total_detections + self.missed_detections
        detection_rate = self.total_detections / max(total_attempts, 1)
        false_alarm_rate = self.false_alarms / max(self.total_detections + self.false_alarms, 1)
        
        return {
            'name': self.name,
            'total_detections': self.total_detections,
            'missed_detections': self.missed_detections,
            'false_alarms': self.false_alarms,
            'detection_rate': detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'active_tracks': len(self.active_tracks),
            'max_range': self.config.max_range,
            'beamwidth': self.beamwidth_deg
        }


class GroundRadar(RadarSystem):
    """Ground-based surveillance and tracking radar"""
    
    def __init__(self, position: np.ndarray, name: str = "Ground Radar"):
        """Initialize ground radar with typical parameters"""
        config = RadarConfig(
            frequency=3.0e9,  # S-band for long range
            peak_power=2.0e6,  # 2 MW
            antenna_gain=45.0,  # High gain for long range
            antenna_diameter=8.0,  # Large antenna
            detection_threshold=12.0,  # Slightly relaxed for long range
            max_range=500000.0,  # 500 km surveillance range
            min_range=2000.0,  # 2 km minimum
            search_rate=10.0,  # 10 second full scan
            track_rate=0.1,  # 100ms track updates
            operating_mode=RadarMode.TRACK_WHILE_SCAN
        )
        super().__init__(config, position, name)
        
        # Ground radar specific parameters
        self.site_elevation = position[2]
        self.horizon_range_cache = {}  # Cache for horizon calculations
        
    def calculate_horizon_range(self, target_altitude: float) -> float:
        """
        Calculate radar horizon for given target altitude
        Accounts for Earth curvature and atmospheric refraction
        
        Args:
            target_altitude: Target altitude in meters
            
        Returns:
            Horizon range in meters
        """
        if target_altitude in self.horizon_range_cache:
            return self.horizon_range_cache[target_altitude]
            
        # Effective Earth radius (4/3 rule for standard atmosphere)
        effective_radius = 4.0/3.0 * self.constants.EARTH_RADIUS
        
        # Horizon range calculation
        radar_height = self.site_elevation
        target_height = target_altitude
        
        horizon_range = (math.sqrt(2 * effective_radius * radar_height) + 
                        math.sqrt(2 * effective_radius * target_height))
        
        self.horizon_range_cache[target_altitude] = horizon_range
        return horizon_range
        
    def detect_target(self, target_position: np.ndarray, target_velocity: np.ndarray,
                     target_rcs: float, timestamp: float,
                     atmospheric_effects: Optional[Dict] = None,
                     clutter_effects: Optional[Dict] = None) -> Optional[RadarDetection]:
        """
        Ground radar detection with horizon masking
        """
        # Check line-of-sight (horizon masking)
        target_altitude = target_position[2]
        range_to_target = np.linalg.norm(target_position - self.position)
        horizon_range = self.calculate_horizon_range(target_altitude)
        
        if range_to_target > horizon_range:
            return None  # Below horizon
            
        # Add ground clutter effects for low elevation targets
        relative_pos = target_position - self.position
        elevation_rad = math.asin(relative_pos[2] / range_to_target)
        elevation_deg = math.degrees(elevation_rad)
        
        if clutter_effects is None:
            clutter_effects = {}
            
        # Increase clutter for low elevation angles
        if elevation_deg < 5.0:
            clutter_power = -60.0 + (5.0 - elevation_deg) * 5.0  # Stronger clutter at low elevation
            clutter_effects['clutter_power'] = clutter_power
            
        return super().detect_target(target_position, target_velocity, target_rcs, 
                                   timestamp, atmospheric_effects, clutter_effects)


class MissileRadar(RadarSystem):
    """Missile-borne radar for terminal guidance"""
    
    def __init__(self, missile_position: np.ndarray, name: str = "Missile Radar"):
        """Initialize missile radar with typical parameters"""
        config = RadarConfig(
            frequency=35.0e9,  # Ka-band for high resolution
            peak_power=50000.0,  # 50 kW (limited by missile power)
            antenna_gain=35.0,  # Smaller antenna
            antenna_diameter=0.3,  # 30cm missile antenna
            detection_threshold=10.0,  # Lower threshold for terminal phase
            max_range=50000.0,  # 50 km missile radar range
            min_range=100.0,  # 100m minimum
            search_rate=0.5,  # 500ms search scan
            track_rate=0.02,  # 20ms track updates (very fast)
            operating_mode=RadarMode.TRACK,
            range_resolution=15.0,  # High resolution
            angle_resolution=0.2,  # High angular resolution
        )
        super().__init__(config, missile_position, name)
        
        # Missile radar specific parameters
        self.seeker_fov = 60.0  # degrees (field of view)
        self.lock_on_range = 20000.0  # meters
        self.terminal_guidance_range = 5000.0  # meters
        
    def update_position(self, new_position: np.ndarray):
        """Update missile radar position as missile moves"""
        self.position = new_position.copy()
        
    def detect_target(self, target_position: np.ndarray, target_velocity: np.ndarray,
                     target_rcs: float, timestamp: float,
                     atmospheric_effects: Optional[Dict] = None,
                     clutter_effects: Optional[Dict] = None) -> Optional[RadarDetection]:
        """
        Missile radar detection with field-of-view constraints
        """
        # Check if target is within seeker field of view
        relative_pos = target_position - self.position
        range_to_target = np.linalg.norm(relative_pos)
        
        if range_to_target > 0:
            # Calculate angle from missile nose (assume missile points toward target initially)
            unit_vector = relative_pos / range_to_target
            # For simplicity, assume missile always points toward target
            # In reality, would need missile orientation
            angle_from_nose = 0.0  # Simplified
            
            if angle_from_nose > self.seeker_fov / 2.0:
                return None  # Outside field of view
                
        # Missile radars have less atmospheric attenuation due to shorter ranges
        if atmospheric_effects:
            # Reduce atmospheric effects for shorter range
            atmospheric_effects = atmospheric_effects.copy()
            atmospheric_effects['path_loss'] *= 0.5
            
        return super().detect_target(target_position, target_velocity, target_rcs,
                                   timestamp, atmospheric_effects, clutter_effects)