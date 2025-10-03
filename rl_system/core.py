"""
Core components: 26D observations with ground radar, coordinate transforms, and safety constraints.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math
from collections import deque


@dataclass
class SafetyLimits:
    """Safety constraints for interceptor actions."""
    max_acceleration: float = 50.0  # m/s^2
    max_angular_rate: float = 5.0   # rad/s
    max_gimbal_angle: float = 0.785  # 45 degrees
    min_thrust: float = 0.0
    max_thrust: float = 1.0
    fuel_depletion_rate: float = 0.1  # kg/s at max thrust


class SensorDelayBuffer:
    """
    Circular buffer for simulating sensor delays in radar measurements.

    Real tactical radar systems have processing and transmission delays of 20-40ms.
    This class implements a configurable delay buffer that holds measurements
    and returns them after the specified delay time.
    """

    def __init__(self, delay_samples: int = 3, buffer_size: Optional[int] = None):
        """
        Initialize sensor delay buffer.

        Args:
            delay_samples: Number of simulation timesteps to delay (default 3 = 30ms at 100Hz)
            buffer_size: Maximum buffer size (default = delay_samples + 1)
        """
        self.delay_samples = max(1, delay_samples)
        self.buffer_size = buffer_size or (self.delay_samples + 1)

        # Use deque for efficient FIFO operations
        self.measurement_buffer = deque(maxlen=self.buffer_size)
        self.detection_buffer = deque(maxlen=self.buffer_size)

        # Track initialization state
        self.initialized = False
        self.samples_received = 0

    def add_measurement(self, measurement: Dict[str, Any], detected: bool = True) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Add new measurement to buffer and return delayed measurement.

        Args:
            measurement: Current measurement data
            detected: Whether target was detected this timestep

        Returns:
            Tuple of (delayed_measurement, delayed_detection_status)
            Returns (None, False) during initial buffer fill period
        """
        # Add current measurement to buffer
        self.measurement_buffer.append(measurement.copy() if measurement else None)
        self.detection_buffer.append(detected)
        self.samples_received += 1

        # Check if we have enough samples for delayed output
        if self.samples_received < self.delay_samples:
            # Still in buffer fill period - return no detection
            return None, False

        # Buffer is ready - return delayed measurement
        self.initialized = True

        # Get delayed measurement (oldest in buffer when buffer is full)
        if len(self.measurement_buffer) > self.delay_samples:
            # Get measurement from delay_samples ago
            delayed_measurement = self.measurement_buffer[0]  # Oldest measurement
            delayed_detected = self.detection_buffer[0]       # Oldest detection

            return delayed_measurement, delayed_detected
        else:
            return None, False

    def reset(self):
        """Reset buffer state for new episode."""
        self.measurement_buffer.clear()
        self.detection_buffer.clear()
        self.initialized = False
        self.samples_received = 0

    def is_initialized(self) -> bool:
        """Check if buffer has been filled and is providing valid delayed measurements."""
        return self.initialized

    def get_delay_time_ms(self, simulation_dt: float) -> float:
        """Get delay time in milliseconds."""
        return self.delay_samples * simulation_dt * 1000.0


@dataclass
class GroundRadarStation:
    """Ground-based early warning and tracking radar station."""
    position: np.ndarray  # Fixed location near defended target

    # Radar specifications (based on AN/TPY-2 THAAD radar)
    max_range: float = 20000.0  # 20km range (much longer than onboard)
    min_elevation_angle: float = 0.087  # 5 degrees in radians
    max_elevation_angle: float = 1.484  # 85 degrees in radians

    # Measurement accuracy (better than onboard radar)
    range_accuracy: float = 10.0  # meters RMS
    velocity_accuracy: float = 2.0  # m/s RMS

    # Environmental factors
    base_quality: float = 0.95  # High-quality ground installation
    weather_sensitivity: float = 0.2  # Degradation factor


class Radar26DObservation:
    """26-dimensional radar observation space with ground radar support."""

    def __init__(self, max_range: float = 10000.0, max_velocity: float = 1000.0,
                 radar_range: float = 5000.0, min_detection_range: float = 50.0,
                 sensor_delay_ms: float = 30.0, simulation_dt: float = 0.01,
                 ground_radar_config: Optional[Dict[str, Any]] = None,
                 radar_beam_width: float = 60.0):
        self.max_range = max_range
        self.max_velocity = max_velocity
        self.radar_range = radar_range  # Onboard radar range
        self.min_detection_range = min_detection_range
        self.radar_beam_width = radar_beam_width  # Beam width in degrees (curriculum-adjusted)
        self.rng = np.random.default_rng()

        # Curriculum parameters (set externally by environment)
        self.onboard_detection_reliability = 1.0  # 0-1, probability multiplier for detection
        self.ground_detection_reliability = 1.0   # 0-1, probability multiplier for ground radar
        self.measurement_noise_level = 0.05       # Base noise level for measurements

        # Sensor delay buffer for onboard radar
        self.simulation_dt = simulation_dt
        delay_samples = int(sensor_delay_ms / (simulation_dt * 1000.0)) if sensor_delay_ms > 0 else 0
        self.sensor_delay_buffer = SensorDelayBuffer(delay_samples) if delay_samples > 0 else None

        # Ground radar configuration
        self.ground_radar_enabled = ground_radar_config.get('enabled', True) if ground_radar_config else True
        if self.ground_radar_enabled and ground_radar_config:
            ground_pos = np.array(ground_radar_config.get('position', [0, 0, 100]), dtype=np.float32)
            self.ground_radar = GroundRadarStation(
                position=ground_pos,
                max_range=ground_radar_config.get('max_range', 20000.0),
                min_elevation_angle=np.radians(ground_radar_config.get('min_elevation_angle', 5.0)),
                max_elevation_angle=np.radians(ground_radar_config.get('max_elevation_angle', 85.0)),
                range_accuracy=ground_radar_config.get('range_accuracy', 10.0),
                velocity_accuracy=ground_radar_config.get('velocity_accuracy', 2.0),
                base_quality=ground_radar_config.get('base_quality', 0.95),
                weather_sensitivity=ground_radar_config.get('weather_sensitivity', 0.2)
            )

            # Data link parameters
            self.max_datalink_range = ground_radar_config.get('max_datalink_range', 50000.0)
            self.datalink_packet_loss = ground_radar_config.get('datalink_packet_loss', 0.05)

            # Ground radar sensor delay (usually higher due to processing + transmission)
            ground_delay_ms = ground_radar_config.get('ground_sensor_delay_ms', 50.0)
            ground_delay_samples = int(ground_delay_ms / (simulation_dt * 1000.0)) if ground_delay_ms > 0 else 0
            self.ground_sensor_delay_buffer = SensorDelayBuffer(ground_delay_samples) if ground_delay_samples > 0 else None
        else:
            self.ground_radar = None
            self.ground_sensor_delay_buffer = None

        # Store last detection info for reward calculation (not passed to policy)
        self._last_detection_info = None
        self._last_ground_detection_info = None

    def seed(self, seed: int):
        """Set random seed for reproducible noise."""
        self.rng = np.random.default_rng(seed)

    def reset_sensor_delays(self):
        """Reset sensor delay buffers for new episode."""
        if self.sensor_delay_buffer:
            self.sensor_delay_buffer.reset()
        if self.ground_sensor_delay_buffer:
            self.ground_sensor_delay_buffer.reset()

    def get_last_detection_info(self) -> Optional[Dict[str, Any]]:
        """
        Get the last detection info for reward calculation.

        This is used internally by the environment for reward shaping but is NOT
        passed to the policy network. The policy only sees the 26D observation vector.
        """
        return self._last_detection_info

    def _compute_ground_radar_detection(self, interceptor: Dict[str, Any], missile: Dict[str, Any],
                                        weather_factor: float = 1.0) -> Dict[str, Any]:
        """
        Simulate ground-based radar detection of missile.

        Key differences from onboard radar:
        - No beam width constraint (scanning radar)
        - Much longer range
        - Fixed position (no platform motion)
        - Can be blocked by terrain/horizon

        Returns radar-only measurements, NO omniscient data.
        """
        if not self.ground_radar_enabled or self.ground_radar is None:
            return {'detected': False, 'reason': 'ground_radar_disabled'}

        int_pos = np.array(interceptor['position'], dtype=np.float32)
        mis_pos = np.array(missile['position'], dtype=np.float32)
        mis_vel = np.array(missile['velocity'], dtype=np.float32)
        int_vel = np.array(interceptor['velocity'], dtype=np.float32)

        ground_pos = self.ground_radar.position

        # Ground to missile vector
        ground_to_missile = mis_pos - ground_pos
        range_to_missile = np.linalg.norm(ground_to_missile)

        # Check range limitation
        if range_to_missile > self.ground_radar.max_range:
            return {'detected': False, 'reason': 'out_of_range'}

        # Check elevation angle (can't see below horizon)
        if range_to_missile > 1e-6:
            elevation = np.arcsin(np.clip(ground_to_missile[2] / range_to_missile, -1.0, 1.0))
            if elevation < self.ground_radar.min_elevation_angle:
                return {'detected': False, 'reason': 'below_horizon'}
            if elevation > self.ground_radar.max_elevation_angle:
                return {'detected': False, 'reason': 'above_coverage'}

        # Check terrain masking (simplified: missile below 50m AGL is masked)
        if mis_pos[2] < 50.0:
            return {'detected': False, 'reason': 'terrain_masking'}

        # Calculate detection probability based on range (with curriculum-based reliability)
        detection_prob = self.ground_radar.base_quality * (1.0 - (range_to_missile / self.ground_radar.max_range) * 0.4)
        detection_prob *= weather_factor  # Weather degradation
        detection_prob *= self.ground_detection_reliability  # Curriculum adjustment

        if self.rng.random() > detection_prob:
            return {'detected': False, 'reason': 'weak_return'}

        # SUCCESS: Generate measurement with realistic errors
        # Relative position: missile to interceptor (as seen from ground perspective)
        true_rel_pos = mis_pos - int_pos

        # Add measurement noise (independent from onboard radar)
        pos_noise = self.rng.normal(0, self.ground_radar.range_accuracy, 3)
        vel_noise = self.rng.normal(0, self.ground_radar.velocity_accuracy, 3)

        measured_rel_pos = true_rel_pos + pos_noise
        measured_rel_vel = mis_vel - int_vel + vel_noise

        return {
            'detected': True,
            'rel_pos': measured_rel_pos,
            'rel_vel': measured_rel_vel,
            'quality': detection_prob,
            'range': range_to_missile
        }

    def _compute_datalink_quality(self, interceptor: Dict[str, Any]) -> float:
        """
        Model ground-to-interceptor data link reliability.

        Factors affecting link quality:
        - Distance (signal strength)
        - Relative velocity (doppler shift)
        - Atmospheric conditions
        - Packet loss
        """
        if not self.ground_radar_enabled or self.ground_radar is None:
            return 0.0

        int_pos = np.array(interceptor['position'], dtype=np.float32)
        int_vel = np.array(interceptor['velocity'], dtype=np.float32)
        ground_pos = self.ground_radar.position

        link_range = np.linalg.norm(int_pos - ground_pos)

        # Range-based quality degradation
        if link_range > self.max_datalink_range:
            return 0.0

        range_factor = 1.0 - (link_range / self.max_datalink_range) ** 2

        # Velocity-based doppler effects (up to 30% degradation at high speed)
        velocity_mag = np.linalg.norm(int_vel)
        doppler_factor = 1.0 - min(velocity_mag / 1000.0, 0.3)

        # Random packet loss
        if self.rng.random() < self.datalink_packet_loss:
            return 0.0  # Packet lost this timestep

        quality = range_factor * doppler_factor * 0.95  # 95% baseline reliability
        return np.clip(quality, 0.0, 1.0)

    def _compute_fusion_confidence(self, onboard_detected: bool, onboard_quality: float,
                                   ground_detected: bool, ground_quality: float,
                                   onboard_pos: np.ndarray, ground_pos: np.ndarray) -> float:
        """
        Calculate confidence in fused radar measurements.

        Higher confidence when:
        - Both radars detect target
        - Measurements agree spatially
        - Both have high individual quality
        """
        if not onboard_detected and not ground_detected:
            return 0.0  # No detection at all

        if onboard_detected and not ground_detected:
            return onboard_quality * 0.5  # Single sensor (onboard only)

        if ground_detected and not onboard_detected:
            return ground_quality * 0.6  # Ground radar only (more reliable)

        # Both detected - check measurement agreement
        position_error = np.linalg.norm(onboard_pos - ground_pos)
        max_acceptable_error = 200.0  # meters (reasonable for independent sensors)

        agreement_factor = 1.0 - min(position_error / max_acceptable_error, 1.0)

        # Weighted fusion confidence
        fusion_confidence = (
            0.35 * onboard_quality +   # Onboard is closer but noisier
            0.50 * ground_quality +    # Ground is more accurate
            0.15 * agreement_factor    # Bonus for agreement
        )

        return np.clip(fusion_confidence, 0.0, 1.0)

    def compute_radar_detection(self, interceptor: Dict[str, Any], missile: Dict[str, Any],
                               radar_quality: float = 1.0, noise_level: float = 0.05,
                               weather_factor: float = 1.0) -> np.ndarray:
        """
        Simulate radar detection from both onboard and ground radars.
        Returns 26D observation vector with realistic sensor delays.

        The interceptor has NO direct access to missile state - only radar returns.
        Ground radar provides additional measurements but with its own limitations.
        """
        # Extract interceptor's own state (perfect knowledge of self)
        int_pos = np.array(interceptor['position'], dtype=np.float32)
        int_vel = np.array(interceptor['velocity'], dtype=np.float32)
        int_quat = np.array(interceptor['orientation'], dtype=np.float32)
        int_fuel = interceptor.get('fuel', 100.0)

        # Extract missile's TRUE state (only for radar simulation)
        true_mis_pos = np.array(missile['position'], dtype=np.float32)
        true_mis_vel = np.array(missile['velocity'], dtype=np.float32)

        # === ONBOARD RADAR DETECTION ===
        true_rel_pos = true_mis_pos - int_pos
        true_range = np.linalg.norm(true_rel_pos)

        onboard_detected = True
        onboard_detection_info = {'detected': True, 'range': true_range, 'radar_quality': radar_quality}

        # Check onboard radar range limitation
        if true_range > self.radar_range:
            onboard_detected = False
            onboard_detection_info['detected'] = False
            onboard_detection_info['reason'] = 'out_of_range'

        # Check radar beam width (curriculum-adjustable cone)
        if onboard_detected:
            int_forward = get_forward_vector(int_quat)
            to_missile = true_rel_pos / (true_range + 1e-6)
            beam_angle = np.arccos(np.clip(np.dot(int_forward, to_missile), -1, 1))
            beam_width_rad = np.radians(self.radar_beam_width)
            if beam_angle > beam_width_rad:
                onboard_detected = False
                onboard_detection_info['detected'] = False
                onboard_detection_info['reason'] = 'outside_beam'

        # Apply radar quality degradation (with curriculum-based reliability)
        if onboard_detected:
            range_factor = 1.0 - (true_range / self.radar_range) * 0.5
            actual_radar_quality = radar_quality * range_factor * self.onboard_detection_reliability

            if self.rng.random() > actual_radar_quality:
                onboard_detected = False
                onboard_detection_info['detected'] = False
                onboard_detection_info['reason'] = 'poor_signal'

        # Prepare onboard measurement
        onboard_measurement = {
            'rel_pos': true_rel_pos.copy(),
            'mis_vel': true_mis_vel.copy(),
            'detection_info': onboard_detection_info.copy()
        }

        # Apply onboard sensor delays
        if self.sensor_delay_buffer:
            delayed_onboard, delayed_onboard_detected = self.sensor_delay_buffer.add_measurement(
                onboard_measurement, onboard_detected
            )
            if delayed_onboard is None:
                delayed_onboard_rel_pos = np.zeros(3)
                delayed_onboard_mis_vel = np.zeros(3)
                delayed_onboard_detected = False
                delayed_onboard_info = {'detected': False, 'reason': 'sensor_delay_initialization', 'radar_quality': 0.0}
            else:
                delayed_onboard_rel_pos = delayed_onboard['rel_pos']
                delayed_onboard_mis_vel = delayed_onboard['mis_vel']
                delayed_onboard_info = delayed_onboard['detection_info']
        else:
            delayed_onboard_rel_pos = true_rel_pos
            delayed_onboard_mis_vel = true_mis_vel
            delayed_onboard_detected = onboard_detected
            delayed_onboard_info = onboard_detection_info

        # === GROUND RADAR DETECTION ===
        ground_detection = self._compute_ground_radar_detection(interceptor, missile, weather_factor)
        ground_detected = ground_detection.get('detected', False)

        if ground_detected:
            ground_rel_pos = ground_detection['rel_pos']
            ground_rel_vel = ground_detection['rel_vel']
            ground_quality = ground_detection['quality']
        else:
            ground_rel_pos = np.zeros(3, dtype=np.float32)
            ground_rel_vel = np.zeros(3, dtype=np.float32)
            ground_quality = 0.0

        # Apply ground radar sensor delays
        if self.ground_sensor_delay_buffer:
            ground_measurement = {
                'rel_pos': ground_rel_pos.copy(),
                'rel_vel': ground_rel_vel.copy(),
                'quality': ground_quality
            }
            delayed_ground, delayed_ground_detected = self.ground_sensor_delay_buffer.add_measurement(
                ground_measurement, ground_detected
            )
            if delayed_ground is None:
                delayed_ground_rel_pos = np.zeros(3)
                delayed_ground_rel_vel = np.zeros(3)
                delayed_ground_detected = False
                delayed_ground_quality = 0.0
            else:
                delayed_ground_rel_pos = delayed_ground['rel_pos']
                delayed_ground_rel_vel = delayed_ground['rel_vel']
                delayed_ground_detected = ground_detected
                delayed_ground_quality = delayed_ground['quality']
        else:
            delayed_ground_rel_pos = ground_rel_pos
            delayed_ground_rel_vel = ground_rel_vel
            delayed_ground_detected = ground_detected
            delayed_ground_quality = ground_quality

        # === DATA LINK QUALITY ===
        datalink_quality = self._compute_datalink_quality(interceptor)

        # === SENSOR FUSION CONFIDENCE ===
        onboard_quality = delayed_onboard_info.get('radar_quality', 0.0) if delayed_onboard_detected else 0.0
        fusion_confidence = self._compute_fusion_confidence(
            delayed_onboard_detected, onboard_quality,
            delayed_ground_detected, delayed_ground_quality,
            delayed_onboard_rel_pos, delayed_ground_rel_pos
        )

        # Store detection info for reward calculation (internal use only)
        self._last_detection_info = delayed_onboard_info.copy()
        self._last_ground_detection_info = {'detected': delayed_ground_detected, 'quality': delayed_ground_quality}

        # Generate 26D observation
        obs = self.compute(
            interceptor, missile,
            delayed_onboard_rel_pos, delayed_onboard_mis_vel, delayed_onboard_detected, delayed_onboard_info,
            delayed_ground_rel_pos, delayed_ground_rel_vel, delayed_ground_detected, delayed_ground_quality,
            datalink_quality, fusion_confidence, noise_level
        )
        return obs
    
    def compute(self, interceptor: Dict[str, Any], missile: Dict[str, Any],
                onboard_rel_pos: np.ndarray, onboard_mis_vel: np.ndarray,
                onboard_detected: bool, onboard_detection_info: Dict[str, Any],
                ground_rel_pos: np.ndarray, ground_rel_vel: np.ndarray,
                ground_detected: bool, ground_quality: float,
                datalink_quality: float, fusion_confidence: float,
                noise_level: float = 0.05) -> np.ndarray:
        """
        Generate 26D observation vector based on radar detections from both sensors.

        ONBOARD RADAR Components:
        [0-2]: Relative position (onboard radar-detected, with noise)
        [3-5]: Relative velocity (onboard radar-detected, with noise)
        [6-8]: Interceptor velocity (internal sensors, perfect)
        [9-11]: Interceptor orientation (internal sensors, perfect)
        [12]: Fuel fraction (internal sensors, perfect)
        [13]: Time to intercept estimate (computed from onboard radar data)
        [14]: Onboard radar lock quality
        [15]: Closing rate (computed from onboard radar data)
        [16]: Off-axis angle (computed from onboard radar data)

        GROUND RADAR Components:
        [17-19]: Ground radar relative position (independent measurement)
        [20-22]: Ground radar relative velocity (independent measurement)
        [23]: Ground radar quality/confidence
        [24]: Data link quality (ground-to-interceptor communication)
        [25]: Multi-radar fusion confidence
        """
        obs = np.zeros(26, dtype=np.float32)

        # Extract interceptor's internal state (perfect self-knowledge)
        int_pos = np.array(interceptor['position'], dtype=np.float32)
        int_vel = np.array(interceptor['velocity'], dtype=np.float32)
        int_quat = np.array(interceptor['orientation'], dtype=np.float32)
        int_fuel = interceptor.get('fuel', 100.0)

        # === ONBOARD RADAR OBSERVATIONS [0-16] ===
        if onboard_detected:
            # Apply radar measurement noise (increases with range)
            range_to_target = np.linalg.norm(onboard_rel_pos)
            range_noise_factor = 1.0 + (range_to_target / self.radar_range) * 2.0

            # Position measurement with range-dependent noise (curriculum-adjusted)
            radar_rel_pos = onboard_rel_pos.copy()
            effective_noise = self.measurement_noise_level  # Use curriculum-adjusted noise level
            if effective_noise > 0:
                pos_noise = self.rng.normal(0, effective_noise * range_noise_factor * range_to_target, 3)
                radar_rel_pos += pos_noise

            # Velocity measurement with doppler noise (curriculum-adjusted)
            radar_rel_vel = onboard_mis_vel - int_vel
            if effective_noise > 0:
                vel_noise_std = effective_noise * range_noise_factor * np.linalg.norm(radar_rel_vel)
                vel_noise = self.rng.normal(0, vel_noise_std, 3)
                radar_rel_vel += vel_noise

            # [0-2] Onboard radar-detected relative position
            obs[0:3] = np.clip(radar_rel_pos / self.max_range, -1.0, 1.0)

            # [3-5] Onboard radar-detected relative velocity
            obs[3:6] = np.clip(radar_rel_vel / self.max_velocity, -1.0, 1.0)

            # Computed values from onboard radar data
            radar_range = np.linalg.norm(radar_rel_pos)
            closing_speed = -np.dot(radar_rel_pos, radar_rel_vel) / (radar_range + 1e-6)

            # [13] Time to intercept estimate
            if closing_speed > 0:
                tti = radar_range / closing_speed
                obs[13] = np.clip(1.0 - tti / 100.0, -1.0, 1.0)
            else:
                obs[13] = -1.0

            # [14] Onboard radar lock quality
            obs[14] = onboard_detection_info.get('radar_quality', 0.0)

            # [15] Closing rate
            obs[15] = np.clip(closing_speed / self.max_velocity, -1.0, 1.0)

            # [16] Off-axis angle
            if radar_range > 1e-6:
                int_forward = get_forward_vector(int_quat)
                to_target = radar_rel_pos / radar_range
                obs[16] = np.dot(int_forward, to_target)
            else:
                obs[16] = 1.0
        else:
            # No onboard detection - all target-related observations are zero/invalid
            obs[0:3] = 0.0
            obs[3:6] = 0.0
            obs[13] = -1.0
            obs[14] = 0.0
            obs[15] = 0.0
            obs[16] = 0.0

        # [6-8] Interceptor's own velocity (perfect internal knowledge)
        obs[6:9] = np.clip(int_vel / self.max_velocity, -1.0, 1.0)

        # [9-11] Interceptor's own orientation (perfect internal knowledge)
        euler = quaternion_to_euler(int_quat)
        obs[9:12] = euler / np.pi

        # [12] Fuel fraction (perfect internal knowledge)
        obs[12] = np.clip(int_fuel / 100.0, 0.0, 1.0)

        # === GROUND RADAR OBSERVATIONS [17-25] ===
        if ground_detected and datalink_quality > 0.1:  # Need data link for ground radar data
            # Apply minimal additional noise to ground radar (already has measurement noise)
            ground_radar_pos = ground_rel_pos.copy()
            ground_radar_vel = ground_rel_vel.copy()

            # [17-19] Ground radar relative position
            obs[17:20] = np.clip(ground_radar_pos / self.max_range, -1.0, 1.0)

            # [20-22] Ground radar relative velocity
            obs[20:23] = np.clip(ground_radar_vel / self.max_velocity, -1.0, 1.0)

            # [23] Ground radar quality
            obs[23] = ground_quality
        else:
            # No ground radar detection or data link failure
            obs[17:20] = 0.0
            obs[20:23] = 0.0
            obs[23] = 0.0

        # [24] Data link quality (always available, shows communication status)
        obs[24] = datalink_quality

        # [25] Multi-radar fusion confidence
        obs[25] = fusion_confidence

        return obs


class CoordinateTransform:
    """Handle coordinate system transformations."""
    
    @staticmethod
    def enu_to_unity(position: np.ndarray, quaternion: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert from ENU right-handed to Unity left-handed coordinates."""
        # Position: ENU (x=East, y=North, z=Up) -> Unity (x=East, y=Up, z=North)
        unity_pos = np.array([position[0], position[2], position[1]], dtype=np.float32)
        
        # Quaternion: adjust for coordinate system change
        w, x, y, z = quaternion
        unity_quat = np.array([w, x, z, y], dtype=np.float32)
        
        return unity_pos, unity_quat
    
    @staticmethod
    def unity_to_enu(position: np.ndarray, quaternion: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert from Unity left-handed to ENU right-handed coordinates."""
        # Position: Unity (x=East, y=Up, z=North) -> ENU (x=East, y=North, z=Up)
        enu_pos = np.array([position[0], position[2], position[1]], dtype=np.float32)
        
        # Quaternion: reverse the adjustment
        w, x, y, z = quaternion
        enu_quat = np.array([w, x, z, y], dtype=np.float32)
        
        return enu_pos, enu_quat


class SafetyClamp:
    """Apply safety constraints to actions."""
    
    def __init__(self, limits: Optional[SafetyLimits] = None):
        self.limits = limits or SafetyLimits()
        
    def apply(self, action: np.ndarray, current_fuel: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply safety constraints to action vector.
        
        Returns:
            Clamped action and info dict with clamping details.
        """
        clamped = action.copy()
        info = {'clamped': False, 'reasons': []}
        
        # Clamp thrust based on fuel availability
        if current_fuel <= 0:
            clamped[0:3] *= 0  # No thrust without fuel
            info['clamped'] = True
            info['reasons'].append('no_fuel')
        
        # Clamp acceleration magnitude
        accel_mag = np.linalg.norm(clamped[0:3])
        if accel_mag > self.limits.max_acceleration:
            clamped[0:3] *= self.limits.max_acceleration / accel_mag
            info['clamped'] = True
            info['reasons'].append('max_acceleration')
        
        # Clamp angular rates
        if len(clamped) > 3:
            angular_mag = np.linalg.norm(clamped[3:6])
            if angular_mag > self.limits.max_angular_rate:
                clamped[3:6] *= self.limits.max_angular_rate / angular_mag
                info['clamped'] = True
                info['reasons'].append('max_angular_rate')
        
        return clamped, info


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w,x,y,z] to euler angles [roll,pitch,yaw]."""
    w, x, y, z = q
    
    # Roll
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    
    # Yaw
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw], dtype=np.float32)


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Convert euler angles [roll,pitch,yaw] to quaternion [w,x,y,z]."""
    roll, pitch, yaw = euler
    
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z], dtype=np.float32)


def get_forward_vector(q: np.ndarray) -> np.ndarray:
    """Get forward direction from quaternion."""
    w, x, y, z = q
    forward = np.array([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y)
    ], dtype=np.float32)
    norm = np.linalg.norm(forward)
    return forward / (norm + 1e-6)