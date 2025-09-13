"""
Core components: 17D observations, coordinate transforms, and safety constraints.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class SafetyLimits:
    """Safety constraints for interceptor actions."""
    max_acceleration: float = 50.0  # m/s^2
    max_angular_rate: float = 5.0   # rad/s
    max_gimbal_angle: float = 0.785  # 45 degrees
    min_thrust: float = 0.0
    max_thrust: float = 1.0
    fuel_depletion_rate: float = 0.1  # kg/s at max thrust


class Radar17DObservation:
    """17-dimensional radar observation space for 1v1 intercept scenarios."""
    
    def __init__(self, max_range: float = 10000.0, max_velocity: float = 1000.0, 
                 radar_range: float = 5000.0, min_detection_range: float = 50.0):
        self.max_range = max_range
        self.max_velocity = max_velocity
        self.radar_range = radar_range  # Maximum radar detection range
        self.min_detection_range = min_detection_range  # Minimum range for accurate tracking
        self.rng = np.random.default_rng()
        
    def seed(self, seed: int):
        """Set random seed for reproducible noise."""
        self.rng = np.random.default_rng(seed)
        
    def compute_radar_detection(self, interceptor: Dict[str, Any], missile: Dict[str, Any], 
                               radar_quality: float = 1.0, noise_level: float = 0.05) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Simulate radar detection of missile from interceptor's sensors only.
        Returns observation vector and detection info.
        
        The interceptor has NO direct access to missile state - only radar returns.
        """
        # Extract interceptor's own state (perfect knowledge of self)
        int_pos = np.array(interceptor['position'], dtype=np.float32)
        int_vel = np.array(interceptor['velocity'], dtype=np.float32)
        int_quat = np.array(interceptor['orientation'], dtype=np.float32)
        int_fuel = interceptor.get('fuel', 100.0)
        
        # Extract missile's TRUE state (only for radar simulation)
        true_mis_pos = np.array(missile['position'], dtype=np.float32)
        true_mis_vel = np.array(missile['velocity'], dtype=np.float32)
        
        # Calculate true range
        true_rel_pos = true_mis_pos - int_pos
        true_range = np.linalg.norm(true_rel_pos)
        
        # Determine if missile is detectable by radar
        detected = True
        detection_info = {'detected': True, 'range': true_range, 'radar_quality': radar_quality}
        
        # Check radar range limitation
        if true_range > self.radar_range:
            detected = False
            detection_info['detected'] = False
            detection_info['reason'] = 'out_of_range'
        
        # Check radar beam width (simplified - assume 60 degree cone)
        if detected:
            int_forward = get_forward_vector(int_quat)
            to_missile = true_rel_pos / (true_range + 1e-6)
            beam_angle = np.arccos(np.clip(np.dot(int_forward, to_missile), -1, 1))
            if beam_angle > np.pi / 3:  # 60 degrees
                detected = False
                detection_info['detected'] = False  
                detection_info['reason'] = 'outside_beam'
        
        # Apply radar quality degradation
        if detected:
            # Degraded detection at long range
            range_factor = 1.0 - (true_range / self.radar_range) * 0.5
            actual_radar_quality = radar_quality * range_factor
            
            # Chance of losing lock based on quality
            if self.rng.random() > actual_radar_quality:
                detected = False
                detection_info['detected'] = False
                detection_info['reason'] = 'poor_signal'
        
        obs = self.compute(interceptor, missile, true_rel_pos, true_mis_vel, detected, 
                          detection_info, noise_level)
        return obs
    
    def compute(self, interceptor: Dict[str, Any], missile: Dict[str, Any], 
                detected_rel_pos: np.ndarray, detected_mis_vel: np.ndarray,
                detected: bool, detection_info: Dict[str, Any], noise_level: float = 0.05) -> np.ndarray:
        """
        Generate 17D observation vector based on radar detection.
        
        Components:
        [0-2]: Relative position (radar-detected, with noise)
        [3-5]: Relative velocity (radar-detected, with noise)  
        [6-8]: Interceptor velocity (internal sensors, perfect)
        [9-11]: Interceptor orientation (internal sensors, perfect)
        [12]: Fuel fraction (internal sensors, perfect)
        [13]: Time to intercept estimate (computed from radar data)
        [14]: Radar lock quality
        [15]: Closing rate (computed from radar data)
        [16]: Off-axis angle (computed from radar data)
        """
        obs = np.zeros(17, dtype=np.float32)
        
        # Extract interceptor's internal state (perfect self-knowledge)
        int_pos = np.array(interceptor['position'], dtype=np.float32)
        int_vel = np.array(interceptor['velocity'], dtype=np.float32)
        int_quat = np.array(interceptor['orientation'], dtype=np.float32)
        int_fuel = interceptor.get('fuel', 100.0)
        
        if detected:
            # Apply radar measurement noise (increases with range)
            range_to_target = np.linalg.norm(detected_rel_pos)
            range_noise_factor = 1.0 + (range_to_target / self.radar_range) * 2.0
            
            # Position measurement with range-dependent noise
            radar_rel_pos = detected_rel_pos.copy()
            if noise_level > 0:
                pos_noise = self.rng.normal(0, noise_level * range_noise_factor * range_to_target, 3)
                radar_rel_pos += pos_noise
            
            # Velocity measurement with doppler noise  
            radar_rel_vel = detected_mis_vel - int_vel
            if noise_level > 0:
                vel_noise_std = noise_level * range_noise_factor * np.linalg.norm(radar_rel_vel)
                vel_noise = self.rng.normal(0, vel_noise_std, 3)
                radar_rel_vel += vel_noise
            
            # [0-2] Radar-detected relative position
            obs[0:3] = np.clip(radar_rel_pos / self.max_range, -1.0, 1.0)
            
            # [3-5] Radar-detected relative velocity
            obs[3:6] = np.clip(radar_rel_vel / self.max_velocity, -1.0, 1.0)
            
            # Computed values from radar data
            radar_range = np.linalg.norm(radar_rel_pos)
            closing_speed = -np.dot(radar_rel_pos, radar_rel_vel) / (radar_range + 1e-6)
            
            # [13] Time to intercept estimate
            if closing_speed > 0:
                tti = radar_range / closing_speed
                obs[13] = np.clip(1.0 - tti / 100.0, -1.0, 1.0)
            else:
                obs[13] = -1.0
            
            # [14] Radar lock quality (from detection info)
            obs[14] = detection_info['radar_quality']
            
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
            # No detection - all target-related observations are zero/invalid
            obs[0:3] = 0.0  # No position data
            obs[3:6] = 0.0  # No velocity data
            obs[13] = -1.0  # No time estimate
            obs[14] = 0.0   # No radar lock
            obs[15] = 0.0   # No closing rate
            obs[16] = 0.0   # No alignment data
        
        # [6-8] Interceptor's own velocity (perfect internal knowledge)
        obs[6:9] = np.clip(int_vel / self.max_velocity, -1.0, 1.0)
        
        # [9-11] Interceptor's own orientation (perfect internal knowledge)
        euler = quaternion_to_euler(int_quat)
        obs[9:12] = euler / np.pi
        
        # [12] Fuel fraction (perfect internal knowledge)
        obs[12] = np.clip(int_fuel / 100.0, 0.0, 1.0)
        
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