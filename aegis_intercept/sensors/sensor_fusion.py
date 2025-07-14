"""
Sensor Fusion and State Estimation System

Implements Kalman filtering and multi-sensor fusion for realistic
state estimation from noisy radar measurements. Replaces perfect
state information with realistic sensor-based tracking.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import math
from scipy.linalg import cholesky, solve_triangular

from .radar_system import RadarDetection, RadarTrack


class TrackQuality(Enum):
    """Track quality levels"""
    TENTATIVE = "tentative"      # New track, low confidence
    CONFIRMED = "confirmed"      # Established track, high confidence
    COASTING = "coasting"        # Track without recent measurements
    LOST = "lost"               # Track lost, marked for deletion


@dataclass
class TrackState:
    """Complete track state with uncertainties"""
    # State vector: [x, y, z, vx, vy, vz, ax, ay, az] (9D for constant acceleration model)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # State covariance matrix (9x9)
    covariance: np.ndarray = field(default_factory=lambda: np.eye(9) * 1000)
    
    # Track metadata
    track_id: int = -1
    last_update_time: float = 0.0
    creation_time: float = 0.0
    quality: TrackQuality = TrackQuality.TENTATIVE
    confidence: float = 0.0
    
    # Measurement history
    measurement_count: int = 0
    consecutive_misses: int = 0
    
    # Prediction state
    predicted_state: np.ndarray = field(default_factory=lambda: np.zeros(9))
    predicted_covariance: np.ndarray = field(default_factory=lambda: np.eye(9))
    innovation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    innovation_covariance: np.ndarray = field(default_factory=lambda: np.eye(3))
    
    def get_full_state_vector(self) -> np.ndarray:
        """Get complete 9D state vector"""
        return np.concatenate([self.position, self.velocity, self.acceleration])
    
    def set_from_state_vector(self, state: np.ndarray):
        """Set state from 9D state vector"""
        self.position = state[0:3].copy()
        self.velocity = state[3:6].copy()
        self.acceleration = state[6:9].copy()


@dataclass
class FusionConfig:
    """Configuration for sensor fusion system"""
    # State model parameters
    process_noise_std: float = 2.0  # m/s² acceleration noise
    position_init_std: float = 100.0  # m initial position uncertainty
    velocity_init_std: float = 50.0  # m/s initial velocity uncertainty
    acceleration_init_std: float = 10.0  # m/s² initial acceleration uncertainty
    
    # Track management
    max_coast_time: float = 5.0  # seconds before track is lost
    confirmation_threshold: int = 3  # measurements needed to confirm
    deletion_threshold: int = 5  # consecutive misses before deletion
    association_gate: float = 4.0  # Chi-squared gate for data association
    
    # Measurement validation
    min_measurement_snr: float = 8.0  # dB minimum SNR for valid measurement
    max_measurement_range: float = 1000000.0  # m maximum valid range
    
    # Performance tuning
    enable_adaptive_noise: bool = True
    enable_imm_filtering: bool = False  # Interacting Multiple Model (future)
    measurement_fusion_method: str = "weighted_average"  # or "kalman"


class KalmanTracker:
    """Extended Kalman Filter for 6DOF target tracking"""
    
    def __init__(self, config: FusionConfig):
        """
        Initialize Kalman tracker
        
        Args:
            config: Fusion system configuration
        """
        self.config = config
        
        # State transition matrix (constant acceleration model)
        self.F = np.eye(9)  # Will be updated with dt
        
        # Measurement matrix (position measurements only)
        self.H = np.zeros((3, 9))
        self.H[0:3, 0:3] = np.eye(3)  # Measure position directly
        
        # Process noise matrix
        self.Q = self._build_process_noise_matrix(1.0)  # Will be scaled by dt
        
        # Measurement noise matrix (will be updated per measurement)
        self.R = np.eye(3) * 100  # Default 10m std in each direction
        
    def _build_process_noise_matrix(self, dt: float) -> np.ndarray:
        """
        Build process noise matrix for constant acceleration model
        
        Args:
            dt: Time step in seconds
            
        Returns:
            9x9 process noise matrix
        """
        # Continuous-time noise model
        sigma_a = self.config.process_noise_std  # acceleration noise std
        
        # Discrete-time process noise (from Van Loan method)
        dt2 = dt * dt
        dt3 = dt2 * dt / 3.0
        dt4 = dt3 * dt / 4.0
        
        # Block structure for x, y, z independently
        block = np.array([
            [dt4, dt3, dt2/2],
            [dt3, dt2, dt],
            [dt2/2, dt, 1.0]
        ]) * sigma_a**2
        
        Q = np.zeros((9, 9))
        for i in range(3):
            start_idx = i * 3
            end_idx = start_idx + 3
            Q[start_idx:end_idx, start_idx:end_idx] = block
            
        return Q
    
    def _update_state_transition_matrix(self, dt: float):
        """Update state transition matrix with time step"""
        # Constant acceleration model: x_k+1 = F * x_k
        # State: [x, y, z, vx, vy, vz, ax, ay, az]
        
        self.F = np.eye(9)
        
        # Position updates
        self.F[0, 3] = dt      # x += vx * dt
        self.F[1, 4] = dt      # y += vy * dt  
        self.F[2, 5] = dt      # z += vz * dt
        self.F[0, 6] = dt*dt/2 # x += ax * dt²/2
        self.F[1, 7] = dt*dt/2 # y += ay * dt²/2
        self.F[2, 8] = dt*dt/2 # z += az * dt²/2
        
        # Velocity updates
        self.F[3, 6] = dt      # vx += ax * dt
        self.F[4, 7] = dt      # vy += ay * dt
        self.F[5, 8] = dt      # vz += az * dt
        
        # Acceleration remains constant (F[6:9, 6:9] = I)
    
    def predict(self, track_state: TrackState, dt: float) -> TrackState:
        """
        Kalman filter prediction step
        
        Args:
            track_state: Current track state
            dt: Time step since last update
            
        Returns:
            Predicted track state
        """
        # Update matrices for this time step
        self._update_state_transition_matrix(dt)
        Q = self._build_process_noise_matrix(dt)
        
        # Get current state
        x = track_state.get_full_state_vector()
        P = track_state.covariance.copy()
        
        # Prediction equations
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + Q
        
        # Create predicted state
        predicted_state = TrackState(
            track_id=track_state.track_id,
            last_update_time=track_state.last_update_time,
            creation_time=track_state.creation_time,
            quality=track_state.quality,
            confidence=track_state.confidence,
            measurement_count=track_state.measurement_count,
            consecutive_misses=track_state.consecutive_misses
        )
        
        predicted_state.set_from_state_vector(x_pred)
        predicted_state.covariance = P_pred
        predicted_state.predicted_state = x_pred
        predicted_state.predicted_covariance = P_pred
        
        return predicted_state
    
    def update(self, predicted_state: TrackState, measurement: np.ndarray,
               measurement_covariance: np.ndarray, timestamp: float) -> TrackState:
        """
        Kalman filter update step
        
        Args:
            predicted_state: Predicted track state
            measurement: 3D position measurement [x, y, z]
            measurement_covariance: 3x3 measurement covariance matrix
            timestamp: Measurement timestamp
            
        Returns:
            Updated track state
        """
        # Get predicted state and covariance
        x_pred = predicted_state.predicted_state
        P_pred = predicted_state.predicted_covariance
        
        # Measurement prediction
        z_pred = self.H @ x_pred  # Predicted measurement
        
        # Innovation
        innovation = measurement - z_pred
        
        # Innovation covariance
        S = self.H @ P_pred @ self.H.T + measurement_covariance
        
        # Kalman gain
        try:
            # Use Cholesky decomposition for numerical stability
            L = cholesky(S, lower=True)
            temp = solve_triangular(L, self.H @ P_pred, lower=True)
            K = solve_triangular(L.T, temp, lower=False).T
        except np.linalg.LinAlgError:
            # Fallback to standard inversion if Cholesky fails
            K = P_pred @ self.H.T @ np.linalg.inv(S)
        
        # State update
        x_updated = x_pred + K @ innovation
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(9) - K @ self.H
        P_updated = I_KH @ P_pred @ I_KH.T + K @ measurement_covariance @ K.T
        
        # Create updated state
        updated_state = TrackState(
            track_id=predicted_state.track_id,
            last_update_time=timestamp,
            creation_time=predicted_state.creation_time,
            quality=predicted_state.quality,
            confidence=min(1.0, predicted_state.confidence + 0.1),
            measurement_count=predicted_state.measurement_count + 1,
            consecutive_misses=0  # Reset miss counter
        )
        
        updated_state.set_from_state_vector(x_updated)
        updated_state.covariance = P_updated
        updated_state.innovation = innovation
        updated_state.innovation_covariance = S
        
        # Update track quality based on performance
        if updated_state.measurement_count >= self.config.confirmation_threshold:
            updated_state.quality = TrackQuality.CONFIRMED
            
        return updated_state
    
    def validate_measurement(self, predicted_state: TrackState, measurement: np.ndarray,
                           measurement_covariance: np.ndarray) -> Tuple[bool, float]:
        """
        Validate measurement using chi-squared gate
        
        Args:
            predicted_state: Predicted track state
            measurement: 3D position measurement
            measurement_covariance: Measurement covariance
            
        Returns:
            Tuple of (is_valid, normalized_distance)
        """
        # Calculate innovation
        z_pred = self.H @ predicted_state.predicted_state
        innovation = measurement - z_pred
        
        # Innovation covariance
        S = self.H @ predicted_state.predicted_covariance @ self.H.T + measurement_covariance
        
        # Normalized squared distance (Mahalanobis distance)
        try:
            inv_S = np.linalg.inv(S)
            normalized_distance = innovation.T @ inv_S @ innovation
        except np.linalg.LinAlgError:
            return False, float('inf')
        
        # Chi-squared test (3 DOF for 3D measurement)
        gate_threshold = self.config.association_gate**2
        is_valid = normalized_distance <= gate_threshold
        
        return is_valid, math.sqrt(normalized_distance)


class SensorFusionSystem:
    """Multi-sensor fusion system for realistic state estimation"""
    
    def __init__(self, config: FusionConfig):
        """
        Initialize sensor fusion system
        
        Args:
            config: Fusion configuration parameters
        """
        self.config = config
        self.kalman_tracker = KalmanTracker(config)
        
        # Track management
        self.active_tracks: Dict[int, TrackState] = {}
        self.next_track_id = 1
        
        # Performance statistics
        self.total_measurements = 0
        self.associated_measurements = 0
        self.false_alarms = 0
        self.track_creations = 0
        self.track_deletions = 0
        
    def process_radar_detections(self, detections: List[RadarDetection], 
                               radar_position: np.ndarray, timestamp: float) -> List[TrackState]:
        """
        Process radar detections and update tracks
        
        Args:
            detections: List of radar detections
            radar_position: 3D position of the radar
            timestamp: Current simulation time
            
        Returns:
            List of updated track states
        """
        self.total_measurements += len(detections)
        
        # Convert detections to Cartesian coordinates
        measurements = []
        measurement_covariances = []
        
        for detection in detections:
            # Validate detection quality
            if detection.snr < self.config.min_measurement_snr:
                continue
            if detection.range > self.config.max_measurement_range:
                continue
                
            # Convert spherical to Cartesian
            range_m = detection.range
            azimuth_rad = math.radians(detection.azimuth)
            elevation_rad = math.radians(detection.elevation)
            
            x = range_m * math.cos(elevation_rad) * math.cos(azimuth_rad)
            y = range_m * math.cos(elevation_rad) * math.sin(azimuth_rad)
            z = range_m * math.sin(elevation_rad)
            
            # Convert to global coordinates
            measurement = np.array([x, y, z]) + radar_position
            measurements.append(measurement)
            
            # Build measurement covariance matrix
            # Convert spherical uncertainties to Cartesian
            range_var = detection.range_std**2
            az_var = (math.radians(detection.azimuth_std))**2
            el_var = (math.radians(detection.elevation_std))**2
            
            # Jacobian matrix for spherical to Cartesian conversion
            J = np.array([
                [math.cos(elevation_rad) * math.cos(azimuth_rad),
                 -range_m * math.cos(elevation_rad) * math.sin(azimuth_rad),
                 -range_m * math.sin(elevation_rad) * math.cos(azimuth_rad)],
                [math.cos(elevation_rad) * math.sin(azimuth_rad),
                 range_m * math.cos(elevation_rad) * math.cos(azimuth_rad),
                 -range_m * math.sin(elevation_rad) * math.sin(azimuth_rad)],
                [math.sin(elevation_rad),
                 0,
                 range_m * math.cos(elevation_rad)]
            ])
            
            # Spherical covariance
            R_spherical = np.diag([range_var, az_var, el_var])
            
            # Transform to Cartesian
            R_cartesian = J @ R_spherical @ J.T
            measurement_covariances.append(R_cartesian)
        
        if not measurements:
            # No valid measurements - just predict existing tracks
            return self._predict_all_tracks(timestamp)
            
        # Data association and tracking
        return self._associate_and_update(measurements, measurement_covariances, timestamp)
    
    def _predict_all_tracks(self, timestamp: float) -> List[TrackState]:
        """Predict all existing tracks to current time"""
        predicted_tracks = []
        tracks_to_remove = []
        
        for track_id, track_state in self.active_tracks.items():
            dt = timestamp - track_state.last_update_time
            
            if dt > self.config.max_coast_time:
                # Track is too old, mark for deletion
                tracks_to_remove.append(track_id)
                continue
                
            # Predict track to current time
            predicted_track = self.kalman_tracker.predict(track_state, dt)
            predicted_track.consecutive_misses += 1
            
            # Update track quality
            if predicted_track.consecutive_misses >= self.config.deletion_threshold:
                predicted_track.quality = TrackQuality.LOST
                tracks_to_remove.append(track_id)
            elif predicted_track.consecutive_misses > 1:
                predicted_track.quality = TrackQuality.COASTING
                
            predicted_tracks.append(predicted_track)
            self.active_tracks[track_id] = predicted_track
            
        # Remove lost tracks
        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
            self.track_deletions += 1
            
        return predicted_tracks
    
    def _associate_and_update(self, measurements: List[np.ndarray],
                            measurement_covariances: List[np.ndarray],
                            timestamp: float) -> List[TrackState]:
        """
        Associate measurements with tracks and update
        Uses nearest neighbor data association
        """
        updated_tracks = []
        unassociated_measurements = list(zip(measurements, measurement_covariances))
        
        # First, predict all existing tracks
        predicted_tracks = {}
        for track_id, track_state in self.active_tracks.items():
            dt = timestamp - track_state.last_update_time
            predicted_track = self.kalman_tracker.predict(track_state, dt)
            predicted_tracks[track_id] = predicted_track
            
        # Associate measurements with tracks using nearest neighbor
        for track_id, predicted_track in predicted_tracks.items():
            if not unassociated_measurements:
                break
                
            best_measurement = None
            best_covariance = None
            best_distance = float('inf')
            best_index = -1
            
            # Find best measurement for this track
            for i, (measurement, covariance) in enumerate(unassociated_measurements):
                is_valid, distance = self.kalman_tracker.validate_measurement(
                    predicted_track, measurement, covariance
                )
                
                if is_valid and distance < best_distance:
                    best_distance = distance
                    best_measurement = measurement
                    best_covariance = covariance
                    best_index = i
                    
            # Update track with best measurement
            if best_measurement is not None:
                updated_track = self.kalman_tracker.update(
                    predicted_track, best_measurement, best_covariance, timestamp
                )
                updated_tracks.append(updated_track)
                self.active_tracks[track_id] = updated_track
                
                # Remove associated measurement
                unassociated_measurements.pop(best_index)
                self.associated_measurements += 1
            else:
                # No measurement associated - track continues coasting
                predicted_track.consecutive_misses += 1
                if predicted_track.consecutive_misses >= self.config.deletion_threshold:
                    predicted_track.quality = TrackQuality.LOST
                else:
                    predicted_track.quality = TrackQuality.COASTING
                    updated_tracks.append(predicted_track)
                    self.active_tracks[track_id] = predicted_track
                    
        # Create new tracks for unassociated measurements
        for measurement, covariance in unassociated_measurements:
            new_track = self._initiate_track(measurement, covariance, timestamp)
            updated_tracks.append(new_track)
            self.active_tracks[new_track.track_id] = new_track
            self.track_creations += 1
            
        return updated_tracks
    
    def _initiate_track(self, measurement: np.ndarray, measurement_covariance: np.ndarray,
                       timestamp: float) -> TrackState:
        """
        Initiate a new track from a measurement
        
        Args:
            measurement: Initial position measurement
            measurement_covariance: Measurement uncertainty
            timestamp: Track creation time
            
        Returns:
            New track state
        """
        # Initialize state vector
        initial_state = np.zeros(9)
        initial_state[0:3] = measurement  # Position from measurement
        # Velocity and acceleration start at zero
        
        # Initialize covariance matrix
        initial_covariance = np.eye(9)
        
        # Position uncertainty from measurement
        initial_covariance[0:3, 0:3] = measurement_covariance
        
        # Velocity uncertainty (large initial uncertainty)
        vel_var = self.config.velocity_init_std**2
        initial_covariance[3:6, 3:6] = np.eye(3) * vel_var
        
        # Acceleration uncertainty
        acc_var = self.config.acceleration_init_std**2
        initial_covariance[6:9, 6:9] = np.eye(3) * acc_var
        
        # Create track state
        new_track = TrackState(
            track_id=self.next_track_id,
            last_update_time=timestamp,
            creation_time=timestamp,
            quality=TrackQuality.TENTATIVE,
            confidence=0.1,
            measurement_count=1,
            consecutive_misses=0
        )
        
        new_track.set_from_state_vector(initial_state)
        new_track.covariance = initial_covariance
        
        self.next_track_id += 1
        return new_track
    
    def get_confirmed_tracks(self) -> List[TrackState]:
        """Get only confirmed, high-quality tracks"""
        return [track for track in self.active_tracks.values() 
                if track.quality == TrackQuality.CONFIRMED]
    
    def get_best_target_estimate(self) -> Optional[TrackState]:
        """
        Get the best target estimate (highest confidence confirmed track)
        
        Returns:
            Best track state or None if no confirmed tracks
        """
        confirmed_tracks = self.get_confirmed_tracks()
        
        if not confirmed_tracks:
            return None
            
        # Return track with highest confidence
        return max(confirmed_tracks, key=lambda t: t.confidence)
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get sensor fusion performance statistics"""
        association_rate = (self.associated_measurements / 
                          max(self.total_measurements, 1))
        
        active_tracks_by_quality = {}
        for quality in TrackQuality:
            count = sum(1 for track in self.active_tracks.values() 
                       if track.quality == quality)
            active_tracks_by_quality[quality.value] = count
            
        return {
            'total_measurements': self.total_measurements,
            'associated_measurements': self.associated_measurements,
            'association_rate': association_rate,
            'false_alarms': self.false_alarms,
            'active_tracks': len(self.active_tracks),
            'confirmed_tracks': active_tracks_by_quality.get('confirmed', 0),
            'track_creations': self.track_creations,
            'track_deletions': self.track_deletions,
            'tracks_by_quality': active_tracks_by_quality
        }