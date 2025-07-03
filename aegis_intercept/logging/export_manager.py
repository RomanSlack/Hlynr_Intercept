"""
Export Manager for AegisIntercept Phase 3

This module provides export functionality for Unity-compatible data structures,
performance metrics, and visualization data. It handles conversion between
Python data formats and Unity-friendly JSON structures.

Features:
- Unity-compatible trajectory export
- Performance metrics dashboard data
- Real-time data streaming
- Video/animation export preparation
- Statistical analysis export
- Custom format support

Author: Coder Agent
Date: Phase 3 Implementation
"""

import json
import numpy as np
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import math
import gzip
import threading
from collections import defaultdict

from .trajectory_logger import TrajectoryLogger, TrajectoryPoint, EpisodeMetrics, LogLevel


class UnityCoordinateSystem(Enum):
    """Unity coordinate system conventions"""
    LEFT_HANDED = "left_handed"  # Unity default (Y up, Z forward, X right)
    RIGHT_HANDED = "right_handed"  # Standard math/physics (Z up, Y forward, X right)


class ExportFormat(Enum):
    """Export format types"""
    UNITY_JSON = "unity_json"
    UNITY_BINARY = "unity_binary"
    CSV_ANALYTICS = "csv_analytics"
    DASHBOARD_JSON = "dashboard_json"
    ANIMATION_DATA = "animation_data"
    REAL_TIME_STREAM = "real_time_stream"


@dataclass
class UnityVector3:
    """Unity-compatible Vector3 structure"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}
    
    @classmethod
    def from_numpy(cls, vec: np.ndarray, coord_system: UnityCoordinateSystem = UnityCoordinateSystem.LEFT_HANDED):
        """Convert numpy array to Unity Vector3"""
        if coord_system == UnityCoordinateSystem.LEFT_HANDED:
            # Convert from right-handed (Z up) to left-handed (Y up)
            return cls(x=vec[0], y=vec[2], z=vec[1])
        else:
            return cls(x=vec[0], y=vec[1], z=vec[2])


@dataclass
class UnityQuaternion:
    """Unity-compatible Quaternion structure"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "w": self.w}
    
    @classmethod
    def from_numpy(cls, quat: np.ndarray, coord_system: UnityCoordinateSystem = UnityCoordinateSystem.LEFT_HANDED):
        """Convert numpy quaternion to Unity Quaternion"""
        if coord_system == UnityCoordinateSystem.LEFT_HANDED:
            # Convert coordinate system: [w, x, y, z] -> Unity [x, z, y, w]
            return cls(x=quat[1], y=quat[3], z=quat[2], w=quat[0])
        else:
            return cls(x=quat[1], y=quat[2], z=quat[3], w=quat[0])


@dataclass
class UnityTransform:
    """Unity-compatible Transform structure"""
    position: UnityVector3
    rotation: UnityQuaternion
    scale: UnityVector3 = None
    
    def __post_init__(self):
        if self.scale is None:
            self.scale = UnityVector3(1.0, 1.0, 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position.to_dict(),
            "rotation": self.rotation.to_dict(),
            "scale": self.scale.to_dict()
        }


@dataclass
class UnityTrajectoryFrame:
    """Single frame of trajectory data for Unity"""
    time: float
    interceptor: UnityTransform
    missile: UnityTransform
    target: UnityVector3
    
    # Visualization data
    interceptor_trail: List[UnityVector3]
    missile_trail: List[UnityVector3]
    
    # State information
    interceptor_velocity: UnityVector3
    missile_velocity: UnityVector3
    fuel_remaining: float
    explosion_active: bool
    
    # Performance metrics
    intercept_distance: float
    closing_velocity: float
    reward: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "interceptor": self.interceptor.to_dict(),
            "missile": self.missile.to_dict(),
            "target": self.target.to_dict(),
            "interceptor_trail": [p.to_dict() for p in self.interceptor_trail],
            "missile_trail": [p.to_dict() for p in self.missile_trail],
            "interceptor_velocity": self.interceptor_velocity.to_dict(),
            "missile_velocity": self.missile_velocity.to_dict(),
            "fuel_remaining": self.fuel_remaining,
            "explosion_active": self.explosion_active,
            "intercept_distance": self.intercept_distance,
            "closing_velocity": self.closing_velocity,
            "reward": self.reward
        }


@dataclass
class UnityEpisodeData:
    """Complete episode data for Unity"""
    episode_id: int
    metadata: Dict[str, Any]
    frames: List[UnityTrajectoryFrame]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "metadata": self.metadata,
            "frames": [frame.to_dict() for frame in self.frames],
            "summary": self.summary
        }


@dataclass
class DashboardMetrics:
    """Performance metrics for dashboard display"""
    # Overall statistics
    total_episodes: int
    success_rate: float
    average_reward: float
    
    # Recent performance
    recent_success_rate: float
    recent_average_reward: float
    recent_episodes: int
    
    # Trends
    success_rate_trend: List[float]
    reward_trend: List[float]
    episode_length_trend: List[float]
    
    # Distribution data
    reward_distribution: Dict[str, int]
    intercept_distance_distribution: Dict[str, int]
    fuel_efficiency_distribution: Dict[str, int]
    
    # Top performances
    best_episodes: List[Dict[str, Any]]
    worst_episodes: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExportManager:
    """Main export manager for Unity and analytics data"""
    
    def __init__(self,
                 export_directory: str = "exports",
                 coordinate_system: UnityCoordinateSystem = UnityCoordinateSystem.LEFT_HANDED,
                 trail_length: int = 50,
                 enable_real_time: bool = False,
                 compression_enabled: bool = True):
        """
        Initialize export manager
        
        Args:
            export_directory: Directory for exported files
            coordinate_system: Unity coordinate system to use
            trail_length: Length of trajectory trails
            enable_real_time: Enable real-time data streaming
            compression_enabled: Enable data compression
        """
        self.export_directory = Path(export_directory)
        self.export_directory.mkdir(parents=True, exist_ok=True)
        
        self.coordinate_system = coordinate_system
        self.trail_length = trail_length
        self.enable_real_time = enable_real_time
        self.compression_enabled = compression_enabled
        
        # Real-time streaming
        self.real_time_data = {}
        self.streaming_lock = threading.Lock()
        
        # Trail management
        self.interceptor_trail = []
        self.missile_trail = []
        
        # Target position (usually static)
        self.target_position = UnityVector3(0, 0, 0)
        
        # Performance tracking
        self.episode_cache = {}
        self.metrics_cache = None
        
        print(f"ExportManager initialized: {export_directory}, coordinate_system={coordinate_system.value}")
    
    def export_episode_for_unity(self,
                                 trajectory_points: List[TrajectoryPoint],
                                 episode_metrics: EpisodeMetrics,
                                 target_position: np.ndarray,
                                 filename: Optional[str] = None) -> str:
        """
        Export episode data in Unity-compatible format
        
        Args:
            trajectory_points: List of trajectory points
            episode_metrics: Episode performance metrics
            target_position: Target position vector
            filename: Optional custom filename
            
        Returns:
            Path to exported file
        """
        # Convert target position
        self.target_position = UnityVector3.from_numpy(target_position, self.coordinate_system)
        
        # Convert trajectory points to Unity frames
        frames = []
        self.interceptor_trail = []
        self.missile_trail = []
        
        for i, point in enumerate(trajectory_points):
            frame = self._convert_trajectory_point_to_unity_frame(point, i)
            frames.append(frame)
        
        # Create episode data
        episode_data = UnityEpisodeData(
            episode_id=episode_metrics.episode_id,
            metadata=self._create_metadata(episode_metrics),
            frames=frames,
            summary=self._create_episode_summary(episode_metrics)
        )
        
        # Save to file
        if filename is None:
            filename = f"unity_episode_{episode_metrics.episode_id:06d}"
        
        filepath = self._save_unity_data(filename, episode_data.to_dict())
        
        # Cache for dashboard
        self.episode_cache[episode_metrics.episode_id] = episode_data
        
        return filepath
    
    def _convert_trajectory_point_to_unity_frame(self, point: TrajectoryPoint, frame_index: int) -> UnityTrajectoryFrame:
        """Convert a trajectory point to Unity frame"""
        # Convert positions and orientations
        interceptor_pos = UnityVector3.from_numpy(point.interceptor_position, self.coordinate_system)
        interceptor_rot = UnityQuaternion.from_numpy(point.interceptor_orientation, self.coordinate_system)
        
        missile_pos = UnityVector3.from_numpy(point.missile_position, self.coordinate_system)
        missile_rot = UnityQuaternion.from_numpy(point.missile_orientation, self.coordinate_system)
        
        # Convert velocities
        interceptor_vel = UnityVector3.from_numpy(point.interceptor_velocity, self.coordinate_system)
        missile_vel = UnityVector3.from_numpy(point.missile_velocity, self.coordinate_system)
        
        # Update trails
        self.interceptor_trail.append(interceptor_pos)
        self.missile_trail.append(missile_pos)
        
        # Keep trail length limited
        if len(self.interceptor_trail) > self.trail_length:
            self.interceptor_trail = self.interceptor_trail[-self.trail_length:]
        if len(self.missile_trail) > self.trail_length:
            self.missile_trail = self.missile_trail[-self.trail_length:]
        
        # Create transforms
        interceptor_transform = UnityTransform(interceptor_pos, interceptor_rot)
        missile_transform = UnityTransform(missile_pos, missile_rot)
        
        # Create frame
        frame = UnityTrajectoryFrame(
            time=point.simulation_time,
            interceptor=interceptor_transform,
            missile=missile_transform,
            target=self.target_position,
            interceptor_trail=self.interceptor_trail.copy(),
            missile_trail=self.missile_trail.copy(),
            interceptor_velocity=interceptor_vel,
            missile_velocity=missile_vel,
            fuel_remaining=point.fuel_remaining,
            explosion_active=point.explosion_command,
            intercept_distance=point.intercept_distance,
            closing_velocity=point.closing_velocity,
            reward=point.step_reward
        )
        
        return frame
    
    def _create_metadata(self, episode_metrics: EpisodeMetrics) -> Dict[str, Any]:
        """Create metadata for Unity episode"""
        return {
            "coordinate_system": self.coordinate_system.value,
            "trail_length": self.trail_length,
            "episode_duration": episode_metrics.simulation_duration,
            "total_frames": episode_metrics.total_steps,
            "export_time": time.time(),
            "success": episode_metrics.success,
            "termination_reason": episode_metrics.termination_reason
        }
    
    def _create_episode_summary(self, episode_metrics: EpisodeMetrics) -> Dict[str, Any]:
        """Create episode summary for Unity"""
        return {
            "performance": {
                "success": episode_metrics.success,
                "total_reward": episode_metrics.total_reward,
                "fuel_efficiency": episode_metrics.fuel_efficiency,
                "intercept_quality": episode_metrics.intercept_quality_score,
                "trajectory_smoothness": episode_metrics.trajectory_smoothness
            },
            "statistics": {
                "max_altitude": episode_metrics.max_interceptor_altitude,
                "max_speed": episode_metrics.max_interceptor_speed,
                "min_distance": episode_metrics.min_intercept_distance,
                "final_distance": episode_metrics.final_intercept_distance
            },
            "control": {
                "average_thrust": episode_metrics.average_thrust_magnitude,
                "average_torque": episode_metrics.average_control_torque,
                "explosion_used": episode_metrics.explosion_used
            }
        }
    
    def export_dashboard_data(self,
                             episode_metrics_list: List[EpisodeMetrics],
                             filename: str = "dashboard_data") -> str:
        """Export data for performance dashboard"""
        dashboard_metrics = self._create_dashboard_metrics(episode_metrics_list)
        
        dashboard_data = {
            "metrics": dashboard_metrics.to_dict(),
            "chart_data": self._create_chart_data(episode_metrics_list),
            "recent_episodes": self._get_recent_episodes_summary(episode_metrics_list[-10:]),
            "export_time": time.time()
        }
        
        filepath = self.export_directory / f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
        
        return str(filepath)
    
    def _create_dashboard_metrics(self, episode_metrics_list: List[EpisodeMetrics]) -> DashboardMetrics:
        """Create dashboard metrics from episode list"""
        if not episode_metrics_list:
            return DashboardMetrics(
                total_episodes=0, success_rate=0.0, average_reward=0.0,
                recent_success_rate=0.0, recent_average_reward=0.0, recent_episodes=0,
                success_rate_trend=[], reward_trend=[], episode_length_trend=[],
                reward_distribution={}, intercept_distance_distribution={}, fuel_efficiency_distribution={},
                best_episodes=[], worst_episodes=[]
            )
        
        total_episodes = len(episode_metrics_list)
        successful_episodes = sum(1 for e in episode_metrics_list if e.success)
        success_rate = successful_episodes / total_episodes
        average_reward = sum(e.total_reward for e in episode_metrics_list) / total_episodes
        
        # Recent performance (last 100 episodes)
        recent_episodes = episode_metrics_list[-100:] if len(episode_metrics_list) >= 100 else episode_metrics_list
        recent_successful = sum(1 for e in recent_episodes if e.success)
        recent_success_rate = recent_successful / len(recent_episodes)
        recent_average_reward = sum(e.total_reward for e in recent_episodes) / len(recent_episodes)
        
        # Trends (last 50 episodes in chunks of 10)
        success_rate_trend = []
        reward_trend = []
        episode_length_trend = []
        
        chunk_size = 10
        for i in range(max(0, len(episode_metrics_list) - 50), len(episode_metrics_list), chunk_size):
            chunk = episode_metrics_list[i:i + chunk_size]
            if chunk:
                chunk_success_rate = sum(1 for e in chunk if e.success) / len(chunk)
                chunk_reward = sum(e.total_reward for e in chunk) / len(chunk)
                chunk_length = sum(e.total_steps for e in chunk) / len(chunk)
                
                success_rate_trend.append(chunk_success_rate)
                reward_trend.append(chunk_reward)
                episode_length_trend.append(chunk_length)
        
        # Distributions
        reward_distribution = self._create_distribution([e.total_reward for e in episode_metrics_list], 
                                                      "reward", bins=10)
        intercept_distance_distribution = self._create_distribution(
            [e.final_intercept_distance for e in episode_metrics_list], "distance", bins=10)
        fuel_efficiency_distribution = self._create_distribution(
            [e.fuel_efficiency for e in episode_metrics_list], "efficiency", bins=10)
        
        # Top and bottom performers
        sorted_by_reward = sorted(episode_metrics_list, key=lambda e: e.total_reward, reverse=True)
        best_episodes = [self._episode_to_summary(e) for e in sorted_by_reward[:5]]
        worst_episodes = [self._episode_to_summary(e) for e in sorted_by_reward[-5:]]
        
        return DashboardMetrics(
            total_episodes=total_episodes,
            success_rate=success_rate,
            average_reward=average_reward,
            recent_success_rate=recent_success_rate,
            recent_average_reward=recent_average_reward,
            recent_episodes=len(recent_episodes),
            success_rate_trend=success_rate_trend,
            reward_trend=reward_trend,
            episode_length_trend=episode_length_trend,
            reward_distribution=reward_distribution,
            intercept_distance_distribution=intercept_distance_distribution,
            fuel_efficiency_distribution=fuel_efficiency_distribution,
            best_episodes=best_episodes,
            worst_episodes=worst_episodes
        )
    
    def _create_distribution(self, values: List[float], label: str, bins: int = 10) -> Dict[str, int]:
        """Create distribution data for histograms"""
        if not values:
            return {}
        
        min_val = min(values)
        max_val = max(values)
        bin_width = (max_val - min_val) / bins
        
        distribution = {}
        for i in range(bins):
            bin_start = min_val + i * bin_width
            bin_end = bin_start + bin_width
            bin_label = f"{bin_start:.1f}-{bin_end:.1f}"
            
            count = sum(1 for v in values if bin_start <= v < bin_end)
            if i == bins - 1:  # Include max value in last bin
                count = sum(1 for v in values if bin_start <= v <= bin_end)
            
            distribution[bin_label] = count
        
        return distribution
    
    def _create_chart_data(self, episode_metrics_list: List[EpisodeMetrics]) -> Dict[str, Any]:
        """Create chart data for visualization"""
        episodes = episode_metrics_list[-100:]  # Last 100 episodes
        
        return {
            "episode_rewards": {
                "x": [e.episode_id for e in episodes],
                "y": [e.total_reward for e in episodes],
                "label": "Episode Rewards"
            },
            "success_rate_rolling": {
                "x": [e.episode_id for e in episodes[10:]],
                "y": self._calculate_rolling_average([1 if e.success else 0 for e in episodes], window=10),
                "label": "Success Rate (Rolling 10)"
            },
            "fuel_efficiency": {
                "x": [e.episode_id for e in episodes],
                "y": [e.fuel_efficiency for e in episodes],
                "label": "Fuel Efficiency"
            },
            "intercept_quality": {
                "x": [e.episode_id for e in episodes],
                "y": [e.intercept_quality_score for e in episodes],
                "label": "Intercept Quality Score"
            }
        }
    
    def _calculate_rolling_average(self, values: List[float], window: int) -> List[float]:
        """Calculate rolling average"""
        if len(values) < window:
            return []
        
        rolling_avg = []
        for i in range(window - 1, len(values)):
            avg = sum(values[i - window + 1:i + 1]) / window
            rolling_avg.append(avg)
        
        return rolling_avg
    
    def _get_recent_episodes_summary(self, recent_episodes: List[EpisodeMetrics]) -> List[Dict[str, Any]]:
        """Get summary of recent episodes"""
        return [self._episode_to_summary(e) for e in recent_episodes]
    
    def _episode_to_summary(self, episode: EpisodeMetrics) -> Dict[str, Any]:
        """Convert episode metrics to summary dict"""
        return {
            "episode_id": episode.episode_id,
            "success": episode.success,
            "total_reward": round(episode.total_reward, 2),
            "total_steps": episode.total_steps,
            "fuel_efficiency": round(episode.fuel_efficiency, 2),
            "intercept_quality": round(episode.intercept_quality_score, 2),
            "final_distance": round(episode.final_intercept_distance, 2),
            "termination_reason": episode.termination_reason
        }
    
    def export_animation_data(self,
                             trajectory_points: List[TrajectoryPoint],
                             target_position: np.ndarray,
                             filename: str,
                             time_step: float = 0.1) -> str:
        """Export trajectory data for animation"""
        # Resample trajectory for smooth animation
        resampled_points = self._resample_trajectory(trajectory_points, time_step)
        
        animation_data = {
            "metadata": {
                "coordinate_system": self.coordinate_system.value,
                "time_step": time_step,
                "total_frames": len(resampled_points),
                "duration": resampled_points[-1].simulation_time if resampled_points else 0
            },
            "target_position": UnityVector3.from_numpy(target_position, self.coordinate_system).to_dict(),
            "frames": []
        }
        
        for point in resampled_points:
            frame_data = {
                "time": point.simulation_time,
                "interceptor": {
                    "position": UnityVector3.from_numpy(point.interceptor_position, self.coordinate_system).to_dict(),
                    "rotation": UnityQuaternion.from_numpy(point.interceptor_orientation, self.coordinate_system).to_dict(),
                    "velocity": UnityVector3.from_numpy(point.interceptor_velocity, self.coordinate_system).to_dict()
                },
                "missile": {
                    "position": UnityVector3.from_numpy(point.missile_position, self.coordinate_system).to_dict(),
                    "rotation": UnityQuaternion.from_numpy(point.missile_orientation, self.coordinate_system).to_dict(),
                    "velocity": UnityVector3.from_numpy(point.missile_velocity, self.coordinate_system).to_dict()
                },
                "effects": {
                    "explosion": point.explosion_command,
                    "fuel_remaining": point.fuel_remaining,
                    "thrust_magnitude": np.linalg.norm(point.thrust_force)
                }
            }
            animation_data["frames"].append(frame_data)
        
        return self._save_unity_data(filename, animation_data)
    
    def _resample_trajectory(self, trajectory_points: List[TrajectoryPoint], time_step: float) -> List[TrajectoryPoint]:
        """Resample trajectory to fixed time steps for smooth animation"""
        if not trajectory_points:
            return []
        
        # Sort by simulation time
        sorted_points = sorted(trajectory_points, key=lambda p: p.simulation_time)
        
        # Create resampled points
        resampled = []
        start_time = sorted_points[0].simulation_time
        end_time = sorted_points[-1].simulation_time
        
        current_time = start_time
        point_index = 0
        
        while current_time <= end_time and point_index < len(sorted_points) - 1:
            # Find the two points to interpolate between
            while (point_index < len(sorted_points) - 1 and 
                   sorted_points[point_index + 1].simulation_time < current_time):
                point_index += 1
            
            if point_index >= len(sorted_points) - 1:
                break
            
            # Interpolate between points
            p1 = sorted_points[point_index]
            p2 = sorted_points[point_index + 1]
            
            if p2.simulation_time == p1.simulation_time:
                alpha = 0.0
            else:
                alpha = (current_time - p1.simulation_time) / (p2.simulation_time - p1.simulation_time)
            
            # Create interpolated point
            interpolated = self._interpolate_trajectory_points(p1, p2, alpha, current_time)
            resampled.append(interpolated)
            
            current_time += time_step
        
        return resampled
    
    def _interpolate_trajectory_points(self, p1: TrajectoryPoint, p2: TrajectoryPoint, 
                                     alpha: float, target_time: float) -> TrajectoryPoint:
        """Interpolate between two trajectory points"""
        # Linear interpolation for positions and velocities
        def lerp(a, b, t):
            return a + t * (b - a)
        
        # Spherical interpolation for quaternions
        def slerp(q1, q2, t):
            dot = np.dot(q1, q2)
            if dot < 0.0:
                q2 = -q2
                dot = -dot
            
            if dot > 0.9995:
                # Linear interpolation for close quaternions
                result = lerp(q1, q2, t)
                return result / np.linalg.norm(result)
            
            theta_0 = np.arccos(abs(dot))
            sin_theta_0 = np.sin(theta_0)
            theta = theta_0 * t
            sin_theta = np.sin(theta)
            
            s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
            s1 = sin_theta / sin_theta_0
            
            return s0 * q1 + s1 * q2
        
        # Create interpolated point
        interpolated = TrajectoryPoint(
            timestamp=lerp(p1.timestamp, p2.timestamp, alpha),
            simulation_time=target_time,
            step_count=int(lerp(p1.step_count, p2.step_count, alpha)),
            interceptor_position=lerp(p1.interceptor_position, p2.interceptor_position, alpha),
            interceptor_velocity=lerp(p1.interceptor_velocity, p2.interceptor_velocity, alpha),
            interceptor_orientation=slerp(p1.interceptor_orientation, p2.interceptor_orientation, alpha),
            interceptor_angular_velocity=lerp(p1.interceptor_angular_velocity, p2.interceptor_angular_velocity, alpha),
            missile_position=lerp(p1.missile_position, p2.missile_position, alpha),
            missile_velocity=lerp(p1.missile_velocity, p2.missile_velocity, alpha),
            missile_orientation=slerp(p1.missile_orientation, p2.missile_orientation, alpha),
            missile_angular_velocity=lerp(p1.missile_angular_velocity, p2.missile_angular_velocity, alpha),
            environment_time=lerp(p1.environment_time, p2.environment_time, alpha),
            fuel_remaining=lerp(p1.fuel_remaining, p2.fuel_remaining, alpha),
            wind_velocity=lerp(p1.wind_velocity, p2.wind_velocity, alpha),
            thrust_force=lerp(p1.thrust_force, p2.thrust_force, alpha),
            control_torque=lerp(p1.control_torque, p2.control_torque, alpha),
            explosion_command=p2.explosion_command,  # Use latest value for boolean
            intercept_distance=lerp(p1.intercept_distance, p2.intercept_distance, alpha),
            missile_target_distance=lerp(p1.missile_target_distance, p2.missile_target_distance, alpha),
            closing_velocity=lerp(p1.closing_velocity, p2.closing_velocity, alpha),
            step_reward=lerp(p1.step_reward, p2.step_reward, alpha),
            cumulative_reward=lerp(p1.cumulative_reward, p2.cumulative_reward, alpha)
        )
        
        return interpolated
    
    def _save_unity_data(self, filename: str, data: Dict[str, Any]) -> str:
        """Save data in Unity-compatible format"""
        filepath = self.export_directory / f"{filename}.json"
        
        try:
            if self.compression_enabled and len(json.dumps(data)) > 1024 * 1024:  # > 1MB
                # Use compressed format for large files
                filepath = self.export_directory / f"{filename}.json.gz"
                with gzip.open(filepath, 'wt') as f:
                    json.dump(data, f, indent=2, separators=(',', ':'))
            else:
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            
            return str(filepath)
            
        except Exception as e:
            print(f"Failed to save Unity data: {e}")
            raise
    
    def start_real_time_streaming(self, port: int = 8080):
        """Start real-time data streaming (placeholder for WebSocket/HTTP server)"""
        if not self.enable_real_time:
            print("Real-time streaming not enabled")
            return
        
        # Placeholder for real-time streaming implementation
        # In a full implementation, this would start a WebSocket or HTTP server
        print(f"Real-time streaming would start on port {port}")
        print("Note: Real-time streaming requires additional networking implementation")
    
    def update_real_time_data(self, trajectory_point: TrajectoryPoint, target_position: np.ndarray):
        """Update real-time streaming data"""
        if not self.enable_real_time:
            return
        
        with self.streaming_lock:
            frame = self._convert_trajectory_point_to_unity_frame(trajectory_point, 0)
            self.real_time_data = {
                "timestamp": time.time(),
                "frame": frame.to_dict(),
                "target": UnityVector3.from_numpy(target_position, self.coordinate_system).to_dict()
            }
    
    def get_real_time_data(self) -> Dict[str, Any]:
        """Get current real-time data"""
        with self.streaming_lock:
            return self.real_time_data.copy()
    
    def clear_cache(self):
        """Clear cached data"""
        self.episode_cache.clear()
        self.metrics_cache = None
        print("Export manager cache cleared")


# Utility functions
def create_export_manager(export_directory: str = "exports",
                         coordinate_system: UnityCoordinateSystem = UnityCoordinateSystem.LEFT_HANDED) -> ExportManager:
    """Create an export manager with default settings"""
    return ExportManager(export_directory=export_directory, coordinate_system=coordinate_system)


def convert_coordinates_for_unity(position: np.ndarray, 
                                 coordinate_system: UnityCoordinateSystem = UnityCoordinateSystem.LEFT_HANDED) -> Dict[str, float]:
    """Convert position vector to Unity coordinate system"""
    unity_pos = UnityVector3.from_numpy(position, coordinate_system)
    return unity_pos.to_dict()


def batch_export_episodes(trajectory_logger: TrajectoryLogger,
                         export_manager: ExportManager,
                         episode_ids: List[int],
                         target_position: np.ndarray) -> List[str]:
    """Batch export multiple episodes for Unity"""
    exported_files = []
    
    for episode_id in episode_ids:
        # This would require integration with TrajectoryLogger to get episode data
        # Placeholder for actual implementation
        print(f"Would export episode {episode_id}")
        # trajectory_points = trajectory_logger.get_episode_trajectory(episode_id)
        # episode_metrics = trajectory_logger.get_episode_metrics(episode_id)
        # filepath = export_manager.export_episode_for_unity(trajectory_points, episode_metrics, target_position)
        # exported_files.append(filepath)
    
    return exported_files