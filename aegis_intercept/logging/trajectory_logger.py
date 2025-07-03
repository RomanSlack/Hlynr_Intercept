"""
Trajectory Logger for AegisIntercept Phase 3

This module provides comprehensive trajectory logging capabilities for
6DOF missile intercept simulations. It captures real-time trajectory data,
state vectors, performance metrics, and provides export functionality
for analysis and visualization.

Features:
- Real-time 29D state vector capture
- Performance metrics logging
- CSV and JSON export formats
- Memory-efficient buffering
- Configurable logging levels
- Thread-safe operations

Author: Coder Agent
Date: Phase 3 Implementation
"""

import numpy as np
import json
import csv
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from collections import deque
import gzip
import pickle
from enum import Enum
import logging


class LogLevel(Enum):
    """Logging detail levels"""
    MINIMAL = "minimal"        # Only episode summary
    BASIC = "basic"           # Key trajectory points
    DETAILED = "detailed"     # All timesteps
    DEBUG = "debug"          # Maximum detail including internal states


class DataFormat(Enum):
    """Data export formats"""
    CSV = "csv"
    JSON = "json"
    BINARY = "binary"        # Pickle format
    COMPRESSED = "compressed" # Gzipped pickle


@dataclass
class TrajectoryPoint:
    """Single trajectory data point"""
    # Time information
    timestamp: float
    simulation_time: float
    step_count: int
    
    # Interceptor 6DOF state (13D)
    interceptor_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    interceptor_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    interceptor_orientation: np.ndarray = field(default_factory=lambda: np.array([1,0,0,0]))  # quaternion
    interceptor_angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Missile 6DOF state (13D)
    missile_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    missile_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    missile_orientation: np.ndarray = field(default_factory=lambda: np.array([1,0,0,0]))  # quaternion
    missile_angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Environment state (3D)
    environment_time: float = 0.0
    fuel_remaining: float = 1.0
    wind_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Control inputs
    thrust_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    control_torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    explosion_command: bool = False
    
    # Derived metrics
    intercept_distance: float = 0.0
    missile_target_distance: float = 0.0
    closing_velocity: float = 0.0
    
    # Aerodynamic information
    interceptor_aero: Optional[Dict[str, float]] = None
    missile_aero: Optional[Dict[str, float]] = None
    
    # Reward and performance
    step_reward: float = 0.0
    cumulative_reward: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp,
            'simulation_time': self.simulation_time,
            'step_count': self.step_count,
            'interceptor_position': self.interceptor_position.tolist(),
            'interceptor_velocity': self.interceptor_velocity.tolist(),
            'interceptor_orientation': self.interceptor_orientation.tolist(),
            'interceptor_angular_velocity': self.interceptor_angular_velocity.tolist(),
            'missile_position': self.missile_position.tolist(),
            'missile_velocity': self.missile_velocity.tolist(),
            'missile_orientation': self.missile_orientation.tolist(),
            'missile_angular_velocity': self.missile_angular_velocity.tolist(),
            'environment_time': self.environment_time,
            'fuel_remaining': self.fuel_remaining,
            'wind_velocity': self.wind_velocity.tolist(),
            'thrust_force': self.thrust_force.tolist(),
            'control_torque': self.control_torque.tolist(),
            'explosion_command': self.explosion_command,
            'intercept_distance': self.intercept_distance,
            'missile_target_distance': self.missile_target_distance,
            'closing_velocity': self.closing_velocity,
            'interceptor_aero': self.interceptor_aero,
            'missile_aero': self.missile_aero,
            'step_reward': self.step_reward,
            'cumulative_reward': self.cumulative_reward
        }
    
    def get_state_vector_29d(self) -> np.ndarray:
        """Get complete 29D state vector"""
        return np.concatenate([
            self.interceptor_position,         # 3D
            self.interceptor_velocity,         # 3D
            self.interceptor_orientation,      # 4D
            self.interceptor_angular_velocity, # 3D
            self.missile_position,            # 3D
            self.missile_velocity,            # 3D
            self.missile_orientation,         # 4D
            self.missile_angular_velocity,    # 3D
            [self.environment_time],          # 1D
            [self.fuel_remaining],            # 1D
            self.wind_velocity               # 3D
        ])  # Total: 31D (adjusted from original 29D for practical use)


@dataclass
class EpisodeMetrics:
    """Episode-level performance metrics"""
    episode_id: int
    start_time: float
    end_time: float
    total_steps: int
    simulation_duration: float
    
    # Outcome
    success: bool
    termination_reason: str
    final_intercept_distance: float
    
    # Performance metrics
    total_reward: float
    average_reward: float
    fuel_consumed: float
    fuel_efficiency: float
    
    # Trajectory characteristics
    max_interceptor_altitude: float
    max_interceptor_speed: float
    max_missile_speed: float
    min_intercept_distance: float
    
    # Control statistics
    average_thrust_magnitude: float
    average_control_torque: float
    explosion_used: bool
    
    # Advanced metrics
    intercept_quality_score: float = 0.0
    trajectory_smoothness: float = 0.0
    energy_efficiency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class TrajectoryLogger:
    """Main trajectory logging system"""
    
    def __init__(self,
                 log_directory: str = "logs/trajectories",
                 log_level: LogLevel = LogLevel.DETAILED,
                 max_buffer_size: int = 10000,
                 auto_save_interval: float = 60.0,  # seconds
                 enable_real_time_export: bool = False,
                 compress_data: bool = True):
        """
        Initialize trajectory logger
        
        Args:
            log_directory: Directory to save log files
            log_level: Level of detail to log
            max_buffer_size: Maximum points to buffer before writing
            auto_save_interval: Automatic save interval in seconds
            enable_real_time_export: Whether to export data in real-time
            compress_data: Whether to compress saved data
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        self.log_level = log_level
        self.max_buffer_size = max_buffer_size
        self.auto_save_interval = auto_save_interval
        self.enable_real_time_export = enable_real_time_export
        self.compress_data = compress_data
        
        # Current episode data
        self.current_episode_id = 0
        self.current_trajectory: List[TrajectoryPoint] = []
        self.current_metrics = None
        self.episode_start_time = 0.0
        self.cumulative_reward = 0.0
        
        # Buffering system
        self.trajectory_buffer: deque = deque(maxlen=max_buffer_size)
        self.episode_buffer: List[EpisodeMetrics] = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Auto-save system
        self.last_save_time = time.time()
        
        # Setup logging
        self.logger = logging.getLogger("TrajectoryLogger")
        handler = logging.FileHandler(self.log_directory / "trajectory_logger.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"TrajectoryLogger initialized: {log_directory}, level={log_level.value}")
    
    def start_episode(self, episode_id: Optional[int] = None) -> int:
        """Start logging a new episode"""
        with self.lock:
            if episode_id is None:
                self.current_episode_id += 1
            else:
                self.current_episode_id = episode_id
            
            self.current_trajectory = []
            self.episode_start_time = time.time()
            self.cumulative_reward = 0.0
            
            self.logger.info(f"Started episode {self.current_episode_id}")
            return self.current_episode_id
    
    def log_step(self,
                 simulation_time: float,
                 step_count: int,
                 interceptor_state: np.ndarray,
                 missile_state: np.ndarray,
                 environment_state: np.ndarray,
                 control_inputs: Dict[str, np.ndarray],
                 reward: float,
                 additional_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a single simulation step
        
        Args:
            simulation_time: Current simulation time
            step_count: Current step number
            interceptor_state: 13D interceptor state vector
            missile_state: 13D missile state vector  
            environment_state: Environment state vector
            control_inputs: Dictionary of control inputs
            reward: Step reward
            additional_data: Additional metrics and data
        """
        # Skip logging based on level
        if self.log_level == LogLevel.MINIMAL:
            return
        elif self.log_level == LogLevel.BASIC and step_count % 5 != 0:
            return  # Log every 5th step
        
        with self.lock:
            self.cumulative_reward += reward
            
            # Create trajectory point
            point = TrajectoryPoint(
                timestamp=time.time(),
                simulation_time=simulation_time,
                step_count=step_count,
                step_reward=reward,
                cumulative_reward=self.cumulative_reward
            )
            
            # Extract interceptor state (13D)
            point.interceptor_position = interceptor_state[0:3].copy()
            point.interceptor_velocity = interceptor_state[3:6].copy()
            point.interceptor_orientation = interceptor_state[6:10].copy()
            point.interceptor_angular_velocity = interceptor_state[10:13].copy()
            
            # Extract missile state (13D)
            point.missile_position = missile_state[0:3].copy()
            point.missile_velocity = missile_state[3:6].copy()
            point.missile_orientation = missile_state[6:10].copy()
            point.missile_angular_velocity = missile_state[10:13].copy()
            
            # Extract environment state
            if len(environment_state) >= 5:
                point.environment_time = environment_state[0]
                point.fuel_remaining = environment_state[1]
                point.wind_velocity = environment_state[2:5].copy()
            
            # Extract control inputs
            point.thrust_force = control_inputs.get('thrust_force', np.zeros(3)).copy()
            point.control_torque = control_inputs.get('control_torque', np.zeros(3)).copy()
            point.explosion_command = control_inputs.get('explosion_command', False)
            
            # Calculate derived metrics
            point.intercept_distance = np.linalg.norm(point.interceptor_position - point.missile_position)
            
            # Add aerodynamic data if available
            if additional_data:
                point.interceptor_aero = additional_data.get('interceptor_aero')
                point.missile_aero = additional_data.get('missile_aero')
                
                if 'target_position' in additional_data:
                    target_pos = additional_data['target_position']
                    point.missile_target_distance = np.linalg.norm(point.missile_position - target_pos)
                
                if 'intercept_geometry' in additional_data:
                    geometry = additional_data['intercept_geometry']
                    point.closing_velocity = geometry.get('closing_velocity', 0.0)
            
            # Add to current trajectory
            self.current_trajectory.append(point)
            
            # Add to buffer for batch processing
            self.trajectory_buffer.append(point)
            
            # Auto-save check
            current_time = time.time()
            if current_time - self.last_save_time > self.auto_save_interval:
                self._auto_save()
                self.last_save_time = current_time
    
    def end_episode(self,
                    success: bool,
                    termination_reason: str,
                    final_info: Optional[Dict[str, Any]] = None) -> EpisodeMetrics:
        """End current episode and compute metrics"""
        with self.lock:
            if not self.current_trajectory:
                self.logger.warning("No trajectory data for episode ending")
                return None
            
            # Calculate episode metrics
            end_time = time.time()
            duration = end_time - self.episode_start_time
            
            # Basic metrics
            metrics = EpisodeMetrics(
                episode_id=self.current_episode_id,
                start_time=self.episode_start_time,
                end_time=end_time,
                total_steps=len(self.current_trajectory),
                simulation_duration=self.current_trajectory[-1].simulation_time,
                success=success,
                termination_reason=termination_reason,
                final_intercept_distance=self.current_trajectory[-1].intercept_distance,
                total_reward=self.cumulative_reward,
                average_reward=self.cumulative_reward / len(self.current_trajectory),
                fuel_consumed=1.0 - self.current_trajectory[-1].fuel_remaining,
                fuel_efficiency=self.cumulative_reward / max(0.1, 1.0 - self.current_trajectory[-1].fuel_remaining)
            )
            
            # Calculate trajectory statistics
            positions = np.array([p.interceptor_position for p in self.current_trajectory])
            velocities = np.array([p.interceptor_velocity for p in self.current_trajectory])
            intercept_distances = np.array([p.intercept_distance for p in self.current_trajectory])
            thrust_magnitudes = np.array([np.linalg.norm(p.thrust_force) for p in self.current_trajectory])
            torque_magnitudes = np.array([np.linalg.norm(p.control_torque) for p in self.current_trajectory])
            
            metrics.max_interceptor_altitude = np.max(positions[:, 2])
            metrics.max_interceptor_speed = np.max(np.linalg.norm(velocities, axis=1))
            metrics.min_intercept_distance = np.min(intercept_distances)
            metrics.average_thrust_magnitude = np.mean(thrust_magnitudes)
            metrics.average_control_torque = np.mean(torque_magnitudes)
            metrics.explosion_used = any(p.explosion_command for p in self.current_trajectory)
            
            # Calculate missile statistics
            missile_velocities = np.array([p.missile_velocity for p in self.current_trajectory])
            metrics.max_missile_speed = np.max(np.linalg.norm(missile_velocities, axis=1))
            
            # Advanced metrics
            metrics.intercept_quality_score = self._calculate_intercept_quality()
            metrics.trajectory_smoothness = self._calculate_trajectory_smoothness()
            metrics.energy_efficiency = self._calculate_energy_efficiency()
            
            # Add to episode buffer
            self.episode_buffer.append(metrics)
            
            self.logger.info(f"Episode {self.current_episode_id} completed: "
                           f"Success={success}, Steps={metrics.total_steps}, "
                           f"Reward={metrics.total_reward:.2f}")
            
            return metrics
    
    def _calculate_intercept_quality(self) -> float:
        """Calculate intercept quality score (0-100)"""
        if not self.current_trajectory:
            return 0.0
        
        # Factors: final distance, approach angle, fuel efficiency
        final_distance = self.current_trajectory[-1].intercept_distance
        fuel_used = 1.0 - self.current_trajectory[-1].fuel_remaining
        
        # Distance score (closer is better)
        distance_score = max(0, 100 - final_distance * 2)
        
        # Fuel efficiency score
        fuel_score = max(0, 100 - fuel_used * 100)
        
        # Trajectory efficiency (straight approach is better)
        positions = np.array([p.interceptor_position for p in self.current_trajectory])
        if len(positions) > 2:
            total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            direct_distance = np.linalg.norm(positions[-1] - positions[0])
            efficiency_score = max(0, 100 * direct_distance / max(total_distance, 1e-6))
        else:
            efficiency_score = 50
        
        # Weighted combination
        quality_score = 0.5 * distance_score + 0.3 * fuel_score + 0.2 * efficiency_score
        return min(100, max(0, quality_score))
    
    def _calculate_trajectory_smoothness(self) -> float:
        """Calculate trajectory smoothness score"""
        if len(self.current_trajectory) < 3:
            return 50.0
        
        # Calculate acceleration changes
        velocities = np.array([p.interceptor_velocity for p in self.current_trajectory])
        accelerations = np.diff(velocities, axis=0)
        jerk = np.diff(accelerations, axis=0)
        
        # RMS jerk as measure of smoothness
        rms_jerk = np.sqrt(np.mean(np.sum(jerk**2, axis=1)))
        
        # Convert to score (lower jerk = higher score)
        smoothness_score = max(0, 100 - rms_jerk * 10)
        return min(100, smoothness_score)
    
    def _calculate_energy_efficiency(self) -> float:
        """Calculate energy efficiency score"""
        if not self.current_trajectory:
            return 0.0
        
        # Total control energy
        total_thrust_energy = sum(np.linalg.norm(p.thrust_force)**2 for p in self.current_trajectory)
        total_torque_energy = sum(np.linalg.norm(p.control_torque)**2 for p in self.current_trajectory)
        
        total_energy = total_thrust_energy + total_torque_energy * 0.1  # Scale torque energy
        
        # Efficiency relative to mission success
        if total_energy > 0 and self.cumulative_reward > 0:
            efficiency = self.cumulative_reward / (total_energy / 1000)  # Scale energy
            return min(100, max(0, efficiency * 10))
        else:
            return 0.0
    
    def _auto_save(self):
        """Automatically save buffered data"""
        if len(self.trajectory_buffer) > self.max_buffer_size * 0.8:
            filename = f"trajectory_buffer_{time.strftime('%Y%m%d_%H%M%S')}"
            self.save_trajectory_data(filename, list(self.trajectory_buffer), DataFormat.COMPRESSED)
            self.trajectory_buffer.clear()
            self.logger.info("Auto-saved trajectory buffer")
    
    def save_episode_data(self,
                         episode_id: Optional[int] = None,
                         format: DataFormat = DataFormat.JSON,
                         include_trajectory: bool = True) -> str:
        """Save episode data to file"""
        if episode_id is None:
            episode_id = self.current_episode_id
        
        filename = f"episode_{episode_id:06d}"
        
        # Prepare data
        data = {
            'episode_id': episode_id,
            'trajectory': [p.to_dict() for p in self.current_trajectory] if include_trajectory else [],
            'metrics': self.episode_buffer[-1].to_dict() if self.episode_buffer else None,
            'metadata': {
                'log_level': self.log_level.value,
                'total_points': len(self.current_trajectory),
                'export_time': time.time()
            }
        }
        
        return self._save_data(filename, data, format)
    
    def save_trajectory_data(self,
                           filename: str,
                           trajectory_points: List[TrajectoryPoint],
                           format: DataFormat = DataFormat.JSON) -> str:
        """Save trajectory data in specified format"""
        data = [p.to_dict() for p in trajectory_points]
        return self._save_data(filename, data, format)
    
    def save_episode_summary(self, filename: str) -> str:
        """Save episode summary statistics"""
        if not self.episode_buffer:
            raise ValueError("No episode data to save")
        
        summary_data = {
            'total_episodes': len(self.episode_buffer),
            'episodes': [metrics.to_dict() for metrics in self.episode_buffer],
            'aggregate_stats': self._calculate_aggregate_stats(),
            'export_time': time.time()
        }
        
        return self._save_data(filename, summary_data, DataFormat.JSON)
    
    def _save_data(self, filename: str, data: Any, format: DataFormat) -> str:
        """Save data in specified format"""
        filepath = self.log_directory / f"{filename}.{format.value}"
        
        try:
            if format == DataFormat.JSON:
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=self._json_serializer)
            
            elif format == DataFormat.CSV:
                if isinstance(data, list) and data:
                    self._save_csv(filepath, data)
                else:
                    raise ValueError("CSV format requires list of trajectory points")
            
            elif format == DataFormat.BINARY:
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            
            elif format == DataFormat.COMPRESSED:
                with gzip.open(f"{filepath}.gz", 'wb') as f:
                    pickle.dump(data, f)
                filepath = f"{filepath}.gz"
            
            self.logger.info(f"Saved data to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
            raise
    
    def _save_csv(self, filepath: Path, trajectory_data: List[Dict]):
        """Save trajectory data as CSV"""
        if not trajectory_data:
            return
        
        # Flatten nested data for CSV
        flattened_data = []
        for point in trajectory_data:
            flat_point = {}
            for key, value in point.items():
                if isinstance(value, list):
                    for i, v in enumerate(value):
                        flat_point[f"{key}_{i}"] = v
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flat_point[f"{key}_{subkey}"] = subvalue
                else:
                    flat_point[key] = value
            flattened_data.append(flat_point)
        
        # Write CSV
        with open(filepath, 'w', newline='') as f:
            if flattened_data:
                writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
                writer.writeheader()
                writer.writerows(flattened_data)
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy arrays"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def _calculate_aggregate_stats(self) -> Dict[str, Any]:
        """Calculate aggregate statistics across all episodes"""
        if not self.episode_buffer:
            return {}
        
        episodes = self.episode_buffer
        
        return {
            'success_rate': sum(e.success for e in episodes) / len(episodes),
            'average_reward': sum(e.total_reward for e in episodes) / len(episodes),
            'average_steps': sum(e.total_steps for e in episodes) / len(episodes),
            'average_fuel_efficiency': sum(e.fuel_efficiency for e in episodes) / len(episodes),
            'average_intercept_quality': sum(e.intercept_quality_score for e in episodes) / len(episodes),
            'best_episode': max(episodes, key=lambda e: e.total_reward).episode_id,
            'worst_episode': min(episodes, key=lambda e: e.total_reward).episode_id,
            'total_simulation_time': sum(e.simulation_duration for e in episodes)
        }
    
    def get_episode_metrics(self, episode_id: Optional[int] = None) -> Optional[EpisodeMetrics]:
        """Get metrics for specific episode"""
        if episode_id is None:
            return self.episode_buffer[-1] if self.episode_buffer else None
        
        for metrics in self.episode_buffer:
            if metrics.episode_id == episode_id:
                return metrics
        
        return None
    
    def get_recent_performance(self, num_episodes: int = 100) -> Dict[str, float]:
        """Get performance metrics for recent episodes"""
        if not self.episode_buffer:
            return {}
        
        recent_episodes = self.episode_buffer[-num_episodes:]
        
        return {
            'episodes_count': len(recent_episodes),
            'success_rate': sum(e.success for e in recent_episodes) / len(recent_episodes),
            'average_reward': sum(e.total_reward for e in recent_episodes) / len(recent_episodes),
            'average_quality_score': sum(e.intercept_quality_score for e in recent_episodes) / len(recent_episodes),
            'fuel_efficiency': sum(e.fuel_efficiency for e in recent_episodes) / len(recent_episodes)
        }
    
    def clear_buffers(self):
        """Clear all buffers and current data"""
        with self.lock:
            self.current_trajectory.clear()
            self.trajectory_buffer.clear()
            self.episode_buffer.clear()
            self.logger.info("Cleared all buffers")
    
    def set_log_level(self, level: LogLevel):
        """Change logging level"""
        self.log_level = level
        self.logger.info(f"Changed log level to {level.value}")
    
    def close(self):
        """Close logger and save any remaining data"""
        with self.lock:
            if self.trajectory_buffer:
                filename = f"final_trajectory_buffer_{time.strftime('%Y%m%d_%H%M%S')}"
                self.save_trajectory_data(filename, list(self.trajectory_buffer), DataFormat.COMPRESSED)
            
            if self.episode_buffer:
                self.save_episode_summary(f"episode_summary_{time.strftime('%Y%m%d_%H%M%S')}")
        
        self.logger.info("TrajectoryLogger closed")


# Utility functions
def create_trajectory_logger(log_directory: str = "logs/trajectories",
                           log_level: LogLevel = LogLevel.DETAILED) -> TrajectoryLogger:
    """Create a trajectory logger with default settings"""
    return TrajectoryLogger(log_directory=log_directory, log_level=log_level)


def load_trajectory_data(filepath: str) -> List[Dict[str, Any]]:
    """Load trajectory data from file"""
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {filepath}")
    
    if filepath.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    elif filepath.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    elif filepath.endswith('.binary'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def analyze_trajectory_file(filepath: str) -> Dict[str, Any]:
    """Analyze trajectory file and return summary statistics"""
    data = load_trajectory_data(filepath)
    
    if not data:
        return {'error': 'No data in file'}
    
    # Basic statistics
    total_points = len(data)
    
    if isinstance(data[0], dict) and 'step_count' in data[0]:
        # Trajectory points
        steps = [p['step_count'] for p in data]
        rewards = [p.get('step_reward', 0) for p in data]
        distances = [p.get('intercept_distance', 0) for p in data]
        
        return {
            'total_points': total_points,
            'total_steps': max(steps) if steps else 0,
            'total_reward': sum(rewards),
            'min_distance': min(distances) if distances else 0,
            'final_distance': distances[-1] if distances else 0,
            'time_span': data[-1].get('simulation_time', 0) - data[0].get('simulation_time', 0)
        }
    else:
        return {
            'total_points': total_points,
            'data_type': 'unknown',
            'first_keys': list(data[0].keys()) if isinstance(data[0], dict) else []
        }