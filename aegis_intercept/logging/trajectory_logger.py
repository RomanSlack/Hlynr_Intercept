"""
Trajectory Logger for AegisIntercept Phase 3.

This module provides comprehensive logging of simulation trajectories,
including step-wise state data, rewards, and performance metrics.
"""

import csv
import json
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import time
from dataclasses import dataclass, asdict
from pyquaternion import Quaternion


@dataclass
class TrajectoryPoint:
    """Single trajectory data point."""
    time: float
    step: int
    
    # Interceptor state
    interceptor_position: List[float]
    interceptor_velocity: List[float]
    interceptor_orientation: List[float]  # [w, x, y, z] quaternion
    interceptor_angular_velocity: List[float]
    
    # Adversary state
    adversary_position: List[float]
    adversary_velocity: List[float]
    adversary_orientation: List[float]  # [w, x, y, z] quaternion
    adversary_angular_velocity: List[float]
    
    # Environment state
    wind_velocity: List[float]
    atmospheric_density: float
    
    # Control inputs
    interceptor_thrust: List[float]
    interceptor_torque: List[float]
    adversary_thrust: List[float]
    adversary_torque: List[float]
    
    # Derived quantities
    intercept_distance: float
    target_distance: float
    relative_velocity: List[float]
    
    # Reward components
    step_reward: float
    distance_reward: float
    fuel_penalty: float
    control_penalty: float
    
    # Mission status
    fuel_remaining: float
    episode_complete: bool
    success: bool


class TrajectoryLogger:
    """
    Comprehensive trajectory logging system.
    
    This class logs detailed trajectory data during simulation runs,
    providing both high-performance logging for training and detailed
    logging for analysis. It supports CSV export and Unity-compatible
    JSON format.
    """
    
    def __init__(self,
                 enable_detailed_logging: bool = True,
                 enable_performance_mode: bool = False,
                 log_frequency: int = 1,
                 max_trajectory_points: int = 10000):
        """
        Initialize trajectory logger.
        
        Args:
            enable_detailed_logging: Enable detailed state logging
            enable_performance_mode: Optimize for training performance
            log_frequency: Log every N steps (1 = every step)
            max_trajectory_points: Maximum points to store in memory
        """
        self.enable_detailed_logging = enable_detailed_logging
        self.enable_performance_mode = enable_performance_mode
        self.log_frequency = log_frequency
        self.max_trajectory_points = max_trajectory_points
        
        # Current trajectory data
        self.current_trajectory: List[TrajectoryPoint] = []
        self.trajectory_metadata: Dict[str, Any] = {}
        
        # Episode tracking
        self.current_episode = 0
        self.total_episodes_logged = 0
        
        # Performance tracking
        self.logging_stats = {
            'total_points_logged': 0,
            'logging_time_ms': 0.0,
            'average_log_time_ms': 0.0
        }
        
        # Buffer for batch operations
        self.log_buffer: List[TrajectoryPoint] = []
        self.buffer_size = 100 if enable_performance_mode else 10
    
    def start_episode(self,
                     episode_metadata: Optional[Dict[str, Any]] = None):
        """
        Start logging a new episode.
        
        Args:
            episode_metadata: Optional metadata for the episode
        """
        # Clear current trajectory
        self.current_trajectory.clear()
        
        # Set metadata
        self.trajectory_metadata = {
            'episode_number': self.current_episode,
            'start_time': time.time(),
            'simulation_time_step': 0.02,  # Default time step
            'total_steps': 0,
            'success': False,
            'final_distance': 0.0,
            'fuel_consumed': 0.0,
            'episode_reward': 0.0
        }
        
        if episode_metadata:
            self.trajectory_metadata.update(episode_metadata)
    
    def log_step(self,
                step: int,
                simulation_time: float,
                interceptor_state: Dict[str, Any],
                adversary_state: Dict[str, Any],
                environment_state: Dict[str, Any],
                control_inputs: Dict[str, Any],
                reward_info: Dict[str, Any],
                mission_status: Dict[str, Any]):
        """
        Log a single simulation step.
        
        Args:
            step: Current step number
            simulation_time: Current simulation time
            interceptor_state: Interceptor state data
            adversary_state: Adversary state data
            environment_state: Environment data
            control_inputs: Control input data
            reward_info: Reward breakdown
            mission_status: Mission status information
        """
        # Skip logging based on frequency
        if step % self.log_frequency != 0:
            return
        
        # Skip detailed logging if in performance mode
        if self.enable_performance_mode and not self.enable_detailed_logging:
            return
        
        start_log_time = time.perf_counter()
        
        try:
            # Create trajectory point
            trajectory_point = self._create_trajectory_point(
                step, simulation_time, interceptor_state, adversary_state,
                environment_state, control_inputs, reward_info, mission_status
            )
            
            # Add to buffer or directly to trajectory
            if self.enable_performance_mode:
                self.log_buffer.append(trajectory_point)
                
                # Flush buffer when full
                if len(self.log_buffer) >= self.buffer_size:
                    self._flush_buffer()
            else:
                self.current_trajectory.append(trajectory_point)
            
            # Limit trajectory size
            if len(self.current_trajectory) > self.max_trajectory_points:
                self.current_trajectory.pop(0)  # Remove oldest point
            
            # Update stats
            self.logging_stats['total_points_logged'] += 1
            
        except Exception as e:
            print(f"Warning: Trajectory logging failed at step {step}: {e}")
        
        # Track logging performance
        log_time_ms = (time.perf_counter() - start_log_time) * 1000
        self.logging_stats['logging_time_ms'] += log_time_ms
        
        if self.logging_stats['total_points_logged'] > 0:
            self.logging_stats['average_log_time_ms'] = (
                self.logging_stats['logging_time_ms'] / 
                self.logging_stats['total_points_logged']
            )
    
    def _create_trajectory_point(self,
                                step: int,
                                simulation_time: float,
                                interceptor_state: Dict[str, Any],
                                adversary_state: Dict[str, Any],
                                environment_state: Dict[str, Any],
                                control_inputs: Dict[str, Any],
                                reward_info: Dict[str, Any],
                                mission_status: Dict[str, Any]) -> TrajectoryPoint:
        """Create a trajectory point from state data."""
        
        # Helper function to convert numpy arrays to lists
        def to_list(arr):
            if isinstance(arr, np.ndarray):
                return arr.tolist()
            elif isinstance(arr, (list, tuple)):
                return list(arr)
            else:
                return [float(arr)] if not hasattr(arr, '__iter__') else list(arr)
        
        # Extract interceptor state
        interceptor_pos = to_list(interceptor_state.get('position', [0, 0, 0]))
        interceptor_vel = to_list(interceptor_state.get('velocity', [0, 0, 0]))
        interceptor_quat = to_list(interceptor_state.get('orientation', [1, 0, 0, 0]))
        interceptor_omega = to_list(interceptor_state.get('angular_velocity', [0, 0, 0]))
        
        # Extract adversary state
        adversary_pos = to_list(adversary_state.get('position', [0, 0, 0]))
        adversary_vel = to_list(adversary_state.get('velocity', [0, 0, 0]))
        adversary_quat = to_list(adversary_state.get('orientation', [1, 0, 0, 0]))
        adversary_omega = to_list(adversary_state.get('angular_velocity', [0, 0, 0]))
        
        # Calculate derived quantities
        intercept_distance = np.linalg.norm(np.array(interceptor_pos) - np.array(adversary_pos))
        target_distance = np.linalg.norm(np.array(adversary_pos))  # Target at origin
        relative_velocity = to_list(np.array(adversary_vel) - np.array(interceptor_vel))
        
        return TrajectoryPoint(
            time=simulation_time,
            step=step,
            
            # Interceptor state
            interceptor_position=interceptor_pos,
            interceptor_velocity=interceptor_vel,
            interceptor_orientation=interceptor_quat,
            interceptor_angular_velocity=interceptor_omega,
            
            # Adversary state
            adversary_position=adversary_pos,
            adversary_velocity=adversary_vel,
            adversary_orientation=adversary_quat,
            adversary_angular_velocity=adversary_omega,
            
            # Environment
            wind_velocity=to_list(environment_state.get('wind_velocity', [0, 0, 0])),
            atmospheric_density=float(environment_state.get('atmospheric_density', 1.225)),
            
            # Control inputs
            interceptor_thrust=to_list(control_inputs.get('interceptor_thrust', [0, 0, 0])),
            interceptor_torque=to_list(control_inputs.get('interceptor_torque', [0, 0, 0])),
            adversary_thrust=to_list(control_inputs.get('adversary_thrust', [0, 0, 0])),
            adversary_torque=to_list(control_inputs.get('adversary_torque', [0, 0, 0])),
            
            # Derived quantities
            intercept_distance=intercept_distance,
            target_distance=target_distance,
            relative_velocity=relative_velocity,
            
            # Reward components
            step_reward=float(reward_info.get('step_reward', 0.0)),
            distance_reward=float(reward_info.get('distance_reward', 0.0)),
            fuel_penalty=float(reward_info.get('fuel_penalty', 0.0)),
            control_penalty=float(reward_info.get('control_penalty', 0.0)),
            
            # Mission status
            fuel_remaining=float(mission_status.get('fuel_remaining', 1.0)),
            episode_complete=bool(mission_status.get('episode_complete', False)),
            success=bool(mission_status.get('success', False))
        )
    
    def _flush_buffer(self):
        """Flush the logging buffer to the main trajectory."""
        self.current_trajectory.extend(self.log_buffer)
        self.log_buffer.clear()
    
    def end_episode(self,
                   final_metadata: Optional[Dict[str, Any]] = None):
        """
        End the current episode and finalize trajectory data.
        
        Args:
            final_metadata: Final episode metadata
        """
        # Flush any remaining buffer
        if self.log_buffer:
            self._flush_buffer()
        
        # Update metadata
        if final_metadata:
            self.trajectory_metadata.update(final_metadata)
        
        self.trajectory_metadata['end_time'] = time.time()
        self.trajectory_metadata['total_steps'] = len(self.current_trajectory)
        self.trajectory_metadata['duration_seconds'] = (
            self.trajectory_metadata['end_time'] - self.trajectory_metadata['start_time']
        )
        
        # Extract final statistics
        if self.current_trajectory:
            final_point = self.current_trajectory[-1]
            self.trajectory_metadata['final_distance'] = final_point.intercept_distance
            self.trajectory_metadata['success'] = final_point.success
            self.trajectory_metadata['episode_reward'] = sum(
                point.step_reward for point in self.current_trajectory
            )
        
        self.total_episodes_logged += 1
        self.current_episode += 1
    
    def export_csv(self, filename: str) -> bool:
        """
        Export current trajectory to CSV file.
        
        Args:
            filename: Output CSV filename
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            with open(filename, 'w', newline='') as csvfile:
                if not self.current_trajectory:
                    return False
                
                # Get field names from the first trajectory point
                fieldnames = list(asdict(self.current_trajectory[0]).keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header
                writer.writeheader()
                
                # Write trajectory data
                for point in self.current_trajectory:
                    writer.writerow(asdict(point))
            
            return True
            
        except Exception as e:
            print(f"Error exporting CSV: {e}")
            return False
    
    def export_json(self, filename: str) -> bool:
        """
        Export current trajectory to JSON file.
        
        Args:
            filename: Output JSON filename
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            export_data = {
                'metadata': self.trajectory_metadata,
                'trajectory': [asdict(point) for point in self.current_trajectory],
                'logging_stats': self.logging_stats
            }
            
            with open(filename, 'w') as jsonfile:
                json.dump(export_data, jsonfile, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error exporting JSON: {e}")
            return False
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Get summary statistics for current trajectory."""
        if not self.current_trajectory:
            return {}
        
        # Calculate summary statistics
        distances = [point.intercept_distance for point in self.current_trajectory]
        rewards = [point.step_reward for point in self.current_trajectory]
        fuel_levels = [point.fuel_remaining for point in self.current_trajectory]
        
        summary = {
            'total_points': len(self.current_trajectory),
            'duration': self.current_trajectory[-1].time - self.current_trajectory[0].time,
            'success': self.current_trajectory[-1].success,
            'final_distance': self.current_trajectory[-1].intercept_distance,
            'min_distance': min(distances),
            'max_distance': max(distances),
            'total_reward': sum(rewards),
            'fuel_consumed': 1.0 - self.current_trajectory[-1].fuel_remaining,
            'average_reward_per_step': np.mean(rewards),
            'distance_statistics': {
                'mean': np.mean(distances),
                'std': np.std(distances),
                'min': min(distances),
                'max': max(distances)
            }
        }
        
        return summary
    
    def clear_trajectory(self):
        """Clear current trajectory data."""
        self.current_trajectory.clear()
        self.log_buffer.clear()
        self.trajectory_metadata.clear()
    
    def get_logging_performance(self) -> Dict[str, float]:
        """Get logging performance statistics."""
        return self.logging_stats.copy()
    
    def set_performance_mode(self, enabled: bool):
        """Enable or disable performance mode."""
        self.enable_performance_mode = enabled
        if enabled:
            self.buffer_size = 100
            self.enable_detailed_logging = False
        else:
            self.buffer_size = 10
            self.enable_detailed_logging = True