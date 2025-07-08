"""
Unity Exporter for AegisIntercept Phase 3.

This module exports trajectory data in Unity-friendly format with proper
coordinate system conversion (left-handed, Y-up) and optimized data structure.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import time
from pyquaternion import Quaternion

from .trajectory_logger import TrajectoryPoint, TrajectoryLogger


class UnityExporter:
    """
    Unity-compatible trajectory exporter.
    
    This class converts simulation trajectory data to Unity-friendly format,
    handling coordinate system conversion and data optimization for real-time
    playback in Unity visualization.
    """
    
    def __init__(self,
                 coordinate_conversion: bool = True,
                 optimize_for_unity: bool = True,
                 decimation_factor: int = 1):
        """
        Initialize Unity exporter.
        
        Args:
            coordinate_conversion: Convert from right-handed to left-handed coordinates
            optimize_for_unity: Optimize data structure for Unity
            decimation_factor: Reduce data points by factor (1 = no reduction)
        """
        self.coordinate_conversion = coordinate_conversion
        self.optimize_for_unity = optimize_for_unity
        self.decimation_factor = max(1, decimation_factor)
    
    def convert_coordinate_system(self, 
                                position: List[float], 
                                rotation: Optional[List[float]] = None) -> Tuple[List[float], Optional[List[float]]]:
        """
        Convert coordinates from right-handed Z-up to left-handed Y-up.
        
        Args:
            position: Position [x, y, z] in right-handed coordinates
            rotation: Optional quaternion [w, x, y, z] in right-handed coordinates
            
        Returns:
            Tuple of (converted_position, converted_rotation)
        """
        if not self.coordinate_conversion:
            return position, rotation
        
        # Convert position: X → X, Y → Z, Z → Y, negate X for left-handed
        unity_position = [-position[0], position[2], position[1]]
        
        unity_rotation = None
        if rotation is not None:
            # Convert quaternion for coordinate system change
            # From right-handed Z-up to left-handed Y-up
            w, x, y, z = rotation
            unity_rotation = [w, -x, z, y]  # Adjust quaternion components
        
        return unity_position, unity_rotation
    
    def create_unity_trajectory(self, 
                               trajectory_data: List[TrajectoryPoint],
                               metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Unity-compatible trajectory data structure.
        
        Args:
            trajectory_data: List of trajectory points
            metadata: Trajectory metadata
            
        Returns:
            Unity-compatible data structure
        """
        # Decimate trajectory data if requested
        if self.decimation_factor > 1:
            trajectory_data = trajectory_data[::self.decimation_factor]
        
        # Convert trajectory points
        unity_trajectory = {
            "metadata": self._create_unity_metadata(metadata),
            "interceptor": self._create_unity_object_data(trajectory_data, "interceptor"),
            "adversary": self._create_unity_object_data(trajectory_data, "adversary"),
            "environment": self._create_unity_environment_data(trajectory_data),
            "events": self._create_unity_events(trajectory_data),
            "performance": self._create_performance_data(trajectory_data)
        }
        
        return unity_trajectory
    
    def _create_unity_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create Unity-compatible metadata."""
        unity_metadata = {
            "version": "1.0",
            "coordinate_system": "left_handed_y_up" if self.coordinate_conversion else "right_handed_z_up",
            "time_step": metadata.get("simulation_time_step", 0.02),
            "total_frames": metadata.get("total_steps", 0),
            "duration_seconds": metadata.get("duration_seconds", 0.0),
            "success": metadata.get("success", False),
            "episode_number": metadata.get("episode_number", 0),
            "final_distance": metadata.get("final_distance", 0.0),
            "episode_reward": metadata.get("episode_reward", 0.0),
            "creation_time": time.time(),
            "decimation_factor": self.decimation_factor
        }
        
        return unity_metadata
    
    def _create_unity_object_data(self, 
                                 trajectory_data: List[TrajectoryPoint], 
                                 object_type: str) -> Dict[str, Any]:
        """Create Unity object data for interceptor or adversary."""
        
        # Extract data based on object type
        if object_type == "interceptor":
            positions = [point.interceptor_position for point in trajectory_data]
            orientations = [point.interceptor_orientation for point in trajectory_data]
            velocities = [point.interceptor_velocity for point in trajectory_data]
            angular_velocities = [point.interceptor_angular_velocity for point in trajectory_data]
            thrust_forces = [point.interceptor_thrust for point in trajectory_data]
            torques = [point.interceptor_torque for point in trajectory_data]
        else:  # adversary
            positions = [point.adversary_position for point in trajectory_data]
            orientations = [point.adversary_orientation for point in trajectory_data]
            velocities = [point.adversary_velocity for point in trajectory_data]
            angular_velocities = [point.adversary_angular_velocity for point in trajectory_data]
            thrust_forces = [point.adversary_thrust for point in trajectory_data]
            torques = [point.adversary_torque for point in trajectory_data]
        
        # Convert coordinate systems
        unity_positions = []
        unity_orientations = []
        unity_velocities = []
        
        for i, pos in enumerate(positions):
            unity_pos, unity_rot = self.convert_coordinate_system(pos, orientations[i])
            unity_vel, _ = self.convert_coordinate_system(velocities[i])
            
            unity_positions.append(unity_pos)
            unity_orientations.append(unity_rot)
            unity_velocities.append(unity_vel)
        
        # Create Unity object data structure
        object_data = {
            "positions": unity_positions,
            "rotations": unity_orientations,
            "velocities": unity_velocities,
            "angular_velocities": angular_velocities,  # Keep as-is for now
            "thrust_forces": thrust_forces,
            "torques": torques,
            "frame_count": len(trajectory_data),
            "object_properties": {
                "mass": 150.0 if object_type == "interceptor" else 200.0,
                "max_thrust": 5000.0 if object_type == "interceptor" else 3000.0,
                "color": [0.2, 0.6, 1.0] if object_type == "interceptor" else [1.0, 0.3, 0.2],
                "trail_enabled": True,
                "trail_length": 50,
                "model_scale": [1.0, 1.0, 1.0]
            }
        }
        
        return object_data
    
    def _create_unity_environment_data(self, trajectory_data: List[TrajectoryPoint]) -> Dict[str, Any]:
        """Create Unity environment data."""
        # Extract environment data
        wind_velocities = [point.wind_velocity for point in trajectory_data]
        atmospheric_densities = [point.atmospheric_density for point in trajectory_data]
        times = [point.time for point in trajectory_data]
        
        # Convert wind velocities to Unity coordinates
        unity_wind_velocities = []
        for wind_vel in wind_velocities:
            unity_wind, _ = self.convert_coordinate_system(wind_vel)
            unity_wind_velocities.append(unity_wind)
        
        environment_data = {
            "wind_velocities": unity_wind_velocities,
            "atmospheric_densities": atmospheric_densities,
            "time_stamps": times,
            "environment_settings": {
                "sky_color": [0.5, 0.7, 1.0],
                "ground_color": [0.3, 0.5, 0.2],
                "fog_enabled": True,
                "fog_density": 0.01,
                "lighting": {
                    "sun_direction": [-0.3, -0.7, 0.6],
                    "ambient_light": [0.4, 0.4, 0.4]
                }
            }
        }
        
        return environment_data
    
    def _create_unity_events(self, trajectory_data: List[TrajectoryPoint]) -> List[Dict[str, Any]]:
        """Create Unity event markers for important trajectory events."""
        events = []
        
        # Find minimum distance event
        min_distance = float('inf')
        min_distance_time = 0.0
        min_distance_frame = 0
        
        for i, point in enumerate(trajectory_data):
            if point.intercept_distance < min_distance:
                min_distance = point.intercept_distance
                min_distance_time = point.time
                min_distance_frame = i
        
        events.append({
            "type": "minimum_distance",
            "time": min_distance_time,
            "frame": min_distance_frame,
            "data": {
                "distance": min_distance,
                "description": f"Closest approach: {min_distance:.2f}m"
            },
            "visual": {
                "color": [1.0, 1.0, 0.0],
                "duration": 2.0,
                "effect": "highlight"
            }
        })
        
        # Find episode end event
        final_point = trajectory_data[-1]
        events.append({
            "type": "episode_end",
            "time": final_point.time,
            "frame": len(trajectory_data) - 1,
            "data": {
                "success": final_point.success,
                "final_distance": final_point.intercept_distance,
                "description": "Mission Complete" if final_point.success else "Mission Failed"
            },
            "visual": {
                "color": [0.0, 1.0, 0.0] if final_point.success else [1.0, 0.0, 0.0],
                "duration": 3.0,
                "effect": "explosion" if final_point.success else "fade"
            }
        })
        
        # Find fuel depletion events
        for i, point in enumerate(trajectory_data):
            if i > 0 and trajectory_data[i-1].fuel_remaining > 0.1 and point.fuel_remaining <= 0.1:
                events.append({
                    "type": "low_fuel",
                    "time": point.time,
                    "frame": i,
                    "data": {
                        "fuel_remaining": point.fuel_remaining,
                        "description": "Low fuel warning"
                    },
                    "visual": {
                        "color": [1.0, 0.5, 0.0],
                        "duration": 1.0,
                        "effect": "warning"
                    }
                })
        
        return events
    
    def _create_performance_data(self, trajectory_data: List[TrajectoryPoint]) -> Dict[str, Any]:
        """Create performance analysis data for Unity display."""
        if not trajectory_data:
            return {}
        
        # Calculate performance metrics
        distances = [point.intercept_distance for point in trajectory_data]
        rewards = [point.step_reward for point in trajectory_data]
        fuel_levels = [point.fuel_remaining for point in trajectory_data]
        
        performance_data = {
            "distance_profile": distances,
            "reward_profile": rewards,
            "fuel_profile": fuel_levels,
            "statistics": {
                "min_distance": min(distances),
                "max_distance": max(distances),
                "mean_distance": float(np.mean(distances)),
                "total_reward": sum(rewards),
                "fuel_consumed": 1.0 - fuel_levels[-1],
                "success_rate": 1.0 if trajectory_data[-1].success else 0.0
            },
            "visualization_settings": {
                "distance_chart_color": [0.0, 0.8, 1.0],
                "reward_chart_color": [0.0, 1.0, 0.0],
                "fuel_chart_color": [1.0, 0.6, 0.0],
                "chart_background": [0.1, 0.1, 0.1, 0.8],
                "update_frequency": 0.1
            }
        }
        
        return performance_data
    
    def export_unity_json(self, 
                         trajectory_logger: TrajectoryLogger,
                         filename: str,
                         include_metadata: bool = True) -> bool:
        """
        Export trajectory data to Unity-compatible JSON file.
        
        Args:
            trajectory_logger: TrajectoryLogger instance with data
            filename: Output filename
            include_metadata: Include additional metadata
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Get trajectory data
            trajectory_data = trajectory_logger.current_trajectory
            metadata = trajectory_logger.trajectory_metadata
            
            if not trajectory_data:
                print("Warning: No trajectory data to export")
                return False
            
            # Create Unity data structure
            unity_data = self.create_unity_trajectory(trajectory_data, metadata)
            
            # Add optional metadata
            if include_metadata:
                unity_data["export_info"] = {
                    "exporter_version": "1.0",
                    "export_time": time.time(),
                    "source_format": "AegisIntercept_Phase3",
                    "coordinate_conversion": self.coordinate_conversion,
                    "decimation_factor": self.decimation_factor,
                    "logging_stats": trajectory_logger.get_logging_performance()
                }
            
            # Write to file
            with open(filename, 'w') as f:
                json.dump(unity_data, f, indent=2)
            
            print(f"Unity trajectory exported to: {filename}")
            print(f"Frames exported: {len(trajectory_data)}")
            print(f"Duration: {unity_data['metadata']['duration_seconds']:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"Error exporting Unity trajectory: {e}")
            return False
    
    def export_multiple_episodes(self,
                                episode_data: List[Tuple[TrajectoryLogger, str]],
                                output_filename: str) -> bool:
        """
        Export multiple episodes to a single Unity-compatible file.
        
        Args:
            episode_data: List of (TrajectoryLogger, episode_name) tuples
            output_filename: Output filename
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            multi_episode_data = {
                "version": "1.0",
                "format": "multi_episode",
                "episode_count": len(episode_data),
                "episodes": {}
            }
            
            for trajectory_logger, episode_name in episode_data:
                if trajectory_logger.current_trajectory:
                    unity_data = self.create_unity_trajectory(
                        trajectory_logger.current_trajectory,
                        trajectory_logger.trajectory_metadata
                    )
                    multi_episode_data["episodes"][episode_name] = unity_data
            
            # Write to file
            with open(output_filename, 'w') as f:
                json.dump(multi_episode_data, f, indent=2)
            
            print(f"Multi-episode Unity data exported to: {output_filename}")
            print(f"Episodes exported: {len(multi_episode_data['episodes'])}")
            
            return True
            
        except Exception as e:
            print(f"Error exporting multi-episode data: {e}")
            return False
    
    def create_unity_config(self, output_filename: str) -> bool:
        """
        Create Unity configuration file for the visualization system.
        
        Args:
            output_filename: Configuration file name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            unity_config = {
                "visualization_settings": {
                    "coordinate_system": "left_handed_y_up" if self.coordinate_conversion else "right_handed_z_up",
                    "default_camera_position": [0, 500, -1000],
                    "default_camera_target": [0, 0, 0],
                    "playback_speed": 1.0,
                    "auto_follow_interceptor": True,
                    "show_trajectories": True,
                    "show_velocity_vectors": True,
                    "show_performance_charts": True
                },
                "object_settings": {
                    "interceptor": {
                        "color": [0.2, 0.6, 1.0],
                        "scale": [2.0, 2.0, 2.0],
                        "trail_color": [0.4, 0.8, 1.0],
                        "trail_width": 0.5
                    },
                    "adversary": {
                        "color": [1.0, 0.3, 0.2],
                        "scale": [2.0, 2.0, 2.0],
                        "trail_color": [1.0, 0.5, 0.3],
                        "trail_width": 0.5
                    },
                    "target": {
                        "color": [0.0, 1.0, 0.0],
                        "scale": [5.0, 5.0, 5.0],
                        "glow_enabled": True
                    }
                },
                "ui_settings": {
                    "show_fps": True,
                    "show_time": True,
                    "show_distance": True,
                    "show_fuel": True,
                    "show_controls": True
                }
            }
            
            with open(output_filename, 'w') as f:
                json.dump(unity_config, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error creating Unity config: {e}")
            return False