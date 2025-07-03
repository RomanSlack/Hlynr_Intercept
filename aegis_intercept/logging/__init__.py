"""
Logging Module for AegisIntercept Phase 3

This module provides comprehensive logging and export capabilities for
6DOF missile intercept simulations, including trajectory logging,
performance metrics, and Unity-compatible data export.
"""

from .trajectory_logger import (
    TrajectoryLogger,
    TrajectoryPoint,
    EpisodeMetrics,
    LogLevel,
    DataFormat,
    create_trajectory_logger,
    load_trajectory_data,
    analyze_trajectory_file
)

from .export_manager import (
    ExportManager,
    UnityCoordinateSystem,
    ExportFormat,
    UnityVector3,
    UnityQuaternion,
    UnityTransform,
    UnityTrajectoryFrame,
    UnityEpisodeData,
    DashboardMetrics,
    create_export_manager,
    convert_coordinates_for_unity,
    batch_export_episodes
)

__all__ = [
    # Trajectory Logger
    'TrajectoryLogger',
    'TrajectoryPoint',
    'EpisodeMetrics',
    'LogLevel',
    'DataFormat',
    'create_trajectory_logger',
    'load_trajectory_data',
    'analyze_trajectory_file',
    
    # Export Manager
    'ExportManager',
    'UnityCoordinateSystem',
    'ExportFormat',
    'UnityVector3',
    'UnityQuaternion',
    'UnityTransform',
    'UnityTrajectoryFrame',
    'UnityEpisodeData',
    'DashboardMetrics',
    'create_export_manager',
    'convert_coordinates_for_unity',
    'batch_export_episodes'
]