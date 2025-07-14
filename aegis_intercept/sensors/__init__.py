"""
Realistic Sensor Systems for AegisIntercept Phase 3

This module implements production-quality radar and sensor systems
for realistic missile defense simulation, replacing the perfect
information system with physics-based sensor modeling.
"""

from .radar_system import (
    RadarSystem, GroundRadar, MissileRadar, RadarConfig,
    RadarDetection, RadarTrack, RadarConstants
)
from .sensor_fusion import (
    SensorFusionSystem, KalmanTracker, TrackState,
    FusionConfig, TrackQuality
)
from .sensor_environment import (
    SensorEnvironment, AtmosphericEffects, WeatherConditions,
    ElectronicWarfare, ClutterModel
)

__all__ = [
    'RadarSystem', 'GroundRadar', 'MissileRadar', 'RadarConfig',
    'RadarDetection', 'RadarTrack', 'RadarConstants',
    'SensorFusionSystem', 'KalmanTracker', 'TrackState',
    'FusionConfig', 'TrackQuality',
    'SensorEnvironment', 'AtmosphericEffects', 'WeatherConditions',
    'ElectronicWarfare', 'ClutterModel'
]