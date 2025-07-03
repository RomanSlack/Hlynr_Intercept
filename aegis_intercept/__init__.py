"""AegisIntercept - Nuclear-Defense Path-Finding Simulator

Advanced 6DOF missile intercept simulation system with curriculum learning,
realistic physics, and comprehensive logging capabilities.

Phase 3 Features:
- 6DOF physics engine with quaternion-based orientation
- Enhanced environment with 29-dimensional state space
- Curriculum learning from 3DOF to advanced 6DOF scenarios
- Comprehensive trajectory logging and Unity export
- Advanced adversary system with sophisticated evasive maneuvers
"""

__version__ = "0.3.0"  # Phase 3 version

# Import main modules
from . import envs
from . import utils
from . import rendering
from . import curriculum
from . import logging
from . import adversary

# Version and feature information
PHASE_3_FEATURES = [
    "6DOF Physics Engine",
    "Enhanced Environment (aegis_6dof_env)",
    "Curriculum Learning System", 
    "Advanced Trajectory Logging",
    "Unity Export Manager",
    "Enhanced Adversary System"
]

def get_version_info():
    """Get detailed version and feature information"""
    return {
        "version": __version__,
        "phase": 3,
        "features": PHASE_3_FEATURES,
        "environments": ["Aegis3DIntercept-v0", "Aegis6DIntercept-v0"],
        "physics_engines": ["3DOF Legacy", "6DOF Advanced"],
        "curriculum_phases": 5,
        "adversary_patterns": 10
    }