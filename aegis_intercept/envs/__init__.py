"""Gymnasium environments for AegisIntercept

Phase 3 includes both legacy 3DOF and advanced 6DOF environments
with curriculum learning support and enhanced physics simulation.
"""

from .aegis_2d_env import Aegis2DInterceptEnv
from .aegis_3d_env import Aegis3DInterceptEnv  
from .aegis_6dof_env import Aegis6DInterceptEnv, DifficultyMode, ActionMode

__all__ = [
    "Aegis2DInterceptEnv",
    "Aegis3DInterceptEnv", 
    "Aegis6DInterceptEnv",
    "DifficultyMode",
    "ActionMode"
]