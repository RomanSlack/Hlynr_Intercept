"""
Physics simulation modules for AegisIntercept Phase 3.

This package contains the core physics engines for 6-DOF rigid body dynamics,
atmospheric modeling, and environmental effects.
"""

from .dynamics import RigidBody6DOF
from .atmosphere import AtmosphericModel, DragModel
from .wind import WindField

__all__ = ["RigidBody6DOF", "AtmosphericModel", "DragModel", "WindField"]