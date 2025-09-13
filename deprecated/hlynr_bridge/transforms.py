"""
Deterministic coordinate transforms between ENU (right-handed) and Unity (left-handed).

This module implements versioned coordinate transformations to ensure deterministic
and reproducible conversion between the Python RL environment (ENU right-handed)
and Unity simulation (left-handed).

Transform version: tfm_v1.0
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TransformVersion:
    """Transform version metadata."""
    version: str
    description: str
    coordinate_mapping: str
    handedness_correction: str


# Supported transform versions
TRANSFORM_VERSIONS = {
    "tfm_v1.0": TransformVersion(
        version="tfm_v1.0",
        description="ENU right-handed to Unity left-handed coordinate transform",
        coordinate_mapping="Unity(x,y,z) = ENU(x,z,y)",
        handedness_correction="Y-axis flip for handedness conversion"
    )
}


class CoordinateTransform:
    """
    Deterministic coordinate transformation between ENU and Unity coordinate systems.
    
    ENU (Python/RL): Right-handed coordinate system
    - X: East (positive = right)
    - Y: North (positive = forward) 
    - Z: Up (positive = upward)
    
    Unity: Left-handed coordinate system
    - X: Right (corresponds to ENU X)
    - Y: Up (corresponds to ENU Z)
    - Z: Forward (corresponds to ENU Y)
    
    Transform: Unity(x,y,z) = ENU(x,z,-y) for handedness correction
    """
    
    def __init__(self, transform_version: str = "tfm_v1.0"):
        """
        Initialize coordinate transform.
        
        Args:
            transform_version: Version identifier for transform
        """
        if transform_version not in TRANSFORM_VERSIONS:
            raise ValueError(f"Unsupported transform version: {transform_version}")
        
        self.version = transform_version
        self.metadata = TRANSFORM_VERSIONS[transform_version]
        
        # Transformation matrices
        # ENU to Unity: [x_u, y_u, z_u] = C_eu * [x_e, y_e, z_e]
        self.C_enu_to_unity = np.array([
            [1.0,  0.0,  0.0],   # Unity X = ENU X (East)
            [0.0,  0.0,  1.0],   # Unity Y = ENU Z (Up)  
            [0.0, -1.0,  0.0]    # Unity Z = -ENU Y (North with handedness flip)
        ], dtype=np.float64)
        
        # Unity to ENU: [x_e, y_e, z_e] = C_ue * [x_u, y_u, z_u]
        self.C_unity_to_enu = self.C_enu_to_unity.T
        
        logger.info(f"Initialized coordinate transform {transform_version}")
    
    def enu_to_unity_position(self, enu_pos: List[float]) -> List[float]:
        """Transform position vector from ENU to Unity coordinates."""
        enu_vec = np.array(enu_pos, dtype=np.float64)
        if len(enu_vec) != 3:
            raise ValueError(f"Position must be 3D vector, got {len(enu_vec)}D")
        
        unity_vec = self.C_enu_to_unity @ enu_vec
        return unity_vec.tolist()
    
    def unity_to_enu_position(self, unity_pos: List[float]) -> List[float]:
        """Transform position vector from Unity to ENU coordinates."""
        unity_vec = np.array(unity_pos, dtype=np.float64)
        if len(unity_vec) != 3:
            raise ValueError(f"Position must be 3D vector, got {len(unity_vec)}D")
        
        enu_vec = self.C_unity_to_enu @ unity_vec
        return enu_vec.tolist()
    
    def enu_to_unity_angular_velocity(self, enu_rates: List[float]) -> List[float]:
        """Transform angular velocity from ENU to Unity coordinates."""
        enu_vec = np.array(enu_rates, dtype=np.float64)
        if len(enu_vec) != 3:
            raise ValueError(f"Angular velocity must be 3D vector, got {len(enu_vec)}D")
        
        unity_vec = self.C_enu_to_unity @ enu_vec
        return unity_vec.tolist()
    
    def unity_to_enu_angular_velocity(self, unity_rates: List[float]) -> List[float]:
        """Transform angular velocity from Unity to ENU coordinates."""
        unity_vec = np.array(unity_rates, dtype=np.float64)
        if len(unity_vec) != 3:
            raise ValueError(f"Angular velocity must be 3D vector, got {len(unity_vec)}D")
        
        enu_vec = self.C_unity_to_enu @ unity_vec
        return enu_vec.tolist()
    
    def transform_state_unity_to_enu(self, unity_state: Dict[str, Any]) -> Dict[str, Any]:
        """Transform complete state from Unity to ENU coordinates."""
        enu_state = {}
        
        if 'pos_m' in unity_state:
            enu_state['pos_m'] = self.unity_to_enu_position(unity_state['pos_m'])
        
        if 'vel_mps' in unity_state:
            enu_state['vel_mps'] = self.unity_to_enu_position(unity_state['vel_mps'])  # Velocity transforms like position
        
        if 'ang_vel_radps' in unity_state:
            enu_state['ang_vel_radps'] = self.unity_to_enu_angular_velocity(unity_state['ang_vel_radps'])
        
        if 'quat_wxyz' in unity_state:
            # Quaternion handling - simplified for now
            enu_state['quat_wxyz'] = unity_state['quat_wxyz']  # TODO: Proper quaternion transformation
        
        return enu_state
    
    def transform_state_enu_to_unity(self, enu_state: Dict[str, Any]) -> Dict[str, Any]:
        """Transform complete state from ENU to Unity coordinates."""
        unity_state = {}
        
        if 'pos_m' in enu_state:
            unity_state['pos_m'] = self.enu_to_unity_position(enu_state['pos_m'])
        
        if 'vel_mps' in enu_state:
            unity_state['vel_mps'] = self.enu_to_unity_position(enu_state['vel_mps'])  # Velocity transforms like position
        
        if 'ang_vel_radps' in enu_state:
            unity_state['ang_vel_radps'] = self.enu_to_unity_angular_velocity(enu_state['ang_vel_radps'])
        
        if 'quat_wxyz' in enu_state:
            # Quaternion handling - simplified for now
            unity_state['quat_wxyz'] = enu_state['quat_wxyz']  # TODO: Proper quaternion transformation
        
        return unity_state


# Global transform instance
_transform_instance = None


def get_transform(transform_version: str = "tfm_v1.0") -> CoordinateTransform:
    """Get global transform instance."""
    global _transform_instance
    
    if _transform_instance is None or _transform_instance.version != transform_version:
        _transform_instance = CoordinateTransform(transform_version)
    
    return _transform_instance


def validate_transform_version(transform_version: str) -> bool:
    """Validate transform version is supported."""
    return transform_version in TRANSFORM_VERSIONS


# Example usage
if __name__ == "__main__":
    # Test coordinate transformations
    transform = get_transform("tfm_v1.0")
    
    print("Coordinate Transform Test")
    print("=" * 40)
    
    # Test position transformation
    enu_pos = [100.0, 200.0, 50.0]  # 100m East, 200m North, 50m Up
    unity_pos = transform.enu_to_unity_position(enu_pos)
    recovered_enu = transform.unity_to_enu_position(unity_pos)
    
    print(f"ENU position: {enu_pos}")
    print(f"Unity position: {unity_pos}")
    print(f"Recovered ENU: {recovered_enu}")
    print(f"Round-trip error: {np.array(enu_pos) - np.array(recovered_enu)}")
    
    # Test angular velocity transformation
    enu_rates = [0.1, 0.2, 0.05]  # rad/s
    unity_rates = transform.enu_to_unity_angular_velocity(enu_rates)
    recovered_rates = transform.unity_to_enu_angular_velocity(unity_rates)
    
    print(f"\\nENU angular rates: {enu_rates}")
    print(f"Unity angular rates: {unity_rates}")
    print(f"Recovered ENU rates: {recovered_rates}")
    print(f"Round-trip error: {np.array(enu_rates) - np.array(recovered_rates)}")
    
    print("\\nTransform test completed successfully!")