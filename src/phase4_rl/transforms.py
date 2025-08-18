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
        """
        Transform position vector from ENU to Unity coordinates.
        
        Args:
            enu_pos: Position [x,y,z] in ENU coordinates (meters)
            
        Returns:
            Position [x,y,z] in Unity coordinates (meters)
        """
        enu_vec = np.array(enu_pos, dtype=np.float64)
        if len(enu_vec) != 3:
            raise ValueError(f"Position must be 3D vector, got {len(enu_vec)}D")
        
        unity_vec = self.C_enu_to_unity @ enu_vec
        return unity_vec.tolist()
    
    def unity_to_enu_position(self, unity_pos: List[float]) -> List[float]:
        """
        Transform position vector from Unity to ENU coordinates.
        
        Args:
            unity_pos: Position [x,y,z] in Unity coordinates (meters)
            
        Returns:
            Position [x,y,z] in ENU coordinates (meters)
        """
        unity_vec = np.array(unity_pos, dtype=np.float64)
        if len(unity_vec) != 3:
            raise ValueError(f"Position must be 3D vector, got {len(unity_vec)}D")
        
        enu_vec = self.C_unity_to_enu @ unity_vec
        return enu_vec.tolist()
    
    def enu_to_unity_quaternion(self, enu_quat: List[float]) -> List[float]:
        """
        Transform quaternion from ENU to Unity coordinates.
        
        This performs a change of basis transformation:
        q_unity = q_transform * q_enu * q_transform_inverse
        
        Args:
            enu_quat: Quaternion [w,x,y,z] in ENU frame
            
        Returns:
            Quaternion [w,x,y,z] in Unity frame
        """
        if len(enu_quat) != 4:
            raise ValueError(f"Quaternion must be 4D, got {len(enu_quat)}D")
        
        # Convert to rotation matrix
        R_enu = self._quaternion_to_matrix(enu_quat)
        
        # Apply coordinate transformation: R_unity = C * R_enu * C^T
        R_unity = self.C_enu_to_unity @ R_enu @ self.C_enu_to_unity.T
        
        # Convert back to quaternion and normalize
        unity_quat = self._matrix_to_quaternion(R_unity)
        unity_quat = self._normalize_quaternion(unity_quat)
        
        return unity_quat
    
    def unity_to_enu_quaternion(self, unity_quat: List[float]) -> List[float]:
        """
        Transform quaternion from Unity to ENU coordinates.
        
        Args:
            unity_quat: Quaternion [w,x,y,z] in Unity frame
            
        Returns:
            Quaternion [w,x,y,z] in ENU frame
        """
        if len(unity_quat) != 4:
            raise ValueError(f"Quaternion must be 4D, got {len(unity_quat)}D")
        
        # Convert to rotation matrix
        R_unity = self._quaternion_to_matrix(unity_quat)
        
        # Apply inverse coordinate transformation: R_enu = C^T * R_unity * C
        R_enu = self.C_unity_to_enu @ R_unity @ self.C_unity_to_enu.T
        
        # Convert back to quaternion and normalize
        enu_quat = self._matrix_to_quaternion(R_enu)
        enu_quat = self._normalize_quaternion(enu_quat)
        
        return enu_quat
    
    def enu_to_unity_velocity(self, enu_vel: List[float]) -> List[float]:
        """Transform velocity vector from ENU to Unity coordinates."""
        return self.enu_to_unity_position(enu_vel)  # Same as position transform
    
    def unity_to_enu_velocity(self, unity_vel: List[float]) -> List[float]:
        """Transform velocity vector from Unity to ENU coordinates."""
        return self.unity_to_enu_position(unity_vel)  # Same as position transform
    
    def enu_to_unity_angular_velocity(self, enu_angvel: List[float]) -> List[float]:
        """Transform angular velocity vector from ENU to Unity coordinates."""
        return self.enu_to_unity_position(enu_angvel)  # Same as vector transform
    
    def unity_to_enu_angular_velocity(self, unity_angvel: List[float]) -> List[float]:
        """Transform angular velocity vector from Unity to ENU coordinates."""
        return self.unity_to_enu_position(unity_angvel)  # Same as vector transform
    
    def _quaternion_to_matrix(self, quat: List[float]) -> np.ndarray:
        """
        Convert quaternion [w,x,y,z] to 3x3 rotation matrix.
        
        Args:
            quat: Quaternion [w,x,y,z]
            
        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = quat
        
        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm < 1e-8:
            raise ValueError("Invalid quaternion: zero magnitude")
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Convert to rotation matrix
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ], dtype=np.float64)
        
        return R
    
    def _matrix_to_quaternion(self, R: np.ndarray) -> List[float]:
        """
        Convert 3x3 rotation matrix to quaternion [w,x,y,z].
        
        Uses Shepperd's method for numerical stability.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Quaternion [w,x,y,z]
        """
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * w
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * x
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * y
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * z
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return [w, x, y, z]
    
    def _normalize_quaternion(self, quat: List[float]) -> List[float]:
        """Normalize quaternion to unit magnitude."""
        w, x, y, z = quat
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        
        if norm < 1e-8:
            raise ValueError("Cannot normalize zero-magnitude quaternion")
        
        return [w/norm, x/norm, y/norm, z/norm]
    
    def transform_state_enu_to_unity(self, enu_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform complete state from ENU to Unity coordinates.
        
        Args:
            enu_state: State dict with ENU coordinates
            
        Returns:
            State dict with Unity coordinates
        """
        unity_state = {}
        
        # Transform position
        if 'pos_m' in enu_state:
            unity_state['pos_m'] = self.enu_to_unity_position(enu_state['pos_m'])
        
        # Transform velocity
        if 'vel_mps' in enu_state:
            unity_state['vel_mps'] = self.enu_to_unity_velocity(enu_state['vel_mps'])
        
        # Transform quaternion
        if 'quat_wxyz' in enu_state:
            unity_state['quat_wxyz'] = self.enu_to_unity_quaternion(enu_state['quat_wxyz'])
        
        # Transform angular velocity
        if 'ang_vel_radps' in enu_state:
            unity_state['ang_vel_radps'] = self.enu_to_unity_angular_velocity(enu_state['ang_vel_radps'])
        
        # Copy non-coordinate fields
        for key, value in enu_state.items():
            if key not in ['pos_m', 'vel_mps', 'quat_wxyz', 'ang_vel_radps']:
                unity_state[key] = value
        
        return unity_state
    
    def transform_state_unity_to_enu(self, unity_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform complete state from Unity to ENU coordinates.
        
        Args:
            unity_state: State dict with Unity coordinates
            
        Returns:
            State dict with ENU coordinates
        """
        enu_state = {}
        
        # Transform position
        if 'pos_m' in unity_state:
            enu_state['pos_m'] = self.unity_to_enu_position(unity_state['pos_m'])
        
        # Transform velocity
        if 'vel_mps' in unity_state:
            enu_state['vel_mps'] = self.unity_to_enu_velocity(unity_state['vel_mps'])
        
        # Transform quaternion
        if 'quat_wxyz' in unity_state:
            enu_state['quat_wxyz'] = self.unity_to_enu_quaternion(unity_state['quat_wxyz'])
        
        # Transform angular velocity
        if 'ang_vel_radps' in unity_state:
            enu_state['ang_vel_radps'] = self.unity_to_enu_angular_velocity(unity_state['ang_vel_radps'])
        
        # Copy non-coordinate fields
        for key, value in unity_state.items():
            if key not in ['pos_m', 'vel_mps', 'quat_wxyz', 'ang_vel_radps']:
                enu_state[key] = value
        
        return enu_state
    
    def round_trip_test(self, enu_pos: List[float], enu_quat: List[float], 
                       tolerance: float = 1e-12) -> Tuple[bool, str]:
        """
        Test round-trip transformation accuracy.
        
        Args:
            enu_pos: Original ENU position
            enu_quat: Original ENU quaternion
            tolerance: Numerical tolerance for comparison
            
        Returns:
            (success, error_message)
        """
        try:
            # Position round-trip: ENU -> Unity -> ENU
            unity_pos = self.enu_to_unity_position(enu_pos)
            enu_pos_recovered = self.unity_to_enu_position(unity_pos)
            
            pos_error = np.linalg.norm(np.array(enu_pos) - np.array(enu_pos_recovered))
            
            # Quaternion round-trip: ENU -> Unity -> ENU
            unity_quat = self.enu_to_unity_quaternion(enu_quat)
            enu_quat_recovered = self.unity_to_enu_quaternion(unity_quat)
            
            # Handle quaternion double cover (q and -q represent same rotation)
            quat_error1 = np.linalg.norm(np.array(enu_quat) - np.array(enu_quat_recovered))
            quat_error2 = np.linalg.norm(np.array(enu_quat) + np.array(enu_quat_recovered))
            quat_error = min(quat_error1, quat_error2)
            
            if pos_error > tolerance:
                return False, f"Position error {pos_error:.2e} > tolerance {tolerance:.2e}"
            
            if quat_error > tolerance:
                return False, f"Quaternion error {quat_error:.2e} > tolerance {tolerance:.2e}"
            
            return True, f"Round-trip successful (pos_err={pos_error:.2e}, quat_err={quat_error:.2e})"
            
        except Exception as e:
            return False, f"Round-trip test failed: {e}"
    
    def get_version_info(self) -> Dict[str, Any]:
        """Get transform version information."""
        return {
            'version': self.version,
            'description': self.metadata.description,
            'coordinate_mapping': self.metadata.coordinate_mapping,
            'handedness_correction': self.metadata.handedness_correction,
            'transformation_matrix': self.C_enu_to_unity.tolist(),
            'inverse_matrix': self.C_unity_to_enu.tolist()
        }


# Global transform instance
_transform_instance = None


def get_transform(transform_version: str = "tfm_v1.0") -> CoordinateTransform:
    """Get global coordinate transform instance."""
    global _transform_instance
    
    if _transform_instance is None or _transform_instance.version != transform_version:
        _transform_instance = CoordinateTransform(transform_version)
    
    return _transform_instance


def validate_transform_version(version: str) -> bool:
    """Validate that transform version is supported."""
    return version in TRANSFORM_VERSIONS


# Convenience functions
def enu_to_unity_state(enu_state: Dict[str, Any], transform_version: str = "tfm_v1.0") -> Dict[str, Any]:
    """Transform state from ENU to Unity coordinates."""
    transform = get_transform(transform_version)
    return transform.transform_state_enu_to_unity(enu_state)


def unity_to_enu_state(unity_state: Dict[str, Any], transform_version: str = "tfm_v1.0") -> Dict[str, Any]:
    """Transform state from Unity to ENU coordinates."""
    transform = get_transform(transform_version)
    return transform.transform_state_unity_to_enu(unity_state)


# Example usage and testing
if __name__ == "__main__":
    # Test coordinate transform
    transform = CoordinateTransform("tfm_v1.0")
    
    # Test cases
    test_cases = [
        # Position tests
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),  # East, Identity rotation
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]),  # North, Identity rotation
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]),  # Up, Identity rotation
        ([100.0, 200.0, 50.0], [0.995, 0.0, 0.1, 0.0]),  # General case
    ]
    
    print("Running coordinate transform tests...")
    print(f"Transform version: {transform.version}")
    print(f"Description: {transform.metadata.description}")
    print()
    
    for i, (pos, quat) in enumerate(test_cases):
        print(f"Test case {i+1}:")
        print(f"  ENU position: {pos}")
        print(f"  ENU quaternion: {quat}")
        
        # Forward transforms
        unity_pos = transform.enu_to_unity_position(pos)
        unity_quat = transform.enu_to_unity_quaternion(quat)
        print(f"  Unity position: {unity_pos}")
        print(f"  Unity quaternion: {unity_quat}")
        
        # Round-trip test
        success, message = transform.round_trip_test(pos, quat)
        print(f"  Round-trip: {'✅' if success else '❌'} {message}")
        print()
    
    print("Transform matrix:")
    print(transform.C_enu_to_unity)
    print()
    print("Inverse matrix:")
    print(transform.C_unity_to_enu)