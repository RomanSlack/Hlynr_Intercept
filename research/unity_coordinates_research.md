# Unity Coordinate Systems Research

## Unity Coordinate System

### Basic Characteristics
- **Left-handed coordinate system**
- **Y-up orientation** (Y axis points upward)
- **Z-forward orientation** (positive Z points forward)
- **X-right orientation** (positive X points right)

### Coordinate System Mapping

From the Unity Data Reference documentation, the coordinate mapping is:
```
Python (ENU Right-Handed) → Unity (Left-Handed)
Unity.X = Python.X  (East)
Unity.Y = Python.Z  (Up) 
Unity.Z = Python.Y  (North)
```

### Transform Components

Unity's Transform system includes:
- **Position**: 3D coordinates (x, y, z)
- **Rotation**: Quaternion representation [x, y, z, w] (note: Unity uses x,y,z,w order)
- **Scale**: 3D scaling factors

### Coordinate Space Types
1. **World Space**: Global coordinate system
2. **Local Space**: Relative to parent object
3. **Screen Space**: 2D screen coordinates

## Quaternion Handling

### Quaternion Representation
- **Unity Order**: (x, y, z, w) - vector components first, scalar last
- **Standard Math Order**: (w, x, y, z) - scalar first, vector components last
- **Normalization Required**: Must maintain unit magnitude for valid rotations

### Quaternion Properties
- Represents rotation as: q = cos(θ/2) + u * sin(θ/2)
- Where u is unit rotation axis, θ is rotation angle
- Composition via multiplication: q_result = q2 * q1 (q1 applied first)
- Avoids gimbal lock issues common with Euler angles

## Coordinate Transformation Implementation

### ENU to Unity Conversion
```python
def enu_to_unity_position(enu_pos: List[float]) -> List[float]:
    """Convert ENU position to Unity position."""
    x_east, y_north, z_up = enu_pos
    return [x_east, z_up, y_north]  # Unity: [X, Y, Z]

def enu_to_unity_quaternion(enu_quat: List[float]) -> List[float]:
    """Convert ENU quaternion [w,x,y,z] to Unity quaternion [x,y,z,w]."""
    w, x, y, z = enu_quat
    # Coordinate frame conversion: swap Y and Z components
    unity_x = x      # East -> East (X)
    unity_y = z      # Up -> Up (Y) 
    unity_z = y      # North -> Forward (Z)
    unity_w = w      # Scalar component unchanged
    return [unity_x, unity_y, unity_z, unity_w]

def enu_to_unity_velocity(enu_vel: List[float]) -> List[float]:
    """Convert ENU velocity to Unity velocity."""
    vx_east, vy_north, vz_up = enu_vel
    return [vx_east, vz_up, vy_north]  # Unity: [X, Y, Z]
```

### Unity to ENU Conversion
```python
def unity_to_enu_position(unity_pos: List[float]) -> List[float]:
    """Convert Unity position to ENU position."""
    x, y, z = unity_pos
    return [x, z, y]  # ENU: [East, North, Up]

def unity_to_enu_quaternion(unity_quat: List[float]) -> List[float]:
    """Convert Unity quaternion [x,y,z,w] to ENU quaternion [w,x,y,z]."""
    x, y, z, w = unity_quat
    # Coordinate frame conversion: swap Y and Z components
    enu_w = w    # Scalar component unchanged
    enu_x = x    # East -> East (X)
    enu_y = z    # Forward -> North (Y)
    enu_z = y    # Up -> Up (Z)
    return [enu_w, enu_x, enu_y, enu_z]

def unity_to_enu_velocity(unity_vel: List[float]) -> List[float]:
    """Convert Unity velocity to ENU velocity."""
    x, y, z = unity_vel
    return [x, z, y]  # ENU: [East, North, Up]
```

## Practical Transformation Class

### Complete Transform Handler
```python
import numpy as np
from typing import List, Dict, Any

class CoordinateTransformer:
    """Handles coordinate transformations between ENU and Unity systems."""
    
    def __init__(self, transform_version: str = "1.0"):
        self.transform_version = transform_version
    
    def enu_state_to_unity(self, enu_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert complete ENU state to Unity format."""
        unity_state = {}
        
        # Position transformation
        if 'position' in enu_state:
            unity_state['position'] = self.enu_to_unity_position(enu_state['position'])
        
        # Quaternion transformation
        if 'quaternion' in enu_state:
            unity_state['rotation'] = self.enu_to_unity_quaternion(enu_state['quaternion'])
        
        # Velocity transformation  
        if 'velocity' in enu_state:
            unity_state['velocity'] = self.enu_to_unity_velocity(enu_state['velocity'])
        
        # Angular velocity transformation
        if 'angular_velocity' in enu_state:
            unity_state['angularVelocity'] = self.enu_to_unity_velocity(enu_state['angular_velocity'])
        
        # Copy non-spatial data directly
        for key in ['fuel_remaining', 'thrust_current', 'status']:
            if key in enu_state:
                unity_state[key] = enu_state[key]
        
        return unity_state
    
    def unity_state_to_enu(self, unity_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Unity state to ENU format."""
        enu_state = {}
        
        # Position transformation
        if 'position' in unity_state:
            enu_state['position'] = self.unity_to_enu_position(unity_state['position'])
        
        # Quaternion transformation
        if 'rotation' in unity_state:
            enu_state['quaternion'] = self.unity_to_enu_quaternion(unity_state['rotation'])
        
        # Velocity transformation
        if 'velocity' in unity_state:
            enu_state['velocity'] = self.unity_to_enu_velocity(unity_state['velocity'])
        
        # Angular velocity transformation
        if 'angularVelocity' in unity_state:
            enu_state['angular_velocity'] = self.unity_to_enu_velocity(unity_state['angularVelocity'])
        
        # Copy non-spatial data directly
        for key in ['fuel_remaining', 'thrust_current', 'status']:
            if key in unity_state:
                enu_state[key] = unity_state[key]
        
        return enu_state
    
    @staticmethod
    def normalize_quaternion(quat: List[float]) -> List[float]:
        """Normalize quaternion to unit magnitude."""
        quat_array = np.array(quat)
        magnitude = np.linalg.norm(quat_array)
        if magnitude > 0:
            return (quat_array / magnitude).tolist()
        else:
            return [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
    
    @staticmethod
    def enu_to_unity_position(enu_pos: List[float]) -> List[float]:
        """Convert ENU position [x_east, y_north, z_up] to Unity [x, y, z]."""
        return [enu_pos[0], enu_pos[2], enu_pos[1]]
    
    @staticmethod
    def enu_to_unity_quaternion(enu_quat: List[float]) -> List[float]:
        """Convert ENU quaternion [w,x,y,z] to Unity [x,y,z,w]."""
        w, x, y, z = enu_quat
        return [x, z, y, w]  # Swap y,z and reorder to x,y,z,w
    
    @staticmethod
    def enu_to_unity_velocity(enu_vel: List[float]) -> List[float]:
        """Convert ENU velocity to Unity velocity."""
        return [enu_vel[0], enu_vel[2], enu_vel[1]]
    
    @staticmethod
    def unity_to_enu_position(unity_pos: List[float]) -> List[float]:
        """Convert Unity position [x, y, z] to ENU [x_east, y_north, z_up]."""
        return [unity_pos[0], unity_pos[2], unity_pos[1]]
    
    @staticmethod
    def unity_to_enu_quaternion(unity_quat: List[float]) -> List[float]:
        """Convert Unity quaternion [x,y,z,w] to ENU [w,x,y,z]."""
        x, y, z, w = unity_quat
        return [w, x, z, y]  # Reorder to w,x,y,z and swap y,z
    
    @staticmethod
    def unity_to_enu_velocity(unity_vel: List[float]) -> List[float]:
        """Convert Unity velocity to ENU velocity."""
        return [unity_vel[0], unity_vel[2], unity_vel[1]]
```

## Validation and Testing

### Round-trip Testing
```python
def test_coordinate_round_trip():
    """Test coordinate transformations maintain consistency."""
    transformer = CoordinateTransformer()
    
    # Test data
    enu_position = [100.0, 200.0, 50.0]  # East, North, Up
    enu_quaternion = [1.0, 0.0, 0.0, 0.0]  # Identity
    enu_velocity = [10.0, 5.0, 2.0]
    
    # ENU -> Unity -> ENU
    unity_pos = transformer.enu_to_unity_position(enu_position)
    enu_pos_back = transformer.unity_to_enu_position(unity_pos)
    
    unity_quat = transformer.enu_to_unity_quaternion(enu_quaternion)
    enu_quat_back = transformer.unity_to_enu_quaternion(unity_quat)
    
    # Verify round-trip accuracy
    assert np.allclose(enu_position, enu_pos_back, atol=1e-10)
    assert np.allclose(enu_quaternion, enu_quat_back, atol=1e-10)
    
    print("Round-trip test passed!")

def test_axis_mapping():
    """Test that axis mapping is correct."""
    transformer = CoordinateTransformer()
    
    # Test unit vectors
    east_unit = [1.0, 0.0, 0.0]
    north_unit = [0.0, 1.0, 0.0] 
    up_unit = [0.0, 0.0, 1.0]
    
    # Convert to Unity
    unity_east = transformer.enu_to_unity_position(east_unit)
    unity_north = transformer.enu_to_unity_position(north_unit)
    unity_up = transformer.enu_to_unity_position(up_unit)
    
    # Verify mapping
    assert unity_east == [1.0, 0.0, 0.0]   # East -> X
    assert unity_north == [0.0, 0.0, 1.0]  # North -> Z
    assert unity_up == [0.0, 1.0, 0.0]     # Up -> Y
    
    print("Axis mapping test passed!")
```

## Key Takeaways for Unity-RL Bridge

1. **Coordinate Frame Conversion**: Always transform between ENU (right-handed) and Unity (left-handed)
2. **Quaternion Order**: Unity uses [x,y,z,w], standard math uses [w,x,y,z]
3. **Axis Mapping**: East→X, North→Z, Up→Y when converting to Unity
4. **Normalization**: Always normalize quaternions after transformation
5. **Round-trip Testing**: Validate transformations maintain accuracy
6. **Version Tracking**: Include transform_version in all data for consistency
7. **Validation**: Test with unit vectors to verify axis mapping correctness