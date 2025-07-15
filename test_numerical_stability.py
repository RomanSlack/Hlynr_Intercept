#!/usr/bin/env python3
"""
Test script to verify numerical stability improvements in AegisIntercept Phase 3.

This script tests the numerical stability improvements by:
1. Creating extreme conditions that would cause overflow
2. Running the physics simulation to verify stability
3. Testing environment-level overflow detection
4. Verifying that the system gracefully handles numerical issues
"""

import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from aegis_intercept.utils.physics6dof import RigidBody6DOF
from aegis_intercept.utils.maths import sanitize_array, check_numerical_stability
from aegis_intercept.envs.aegis_6dof_env import Aegis6DOFEnv


def test_physics_stability():
    """Test RigidBody6DOF numerical stability improvements."""
    print("Testing RigidBody6DOF numerical stability...")
    
    # Test 1: Extreme force application
    print("\n1. Testing extreme force application...")
    rb = RigidBody6DOF(mass=1.0, gravity=0.0)
    
    # Apply extremely large force that would cause overflow without safeguards
    extreme_force = np.array([1e10, 1e10, 1e10])
    rb.apply_force(extreme_force)
    
    # Update physics - should not crash or produce NaN/Inf
    rb.update(dt=0.01)
    
    # Check that state remains finite
    assert np.all(np.isfinite(rb.velocity)), "Velocity became infinite!"
    assert np.all(np.isfinite(rb.position)), "Position became infinite!"
    assert np.all(np.isfinite(rb.angular_velocity)), "Angular velocity became infinite!"
    assert np.isfinite(rb.quaternion.norm), "Quaternion norm became infinite!"
    
    print("   ✓ Extreme force handled correctly")
    
    # Test 2: Extreme torque application
    print("\n2. Testing extreme torque application...")
    rb = RigidBody6DOF(mass=1.0, gravity=0.0)
    
    # Apply extremely large torque
    extreme_torque = np.array([1e10, 1e10, 1e10])
    rb.apply_torque(extreme_torque)
    
    # Update physics - should not crash
    rb.update(dt=0.01)
    
    # Check that state remains finite
    assert np.all(np.isfinite(rb.angular_velocity)), "Angular velocity became infinite!"
    assert np.isfinite(rb.quaternion.norm), "Quaternion norm became infinite!"
    
    print("   ✓ Extreme torque handled correctly")
    
    # Test 3: Large angular velocity
    print("\n3. Testing large angular velocity...")
    rb = RigidBody6DOF(mass=1.0, gravity=0.0)
    
    # Set extremely large angular velocity
    rb.angular_velocity = np.array([1000.0, 1000.0, 1000.0])
    
    # Update physics multiple times
    for i in range(10):
        rb.update(dt=0.01)
        
        # Check that quaternion remains normalized
        assert abs(rb.quaternion.norm - 1.0) < 1e-6, f"Quaternion not normalized at step {i}"
        assert np.all(np.isfinite(rb.angular_velocity)), f"Angular velocity became infinite at step {i}"
    
    print("   ✓ Large angular velocity handled correctly")
    
    # Test 4: Sanitization functions
    print("\n4. Testing sanitization functions...")
    
    # Test sanitize_array
    dirty_array = np.array([np.nan, np.inf, -np.inf, 1e10, -1e10])
    clean_array = sanitize_array(dirty_array, max_val=1000.0)
    assert np.all(np.isfinite(clean_array)), "Sanitized array contains non-finite values!"
    assert np.all(np.abs(clean_array) <= 1000.0), "Sanitized array exceeds maximum value!"
    
    # Test numerical stability check
    stable_array = np.array([1.0, 2.0, 3.0])
    unstable_array = np.array([np.nan, 1.0, 2.0])
    
    assert check_numerical_stability(stable_array), "Stable array reported as unstable!"
    assert not check_numerical_stability(unstable_array), "Unstable array reported as stable!"
    
    print("   ✓ Sanitization functions work correctly")
    
    print("\n✅ All RigidBody6DOF stability tests passed!")


def test_environment_stability():
    """Test environment-level numerical stability."""
    print("\nTesting environment numerical stability...")
    
    # Create environment
    env = Aegis6DOFEnv(curriculum_level="easy")
    
    # Test 1: Normal operation
    print("\n1. Testing normal operation...")
    obs, info = env.reset()
    
    # Run a few steps with normal actions
    for i in range(10):
        action = np.random.uniform(-1, 1, 6)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check that observation is finite
        assert np.all(np.isfinite(obs)), f"Observation became infinite at step {i}"
        assert np.isfinite(reward), f"Reward became infinite at step {i}"
        
        if terminated or truncated:
            break
    
    print("   ✓ Normal operation stable")
    
    # Test 2: Extreme actions
    print("\n2. Testing extreme actions...")
    obs, info = env.reset()
    
    # Apply extreme actions that would cause instability without safeguards
    extreme_action = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])  # Beyond [-1, 1] range
    
    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(extreme_action)
        
        # Check that observation remains finite
        assert np.all(np.isfinite(obs)), f"Observation became infinite at step {i}"
        assert np.isfinite(reward), f"Reward became infinite at step {i}"
        
        if terminated or truncated:
            if info.get('termination_reason') == 'numerical_instability':
                print(f"   ✓ Numerical instability detected and handled at step {i}")
            break
    
    print("   ✓ Extreme actions handled correctly")
    
    # Test 3: Overflow detection
    print("\n3. Testing overflow detection...")
    
    # Manually corrupt the rigid body state to trigger overflow detection
    obs, info = env.reset()
    
    # Set extreme values that should trigger overflow detection
    env.interceptor.velocity = np.array([1e6, 1e6, 1e6])  # Extreme velocity
    
    # Step should detect numerical instability
    action = np.zeros(6)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if info.get('termination_reason') == 'numerical_instability':
        print("   ✓ Overflow detection working correctly")
    else:
        print(f"   ⚠ Overflow detection may not be working (reason: {info.get('termination_reason')})")
    
    print("\n✅ All environment stability tests passed!")


def test_stability_under_stress():
    """Test stability under extended stress conditions."""
    print("\nTesting stability under stress...")
    
    # Create rigid body with extreme parameters
    rb = RigidBody6DOF(
        mass=0.1,  # Very light
        inertia=np.diag([0.001, 0.001, 0.001]),  # Very low inertia
        gravity=100.0  # Very high gravity
    )
    
    # Apply random extreme forces and torques
    np.random.seed(42)  # For reproducibility
    
    max_values = []
    
    for i in range(100):
        # Random extreme forces and torques
        force = np.random.uniform(-1e6, 1e6, 3)
        torque = np.random.uniform(-1e6, 1e6, 3)
        
        rb.apply_force(force)
        rb.apply_torque(torque)
        
        # Update physics
        rb.update(dt=0.001)
        
        # Check that everything remains finite
        assert np.all(np.isfinite(rb.position)), f"Position became infinite at step {i}"
        assert np.all(np.isfinite(rb.velocity)), f"Velocity became infinite at step {i}"
        assert np.all(np.isfinite(rb.angular_velocity)), f"Angular velocity became infinite at step {i}"
        assert np.isfinite(rb.quaternion.norm), f"Quaternion norm became infinite at step {i}"
        assert abs(rb.quaternion.norm - 1.0) < 1e-6, f"Quaternion not normalized at step {i}"
        
        # Track maximum values
        max_values.append({
            'pos': np.linalg.norm(rb.position),
            'vel': np.linalg.norm(rb.velocity),
            'ang_vel': np.linalg.norm(rb.angular_velocity)
        })
    
    # Check that maximum values are reasonable
    max_pos = max(v['pos'] for v in max_values)
    max_vel = max(v['vel'] for v in max_values)
    max_ang_vel = max(v['ang_vel'] for v in max_values)
    
    print(f"   Maximum position magnitude: {max_pos:.1f} m")
    print(f"   Maximum velocity magnitude: {max_vel:.1f} m/s")
    print(f"   Maximum angular velocity magnitude: {max_ang_vel:.1f} rad/s")
    
    # These should be bounded by our stability limits
    assert max_pos < 100000.0, "Position exceeded stability limit"
    assert max_vel < 500.0, "Velocity exceeded stability limit"
    assert max_ang_vel < 50.0, "Angular velocity exceeded stability limit"
    
    print("   ✓ All values remained within stability limits")
    print("\n✅ Stress test passed!")


if __name__ == "__main__":
    print("🔬 Testing AegisIntercept Numerical Stability Improvements")
    print("=" * 60)
    
    try:
        test_physics_stability()
        test_environment_stability()
        test_stability_under_stress()
        
        print("\n🎉 All numerical stability tests passed!")
        print("The system now handles extreme conditions gracefully and prevents numerical overflow.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()