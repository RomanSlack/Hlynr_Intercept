"""
Test suite for 6-DOF physics validation in AegisIntercept Phase 3.

This module provides comprehensive tests for the RigidBody6DOF class,
including gravity parameter validation, physics integration, and
numerical accuracy checks.
"""

import pytest
import numpy as np
from pyquaternion import Quaternion

from aegis_intercept.utils.physics6dof import RigidBody6DOF


class TestRigidBody6DOFGravity:
    """Test gravity parameter handling in RigidBody6DOF."""
    
    def test_default_gravity(self):
        """Test default Earth gravity initialization."""
        rb = RigidBody6DOF(mass=1.0)
        expected_gravity = np.array([0, 0, -9.81])
        np.testing.assert_array_almost_equal(rb.gravity, expected_gravity)
    
    def test_scalar_gravity(self):
        """Test scalar gravity input."""
        gravity_value = 9.80665
        rb = RigidBody6DOF(mass=1.0, gravity=gravity_value)
        expected_gravity = np.array([0, 0, -gravity_value])
        np.testing.assert_array_almost_equal(rb.gravity, expected_gravity)
    
    def test_vector_gravity(self):
        """Test vector gravity input."""
        gravity_vector = np.array([0.1, 0.2, -9.81])
        rb = RigidBody6DOF(mass=1.0, gravity=gravity_vector)
        np.testing.assert_array_almost_equal(rb.gravity, gravity_vector)
    
    def test_zero_gravity(self):
        """Test zero gravity input."""
        rb = RigidBody6DOF(mass=1.0, gravity=0.0)
        expected_gravity = np.array([0, 0, 0])
        np.testing.assert_array_almost_equal(rb.gravity, expected_gravity)
    
    def test_list_gravity(self):
        """Test gravity input as list."""
        gravity_list = [0.5, -0.3, -9.81]
        rb = RigidBody6DOF(mass=1.0, gravity=gravity_list)
        expected_gravity = np.array(gravity_list)
        np.testing.assert_array_almost_equal(rb.gravity, expected_gravity)
    
    def test_invalid_gravity_shape(self):
        """Test invalid gravity vector shape."""
        with pytest.raises(ValueError, match="Gravity must be a scalar or 3-element array"):
            RigidBody6DOF(mass=1.0, gravity=[1, 2])  # Wrong shape
        
        with pytest.raises(ValueError, match="Gravity must be a scalar or 3-element array"):
            RigidBody6DOF(mass=1.0, gravity=[1, 2, 3, 4])  # Wrong shape
    
    def test_gravity_effect_on_motion(self):
        """Test that gravity affects motion correctly."""
        # Test with Earth gravity
        rb = RigidBody6DOF(mass=1.0, gravity=9.81)
        initial_velocity = rb.velocity.copy()
        
        # No applied forces - should fall under gravity
        rb.update(dt=1.0)
        
        # After 1 second, velocity should be [0, 0, -9.81]
        expected_velocity = initial_velocity + np.array([0, 0, -9.81])
        np.testing.assert_array_almost_equal(rb.velocity, expected_velocity)
        
        # Position should have changed by velocity * time (Euler integration)
        # With Euler integration: x = x0 + v0*dt + 0.5*a*dt^2 becomes x = x0 + (v0 + a*dt)*dt
        # Since v0=0 and the velocity is updated first: x = 0 + (-9.81)*1 = -9.81
        expected_position = np.array([0, 0, -9.81])  # Euler integration result
        np.testing.assert_array_almost_equal(rb.position, expected_position, decimal=3)
    
    def test_custom_gravity_motion(self):
        """Test motion under custom gravity."""
        custom_gravity = np.array([1.0, 2.0, -5.0])
        rb = RigidBody6DOF(mass=2.0, gravity=custom_gravity)
        
        # No applied forces - should accelerate under custom gravity
        rb.update(dt=1.0)
        
        # After 1 second, velocity should equal gravity vector
        np.testing.assert_array_almost_equal(rb.velocity, custom_gravity)
    
    def test_zero_gravity_motion(self):
        """Test motion with zero gravity."""
        rb = RigidBody6DOF(mass=1.0, gravity=0.0)
        initial_velocity = np.array([1.0, 2.0, 3.0])
        rb.velocity = initial_velocity.copy()
        
        # No applied forces and no gravity - should maintain constant velocity
        rb.update(dt=1.0)
        
        # Velocity should remain unchanged
        np.testing.assert_array_almost_equal(rb.velocity, initial_velocity)
        
        # Position should change by velocity * time
        expected_position = initial_velocity * 1.0
        np.testing.assert_array_almost_equal(rb.position, expected_position)


class TestRigidBody6DOFPhysics:
    """Test physics integration and numerical accuracy."""
    
    def test_mass_conservation(self):
        """Test that mass is conserved."""
        mass = 100.0
        rb = RigidBody6DOF(mass=mass)
        assert rb.mass == mass
        
        # Mass should not change during updates
        rb.update(dt=0.1)
        assert rb.mass == mass
    
    def test_force_application(self):
        """Test force application and integration."""
        rb = RigidBody6DOF(mass=1.0, gravity=0.0)  # No gravity for clean test
        
        # Apply force of 1N in x direction
        force = np.array([1.0, 0.0, 0.0])
        rb.apply_force(force)
        
        # Update physics
        rb.update(dt=1.0)
        
        # With F=ma, acceleration should be 1 m/s²
        # After 1 second, velocity should be 1 m/s
        expected_velocity = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(rb.velocity, expected_velocity)
    
    def test_torque_application(self):
        """Test torque application and angular dynamics."""
        inertia = np.diag([1.0, 1.0, 1.0])  # Unit inertia for simple calculation
        rb = RigidBody6DOF(mass=1.0, inertia=inertia, gravity=0.0)
        
        # Apply torque around z-axis
        torque = np.array([0.0, 0.0, 1.0])
        rb.apply_torque(torque)
        
        # Update physics
        rb.update(dt=1.0)
        
        # With τ = I*α, angular acceleration should be 1 rad/s²
        # After 1 second, angular velocity should be 1 rad/s
        expected_angular_velocity = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(rb.angular_velocity, expected_angular_velocity)
    
    def test_quaternion_normalization(self):
        """Test that quaternion remains normalized."""
        rb = RigidBody6DOF()
        
        # Initial quaternion should be normalized
        assert abs(rb.quaternion.norm - 1.0) < 1e-10
        
        # Apply angular velocity and update
        rb.angular_velocity = np.array([0.1, 0.2, 0.3])
        rb.update(dt=0.1)
        
        # Quaternion should still be normalized
        assert abs(rb.quaternion.norm - 1.0) < 1e-10
    
    def test_energy_conservation(self):
        """Test energy conservation in free flight."""
        rb = RigidBody6DOF(mass=1.0, gravity=0.0)  # No gravity for energy conservation
        
        # Give initial velocity
        initial_velocity = np.array([1.0, 2.0, 3.0])
        rb.velocity = initial_velocity.copy()
        
        # Give initial angular velocity
        initial_angular_velocity = np.array([0.1, 0.2, 0.3])
        rb.angular_velocity = initial_angular_velocity.copy()
        
        # Calculate initial energy
        initial_energy = rb.get_kinetic_energy()
        
        # Update without external forces
        for _ in range(10):
            rb.update(dt=0.01)
        
        # Energy should be conserved (within numerical precision)
        final_energy = rb.get_kinetic_energy()
        np.testing.assert_almost_equal(initial_energy, final_energy, decimal=6)
    
    def test_copy_functionality(self):
        """Test rigid body copy functionality."""
        original = RigidBody6DOF(
            mass=50.0,
            inertia=np.diag([1.0, 2.0, 3.0]),
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([0.1, 0.2, 0.3]),
            quaternion=Quaternion(axis=[0, 0, 1], angle=0.5),
            angular_velocity=np.array([0.01, 0.02, 0.03]),
            gravity=9.80665
        )
        
        # Create copy
        copy = original.copy()
        
        # Verify all properties are copied correctly
        assert copy.mass == original.mass
        np.testing.assert_array_equal(copy.inertia, original.inertia)
        np.testing.assert_array_equal(copy.position, original.position)
        np.testing.assert_array_equal(copy.velocity, original.velocity)
        assert copy.quaternion == original.quaternion
        np.testing.assert_array_equal(copy.angular_velocity, original.angular_velocity)
        np.testing.assert_array_equal(copy.gravity, original.gravity)
        
        # Verify they are independent objects
        copy.velocity[0] = 999.0
        assert original.velocity[0] != 999.0
    
    def test_reset_functionality(self):
        """Test rigid body reset functionality."""
        rb = RigidBody6DOF()
        
        # Modify state
        rb.position = np.array([1.0, 2.0, 3.0])
        rb.velocity = np.array([0.1, 0.2, 0.3])
        rb.angular_velocity = np.array([0.01, 0.02, 0.03])
        
        # Reset to new state
        new_position = np.array([5.0, 6.0, 7.0])
        new_velocity = np.array([0.5, 0.6, 0.7])
        rb.reset(position=new_position, velocity=new_velocity)
        
        # Verify reset worked
        np.testing.assert_array_equal(rb.position, new_position)
        np.testing.assert_array_equal(rb.velocity, new_velocity)
        np.testing.assert_array_equal(rb.angular_velocity, np.zeros(3))  # Default reset
    
    def test_coordinate_transformations(self):
        """Test coordinate frame transformations."""
        rb = RigidBody6DOF()
        
        # Rotate 90 degrees around z-axis
        rb.quaternion = Quaternion(axis=[0, 0, 1], angle=np.pi/2)
        
        # Test vector transformation
        world_vector = np.array([1.0, 0.0, 0.0])
        body_vector = rb.transform_vector_to_body(world_vector)
        
        # After 90° rotation, x in world becomes -y in body (right-hand rule)
        expected_body_vector = np.array([0.0, -1.0, 0.0])
        np.testing.assert_array_almost_equal(body_vector, expected_body_vector)
        
        # Test inverse transformation
        world_vector_back = rb.transform_vector_to_world(body_vector)
        np.testing.assert_array_almost_equal(world_vector_back, world_vector)


class TestRigidBody6DOFIntegration:
    """Integration tests for environment compatibility."""
    
    def test_environment_compatibility(self):
        """Test that the rigid body works with environment parameters."""
        # This should match the parameters used in Aegis6DOFEnv
        interceptor = RigidBody6DOF(
            mass=150.0,
            inertia=np.diag([10.0, 50.0, 50.0]),
            gravity=9.80665
        )
        
        adversary = RigidBody6DOF(
            mass=200.0,
            inertia=np.diag([15.0, 60.0, 60.0]),
            gravity=9.80665
        )
        
        # Both should initialize without error
        assert interceptor.mass == 150.0
        assert adversary.mass == 200.0
        
        # Both should have correct gravity
        expected_gravity = np.array([0, 0, -9.80665])
        np.testing.assert_array_almost_equal(interceptor.gravity, expected_gravity)
        np.testing.assert_array_almost_equal(adversary.gravity, expected_gravity)
        
        # Both should update without error
        interceptor.update(dt=0.01)
        adversary.update(dt=0.01)
    
    def test_numerical_stability(self):
        """Test numerical stability over many time steps."""
        rb = RigidBody6DOF(mass=1.0, gravity=9.81)
        
        # Small time step simulation
        dt = 0.001
        total_time = 1.0
        num_steps = int(total_time / dt)
        
        # Run simulation
        for _ in range(num_steps):
            rb.update(dt=dt)
        
        # Check that we get expected results (free fall)
        # v = g*t, s = 0.5*g*t^2
        expected_velocity = np.array([0, 0, -9.81 * total_time])
        expected_position = np.array([0, 0, -0.5 * 9.81 * total_time**2])
        
        np.testing.assert_array_almost_equal(rb.velocity, expected_velocity, decimal=2)
        np.testing.assert_array_almost_equal(rb.position, expected_position, decimal=2)
        
        # Quaternion should remain normalized
        assert abs(rb.quaternion.norm - 1.0) < 1e-6