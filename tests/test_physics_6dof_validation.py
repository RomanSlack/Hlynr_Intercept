"""
Comprehensive Physics Validation Tests for 6DOF System

This module contains extensive tests to validate the correctness and accuracy
of the 6DOF physics engine including:
- Quaternion operations and consistency
- Numerical integration accuracy and stability
- Conservation laws validation
- Physics benchmarking against known solutions
- Aerodynamic modeling verification

Author: Tester Agent
Date: Phase 3 Testing Framework
"""

import pytest
import numpy as np
import time
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt

from aegis_intercept.utils.physics6dof import (
    RigidBody6DOF, VehicleType, QuaternionUtils, PhysicsConstants,
    AtmosphericModel, distance_6dof, intercept_geometry_6dof,
    AERODYNAMIC_PROPERTIES
)


class TestQuaternionUtils:
    """Test quaternion operations for correctness and numerical stability"""
    
    def test_quaternion_normalization(self):
        """Test quaternion normalization accuracy"""
        # Test various quaternions
        test_quats = [
            np.array([1.0, 0.0, 0.0, 0.0]),  # Identity
            np.array([0.707, 0.707, 0.0, 0.0]),  # 90-degree rotation
            np.array([0.5, 0.5, 0.5, 0.5]),  # 120-degree rotation
            np.array([2.0, 3.0, 4.0, 5.0]),  # Unnormalized
            np.array([1e-10, 1e-10, 1e-10, 1e-10]),  # Very small
        ]
        
        for quat in test_quats:
            normalized = QuaternionUtils.normalize(quat)
            
            # Check unit length
            norm = np.linalg.norm(normalized)
            assert abs(norm - 1.0) < 1e-12, f"Quaternion norm {norm} not unity"
            
            # Check that very small quaternions return identity
            if np.linalg.norm(quat) < PhysicsConstants.QUATERNION_NORMALIZE_THRESHOLD:
                np.testing.assert_array_almost_equal(normalized, [1, 0, 0, 0])
    
    def test_quaternion_multiplication_properties(self):
        """Test quaternion multiplication properties (associativity, etc.)"""
        q1 = QuaternionUtils.normalize(np.array([1.0, 2.0, 3.0, 4.0]))
        q2 = QuaternionUtils.normalize(np.array([2.0, 1.0, 4.0, 3.0]))
        q3 = QuaternionUtils.normalize(np.array([3.0, 4.0, 1.0, 2.0]))
        
        # Test associativity: (q1 * q2) * q3 = q1 * (q2 * q3)
        left = QuaternionUtils.multiply(QuaternionUtils.multiply(q1, q2), q3)
        right = QuaternionUtils.multiply(q1, QuaternionUtils.multiply(q2, q3))
        np.testing.assert_array_almost_equal(left, right, decimal=12)
        
        # Test identity multiplication
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        result = QuaternionUtils.multiply(q1, identity)
        np.testing.assert_array_almost_equal(result, q1, decimal=12)
        
        # Test conjugate property: q * q_conj = |q|^2 * identity (for unit quaternions)
        q_conj = QuaternionUtils.conjugate(q1)
        result = QuaternionUtils.multiply(q1, q_conj)
        expected = np.array([1.0, 0.0, 0.0, 0.0])  # For unit quaternions
        np.testing.assert_array_almost_equal(result, expected, decimal=10)
    
    def test_rotation_matrix_consistency(self):
        """Test quaternion to rotation matrix conversion consistency"""
        # Test known rotations
        test_cases = [
            # 90-degree rotation around x-axis
            (np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0, 0]), 'x', np.pi/2),
            # 90-degree rotation around y-axis  
            (np.array([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0]), 'y', np.pi/2),
            # 90-degree rotation around z-axis
            (np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]), 'z', np.pi/2),
        ]
        
        for quat, axis, angle in test_cases:
            R = QuaternionUtils.to_rotation_matrix(quat)
            
            # Check that R is orthogonal
            should_be_identity = R @ R.T
            np.testing.assert_array_almost_equal(should_be_identity, np.eye(3), decimal=10)
            
            # Check determinant = 1 (proper rotation)
            assert abs(np.linalg.det(R) - 1.0) < 1e-10
            
            # Test specific known rotations
            if axis == 'x':
                test_vector = np.array([0, 1, 0])
                expected = np.array([0, 0, 1]) if angle > 0 else np.array([0, 0, -1])
                result = R @ test_vector
                np.testing.assert_array_almost_equal(result, expected, decimal=10)
    
    def test_angular_velocity_integration(self):
        """Test angular velocity to quaternion integration"""
        dt = 0.01
        omega_cases = [
            np.array([1.0, 0.0, 0.0]),  # Rotation around x
            np.array([0.0, 1.0, 0.0]),  # Rotation around y
            np.array([0.0, 0.0, 1.0]),  # Rotation around z
            np.array([1.0, 1.0, 1.0]),  # Combined rotation
        ]
        
        for omega in omega_cases:
            dq = QuaternionUtils.from_angular_velocity(omega, dt)
            
            # Check that quaternion is normalized
            norm = np.linalg.norm(dq)
            assert abs(norm - 1.0) < 1e-12
            
            # For small dt, the rotation angle should be approximately |omega| * dt
            angle = 2 * np.arccos(abs(dq[0]))  # Angle from quaternion
            expected_angle = np.linalg.norm(omega) * dt
            assert abs(angle - expected_angle) < 1e-6


class TestNumericalIntegration:
    """Test numerical integration accuracy and stability"""
    
    def test_integration_stability(self):
        """Test that integration remains stable over time"""
        # Create a simple rigid body
        rigid_body = RigidBody6DOF(
            VehicleType.MISSILE,
            initial_position=np.array([0, 0, 1000]),
            initial_velocity=np.array([100, 0, 0]),
            initial_orientation=np.array([1, 0, 0, 0]),
            initial_angular_velocity=np.array([0.1, 0.1, 0.1])
        )
        
        # Set constant thrust
        rigid_body.set_control_inputs(np.array([1000, 0, 0]), np.array([10, 10, 10]))
        
        dt = 0.01
        num_steps = 1000
        wind_velocity = np.zeros(3)
        
        # Track energy for stability check
        initial_energy = self._compute_total_energy(rigid_body)
        energies = [initial_energy]
        
        for i in range(num_steps):
            rigid_body.step(dt, i * dt, wind_velocity)
            energy = self._compute_total_energy(rigid_body)
            energies.append(energy)
            
            # Check quaternion remains normalized
            quat_norm = np.linalg.norm(rigid_body.orientation)
            assert abs(quat_norm - 1.0) < 1e-10, f"Quaternion denormalized at step {i}"
            
            # Check velocities don't explode
            vel_mag = np.linalg.norm(rigid_body.velocity)
            omega_mag = np.linalg.norm(rigid_body.angular_velocity)
            assert vel_mag < PhysicsConstants.VELOCITY_LIMIT, f"Velocity exploded: {vel_mag}"
            assert omega_mag < PhysicsConstants.ANGULAR_VELOCITY_LIMIT, f"Angular velocity exploded: {omega_mag}"
        
        # Energy should not grow without bound (basic stability check)
        max_energy = max(energies)
        assert max_energy < initial_energy * 10, "Energy grew too much - integration unstable"
    
    def test_integration_accuracy_simple_motion(self):
        """Test integration accuracy against analytical solutions"""
        # Test simple constant acceleration case
        rigid_body = RigidBody6DOF(
            VehicleType.INTERCEPTOR,
            initial_position=np.zeros(3),
            initial_velocity=np.array([10, 0, 0]),
            initial_orientation=np.array([1, 0, 0, 0]),
            initial_angular_velocity=np.zeros(3)
        )
        
        # Apply constant force (no aerodynamics for this test)
        force = np.array([100, 0, 0])  # 100N force
        rigid_body.set_control_inputs(force, np.zeros(3))
        
        dt = 0.001  # Small timestep for accuracy
        t_final = 1.0
        num_steps = int(t_final / dt)
        
        # Disable aerodynamics by moving to space (high altitude)
        rigid_body.position[2] = 100000  # 100km altitude
        
        for i in range(num_steps):
            rigid_body.step(dt, i * dt, np.zeros(3))
        
        # Compare with analytical solution
        mass = rigid_body.aero_props.mass
        accel = force[0] / mass
        
        # Analytical solution: x = v0*t + 0.5*a*t^2, v = v0 + a*t
        expected_pos_x = 10 * t_final + 0.5 * accel * t_final**2
        expected_vel_x = 10 + accel * t_final
        
        # Check position accuracy (account for gravity)
        pos_error = abs(rigid_body.position[0] - expected_pos_x)
        vel_error = abs(rigid_body.velocity[0] - expected_vel_x)
        
        assert pos_error < 0.01, f"Position error too large: {pos_error}"
        assert vel_error < 0.001, f"Velocity error too large: {vel_error}"
    
    def test_conservation_of_momentum(self):
        """Test conservation of momentum in absence of external forces"""
        rigid_body = RigidBody6DOF(
            VehicleType.MISSILE,
            initial_position=np.array([0, 0, 100000]),  # High altitude (no air)
            initial_velocity=np.array([50, 30, 20]),
            initial_orientation=np.array([1, 0, 0, 0]),
            initial_angular_velocity=np.array([0.5, 0.3, 0.2])
        )
        
        # No control inputs - should conserve momentum
        rigid_body.set_control_inputs(np.zeros(3), np.zeros(3))
        
        initial_linear_momentum = rigid_body.aero_props.mass * rigid_body.velocity
        initial_angular_momentum = (rigid_body.aero_props.inertia_tensor @ 
                                   rigid_body.angular_velocity)
        
        dt = 0.01
        for i in range(500):  # 5 seconds
            rigid_body.step(dt, i * dt, np.zeros(3))
        
        final_linear_momentum = rigid_body.aero_props.mass * rigid_body.velocity
        final_angular_momentum = (rigid_body.aero_props.inertia_tensor @ 
                                 rigid_body.angular_velocity)
        
        # Check momentum conservation (allow for numerical errors and gravity)
        linear_error = np.linalg.norm(final_linear_momentum - initial_linear_momentum)
        angular_error = np.linalg.norm(final_angular_momentum - initial_angular_momentum)
        
        # Allow for gravity effect on linear momentum
        gravity_effect = rigid_body.aero_props.mass * PhysicsConstants.GRAVITY * 5.0
        assert linear_error < gravity_effect + 0.1, f"Linear momentum not conserved: {linear_error}"
        assert angular_error < 0.1, f"Angular momentum not conserved: {angular_error}"
    
    def _compute_total_energy(self, rigid_body: RigidBody6DOF) -> float:
        """Compute total kinetic energy of rigid body"""
        # Translational kinetic energy
        ke_trans = 0.5 * rigid_body.aero_props.mass * np.linalg.norm(rigid_body.velocity)**2
        
        # Rotational kinetic energy
        I_omega = rigid_body.aero_props.inertia_tensor @ rigid_body.angular_velocity
        ke_rot = 0.5 * np.dot(rigid_body.angular_velocity, I_omega)
        
        # Potential energy (gravitational)
        pe = rigid_body.aero_props.mass * PhysicsConstants.GRAVITY * rigid_body.position[2]
        
        return ke_trans + ke_rot + pe


class TestAerodynamicModeling:
    """Test aerodynamic force and torque calculations"""
    
    def test_drag_force_scaling(self):
        """Test that drag forces scale correctly with velocity squared"""
        rigid_body = RigidBody6DOF(
            VehicleType.INTERCEPTOR,
            initial_position=np.array([0, 0, 1000]),
            initial_velocity=np.array([100, 0, 0]),
            initial_orientation=np.array([1, 0, 0, 0]),
            initial_angular_velocity=np.zeros(3)
        )
        
        velocities = [50, 100, 200, 300]  # m/s
        drag_forces = []
        
        for vel in velocities:
            air_velocity = np.array([vel, 0, 0])
            air_density = AtmosphericModel.get_air_density(1000)
            
            forces, _ = rigid_body.compute_aerodynamic_forces_torques(air_velocity, air_density)
            drag_magnitude = np.linalg.norm(forces)
            drag_forces.append(drag_magnitude)
        
        # Check quadratic scaling
        for i in range(1, len(velocities)):
            velocity_ratio = velocities[i] / velocities[0]
            force_ratio = drag_forces[i] / drag_forces[0]
            expected_ratio = velocity_ratio**2
            
            relative_error = abs(force_ratio - expected_ratio) / expected_ratio
            assert relative_error < 0.05, f"Drag scaling error: {relative_error}"
    
    def test_angle_of_attack_effects(self):
        """Test angle of attack calculations and effects"""
        rigid_body = RigidBody6DOF(VehicleType.MISSILE)
        
        # Test various air velocities with different angles
        test_cases = [
            (np.array([100, 0, 0]), 0.0),      # Zero AoA
            (np.array([100, 0, 10]), np.arctan(0.1)),  # Small positive AoA
            (np.array([100, 0, -10]), -np.arctan(0.1)), # Small negative AoA
            (np.array([100, 50, 0]), 0.0),     # Sideslip only
        ]
        
        for air_velocity, expected_alpha in test_cases:
            alpha, beta = rigid_body.compute_aerodynamic_angles(air_velocity)
            
            if abs(expected_alpha) < 1e-10:  # Zero AoA case
                assert abs(alpha) < 1e-10, f"Expected zero AoA, got {alpha}"
            else:
                relative_error = abs(alpha - expected_alpha) / abs(expected_alpha)
                assert relative_error < 0.01, f"AoA calculation error: {relative_error}"
    
    def test_atmospheric_model_consistency(self):
        """Test atmospheric model for physical consistency"""
        altitudes = [0, 1000, 5000, 10000, 20000]  # meters
        
        previous_density = float('inf')
        previous_temp = float('inf')
        
        for alt in altitudes:
            density = AtmosphericModel.get_air_density(alt)
            sound_speed = AtmosphericModel.get_speed_of_sound(alt)
            
            # Density should decrease with altitude
            assert density < previous_density, f"Density increased with altitude at {alt}m"
            previous_density = density
            
            # Check reasonable values
            assert density > 0, f"Negative density at {alt}m"
            assert sound_speed > 200, f"Unrealistic sound speed at {alt}m: {sound_speed}"
            assert sound_speed < 400, f"Unrealistic sound speed at {alt}m: {sound_speed}"


class TestPhysicsConstants:
    """Test physics constants for reasonableness"""
    
    def test_constants_validity(self):
        """Test that physics constants are reasonable"""
        constants = PhysicsConstants()
        
        # Check basic physics constants
        assert constants.AIR_DENSITY > 1.0 and constants.AIR_DENSITY < 1.5
        assert constants.SPEED_OF_SOUND > 300 and constants.SPEED_OF_SOUND < 400
        assert constants.GRAVITY > 9.5 and constants.GRAVITY < 10.0
        
        # Check integration parameters
        assert constants.INTEGRATION_SUBSTEPS >= 1
        assert constants.MIN_TIMESTEP > 0
        assert constants.MAX_TIMESTEP > constants.MIN_TIMESTEP
        
        # Check limits
        assert constants.VELOCITY_LIMIT > 100  # Should allow supersonic
        assert constants.ANGULAR_VELOCITY_LIMIT > 10  # Should allow reasonable spin


class TestGeometryCalculations:
    """Test geometric calculation functions"""
    
    def test_distance_calculation(self):
        """Test distance calculation accuracy"""
        test_cases = [
            (np.array([0, 0, 0]), np.array([3, 4, 0]), 5.0),  # 3-4-5 triangle
            (np.array([1, 1, 1]), np.array([4, 5, 1]), 5.0),  # Same triangle shifted
            (np.array([0, 0, 0]), np.array([0, 0, 10]), 10.0), # Vertical distance
        ]
        
        for pos1, pos2, expected_dist in test_cases:
            calculated_dist = distance_6dof(pos1, pos2)
            assert abs(calculated_dist - expected_dist) < 1e-10
    
    def test_intercept_geometry_calculation(self):
        """Test intercept geometry calculations"""
        # Head-on approach
        pos1 = np.array([0, 0, 0])
        vel1 = np.array([100, 0, 0])
        pos2 = np.array([1000, 0, 0])
        vel2 = np.array([-50, 0, 0])
        
        geometry = intercept_geometry_6dof(pos1, vel1, pos2, vel2)
        
        # Should be closing
        assert geometry['closing_velocity'] > 0, "Should be closing approach"
        
        # Should have reasonable time to closest approach
        assert geometry['time_to_closest_approach'] > 0
        assert geometry['time_to_closest_approach'] < 100  # Less than 100 seconds
        
        # Aspect angle should be close to pi (opposite directions)
        assert abs(geometry['aspect_angle'] - np.pi) < 0.1


class TestPerformanceBenchmarks:
    """Performance benchmarks for 6DOF physics"""
    
    def test_single_step_performance(self):
        """Benchmark single physics step performance"""
        rigid_body = RigidBody6DOF(VehicleType.INTERCEPTOR)
        rigid_body.set_control_inputs(np.array([1000, 0, 0]), np.array([100, 0, 0]))
        
        dt = 0.01
        wind_velocity = np.array([5, 2, 0])
        
        # Warmup
        for _ in range(10):
            rigid_body.step(dt, 0.0, wind_velocity)
        
        # Benchmark
        start_time = time.time()
        num_steps = 1000
        
        for i in range(num_steps):
            rigid_body.step(dt, i * dt, wind_velocity)
        
        elapsed = time.time() - start_time
        steps_per_second = num_steps / elapsed
        
        # Should be able to do at least 10,000 steps per second
        assert steps_per_second > 10000, f"Performance too slow: {steps_per_second} steps/sec"
        
        print(f"6DOF Physics Performance: {steps_per_second:.0f} steps/second")
    
    def test_memory_usage_stability(self):
        """Test that memory usage doesn't grow during simulation"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        rigid_body = RigidBody6DOF(VehicleType.MISSILE)
        rigid_body.set_control_inputs(np.array([500, 100, 50]), np.array([50, 25, 10]))
        
        dt = 0.01
        wind_velocity = np.array([10, 5, 0])
        
        # Run for extended period
        for i in range(5000):
            rigid_body.step(dt, i * dt, wind_velocity)
            
            # Check memory every 1000 steps
            if i % 1000 == 0:
                current_memory = process.memory_info().rss
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be minimal (less than 10MB)
                assert memory_growth < 10 * 1024 * 1024, f"Memory leak detected: {memory_growth/1024/1024:.1f}MB growth"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])