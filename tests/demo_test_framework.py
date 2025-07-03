#!/usr/bin/env python3
"""
Demo Script for 6DOF Test Framework

This script demonstrates how to use the comprehensive test framework
for validating the 6DOF system implementation. It shows:
- Running different test categories
- Performance monitoring
- Result analysis
- Test framework capabilities

Author: Tester Agent  
Date: Phase 3 Testing Framework
"""

import sys
import os
import time
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_runners import TestRunner, CITestRunner, TestRunConfig
from tests.conftest import validate_test_environment, create_test_environment, run_basic_episode


def demo_quick_validation():
    """Demonstrate quick validation testing"""
    print("🚀 Demo: Quick Validation Testing")
    print("=" * 50)
    
    runner = TestRunner()
    
    # Run quick validation
    results = runner.run_quick_validation()
    
    print(f"✅ Quick validation completed in {results['duration']:.1f} seconds")
    print(f"📊 Status: {results['status']}")
    
    if results['status'] == 'passed':
        print("🎉 All quick validation tests passed!")
    else:
        print("⚠️  Some tests failed - check detailed output")
    
    return results


def demo_environment_testing():
    """Demonstrate environment testing capabilities"""
    print("\n🧪 Demo: Environment Testing")
    print("=" * 50)
    
    # Test environment creation
    print("Creating test environments...")
    
    try:
        # Create different environment configurations
        env_6dof = create_test_environment('default')
        print("✅ 6DOF environment created successfully")
        
        env_legacy = create_test_environment('legacy')  
        print("✅ Legacy 3DOF environment created successfully")
        
        # Run basic episodes
        print("\nRunning test episodes...")
        
        episode_data_6dof = run_basic_episode(env_6dof, seed=42, policy='pursuit')
        print(f"✅ 6DOF episode completed: {episode_data_6dof['episode_length']} steps, "
              f"reward: {episode_data_6dof['total_reward']:.2f}")
        
        episode_data_legacy = run_basic_episode(env_legacy, seed=42, policy='random')
        print(f"✅ Legacy episode completed: {episode_data_legacy['episode_length']} steps, "
              f"reward: {episode_data_legacy['total_reward']:.2f}")
        
        # Clean up
        env_6dof.close() if hasattr(env_6dof, 'close') else None
        env_legacy.close() if hasattr(env_legacy, 'close') else None
        
        return True
        
    except Exception as e:
        print(f"❌ Environment testing failed: {e}")
        return False


def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities"""
    print("\n⚡ Demo: Performance Monitoring")
    print("=" * 50)
    
    from tests.conftest import measure_performance
    
    # Create test environment
    env = create_test_environment('fast')
    
    def test_performance():
        """Test function for performance measurement"""
        total_steps = 0
        
        for episode in range(10):
            obs, _ = env.reset(seed=episode)
            
            for step in range(50):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_steps += 1
                
                if terminated or truncated:
                    break
        
        return total_steps
    
    # Measure performance
    perf_results = measure_performance(test_performance)
    
    total_steps = perf_results['result']
    duration = perf_results['duration']
    memory_growth = perf_results['memory_growth'] / (1024 * 1024)  # MB
    
    steps_per_second = total_steps / duration if duration > 0 else 0
    
    print(f"📈 Performance Results:")
    print(f"   Total steps: {total_steps}")
    print(f"   Duration: {duration:.3f} seconds")
    print(f"   Steps/second: {steps_per_second:.0f}")
    print(f"   Memory growth: {memory_growth:.1f} MB")
    
    # Performance thresholds
    if steps_per_second > 1000:
        print("✅ Performance meets requirements (>1000 steps/sec)")
    else:
        print("⚠️  Performance below threshold")
    
    if memory_growth < 50:
        print("✅ Memory usage acceptable (<50MB growth)")
    else:
        print("⚠️  High memory usage detected")
    
    env.close() if hasattr(env, 'close') else None
    
    return perf_results


def demo_physics_validation():
    """Demonstrate physics validation testing"""
    print("\n🔬 Demo: Physics Validation") 
    print("=" * 50)
    
    from aegis_intercept.utils.physics6dof import RigidBody6DOF, VehicleType, QuaternionUtils
    import numpy as np
    
    print("Testing quaternion operations...")
    
    # Test quaternion normalization
    test_quat = np.array([2.0, 3.0, 4.0, 5.0])
    normalized = QuaternionUtils.normalize(test_quat)
    norm = np.linalg.norm(normalized)
    
    print(f"   Original quaternion: {test_quat}")
    print(f"   Normalized quaternion: {normalized}")
    print(f"   Norm: {norm:.10f}")
    
    if abs(norm - 1.0) < 1e-10:
        print("✅ Quaternion normalization accurate")
    else:
        print("❌ Quaternion normalization failed")
    
    print("\nTesting 6DOF physics integration...")
    
    # Create rigid body
    rigid_body = RigidBody6DOF(
        VehicleType.INTERCEPTOR,
        initial_position=np.array([0, 0, 1000]),
        initial_velocity=np.array([100, 0, 0]),
        initial_orientation=np.array([1, 0, 0, 0]),
        initial_angular_velocity=np.array([0.1, 0.1, 0.1])
    )
    
    # Apply controls and simulate
    rigid_body.set_control_inputs(np.array([1000, 0, 0]), np.array([10, 10, 10]))
    
    initial_pos = rigid_body.position.copy()
    
    # Simulate for 1 second
    dt = 0.01
    steps = 100
    wind = np.array([5, 2, 0])
    
    for i in range(steps):
        rigid_body.step(dt, i * dt, wind)
    
    final_pos = rigid_body.position.copy()
    distance_moved = np.linalg.norm(final_pos - initial_pos)
    
    print(f"   Initial position: {initial_pos}")
    print(f"   Final position: {final_pos}")
    print(f"   Distance moved: {distance_moved:.1f} m")
    
    if 10 < distance_moved < 1000:
        print("✅ Physics simulation producing reasonable results")
    else:
        print("⚠️  Physics simulation results may be unrealistic")
    
    return True


def demo_test_categories():
    """Demonstrate different test categories"""
    print("\n📋 Demo: Test Categories")
    print("=" * 50)
    
    runner = TestRunner()
    
    # Show available test categories
    categories = [
        ('Physics Validation', 'Validates physics engine accuracy'),
        ('Environment Validation', 'Validates environment behavior'),
        ('Curriculum Learning', 'Validates curriculum progression'),
        ('Integration Tests', 'Validates component integration'),
        ('Performance Tests', 'Benchmarks computational performance'),
        ('Regression Tests', 'Prevents capability regression'),
        ('Real-World Validation', 'Validates real-world applicability')
    ]
    
    print("Available test categories:")
    for name, description in categories:
        print(f"   🎯 {name}: {description}")
    
    print(f"\nTest framework located at: {runner.base_dir}")
    print(f"Results will be saved to: {runner.results_dir}")
    
    return True


def demo_ci_integration():
    """Demonstrate CI/CD integration capabilities"""
    print("\n🔄 Demo: CI/CD Integration")
    print("=" * 50)
    
    ci_runner = CITestRunner()
    
    print("CI/CD test capabilities:")
    print("   ⚡ Quick validation for pull requests")
    print("   🌙 Nightly comprehensive testing")
    print("   📊 Automated performance monitoring")
    print("   🚨 Regression detection")
    print("   📈 Performance benchmarking")
    
    # Example CI configuration
    ci_config = TestRunConfig(
        test_categories=['ci'],
        include_slow=False,
        include_performance=False,
        max_duration=600,  # 10 minutes
        output_format='summary',
        fail_fast=True
    )
    
    print(f"\nExample CI configuration:")
    print(f"   Max duration: {ci_config.max_duration} seconds")
    print(f"   Fail fast: {ci_config.fail_fast}")
    print(f"   Include slow tests: {ci_config.include_slow}")
    print(f"   Output format: {ci_config.output_format}")
    
    return True


def main():
    """Run the complete test framework demonstration"""
    print("🎯 AegisIntercept 6DOF Test Framework Demo")
    print("=" * 60)
    print()
    
    # Validate test environment first
    print("🔍 Validating test environment...")
    if not validate_test_environment():
        print("❌ Test environment validation failed!")
        print("Please ensure the aegis_intercept package is properly installed.")
        return False
    
    print("✅ Test environment validation passed!")
    
    try:
        # Run demonstrations
        demo_results = {}
        
        demo_results['quick_validation'] = demo_quick_validation()
        demo_results['environment_testing'] = demo_environment_testing()
        demo_results['performance_monitoring'] = demo_performance_monitoring()
        demo_results['physics_validation'] = demo_physics_validation()
        demo_results['test_categories'] = demo_test_categories()
        demo_results['ci_integration'] = demo_ci_integration()
        
        # Summary
        print("\n🏁 Demo Summary")
        print("=" * 50)
        
        successful_demos = sum(1 for result in demo_results.values() 
                             if result and result != {'status': 'failed'})
        total_demos = len(demo_results)
        
        print(f"✅ Successful demonstrations: {successful_demos}/{total_demos}")
        
        if successful_demos == total_demos:
            print("🎉 All demonstrations completed successfully!")
            print("\n📖 Next Steps:")
            print("   1. Run 'python tests/test_runners.py quick' for basic validation")
            print("   2. Run 'python tests/test_runners.py full' for comprehensive testing")
            print("   3. Use 'pytest tests/' for custom test execution")
            print("   4. Check tests/README.md for detailed documentation")
        else:
            print("⚠️  Some demonstrations had issues - check output above")
        
        return successful_demos == total_demos
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)