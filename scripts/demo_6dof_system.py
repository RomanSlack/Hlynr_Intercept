#!/usr/bin/env python3
"""
AegisIntercept Phase 3 - 6DOF System Demonstration

This script demonstrates the key features of the Phase 3 6DOF system:
- 6DOF physics engine with realistic aerodynamics
- Enhanced environment with multiple difficulty modes
- Curriculum learning progression
- Advanced adversary behaviors
- Comprehensive logging and Unity export

Author: Coder Agent
Date: Phase 3 Implementation
"""

import numpy as np
import time
import json
from pathlib import Path

# AegisIntercept Phase 3 imports
import aegis_intercept
from aegis_intercept.envs import Aegis6DInterceptEnv, DifficultyMode, ActionMode
from aegis_intercept.utils.physics6dof import RigidBody6DOF, VehicleType, QuaternionUtils
from aegis_intercept.curriculum import CurriculumManager, CurriculumPhase, create_curriculum_manager
from aegis_intercept.logging import TrajectoryLogger, ExportManager, LogLevel
from aegis_intercept.adversary import EnhancedAdversary, create_default_adversary_config


def demo_version_info():
    """Demonstrate version and feature information"""
    print("🚀 AegisIntercept Phase 3 - 6DOF System Demo")
    print("=" * 50)
    
    version_info = aegis_intercept.get_version_info()
    print(f"Version: {version_info['version']}")
    print(f"Phase: {version_info['phase']}")
    print(f"Available Environments: {', '.join(version_info['environments'])}")
    print(f"Physics Engines: {', '.join(version_info['physics_engines'])}")
    print()
    
    print("🎯 Phase 3 Features:")
    for feature in version_info['features']:
        print(f"  • {feature}")
    print()


def demo_6dof_physics():
    """Demonstrate 6DOF physics engine capabilities"""
    print("🔬 6DOF Physics Engine Demo")
    print("-" * 30)
    
    # Create interceptor and missile
    interceptor_pos = np.array([300.0, 300.0, 50.0])
    interceptor_vel = np.array([0.0, 0.0, 20.0])
    interceptor_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    interceptor_omega = np.zeros(3)
    
    interceptor = RigidBody6DOF(
        VehicleType.INTERCEPTOR,
        interceptor_pos, interceptor_vel, interceptor_quat, interceptor_omega
    )
    
    missile_pos = np.array([500.0, 500.0, 300.0])
    missile_vel = np.array([-30.0, -30.0, -10.0])
    missile_quat = np.array([0.9659, 0.0, 0.2588, 0.0])  # 30 degree pitch
    missile_omega = np.array([0.0, 0.1, 0.0])
    
    missile = RigidBody6DOF(
        VehicleType.MISSILE,
        missile_pos, missile_vel, missile_quat, missile_omega
    )
    
    print(f"Interceptor initial state:")
    print(f"  Position: {interceptor.position}")
    print(f"  Velocity: {interceptor.velocity}")
    print(f"  Orientation (quat): {interceptor.orientation}")
    print(f"  Euler angles: {np.degrees(QuaternionUtils.to_euler(interceptor.orientation))}")
    print()
    
    print(f"Missile initial state:")
    print(f"  Position: {missile.position}")
    print(f"  Velocity: {missile.velocity}")
    print(f"  Orientation (quat): {missile.orientation}")
    print(f"  Euler angles: {np.degrees(QuaternionUtils.to_euler(missile.orientation))}")
    print()
    
    # Simulate a few steps
    dt = 0.05
    wind = np.array([5.0, 2.0, 0.0])
    
    print("Simulating 5 physics steps...")
    for step in range(5):
        # Apply some control inputs
        interceptor.set_control_inputs(np.array([1000.0, 0.0, 200.0]), np.array([0.0, 50.0, 0.0]))
        missile.set_control_inputs(np.array([800.0, 0.0, 0.0]), np.array([0.0, 0.0, 10.0]))
        
        # Step physics
        interceptor.step(dt, step * dt, wind)
        missile.step(dt, step * dt, wind)
        
        # Calculate intercept distance
        distance = np.linalg.norm(interceptor.position - missile.position)
        
        print(f"  Step {step + 1}: Intercept distance = {distance:.1f}m")
    
    # Show aerodynamic info
    print(f"\nInterceptor aerodynamics:")
    aero_info = interceptor.get_aerodynamic_info()
    print(f"  Angle of attack: {np.degrees(aero_info['angle_of_attack']):.1f}°")
    print(f"  Mach number: {aero_info['mach_number']:.2f}")
    print(f"  Dynamic pressure: {aero_info['dynamic_pressure']:.1f} Pa")
    print()


def demo_enhanced_environment():
    """Demonstrate enhanced 6DOF environment"""
    print("🌍 Enhanced 6DOF Environment Demo")
    print("-" * 35)
    
    # Test different difficulty modes
    difficulty_modes = [
        (DifficultyMode.SIMPLIFIED_6DOF, "Simplified 6DOF"),
        (DifficultyMode.FULL_6DOF, "Full 6DOF"),
        (DifficultyMode.EXPERT_6DOF, "Expert 6DOF")
    ]
    
    for difficulty, name in difficulty_modes:
        print(f"Testing {name} mode...")
        
        env = Aegis6DInterceptEnv(
            difficulty_mode=difficulty,
            action_mode=ActionMode.ACCELERATION_6DOF,
            world_size=400.0,
            max_steps=200,
            enable_wind=True,
            wind_strength=1.0
        )
        
        obs, info = env.reset(seed=42)
        
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.shape}")
        print(f"  Initial observation sample: {obs[:5]}...")  # First 5 elements
        
        # Run a few steps with random actions
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"  Completed {step + 1} steps, total reward: {total_reward:.2f}")
        print(f"  Final intercept distance: {info.get('intercept_distance', 'N/A')}")
        env.close()
        print()


def demo_curriculum_learning():
    """Demonstrate curriculum learning system"""
    print("🎓 Curriculum Learning Demo")
    print("-" * 28)
    
    # Create curriculum manager
    curriculum_manager = create_curriculum_manager()
    
    print(f"Initial phase: {curriculum_manager.current_phase.value}")
    
    # Show phase configurations
    current_config = curriculum_manager.get_current_phase_config()
    print(f"Current phase config:")
    print(f"  Difficulty mode: {current_config.difficulty_mode.value}")
    print(f"  Action mode: {current_config.action_mode.value}")
    print(f"  World size: {current_config.world_size}")
    print(f"  Max steps: {current_config.max_steps}")
    print(f"  Success rate threshold: {current_config.success_rate_threshold}")
    print()
    
    # Simulate some episodes with performance updates
    print("Simulating curriculum progression...")
    episode_rewards = [15.2, 12.8, 18.5, 16.1, 19.3, 14.7, 17.8, 20.1, 16.9, 18.4]
    
    for i, reward in enumerate(episode_rewards):
        success = reward > 15.0
        fuel_used = np.random.uniform(0.3, 0.8)
        intercept_time = np.random.uniform(8.0, 15.0)
        
        curriculum_manager.update_performance(reward, success, fuel_used, intercept_time)
        
        if i % 3 == 0:  # Show status every few episodes
            status = curriculum_manager.get_curriculum_status()
            print(f"  Episode {i + 1}: Phase = {status['current_phase']}, "
                  f"Success rate = {status['phase_progress']['recent_success_rate']:.2f}")
    
    # Show final status
    final_status = curriculum_manager.get_curriculum_status()
    print(f"\nFinal curriculum status:")
    print(f"  Current phase: {final_status['current_phase']}")
    print(f"  Episodes completed: {final_status['phase_progress']['episodes_completed']}")
    print(f"  Success rate: {final_status['phase_progress']['success_rate']:.2f}")
    print(f"  Advancement criteria met: {final_status['advancement_criteria']['criteria_met']}")
    print()


def demo_adversary_system():
    """Demonstrate enhanced adversary system"""
    print("🎯 Enhanced Adversary System Demo")
    print("-" * 34)
    
    # Create adversary with different difficulty levels
    difficulties = ["easy", "medium", "hard", "expert"]
    
    for difficulty in difficulties:
        config = create_default_adversary_config(difficulty)
        print(f"{difficulty.title()} Adversary Configuration:")
        print(f"  Aggressiveness: {config.aggressiveness}")
        print(f"  Intelligence: {config.intelligence}")
        print(f"  Skill level: {config.skill_level}")
        print(f"  Max G-force: {config.max_g_force}")
        print(f"  Available patterns: {len(config.available_patterns)}")
        print(f"  Guidance mode: {config.guidance_mode.value}")
        print()
    
    # Show evasion patterns
    from aegis_intercept.adversary import EvasionPattern
    print("Available Evasion Patterns:")
    for pattern in EvasionPattern:
        if pattern != EvasionPattern.NONE:
            print(f"  • {pattern.value.replace('_', ' ').title()}")
    print()


def demo_logging_system():
    """Demonstrate comprehensive logging system"""
    print("📊 Comprehensive Logging System Demo")
    print("-" * 37)
    
    # Create trajectory logger
    logger = TrajectoryLogger(
        log_directory="demo_logs",
        log_level=LogLevel.DETAILED,
        max_buffer_size=100
    )
    
    print(f"Trajectory logger created with log level: {logger.log_level.value}")
    
    # Start episode
    episode_id = logger.start_episode()
    print(f"Started episode {episode_id}")
    
    # Simulate trajectory logging
    print("Logging trajectory points...")
    for step in range(10):
        # Simulate 6DOF state data
        interceptor_state = np.random.uniform(-100, 100, 13)  # 13D state
        missile_state = np.random.uniform(-100, 100, 13)
        env_state = np.array([step * 0.1, 0.8, 5.0, 2.0, 0.0])  # time, fuel, wind
        
        control_inputs = {
            'thrust_force': np.random.uniform(-1000, 1000, 3),
            'control_torque': np.random.uniform(-100, 100, 3),
            'explosion_command': False
        }
        
        reward = np.random.uniform(-1, 5)
        
        logger.log_step(
            step * 0.1, step, interceptor_state, missile_state, 
            env_state, control_inputs, reward
        )
    
    # End episode
    metrics = logger.end_episode(True, "successful_intercept")
    print(f"Episode completed:")
    print(f"  Total steps: {metrics.total_steps}")
    print(f"  Total reward: {metrics.total_reward:.2f}")
    print(f"  Intercept quality score: {metrics.intercept_quality_score:.1f}")
    print()
    
    # Save episode data
    filepath = logger.save_episode_data(format=logger.DataFormat.JSON)
    print(f"Episode data saved to: {filepath}")
    
    logger.close()
    print()


def demo_unity_export():
    """Demonstrate Unity export capabilities"""
    print("🎮 Unity Export System Demo")
    print("-" * 29)
    
    # Create export manager
    from aegis_intercept.logging import UnityCoordinateSystem
    export_manager = ExportManager(
        export_directory="demo_exports",
        coordinate_system=UnityCoordinateSystem.LEFT_HANDED,
        trail_length=20
    )
    
    print(f"Export manager created with coordinate system: {export_manager.coordinate_system.value}")
    
    # Create sample trajectory data
    from aegis_intercept.logging import TrajectoryPoint, EpisodeMetrics
    
    trajectory_points = []
    for i in range(20):
        point = TrajectoryPoint(
            timestamp=time.time() + i * 0.1,
            simulation_time=i * 0.1,
            step_count=i,
            interceptor_position=np.array([300 + i * 2, 300 + i * 1, 50 + i * 3]),
            interceptor_velocity=np.array([20, 10, 30]),
            interceptor_orientation=np.array([1, 0, 0, 0]),
            interceptor_angular_velocity=np.zeros(3),
            missile_position=np.array([500 - i * 3, 500 - i * 2, 300 - i * 5]),
            missile_velocity=np.array([-30, -20, -50]),
            missile_orientation=np.array([0.9659, 0, 0.2588, 0]),
            missile_angular_velocity=np.array([0, 0.1, 0]),
            fuel_remaining=1.0 - i * 0.02,
            intercept_distance=np.linalg.norm(np.array([200 - i * 5, 200 - i * 3, 250 - i * 8])),
            step_reward=1.0 + np.random.uniform(-0.5, 0.5)
        )
        trajectory_points.append(point)
    
    # Create episode metrics
    episode_metrics = EpisodeMetrics(
        episode_id=1,
        start_time=time.time(),
        end_time=time.time() + 2.0,
        total_steps=20,
        simulation_duration=2.0,
        success=True,
        termination_reason="successful_intercept",
        final_intercept_distance=15.0,
        total_reward=25.5,
        average_reward=1.275,
        fuel_consumed=0.4,
        fuel_efficiency=63.75,
        max_interceptor_altitude=109.0,
        max_interceptor_speed=37.4,
        max_missile_speed=59.2,
        min_intercept_distance=15.0,
        average_thrust_magnitude=850.0,
        average_control_torque=45.2,
        explosion_used=False
    )
    
    # Export for Unity
    target_position = np.array([300, 300, 0])
    filepath = export_manager.export_episode_for_unity(
        trajectory_points, episode_metrics, target_position, "demo_episode"
    )
    
    print(f"Unity episode data exported to: {filepath}")
    
    # Export dashboard data
    dashboard_filepath = export_manager.export_dashboard_data([episode_metrics], "demo_dashboard")
    print(f"Dashboard data exported to: {dashboard_filepath}")
    
    # Export animation data
    animation_filepath = export_manager.export_animation_data(
        trajectory_points, target_position, "demo_animation", time_step=0.1
    )
    print(f"Animation data exported to: {animation_filepath}")
    print()


def demo_integration_example():
    """Demonstrate complete system integration"""
    print("🔧 Complete System Integration Demo")
    print("-" * 36)
    
    # Create all components
    curriculum_manager = create_curriculum_manager()
    env_config = curriculum_manager.get_environment_config()
    
    env = Aegis6DInterceptEnv(
        difficulty_mode=env_config['difficulty_mode'],
        action_mode=env_config['action_mode'],
        world_size=env_config['world_size'],
        max_steps=50,  # Short demo
        enable_wind=True
    )
    
    trajectory_logger = TrajectoryLogger(
        log_directory="integration_demo_logs",
        log_level=LogLevel.BASIC
    )
    
    export_manager = ExportManager(export_directory="integration_demo_exports")
    
    print("All components initialized successfully!")
    print(f"Environment: {env_config['difficulty_mode'].value} mode")
    print(f"Curriculum Phase: {curriculum_manager.current_phase.value}")
    
    # Run a complete episode
    print("\nRunning integrated episode...")
    obs, info = env.reset(seed=123)
    episode_id = trajectory_logger.start_episode()
    
    total_reward = 0
    step_count = 0
    
    while True:
        # Random action for demo
        action = env.action_space.sample()
        
        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Log trajectory (simplified for demo)
        if hasattr(env, 'interceptor_6dof') and env.interceptor_6dof is not None:
            interceptor_state = env.interceptor_6dof.get_state_vector()
            missile_state = env.missile_6dof.get_state_vector()
            env_state = np.array([step_count * env.dt, env.fuel_remaining / env.max_fuel, 0, 0, 0])
            
            control_inputs = {
                'thrust_force': action[:3] * 1000,
                'control_torque': action[3:6] * 100 if len(action) > 6 else np.zeros(3),
                'explosion_command': action[-1] > 0.5
            }
            
            trajectory_logger.log_step(
                step_count * env.dt, step_count, interceptor_state, missile_state,
                env_state, control_inputs, reward, info
            )
        
        if terminated or truncated:
            break
        
        obs = next_obs
    
    # Complete episode
    success = total_reward > 10.0
    episode_metrics = trajectory_logger.end_episode(success, "demo_completion")
    
    print(f"Episode completed:")
    print(f"  Steps: {step_count}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Success: {success}")
    print(f"  Final distance: {info.get('intercept_distance', 'N/A')}")
    
    # Update curriculum
    curriculum_manager.update_performance(total_reward, success, 0.3, step_count * env.dt)
    
    # Save and export data
    trajectory_logger.save_episode_data()
    if hasattr(env, 'target_pos'):
        export_manager.export_episode_for_unity(
            trajectory_logger.current_trajectory, episode_metrics, env.target_pos
        )
    
    # Cleanup
    env.close()
    trajectory_logger.close()
    
    print("Integration demo completed successfully! ✅")
    print()


def main():
    """Run all demonstrations"""
    print("🎮 AegisIntercept Phase 3 - Complete System Demonstration")
    print("=" * 60)
    print()
    
    # Run all demos
    demo_version_info()
    demo_6dof_physics()
    demo_enhanced_environment()
    demo_curriculum_learning()
    demo_adversary_system()
    demo_logging_system()
    demo_unity_export()
    demo_integration_example()
    
    print("🎉 All demonstrations completed successfully!")
    print()
    print("Next steps:")
    print("• Run 'python scripts/train_ppo_phase3_6dof.py' to start training")
    print("• Customize curriculum configs in curriculum/configs/")
    print("• Export trajectory data for Unity visualization")
    print("• Experiment with different adversary configurations")
    print()
    print("For more information, check the documentation and example scripts.")


if __name__ == "__main__":
    main()