#!/usr/bin/env python3
"""Simple test for 6DOF physics without logging."""

import numpy as np
from radar_env import RadarEnv

def test_6dof():
    """Test basic 6DOF physics functionality."""
    print("Testing 6DOF RadarEnv...")
    
    # Create environment with explicit 3D config
    config = {
        'environment': {
            'num_missiles': 1,
            'num_interceptors': 1,
            'max_episode_steps': 100
        },
        'spawn': {
            'missile_spawn_area': [[-100, -100, 200], [100, 100, 500]],
            'interceptor_spawn_area': [[400, 400, 50], [600, 600, 200]],
            'target_area': [[800, 800, 0], [1000, 1000, 10]]
        },
        'radar': {
            'range': 1000.0,
            'noise_level': 0.05
        }
    }
    
    env = RadarEnv(config=config)
    
    print(f"Missile positions shape: {env.missile_positions.shape}")
    print(f"Missile velocities shape: {env.missile_velocities.shape}")
    print(f"Interceptor positions shape: {env.interceptor_positions.shape}")
    print(f"Interceptor velocities shape: {env.interceptor_velocities.shape}")
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    print(f"Initial missile position: {env.missile_positions[0]}")
    print(f"Initial missile velocity: {env.missile_velocities[0]}")
    print(f"Initial interceptor position: {env.interceptor_positions[0]}")
    print(f"Initial interceptor velocity: {env.interceptor_velocities[0]}")
    print(f"Target position: {env.target_positions[0]}")
    
    # Test a few steps
    for step in range(5):
        action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])  # Small thrust in X
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step+1}:")
        print(f"  Interceptor pos: {env.interceptor_positions[0]}")
        print(f"  Interceptor vel: {env.interceptor_velocities[0]}")
        print(f"  Interceptor orientation: {env.interceptor_orientations[0]}")
        print(f"  Interceptor angular vel: {env.interceptor_angular_velocities[0]}")
        print(f"  Reward: {reward:.3f}")
        
        if terminated or truncated:
            break
    
    print("6DOF test completed!")

if __name__ == "__main__":
    test_6dof()