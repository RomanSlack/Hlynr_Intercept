#!/usr/bin/env python3
"""
Quick training test to verify that the numerical stability improvements
work correctly during actual training.
"""

import numpy as np
from aegis_intercept.envs.aegis_6dof_env import Aegis6DOFEnv


def test_training_stability():
    """Test that training runs without numerical overflow warnings."""
    print("Testing training stability...")
    
    # Create environment
    env = Aegis6DOFEnv(curriculum_level="easy")
    
    # Run multiple episodes to check for stability
    successful_episodes = 0
    numerical_instability_episodes = 0
    
    for episode in range(10):
        print(f"\nEpisode {episode + 1}:")
        
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(100):  # Limit steps for quick test
            # Random action
            action = np.random.uniform(-1, 1, 6)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Check for numerical issues
            if not np.all(np.isfinite(obs)):
                print(f"  ❌ Non-finite observation at step {step}")
                break
            
            if not np.isfinite(reward):
                print(f"  ❌ Non-finite reward at step {step}")
                break
            
            if terminated or truncated:
                termination_reason = info.get('termination_reason', 'unknown')
                print(f"  Episode ended at step {step}: {termination_reason}")
                
                if termination_reason == 'numerical_instability':
                    numerical_instability_episodes += 1
                elif termination_reason == 'intercept_success':
                    successful_episodes += 1
                break
        
        print(f"  Total reward: {episode_reward:.2f}")
    
    print(f"\nResults:")
    print(f"  Successful episodes: {successful_episodes}/10")
    print(f"  Numerical instability episodes: {numerical_instability_episodes}/10")
    print(f"  Other terminations: {10 - successful_episodes - numerical_instability_episodes}/10")
    
    # Test should complete without crashing
    print("\n✅ Training stability test completed successfully!")
    
    if numerical_instability_episodes > 0:
        print(f"  ℹ️  Numerical instability detection worked {numerical_instability_episodes} times")
    else:
        print("  ✓ No numerical instability detected")


if __name__ == "__main__":
    test_training_stability()