#!/usr/bin/env python3
"""
Test random baseline performance for comparison with trained models.
"""

import numpy as np
from fast_sim_env import make_fast_sim_env


def test_random_baseline(scenario_name="easy", num_episodes=50):
    """
    Test random baseline performance.
    
    Args:
        scenario_name: Scenario to test on
        num_episodes: Number of episodes to run
        
    Returns:
        Average reward of random baseline
    """
    print(f"Testing random baseline on {scenario_name} scenario ({num_episodes} episodes)")
    
    # Create environment
    env = make_fast_sim_env(scenario_name)
    
    episode_rewards = []
    episode_lengths = []
    successes = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)  # Use different seed for each episode
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # Random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Simple success check (positive reward generally means good performance)
        if episode_reward > 0:
            successes += 1
        
        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f}")
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = successes / num_episodes
    
    print(f"\nRandom Baseline Results ({scenario_name}):")
    print(f"  Episodes: {num_episodes}")
    print(f"  Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"  Min reward: {min_reward:.2f}")
    print(f"  Max reward: {max_reward:.2f}")
    print(f"  Average length: {avg_length:.1f}")
    print(f"  Success rate: {success_rate:.1%}")
    
    env.close()
    
    return avg_reward


def calculate_performance_improvement(trained_reward, random_reward):
    """Calculate percentage improvement over random baseline."""
    if random_reward == 0:
        return float('inf') if trained_reward > 0 else 0
    
    improvement = ((trained_reward - random_reward) / abs(random_reward)) * 100
    return improvement


if __name__ == "__main__":
    # Test random baseline
    random_reward = test_random_baseline("easy", 50)
    
    # Compare with known trained model performance
    trained_reward = 95.57  # From previous test
    
    improvement = calculate_performance_improvement(trained_reward, random_reward)
    
    print(f"\n" + "="*50)
    print(f"Performance Comparison:")
    print(f"  Random baseline: {random_reward:.2f}")
    print(f"  Trained model: {trained_reward:.2f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    if improvement >= 25:
        print(f"✅ Trained model beats random baseline by {improvement:.1f}% (≥ 25% required)")
    else:
        print(f"❌ Trained model only beats random baseline by {improvement:.1f}% (< 25% required)")
        
    print(f"  Required for 25% improvement: {random_reward * 1.25:.2f}")