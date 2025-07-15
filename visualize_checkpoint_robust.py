#!/usr/bin/env python3
"""
Robust checkpoint visualization script for AegisIntercept Phase 3.

This script provides a more robust way to visualize checkpoints by handling
numpy compatibility issues and providing fallback visualization methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import pickle
from pathlib import Path
import sys
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from aegis_intercept.envs.aegis_6dof_env import Aegis6DOFEnv


def load_checkpoint_data(checkpoint_dir):
    """Load checkpoint data and metadata."""
    checkpoint_path = Path(checkpoint_dir)
    
    # Load training state
    training_state_path = checkpoint_path / "training_state.json"
    if training_state_path.exists():
        with open(training_state_path, 'r') as f:
            training_state = json.load(f)
    else:
        training_state = {}
    
    # Try to load VecNormalize stats
    vec_normalize_path = checkpoint_path / "vec_normalize.pkl"
    vec_normalize_stats = None
    if vec_normalize_path.exists():
        try:
            with open(vec_normalize_path, 'rb') as f:
                vec_normalize_stats = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load VecNormalize stats: {e}")
    
    return training_state, vec_normalize_stats


def visualize_training_progress(training_state):
    """Visualize training progress from checkpoint data."""
    if not training_state:
        print("No training state data available")
        return
    
    print("\n=== Training Progress Analysis ===")
    print(f"Checkpoint Version: {training_state.get('checkpoint_version', 'Unknown')}")
    print(f"Total Timesteps: {training_state.get('num_timesteps', 'Unknown'):,}")
    print(f"Model Class: {training_state.get('model_class', 'Unknown')}")
    
    # Environment info
    env_info = training_state.get('env_info', {})
    print(f"Number of Environments: {env_info.get('n_envs', 'Unknown')}")
    print(f"Observation Space: {env_info.get('observation_space', 'Unknown')}")
    print(f"Action Space: {env_info.get('action_space', 'Unknown')}")
    
    # Training progress
    progress = training_state.get('training_progress', {})
    print(f"Wall Time Elapsed: {progress.get('wall_time_elapsed', 0):.1f} seconds")
    print(f"Timesteps Per Second: {progress.get('timesteps_per_second', 0):.1f}")
    
    # Curriculum info
    curriculum = training_state.get('callback_states', {}).get('curriculum', {})
    episode_count = curriculum.get('episode_count', 0)
    print(f"Episodes Completed: {episode_count}")
    
    # Episode rewards analysis
    episode_rewards = curriculum.get('episode_rewards', [])
    if episode_rewards:
        print(f"\n=== Episode Rewards Analysis ===")
        print(f"Total Episodes: {len(episode_rewards)}")
        print(f"Average Reward: {np.mean(episode_rewards):.2f}")
        print(f"Best Reward: {np.max(episode_rewards):.2f}")
        print(f"Worst Reward: {np.min(episode_rewards):.2f}")
        print(f"Reward Std: {np.std(episode_rewards):.2f}")
        
        # Plot reward progression
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Raw reward progression
        ax1.plot(episode_rewards, 'b-', alpha=0.7, linewidth=1)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Progress: Episode Rewards')
        ax1.grid(True, alpha=0.3)
        
        # Moving average
        if len(episode_rewards) > 10:
            window_size = min(50, len(episode_rewards) // 10)
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(range(window_size-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window_size})')
            ax1.legend()
        
        # Reward distribution
        ax2.hist(episode_rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Reward')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Reward Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Success rate analysis
    success_rate = curriculum.get('success_rate', 0)
    print(f"\n=== Performance Metrics ===")
    print(f"Success Rate: {success_rate:.2%}")
    
    # Current curriculum level
    current_level = curriculum.get('current_level', 'Unknown')
    print(f"Current Curriculum Level: {current_level}")


def visualize_environment_demo(num_episodes=3):
    """Run environment demo with visualization."""
    print(f"\n=== Environment Demo ({num_episodes} episodes) ===")
    
    # Create environment
    env = Aegis6DOFEnv(curriculum_level="easy")
    
    # Run episodes
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        obs, info = env.reset()
        
        # Setup 3D visualization
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Storage for trajectory
        interceptor_trajectory = []
        adversary_trajectory = []
        
        step = 0
        done = False
        total_reward = 0
        
        while not done and step < 200:  # Limit steps for demo
            # Random action for demo
            action = np.random.uniform(-0.5, 0.5, 6)  # Smaller actions for stability
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # Store positions
            interceptor_pos = env.interceptor.get_position()
            adversary_pos = env.adversary.get_position()
            
            interceptor_trajectory.append(interceptor_pos.copy())
            adversary_trajectory.append(adversary_pos.copy())
            
            step += 1
            
            # Update visualization every 20 steps
            if step % 20 == 0:
                ax.clear()
                
                # Plot trajectories
                if len(interceptor_trajectory) > 1:
                    traj = np.array(interceptor_trajectory)
                    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', label='Interceptor', linewidth=2)
                    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='blue', s=100, marker='o')
                
                if len(adversary_trajectory) > 1:
                    traj = np.array(adversary_trajectory)
                    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r-', label='Adversary', linewidth=2)
                    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='red', s=100, marker='^')
                
                # Plot target
                target_pos = env.target_position
                ax.scatter(target_pos[0], target_pos[1], target_pos[2], c='green', s=200, marker='*', label='Target')
                
                # Set labels and title
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                
                distance = info.get("intercept_distance", 0)
                ax.set_title(f'Episode {episode+1}, Step {step}: Distance {distance:.1f}m, Reward {total_reward:.2f}')
                ax.legend()
                
                # Set reasonable bounds
                max_range = 3000
                ax.set_xlim([-max_range, max_range])
                ax.set_ylim([-max_range, max_range])
                ax.set_zlim([0, 2000])
                
                plt.draw()
                plt.pause(0.1)
        
        # Final plot
        ax.clear()
        
        # Plot final trajectories
        if len(interceptor_trajectory) > 1:
            traj = np.array(interceptor_trajectory)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', label='Interceptor', linewidth=2)
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='blue', s=200, marker='o')
        
        if len(adversary_trajectory) > 1:
            traj = np.array(adversary_trajectory)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r-', label='Adversary', linewidth=2)
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='red', s=200, marker='^')
        
        # Plot target
        target_pos = env.target_position
        ax.scatter(target_pos[0], target_pos[1], target_pos[2], c='green', s=300, marker='*', label='Target')
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        distance = info.get("intercept_distance", 0)
        termination_reason = info.get('termination_reason', 'unknown')
        ax.set_title(f'Episode {episode+1} Final: {termination_reason}\nDistance: {distance:.1f}m, Reward: {total_reward:.2f}, Steps: {step}')
        ax.legend()
        
        # Set reasonable bounds
        max_range = 3000
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, 2000])
        
        plt.draw()
        plt.pause(2.0)
        
        # Results
        success = distance < 20.0
        status = "SUCCESS" if success else "FAILED"
        
        print(f"Episode {episode + 1} Results:")
        print(f"  Status: {status}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {step}")
        print(f"  Final Distance: {distance:.1f}m")
        print(f"  Termination Reason: {termination_reason}")
        
        if episode < num_episodes - 1:
            plt.close()
    
    plt.show()


def visualize_observation_space(vec_normalize_stats):
    """Visualize observation space statistics."""
    if not vec_normalize_stats:
        print("No VecNormalize stats available")
        return
    
    print("\n=== Observation Space Analysis ===")
    
    # Get observation statistics
    obs_rms = vec_normalize_stats.obs_rms
    obs_mean = obs_rms.mean
    obs_var = obs_rms.var
    obs_count = obs_rms.count
    
    print(f"Observation Dimension: {len(obs_mean)}")
    print(f"Observation Count: {obs_count}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Observation means
    ax1.bar(range(len(obs_mean)), obs_mean, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Observation Dimension')
    ax1.set_ylabel('Mean Value')
    ax1.set_title('Observation Space: Mean Values')
    ax1.grid(True, alpha=0.3)
    
    # Observation variances (log scale for better visualization)
    ax2.bar(range(len(obs_var)), np.log10(obs_var + 1e-10), alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Observation Dimension')
    ax2.set_ylabel('Log10(Variance)')
    ax2.set_title('Observation Space: Log Variance')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Robust AegisIntercept checkpoint visualization')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Checkpoint directory')
    parser.add_argument('--demo-episodes', type=int, default=3, help='Number of demo episodes to run')
    parser.add_argument('--skip-demo', action='store_true', help='Skip environment demo')
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint_dir).exists():
        print(f"Error: Checkpoint directory {args.checkpoint_dir} does not exist")
        return 1
    
    print(f"🔍 Analyzing checkpoint: {args.checkpoint_dir}")
    
    # Load checkpoint data
    training_state, vec_normalize_stats = load_checkpoint_data(args.checkpoint_dir)
    
    # Visualize training progress
    visualize_training_progress(training_state)
    
    # Visualize observation space
    if vec_normalize_stats:
        visualize_observation_space(vec_normalize_stats)
    
    # Run environment demo
    if not args.skip_demo:
        visualize_environment_demo(args.demo_episodes)
    
    print("\n✅ Visualization complete!")
    return 0


if __name__ == "__main__":
    exit(main())