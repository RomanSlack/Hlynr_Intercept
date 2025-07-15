#!/usr/bin/env python3
"""
Static checkpoint visualization for AegisIntercept Phase 3.

This script creates static matplotlib plots of checkpoint data without
running the environment interactively.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from pathlib import Path
import argparse


def load_checkpoint_data(checkpoint_dir):
    """Load checkpoint data and metadata."""
    checkpoint_path = Path(checkpoint_dir)
    
    # Load training state
    training_state_path = checkpoint_path / "training_state.json"
    training_state = {}
    if training_state_path.exists():
        with open(training_state_path, 'r') as f:
            training_state = json.load(f)
    
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


def create_training_plots(training_state, output_dir="visualization_output"):
    """Create static plots of training progress."""
    if not training_state:
        print("No training state data available")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
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
        
        # Create comprehensive training plots
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Raw reward progression
        ax1 = plt.subplot(2, 3, 1)
        plt.plot(episode_rewards, 'b-', alpha=0.7, linewidth=1)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Progress: Episode Rewards')
        plt.grid(True, alpha=0.3)
        
        # Moving average
        if len(episode_rewards) > 10:
            window_size = min(50, len(episode_rewards) // 10)
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window_size})')
            plt.legend()
        
        # Plot 2: Reward distribution
        ax2 = plt.subplot(2, 3, 2)
        plt.hist(episode_rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Reward statistics over time
        ax3 = plt.subplot(2, 3, 3)
        if len(episode_rewards) > 20:
            chunk_size = len(episode_rewards) // 20
            chunks = [episode_rewards[i:i+chunk_size] for i in range(0, len(episode_rewards), chunk_size)]
            chunk_means = [np.mean(chunk) for chunk in chunks if chunk]
            chunk_stds = [np.std(chunk) for chunk in chunks if chunk]
            
            x = np.arange(len(chunk_means))
            plt.errorbar(x, chunk_means, yerr=chunk_stds, fmt='o-', capsize=5, capthick=2)
            plt.xlabel('Training Phase')
            plt.ylabel('Reward')
            plt.title('Reward Evolution (Mean ± Std)')
            plt.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative reward
        ax4 = plt.subplot(2, 3, 4)
        cumulative_rewards = np.cumsum(episode_rewards)
        plt.plot(cumulative_rewards, 'g-', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Reward Progress')
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Recent performance (last 100 episodes)
        ax5 = plt.subplot(2, 3, 5)
        if len(episode_rewards) > 100:
            recent_rewards = episode_rewards[-100:]
            plt.plot(recent_rewards, 'purple', linewidth=2)
            plt.axhline(y=np.mean(recent_rewards), color='red', linestyle='--', label=f'Mean: {np.mean(recent_rewards):.0f}')
            plt.xlabel('Episode (Last 100)')
            plt.ylabel('Reward')
            plt.title('Recent Performance')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 6: Performance trends
        ax6 = plt.subplot(2, 3, 6)
        if len(episode_rewards) > 10:
            # Calculate rolling statistics
            window = min(25, len(episode_rewards) // 4)
            rolling_mean = []
            rolling_std = []
            
            for i in range(window, len(episode_rewards)):
                window_data = episode_rewards[i-window:i]
                rolling_mean.append(np.mean(window_data))
                rolling_std.append(np.std(window_data))
            
            x = np.arange(window, len(episode_rewards))
            plt.plot(x, rolling_mean, 'orange', linewidth=2, label='Rolling Mean')
            plt.fill_between(x, np.array(rolling_mean) - np.array(rolling_std), 
                           np.array(rolling_mean) + np.array(rolling_std), alpha=0.3, color='orange')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title(f'Rolling Statistics (Window={window})')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "training_progress.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create reward analysis summary
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Box plot of rewards by training phase
        if len(episode_rewards) > 50:
            phase_size = len(episode_rewards) // 5
            phases = []
            labels = []
            
            for i in range(5):
                start_idx = i * phase_size
                end_idx = (i + 1) * phase_size if i < 4 else len(episode_rewards)
                phase_rewards = episode_rewards[start_idx:end_idx]
                phases.append(phase_rewards)
                labels.append(f'Phase {i+1}\\n({start_idx}-{end_idx})')
            
            ax.boxplot(phases, labels=labels)
            ax.set_ylabel('Reward')
            ax.set_title('Reward Distribution by Training Phase')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / "reward_phases.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    # Success rate analysis
    success_rate = curriculum.get('success_rate', 0)
    print(f"\n=== Performance Metrics ===")
    print(f"Success Rate: {success_rate:.2%}")
    
    # Current curriculum level
    current_level = curriculum.get('current_level', 'Unknown')
    print(f"Current Curriculum Level: {current_level}")
    
    # Create summary statistics plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Summary statistics
    if episode_rewards:
        stats = {
            'Total Episodes': len(episode_rewards),
            'Success Rate (%)': success_rate * 100,
            'Average Reward': np.mean(episode_rewards),
            'Best Reward': np.max(episode_rewards),
            'Worst Reward': np.min(episode_rewards),
            'Reward Std': np.std(episode_rewards),
            'Timesteps (K)': training_state.get('num_timesteps', 0) / 1000,
            'Training Time (hrs)': progress.get('wall_time_elapsed', 0) / 3600
        }
        
        # Create text summary
        summary_text = f"""
AegisIntercept Phase 3 - Training Summary

Checkpoint: {Path(args.checkpoint_dir).name}
Model: {training_state.get('model_class', 'Unknown')}
Environments: {env_info.get('n_envs', 'Unknown')}

Training Progress:
• Total Timesteps: {training_state.get('num_timesteps', 0):,}
• Episodes Completed: {len(episode_rewards)}
• Training Time: {progress.get('wall_time_elapsed', 0)/3600:.1f} hours
• Speed: {progress.get('timesteps_per_second', 0):.0f} timesteps/sec

Performance:
• Success Rate: {success_rate:.2%}
• Average Reward: {np.mean(episode_rewards):.2f}
• Best Reward: {np.max(episode_rewards):.2f}
• Worst Reward: {np.min(episode_rewards):.2f}
• Reward Std: {np.std(episode_rewards):.2f}

Curriculum:
• Current Level: {current_level}
• Observation Space: {env_info.get('observation_space', 'Unknown')}
• Action Space: {env_info.get('action_space', 'Unknown')}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / "training_summary.png", dpi=300, bbox_inches='tight')
        plt.show()


def create_observation_plots(vec_normalize_stats, output_dir="visualization_output"):
    """Create plots of observation space statistics."""
    if not vec_normalize_stats:
        print("No VecNormalize stats available")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n=== Observation Space Analysis ===")
    
    # Get observation statistics
    obs_rms = vec_normalize_stats.obs_rms
    obs_mean = obs_rms.mean
    obs_var = obs_rms.var
    obs_count = obs_rms.count
    
    print(f"Observation Dimension: {len(obs_mean)}")
    print(f"Observation Count: {obs_count}")
    
    # Create observation space visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Observation means
    ax1.bar(range(len(obs_mean)), obs_mean, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Observation Dimension')
    ax1.set_ylabel('Mean Value')
    ax1.set_title('Observation Space: Mean Values')
    ax1.grid(True, alpha=0.3)
    
    # Observation variances (log scale)
    ax2.bar(range(len(obs_var)), np.log10(obs_var + 1e-10), alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Observation Dimension')
    ax2.set_ylabel('Log10(Variance)')
    ax2.set_title('Observation Space: Log Variance')
    ax2.grid(True, alpha=0.3)
    
    # Observation standard deviations
    ax3.bar(range(len(obs_var)), np.sqrt(obs_var), alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_xlabel('Observation Dimension')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_title('Observation Space: Standard Deviations')
    ax3.grid(True, alpha=0.3)
    
    # Normalized observations (mean/std)
    normalized_obs = obs_mean / (np.sqrt(obs_var) + 1e-8)
    ax4.bar(range(len(normalized_obs)), normalized_obs, alpha=0.7, color='gold', edgecolor='black')
    ax4.set_xlabel('Observation Dimension')
    ax4.set_ylabel('Normalized Value (Mean/Std)')
    ax4.set_title('Observation Space: Normalized Values')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "observation_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Static AegisIntercept checkpoint visualization')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Checkpoint directory')
    parser.add_argument('--output-dir', type=str, default='visualization_output', help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint_dir).exists():
        print(f"Error: Checkpoint directory {args.checkpoint_dir} does not exist")
        return 1
    
    print(f"🔍 Analyzing checkpoint: {args.checkpoint_dir}")
    
    # Load checkpoint data
    training_state, vec_normalize_stats = load_checkpoint_data(args.checkpoint_dir)
    
    # Create training plots
    create_training_plots(training_state, args.output_dir)
    
    # Create observation plots
    if vec_normalize_stats:
        create_observation_plots(vec_normalize_stats, args.output_dir)
    
    print(f"\n✅ Visualization complete! Plots saved to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    exit(main())