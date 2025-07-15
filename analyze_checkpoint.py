#!/usr/bin/env python3
"""
Command-line checkpoint analysis for AegisIntercept Phase 3.

This script analyzes checkpoint data and saves plots without displaying them.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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


def analyze_checkpoint(checkpoint_dir, output_dir="visualization_output"):
    """Analyze checkpoint and create plots."""
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"AEGISINTERCEPT PHASE 3 - CHECKPOINT ANALYSIS")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path.name}")
    
    # Load data
    training_state, vec_normalize_stats = load_checkpoint_data(checkpoint_dir)
    
    if not training_state:
        print("❌ No training state data found")
        return
    
    # Basic info
    print(f"\n📊 BASIC INFORMATION")
    print(f"  Checkpoint Version: {training_state.get('checkpoint_version', 'Unknown')}")
    print(f"  Model Class: {training_state.get('model_class', 'Unknown')}")
    print(f"  Total Timesteps: {training_state.get('num_timesteps', 0):,}")
    
    # Environment info
    env_info = training_state.get('env_info', {})
    print(f"  Environments: {env_info.get('n_envs', 'Unknown')}")
    print(f"  Observation Space: {env_info.get('observation_space', 'Unknown')}")
    print(f"  Action Space: {env_info.get('action_space', 'Unknown')}")
    
    # Training performance
    progress = training_state.get('training_progress', {})
    wall_time = progress.get('wall_time_elapsed', 0)
    timesteps_per_sec = progress.get('timesteps_per_second', 0)
    
    print(f"\n⏱️  TRAINING PERFORMANCE")
    print(f"  Wall Time: {wall_time:.1f} seconds ({wall_time/3600:.2f} hours)")
    print(f"  Speed: {timesteps_per_sec:.0f} timesteps/second")
    
    # Curriculum analysis
    curriculum = training_state.get('callback_states', {}).get('curriculum', {})
    episode_count = curriculum.get('episode_count', 0)
    success_rate = curriculum.get('success_rate', 0)
    current_level = curriculum.get('current_level', 'Unknown')
    
    print(f"\n🎯 CURRICULUM LEARNING")
    print(f"  Episodes Completed: {episode_count}")
    print(f"  Success Rate: {success_rate:.2%}")
    print(f"  Current Level: {current_level}")
    
    # Episode rewards analysis
    episode_rewards = curriculum.get('episode_rewards', [])
    if episode_rewards:
        print(f"\n📈 EPISODE REWARDS ANALYSIS")
        print(f"  Total Episodes: {len(episode_rewards)}")
        print(f"  Average Reward: {np.mean(episode_rewards):,.2f}")
        print(f"  Best Reward: {np.max(episode_rewards):,.2f}")
        print(f"  Worst Reward: {np.min(episode_rewards):,.2f}")
        print(f"  Reward Std: {np.std(episode_rewards):,.2f}")
        print(f"  Median Reward: {np.median(episode_rewards):,.2f}")
        
        # Recent performance
        if len(episode_rewards) > 50:
            recent_rewards = episode_rewards[-50:]
            print(f"  Recent Avg (last 50): {np.mean(recent_rewards):,.2f}")
            print(f"  Recent Best (last 50): {np.max(recent_rewards):,.2f}")
        
        # Create reward progression plot
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Raw rewards
        plt.subplot(2, 3, 1)
        plt.plot(episode_rewards, 'b-', alpha=0.7, linewidth=1)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards')
        plt.grid(True, alpha=0.3)
        
        # Add moving average
        if len(episode_rewards) > 20:
            window_size = min(50, len(episode_rewards) // 10)
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')
            plt.legend()
        
        # Plot 2: Reward distribution
        plt.subplot(2, 3, 2)
        plt.hist(episode_rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative rewards
        plt.subplot(2, 3, 3)
        cumulative_rewards = np.cumsum(episode_rewards)
        plt.plot(cumulative_rewards, 'g-', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Recent performance
        plt.subplot(2, 3, 4)
        if len(episode_rewards) > 100:
            recent_rewards = episode_rewards[-100:]
            plt.plot(recent_rewards, 'purple', linewidth=2)
            plt.axhline(y=np.mean(recent_rewards), color='red', linestyle='--', label=f'Mean: {np.mean(recent_rewards):.0f}')
            plt.xlabel('Episode (Last 100)')
            plt.ylabel('Reward')
            plt.title('Recent Performance')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Plot 5: Training phases
        plt.subplot(2, 3, 5)
        if len(episode_rewards) > 50:
            phase_size = len(episode_rewards) // 5
            phases = []
            labels = []
            
            for i in range(5):
                start_idx = i * phase_size
                end_idx = (i + 1) * phase_size if i < 4 else len(episode_rewards)
                phase_rewards = episode_rewards[start_idx:end_idx]
                phases.append(phase_rewards)
                labels.append(f'Phase {i+1}')
            
            plt.boxplot(phases, labels=labels)
            plt.ylabel('Reward')
            plt.title('Reward by Training Phase')
            plt.grid(True, alpha=0.3)
        
        # Plot 6: Performance trends
        plt.subplot(2, 3, 6)
        if len(episode_rewards) > 20:
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
        plt.savefig(output_path / "reward_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Reward analysis plot saved to: {output_path}/reward_analysis.png")
    
    # Observation space analysis
    if vec_normalize_stats:
        print(f"\n🔍 OBSERVATION SPACE ANALYSIS")
        
        obs_rms = vec_normalize_stats.obs_rms
        obs_mean = obs_rms.mean
        obs_var = obs_rms.var
        obs_count = obs_rms.count
        
        print(f"  Observation Dimensions: {len(obs_mean)}")
        print(f"  Observation Count: {obs_count:,}")
        print(f"  Mean Range: [{np.min(obs_mean):.3f}, {np.max(obs_mean):.3f}]")
        print(f"  Variance Range: [{np.min(obs_var):.3f}, {np.max(obs_var):.3f}]")
        
        # Create observation plots
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Observation means
        plt.subplot(2, 2, 1)
        plt.bar(range(len(obs_mean)), obs_mean, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Observation Dimension')
        plt.ylabel('Mean Value')
        plt.title('Observation Means')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Observation variances (log scale)
        plt.subplot(2, 2, 2)
        plt.bar(range(len(obs_var)), np.log10(obs_var + 1e-10), alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('Observation Dimension')
        plt.ylabel('Log10(Variance)')
        plt.title('Observation Variances (Log Scale)')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Standard deviations
        plt.subplot(2, 2, 3)
        plt.bar(range(len(obs_var)), np.sqrt(obs_var), alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Observation Dimension')
        plt.ylabel('Standard Deviation')
        plt.title('Observation Standard Deviations')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Normalized values
        plt.subplot(2, 2, 4)
        normalized_obs = obs_mean / (np.sqrt(obs_var) + 1e-8)
        plt.bar(range(len(normalized_obs)), normalized_obs, alpha=0.7, color='gold', edgecolor='black')
        plt.xlabel('Observation Dimension')
        plt.ylabel('Normalized Value')
        plt.title('Normalized Observations (Mean/Std)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "observation_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Observation analysis plot saved to: {output_path}/observation_analysis.png")
    
    # Create summary report
    avg_reward = np.mean(episode_rewards) if episode_rewards else 0
    best_reward = np.max(episode_rewards) if episode_rewards else 0
    recent_perf = np.mean(episode_rewards[-50:]) if len(episode_rewards) > 50 else 0
    
    summary_text = f"""
AegisIntercept Phase 3 - Training Summary Report
{'='*50}

Checkpoint: {checkpoint_path.name}
Generated: {checkpoint_path.stat().st_mtime}

TRAINING OVERVIEW:
• Model: {training_state.get('model_class', 'Unknown')}
• Timesteps: {training_state.get('num_timesteps', 0):,}
• Episodes: {len(episode_rewards) if episode_rewards else 0}
• Training Time: {wall_time/3600:.2f} hours
• Speed: {timesteps_per_sec:.0f} timesteps/second

PERFORMANCE METRICS:
• Success Rate: {success_rate:.2%}
• Current Level: {current_level}
• Average Reward: {avg_reward:,.2f if episode_rewards else 'N/A'}
• Best Reward: {best_reward:,.2f if episode_rewards else 'N/A'}
• Recent Performance: {recent_perf:,.2f if len(episode_rewards) > 50 else 'N/A'}

ENVIRONMENT SETUP:
• Parallel Environments: {env_info.get('n_envs', 'Unknown')}
• Observation Space: {env_info.get('observation_space', 'Unknown')}
• Action Space: {env_info.get('action_space', 'Unknown')}

OBSERVATION STATISTICS:
• Dimensions: {len(obs_mean) if vec_normalize_stats else 'N/A'}
• Sample Count: {obs_count:,} if vec_normalize_stats else 'N/A'
• Mean Range: [{np.min(obs_mean):.3f}, {np.max(obs_mean):.3f}] if vec_normalize_stats else 'N/A'
• Variance Range: [{np.min(obs_var):.3f}, {np.max(obs_var):.3f}] if vec_normalize_stats else 'N/A'
"""
    
    # Save summary report
    with open(output_path / "training_summary.txt", 'w') as f:
        f.write(summary_text)
    
    print(f"\n📄 Summary report saved to: {output_path}/training_summary.txt")
    print(f"\n{'='*60}")
    print(f"✅ ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path}/")
    print(f"• reward_analysis.png - Training progress plots")
    print(f"• observation_analysis.png - Observation space analysis")
    print(f"• training_summary.txt - Detailed text report")


def main():
    parser = argparse.ArgumentParser(description='Analyze AegisIntercept checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Checkpoint directory')
    parser.add_argument('--output-dir', type=str, default='checkpoint_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint_dir).exists():
        print(f"❌ Error: Checkpoint directory {args.checkpoint_dir} does not exist")
        return 1
    
    analyze_checkpoint(args.checkpoint_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    exit(main())