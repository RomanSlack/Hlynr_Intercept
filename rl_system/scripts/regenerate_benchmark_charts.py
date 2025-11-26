#!/usr/bin/env python3
"""
Regenerate benchmark charts from existing results.

Usage:
    python scripts/regenerate_benchmark_charts.py benchmark_results/benchmark_20251126_163408/
"""

import sys
import json
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    print("Error: matplotlib required. Install with: pip install matplotlib")
    sys.exit(1)


def normalize_ppo_results(ppo_raw: dict) -> dict:
    """Normalize PPO results to match HRL field names."""
    # PPO uses different field names than HRL
    normalized = {
        'success_rate': ppo_raw.get('success_rate', 0),
        'mean_reward': ppo_raw.get('mean_reward') or ppo_raw.get('avg_reward', 0),
        'std_reward': ppo_raw.get('std_reward', 0),
        'mean_min_distance': ppo_raw.get('mean_min_distance', 0),
        'best_min_distance': ppo_raw.get('best_min_distance', 0),
        'mean_episode_length': ppo_raw.get('mean_episode_length') or ppo_raw.get('avg_steps', 0),
        'std_episode_length': ppo_raw.get('std_episode_length', 0),
        'mean_fuel_used': ppo_raw.get('mean_fuel_used', 0),
        'n_episodes': ppo_raw.get('n_episodes') or ppo_raw.get('num_episodes', 0),
        'episodes': ppo_raw.get('episodes', []),
    }

    # Calculate missing stats from episodes if available
    episodes = ppo_raw.get('episodes', [])
    if episodes:
        # Calculate success rate from episodes
        successes = sum(1 for ep in episodes if ep.get('outcome') == 'intercepted')
        normalized['success_rate'] = successes / len(episodes) if episodes else 0

        # Calculate min distances
        min_dists = [ep.get('min_distance') for ep in episodes if ep.get('min_distance') is not None]
        if min_dists:
            normalized['mean_min_distance'] = np.mean(min_dists)
            normalized['best_min_distance'] = min(min_dists)

        # Calculate reward stats
        rewards = [ep.get('total_reward', 0) for ep in episodes]
        if rewards:
            normalized['mean_reward'] = np.mean(rewards)
            normalized['std_reward'] = np.std(rewards)

        # Calculate episode length stats
        lengths = [ep.get('steps', 0) for ep in episodes]
        if lengths:
            normalized['mean_episode_length'] = np.mean(lengths)
            normalized['std_episode_length'] = np.std(lengths)

        # Calculate fuel usage
        fuel_used = [ep.get('fuel_used', 0) for ep in episodes]
        if fuel_used:
            normalized['mean_fuel_used'] = np.mean(fuel_used)

    return normalized


def generate_charts(ppo_results: dict, hrl_results: dict, output_dir: Path):
    """Generate comparison charts."""

    # Normalize PPO results
    ppo = normalize_ppo_results(ppo_results)
    hrl = hrl_results  # HRL already has correct field names

    print(f"\nNormalized PPO results:")
    for k in ['success_rate', 'mean_reward', 'mean_min_distance', 'mean_episode_length', 'n_episodes']:
        print(f"  {k}: {ppo.get(k, 'N/A')}")

    print(f"\nHRL results:")
    for k in ['success_rate', 'mean_reward', 'mean_min_distance', 'mean_episode_length', 'n_episodes']:
        print(f"  {k}: {hrl.get(k, 'N/A')}")

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Colors
    ppo_color = '#2196F3'  # Blue
    hrl_color = '#4CAF50'  # Green

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Success Rate Comparison (Bar Chart)
    ax1 = fig.add_subplot(2, 3, 1)
    success_rates = [
        ppo.get('success_rate', 0) * 100,
        hrl.get('success_rate', 0) * 100
    ]
    bars = ax1.bar(['PPO Baseline', 'HRL System'], success_rates,
                   color=[ppo_color, hrl_color], edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    for bar, rate in zip(bars, success_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 2. Mean Reward Comparison
    ax2 = fig.add_subplot(2, 3, 2)
    rewards = [
        ppo.get('mean_reward', 0),
        hrl.get('mean_reward', 0)
    ]
    reward_stds = [
        ppo.get('std_reward', 0),
        hrl.get('std_reward', 0)
    ]
    bars = ax2.bar(['PPO Baseline', 'HRL System'], rewards,
                   yerr=reward_stds, capsize=5,
                   color=[ppo_color, hrl_color], edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Mean Reward', fontsize=12)
    ax2.set_title('Reward Comparison', fontsize=14, fontweight='bold')
    for bar, reward in zip(bars, rewards):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{reward:.0f}', ha='center', va='bottom', fontsize=11)

    # 3. Minimum Distance Comparison
    ax3 = fig.add_subplot(2, 3, 3)
    min_dists = [
        ppo.get('mean_min_distance', 0),
        hrl.get('mean_min_distance', 0)
    ]
    bars = ax3.bar(['PPO Baseline', 'HRL System'], min_dists,
                   color=[ppo_color, hrl_color], edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Mean Min Distance (m)', fontsize=12)
    ax3.set_title('Closest Approach Distance', fontsize=14, fontweight='bold')
    for bar, dist in zip(bars, min_dists):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{dist:.1f}m', ha='center', va='bottom', fontsize=11)

    # 4. Episode Length Comparison
    ax4 = fig.add_subplot(2, 3, 4)
    lengths = [
        ppo.get('mean_episode_length', 0),
        hrl.get('mean_episode_length', 0)
    ]
    length_stds = [
        ppo.get('std_episode_length', 0),
        hrl.get('std_episode_length', 0)
    ]
    bars = ax4.bar(['PPO Baseline', 'HRL System'], lengths,
                   yerr=length_stds, capsize=5,
                   color=[ppo_color, hrl_color], edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Mean Episode Length (steps)', fontsize=12)
    ax4.set_title('Episode Length Comparison', fontsize=14, fontweight='bold')
    for bar, length in zip(bars, lengths):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f'{length:.0f}', ha='center', va='bottom', fontsize=11)

    # 5. Fuel Efficiency Comparison
    ax5 = fig.add_subplot(2, 3, 5)
    fuel_used = [
        ppo.get('mean_fuel_used', 0) * 100,
        hrl.get('mean_fuel_used', 0) * 100
    ]
    bars = ax5.bar(['PPO Baseline', 'HRL System'], fuel_used,
                   color=[ppo_color, hrl_color], edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Mean Fuel Used (%)', fontsize=12)
    ax5.set_title('Fuel Efficiency', fontsize=14, fontweight='bold')
    for bar, fuel in zip(bars, fuel_used):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{fuel:.1f}%', ha='center', va='bottom', fontsize=11)

    # 6. Summary Statistics Table
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    # Calculate improvement percentages
    ppo_sr = ppo.get('success_rate', 0)
    hrl_sr = hrl.get('success_rate', 0)
    sr_improvement = ((hrl_sr - ppo_sr) / max(ppo_sr, 0.01)) * 100 if ppo_sr > 0 else 0

    ppo_reward = ppo.get('mean_reward', 0)
    hrl_reward = hrl.get('mean_reward', 0)
    reward_improvement = ((hrl_reward - ppo_reward) / max(abs(ppo_reward), 1)) * 100 if ppo_reward != 0 else 0

    summary_text = f"""
    BENCHMARK SUMMARY
    {'='*40}

    Success Rate:
      PPO:  {ppo.get('success_rate', 0)*100:.1f}%
      HRL:  {hrl.get('success_rate', 0)*100:.1f}%
      Change: {sr_improvement:+.1f}%

    Mean Reward:
      PPO:  {ppo.get('mean_reward', 0):.1f}
      HRL:  {hrl.get('mean_reward', 0):.1f}
      Change: {reward_improvement:+.1f}%

    Best Min Distance:
      PPO:  {ppo.get('best_min_distance', 0):.1f}m
      HRL:  {hrl.get('best_min_distance', 0):.1f}m

    Episodes: {ppo.get('n_episodes', 0)} PPO, {hrl.get('n_episodes', 0)} HRL
    """

    ax6.text(0.1, 0.5, summary_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    chart_path = output_dir / "benchmark_comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    print(f"\nChart saved to: {chart_path}")

    plt.close()

    # Generate standalone success rate chart
    fig, ax = plt.subplots(figsize=(8, 6))
    success_rates = [
        ppo.get('success_rate', 0) * 100,
        hrl.get('success_rate', 0) * 100
    ]
    bars = ax.bar(['PPO Baseline', 'HRL System'], success_rates,
                  color=[ppo_color, hrl_color], edgecolor='black', linewidth=2)
    ax.set_ylabel('Success Rate (%)', fontsize=14)
    ax.set_title('Intercept Success Rate: PPO vs HRL', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 100)

    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{rate:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add improvement annotation
    if success_rates[0] > 0:
        improvement = ((success_rates[1] - success_rates[0]) / success_rates[0]) * 100
        color = 'green' if improvement > 0 else 'red'
        ax.annotate(f'{improvement:+.0f}% {"improvement" if improvement > 0 else "decrease"}',
                   xy=(1, success_rates[1]), xytext=(1.2, max(success_rates) * 0.8),
                   fontsize=12, color=color,
                   arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig(output_dir / "success_rate_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Success rate chart saved to: {output_dir / 'success_rate_comparison.png'}")
    plt.close()

    # Generate distance distribution if episode data available
    ppo_episodes = ppo.get('episodes', [])
    hrl_episodes = hrl.get('episodes', [])

    if ppo_episodes and hrl_episodes:
        fig, ax = plt.subplots(figsize=(10, 6))

        ppo_min_dists = [ep.get('min_distance', 0) for ep in ppo_episodes
                        if ep.get('min_distance') is not None]
        hrl_min_dists = [ep.get('min_distance', 0) for ep in hrl_episodes
                        if ep.get('min_distance') is not None]

        if ppo_min_dists and hrl_min_dists:
            max_dist = max(max(ppo_min_dists), max(hrl_min_dists)) * 1.1
            bins = np.linspace(0, max_dist, 20)
            ax.hist(ppo_min_dists, bins=bins, alpha=0.6, label=f'PPO Baseline (n={len(ppo_min_dists)})', color=ppo_color)
            ax.hist(hrl_min_dists, bins=bins, alpha=0.6, label=f'HRL System (n={len(hrl_min_dists)})', color=hrl_color)
            ax.set_xlabel('Minimum Distance (m)', fontsize=14)
            ax.set_ylabel('Frequency', fontsize=14)
            ax.set_title('Distribution of Closest Approach Distances', fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.axvline(x=100, color='red', linestyle='--', linewidth=2, label='Intercept Radius (100m)')

            plt.tight_layout()
            plt.savefig(output_dir / "distance_distribution.png", dpi=150, bbox_inches='tight')
            print(f"Distance distribution saved to: {output_dir / 'distance_distribution.png'}")
            plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/regenerate_benchmark_charts.py <benchmark_dir>")
        print("Example: python scripts/regenerate_benchmark_charts.py benchmark_results/benchmark_20251126_163408/")
        sys.exit(1)

    benchmark_dir = Path(sys.argv[1])

    if not benchmark_dir.exists():
        print(f"Error: Directory not found: {benchmark_dir}")
        sys.exit(1)

    # Load PPO results
    ppo_dirs = sorted(benchmark_dir.glob("ppo_results/offline_run_*"))
    if not ppo_dirs:
        print("Error: No PPO results found")
        sys.exit(1)

    ppo_summary = ppo_dirs[-1] / "summary.json"
    print(f"Loading PPO results from: {ppo_summary}")
    with open(ppo_summary) as f:
        ppo_results = json.load(f)

    # Load HRL results
    hrl_dirs = sorted(benchmark_dir.glob("hrl_results/hrl_offline_run_*"))
    if not hrl_dirs:
        print("Error: No HRL results found")
        sys.exit(1)

    hrl_summary = hrl_dirs[-1] / "summary.json"
    print(f"Loading HRL results from: {hrl_summary}")
    with open(hrl_summary) as f:
        hrl_results = json.load(f)

    # Generate charts
    generate_charts(ppo_results, hrl_results, benchmark_dir)

    print(f"\n{'='*60}")
    print("Charts regenerated successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
