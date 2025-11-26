#!/usr/bin/env python3
"""
Benchmark Comparison Script: PPO vs HRL

Runs both PPO baseline and HRL system on identical scenarios,
then generates comparison charts and statistics.

Usage:
    python scripts/benchmark_comparison.py \
        --ppo-model checkpoints/best_model.zip \
        --hrl-selector checkpoints/hrl/selector/.../best/best_model.zip \
        --hrl-search checkpoints/hrl/specialists/search/.../final/model.zip \
        --hrl-track checkpoints/hrl/specialists/track/.../final/model.zip \
        --hrl-terminal checkpoints/hrl/specialists/terminal/.../final/model.zip \
        --episodes 50 \
        --seed 42
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import subprocess

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Charts will not be generated.")


def run_ppo_inference(
    model_path: str,
    config_path: str,
    episodes: int,
    seed: Optional[int],
    output_dir: Path
) -> Dict[str, Any]:
    """Run PPO offline inference and return results."""
    print(f"\n{'='*60}")
    print("Running PPO Baseline Inference")
    print(f"{'='*60}")

    cmd = [
        "python", "inference.py",
        "--model", model_path,
        "--mode", "offline",
        "--episodes", str(episodes),
        "--config", config_path,
        "--output", str(output_dir / "ppo_results")
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"PPO inference failed:\n{result.stderr}")
        return None

    # Find and load results
    ppo_dirs = sorted(output_dir.glob("ppo_results/offline_run_*"))
    if not ppo_dirs:
        print("No PPO results found")
        return None

    latest_dir = ppo_dirs[-1]
    summary_path = latest_dir / "summary.json"

    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


def run_hrl_inference(
    selector_path: str,
    search_path: str,
    track_path: str,
    terminal_path: str,
    config_path: str,
    episodes: int,
    seed: Optional[int],
    output_dir: Path
) -> Dict[str, Any]:
    """Run HRL inference and return results."""
    print(f"\n{'='*60}")
    print("Running HRL System Inference")
    print(f"{'='*60}")

    cmd = [
        "python", "scripts/evaluate_hrl.py",
        "--selector", selector_path,
        "--search", search_path,
        "--track", track_path,
        "--terminal", terminal_path,
        "--episodes", str(episodes),
        "--config", config_path,
        "--output", str(output_dir / "hrl_results")
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"HRL inference failed:\n{result.stderr}")
        return None

    # Find and load results
    hrl_dirs = sorted(output_dir.glob("hrl_results/hrl_offline_run_*"))
    if not hrl_dirs:
        print("No HRL results found")
        return None

    latest_dir = hrl_dirs[-1]
    summary_path = latest_dir / "summary.json"

    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return None


def generate_comparison_charts(
    ppo_results: Dict[str, Any],
    hrl_results: Dict[str, Any],
    output_dir: Path
):
    """Generate comparison charts."""
    if not HAS_MATPLOTLIB:
        print("Skipping charts (matplotlib not available)")
        return

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
        ppo_results.get('success_rate', 0) * 100,
        hrl_results.get('success_rate', 0) * 100
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
        ppo_results.get('mean_reward', 0),
        hrl_results.get('mean_reward', 0)
    ]
    reward_stds = [
        ppo_results.get('std_reward', 0),
        hrl_results.get('std_reward', 0)
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
        ppo_results.get('mean_min_distance', 0),
        hrl_results.get('mean_min_distance', 0)
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
        ppo_results.get('mean_episode_length', 0),
        hrl_results.get('mean_episode_length', 0)
    ]
    length_stds = [
        ppo_results.get('std_episode_length', 0),
        hrl_results.get('std_episode_length', 0)
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
        ppo_results.get('mean_fuel_used', 0) * 100,
        hrl_results.get('mean_fuel_used', 0) * 100
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
    sr_improvement = ((hrl_results.get('success_rate', 0) - ppo_results.get('success_rate', 0))
                      / max(ppo_results.get('success_rate', 0.01), 0.01)) * 100
    reward_improvement = ((hrl_results.get('mean_reward', 0) - ppo_results.get('mean_reward', 1))
                          / max(abs(ppo_results.get('mean_reward', 1)), 1)) * 100

    summary_text = f"""
    BENCHMARK SUMMARY
    {'='*40}

    Success Rate:
      PPO:  {ppo_results.get('success_rate', 0)*100:.1f}%
      HRL:  {hrl_results.get('success_rate', 0)*100:.1f}%
      Change: {sr_improvement:+.1f}%

    Mean Reward:
      PPO:  {ppo_results.get('mean_reward', 0):.1f}
      HRL:  {hrl_results.get('mean_reward', 0):.1f}
      Change: {reward_improvement:+.1f}%

    Best Min Distance:
      PPO:  {ppo_results.get('best_min_distance', 0):.1f}m
      HRL:  {hrl_results.get('best_min_distance', 0):.1f}m

    Episodes: {ppo_results.get('n_episodes', 0)}
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

    # Also save individual charts
    save_individual_charts(ppo_results, hrl_results, output_dir, ppo_color, hrl_color)

    plt.close()


def save_individual_charts(
    ppo_results: Dict[str, Any],
    hrl_results: Dict[str, Any],
    output_dir: Path,
    ppo_color: str,
    hrl_color: str
):
    """Save individual high-quality charts."""

    # Success Rate Chart (larger, standalone)
    fig, ax = plt.subplots(figsize=(8, 6))
    success_rates = [
        ppo_results.get('success_rate', 0) * 100,
        hrl_results.get('success_rate', 0) * 100
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
        ax.annotate(f'{improvement:+.0f}% improvement',
                   xy=(1, success_rates[1]), xytext=(1.3, success_rates[1] - 10),
                   fontsize=12, color='green' if improvement > 0 else 'red',
                   arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    plt.savefig(output_dir / "success_rate_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Distance Distribution Chart (if episode data available)
    if 'episodes' in ppo_results and 'episodes' in hrl_results:
        fig, ax = plt.subplots(figsize=(10, 6))

        ppo_min_dists = [ep.get('min_distance', 0) for ep in ppo_results['episodes']
                        if ep.get('min_distance') is not None]
        hrl_min_dists = [ep.get('min_distance', 0) for ep in hrl_results['episodes']
                        if ep.get('min_distance') is not None]

        if ppo_min_dists and hrl_min_dists:
            bins = np.linspace(0, max(max(ppo_min_dists), max(hrl_min_dists)) * 1.1, 20)
            ax.hist(ppo_min_dists, bins=bins, alpha=0.6, label='PPO Baseline', color=ppo_color)
            ax.hist(hrl_min_dists, bins=bins, alpha=0.6, label='HRL System', color=hrl_color)
            ax.set_xlabel('Minimum Distance (m)', fontsize=14)
            ax.set_ylabel('Frequency', fontsize=14)
            ax.set_title('Distribution of Closest Approach Distances', fontsize=16, fontweight='bold')
            ax.legend(fontsize=12)
            ax.axvline(x=100, color='red', linestyle='--', linewidth=2, label='Intercept Radius (100m)')

            plt.tight_layout()
            plt.savefig(output_dir / "distance_distribution.png", dpi=150, bbox_inches='tight')
            plt.close()


def generate_text_report(
    ppo_results: Dict[str, Any],
    hrl_results: Dict[str, Any],
    output_dir: Path
):
    """Generate a text report."""

    report_lines = [
        "=" * 70,
        "BENCHMARK COMPARISON REPORT: PPO vs HRL",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "-" * 70,
        "CONFIGURATION",
        "-" * 70,
        f"Episodes per system: {ppo_results.get('n_episodes', 'N/A')}",
        "",
        "-" * 70,
        "SUCCESS RATE",
        "-" * 70,
        f"  PPO Baseline:  {ppo_results.get('success_rate', 0)*100:6.1f}%",
        f"  HRL System:    {hrl_results.get('success_rate', 0)*100:6.1f}%",
    ]

    # Calculate improvement
    ppo_sr = ppo_results.get('success_rate', 0)
    hrl_sr = hrl_results.get('success_rate', 0)
    if ppo_sr > 0:
        improvement = ((hrl_sr - ppo_sr) / ppo_sr) * 100
        report_lines.append(f"  Improvement:   {improvement:+6.1f}%")

    report_lines.extend([
        "",
        "-" * 70,
        "REWARD METRICS",
        "-" * 70,
        f"  PPO Mean Reward:  {ppo_results.get('mean_reward', 0):10.1f} +/- {ppo_results.get('std_reward', 0):.1f}",
        f"  HRL Mean Reward:  {hrl_results.get('mean_reward', 0):10.1f} +/- {hrl_results.get('std_reward', 0):.1f}",
        "",
        "-" * 70,
        "PRECISION METRICS (Minimum Distance)",
        "-" * 70,
        f"  PPO Mean Min Distance:   {ppo_results.get('mean_min_distance', 0):8.1f}m",
        f"  HRL Mean Min Distance:   {hrl_results.get('mean_min_distance', 0):8.1f}m",
        f"  PPO Best (Closest):      {ppo_results.get('best_min_distance', 0):8.1f}m",
        f"  HRL Best (Closest):      {hrl_results.get('best_min_distance', 0):8.1f}m",
        "",
        "-" * 70,
        "EFFICIENCY METRICS",
        "-" * 70,
        f"  PPO Mean Episode Length: {ppo_results.get('mean_episode_length', 0):8.1f} steps",
        f"  HRL Mean Episode Length: {hrl_results.get('mean_episode_length', 0):8.1f} steps",
        f"  PPO Mean Fuel Used:      {ppo_results.get('mean_fuel_used', 0)*100:8.1f}%",
        f"  HRL Mean Fuel Used:      {hrl_results.get('mean_fuel_used', 0)*100:8.1f}%",
    ])

    # HRL-specific metrics
    if 'option_usage_percentages' in hrl_results:
        report_lines.extend([
            "",
            "-" * 70,
            "HRL OPTION USAGE",
            "-" * 70,
        ])
        for opt, pct in hrl_results['option_usage_percentages'].items():
            report_lines.append(f"  {opt:10s}: {pct:5.1f}%")

    report_lines.extend([
        "",
        "=" * 70,
        "CONCLUSION",
        "=" * 70,
    ])

    # Determine winner
    if hrl_sr > ppo_sr:
        report_lines.append(f"  HRL outperforms PPO by {(hrl_sr - ppo_sr)*100:.1f} percentage points in success rate.")
    elif ppo_sr > hrl_sr:
        report_lines.append(f"  PPO outperforms HRL by {(ppo_sr - hrl_sr)*100:.1f} percentage points in success rate.")
    else:
        report_lines.append("  Both systems have equal success rates.")

    report_lines.append("=" * 70)

    report_text = "\n".join(report_lines)

    # Print to console
    print("\n" + report_text)

    # Save to file
    report_path = output_dir / "benchmark_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark comparison between PPO and HRL systems",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # PPO arguments
    parser.add_argument("--ppo-model", type=str, required=True,
                       help="Path to PPO model checkpoint")

    # HRL arguments
    parser.add_argument("--hrl-selector", type=str, required=True,
                       help="Path to HRL selector model")
    parser.add_argument("--hrl-search", type=str, required=True,
                       help="Path to HRL search specialist")
    parser.add_argument("--hrl-track", type=str, required=True,
                       help="Path to HRL track specialist")
    parser.add_argument("--hrl-terminal", type=str, required=True,
                       help="Path to HRL terminal specialist")

    # Common arguments
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--episodes", type=int, default=50,
                       help="Number of episodes per system")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--skip-ppo", action="store_true",
                       help="Skip PPO inference (use existing results)")
    parser.add_argument("--skip-hrl", action="store_true",
                       help="Skip HRL inference (use existing results)")

    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"benchmark_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("BENCHMARK COMPARISON: PPO vs HRL")
    print(f"{'='*60}")
    print(f"Episodes: {args.episodes}")
    print(f"Seed: {args.seed if args.seed else 'Random'}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Run PPO inference
    if not args.skip_ppo:
        ppo_results = run_ppo_inference(
            args.ppo_model, args.config, args.episodes, args.seed, output_dir
        )
    else:
        print("Skipping PPO inference...")
        ppo_results = None

    # Run HRL inference
    if not args.skip_hrl:
        hrl_results = run_hrl_inference(
            args.hrl_selector, args.hrl_search, args.hrl_track, args.hrl_terminal,
            args.config, args.episodes, args.seed, output_dir
        )
    else:
        print("Skipping HRL inference...")
        hrl_results = None

    # Check results
    if ppo_results is None or hrl_results is None:
        print("\nError: Could not load results from one or both systems.")
        print("Check the inference output above for errors.")
        return 1

    # Save combined results
    combined_results = {
        'timestamp': timestamp,
        'episodes': args.episodes,
        'seed': args.seed,
        'ppo': ppo_results,
        'hrl': hrl_results
    }

    with open(output_dir / "combined_results.json", 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)

    # Generate report
    generate_text_report(ppo_results, hrl_results, output_dir)

    # Generate charts
    generate_comparison_charts(ppo_results, hrl_results, output_dir)

    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"  - benchmark_report.txt")
    print(f"  - benchmark_comparison.png")
    print(f"  - success_rate_comparison.png")
    print(f"  - distance_distribution.png")
    print(f"  - combined_results.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
