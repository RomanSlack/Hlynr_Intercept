#!/usr/bin/env python3
"""
Simple metrics analyzer - generates HTML report from inference metrics.jsonl

Usage:
    python analyze_metrics.py inference_results/offline_run_*/*/metrics.jsonl
    python analyze_metrics.py inference_results/offline_run_20251003_114022/offline_inference_20251003_114022/metrics.jsonl
"""

import json
import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime


def load_metrics(jsonl_path):
    """Load metrics from JSONL file."""
    metrics = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    return metrics


def analyze_metrics(metrics):
    """Calculate statistics from metrics."""
    # Check if volley mode
    is_volley = any(m.get('volley_mode', False) for m in metrics)

    outcomes = [m['outcome'] for m in metrics]
    distances = [m['final_distance'] for m in metrics]
    rewards = [m['total_reward'] for m in metrics]
    steps = [m['steps'] for m in metrics]
    fuel_used = [m['fuel_used'] for m in metrics]

    if is_volley:
        # Volley mode metrics
        all_intercepted = [o == 'all_intercepted' for o in outcomes]
        partial = [o == 'partial_interception' for o in outcomes]
        failed = [o == 'failed' for o in outcomes]

        missiles_intercepted = [m.get('missiles_intercepted', 0) for m in metrics]
        volley_sizes = [m.get('volley_size', 1) for m in metrics]

        # Extract per-missile minimum distances for accurate plotting
        # Fall back to episode-level distance if per-missile distances not available (old data)
        all_missile_distances = []
        has_per_missile_data = any('missile_min_distances' in m for m in metrics)

        if has_per_missile_data:
            for m in metrics:
                min_dists = m.get('missile_min_distances', [])
                all_missile_distances.extend(min_dists)
        else:
            # Backward compatibility: approximate per-missile distances
            # Intercepted missiles assumed to be at intercept radius (150m)
            # Non-intercepted missiles use final_distance
            for m in metrics:
                volley_sz = m.get('volley_size', 1)
                intercepted = m.get('missiles_intercepted', 0)
                episode_dist = m.get('final_distance', 0)

                # Approximate: intercepted missiles at 150m, rest at final_distance
                distances = [150.0] * intercepted + [episode_dist] * (volley_sz - intercepted)
                all_missile_distances.extend(distances)

        # Calculate rates
        all_intercept_rate = np.mean(all_intercepted) * 100
        partial_rate = np.mean(partial) * 100
        fail_rate = np.mean(failed) * 100

        # Overall missile interception rate
        total_missiles = sum(volley_sizes)
        total_intercepted = sum(missiles_intercepted)
        overall_rate = (total_intercepted / total_missiles * 100) if total_missiles > 0 else 0

        # Separate by outcome type (using per-missile distances)
        all_success_distances = []
        partial_distances = []
        fail_distances = []

        if has_per_missile_data:
            for m, o in zip(metrics, outcomes):
                min_dists = m.get('missile_min_distances', [])
                if o == 'all_intercepted':
                    all_success_distances.extend(min_dists)
                elif o == 'partial_interception':
                    partial_distances.extend(min_dists)
                else:
                    fail_distances.extend(min_dists)
        else:
            # Backward compatibility: approximate per-missile distances
            for m, o in zip(metrics, outcomes):
                episode_dist = m.get('final_distance', 0)
                volley_sz = m.get('volley_size', 1)
                intercepted = m.get('missiles_intercepted', 0)

                # Approximate: intercepted at 150m, missed at final_distance
                dists = [150.0] * intercepted + [episode_dist] * (volley_sz - intercepted)

                if o == 'all_intercepted':
                    all_success_distances.extend(dists)
                elif o == 'partial_interception':
                    partial_distances.extend(dists)
                else:
                    fail_distances.extend(dists)

        stats = {
            'volley_mode': True,
            'total_episodes': len(metrics),
            'volley_size': volley_sizes[0] if volley_sizes else 1,
            'total_missiles': total_missiles,
            'total_intercepted': total_intercepted,
            'all_intercept_count': sum(all_intercepted),
            'partial_count': sum(partial),
            'fail_count': sum(failed),
            'all_intercept_rate': all_intercept_rate,
            'partial_rate': partial_rate,
            'fail_rate': fail_rate,
            'overall_interception_rate': overall_rate,
            'avg_missiles_per_episode': np.mean(missiles_intercepted),
            'avg_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'avg_reward': np.mean(rewards),
            'avg_steps': np.mean(steps),
            'avg_fuel_used': np.mean(fuel_used),
            'all_success_avg_distance': np.mean(all_success_distances) if all_success_distances else 0,
            'partial_avg_distance': np.mean(partial_distances) if partial_distances else 0,
            'fail_avg_distance': np.mean(fail_distances) if fail_distances else 0,
        }

        return stats, {
            'distances': distances,
            'rewards': rewards,
            'steps': steps,
            'fuel_used': fuel_used,
            'all_intercepted': all_intercepted,
            'partial': partial,
            'failed': failed,
            'missiles_intercepted': missiles_intercepted,
            'volley_sizes': volley_sizes,
            'all_missile_distances': all_missile_distances,  # Per-missile distances for plotting
            'all_success_distances': all_success_distances,
            'partial_distances': partial_distances,
            'fail_distances': fail_distances
        }
    else:
        # Standard single-missile mode
        intercepted = [o == 'intercepted' for o in outcomes]
        intercept_rate = np.mean(intercepted) * 100

        # Separate successful and failed
        success_distances = [d for d, i in zip(distances, intercepted) if i]
        fail_distances = [d for d, i in zip(distances, intercepted) if not i]

        stats = {
            'volley_mode': False,
            'total_episodes': len(metrics),
            'intercept_count': sum(intercepted),
            'intercept_rate': intercept_rate,
            'avg_distance': np.mean(distances),
            'median_distance': np.median(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'avg_reward': np.mean(rewards),
            'avg_steps': np.mean(steps),
            'avg_fuel_used': np.mean(fuel_used),
            'success_avg_distance': np.mean(success_distances) if success_distances else 0,
            'fail_avg_distance': np.mean(fail_distances) if fail_distances else 0,
        }

        return stats, {
            'distances': distances,
            'rewards': rewards,
            'steps': steps,
            'fuel_used': fuel_used,
            'intercepted': intercepted
        }


def generate_plots(data, output_dir, is_volley=False):
    """Generate matplotlib plots."""
    plots = {}

    # 1. Intercept Success Rate
    fig, ax = plt.subplots(figsize=(8, 6))
    if is_volley:
        # Volley mode: three categories
        all_count = sum(data['all_intercepted'])
        partial_count = sum(data['partial'])
        fail_count = sum(data['failed'])
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        labels = ['All Intercepted', 'Partial', 'Failed']
        ax.pie([all_count, partial_count, fail_count], labels=labels,
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Episode Outcomes (Volley Mode)', fontsize=14, fontweight='bold')
    else:
        # Standard mode: two categories
        success_count = sum(data['intercepted'])
        fail_count = len(data['intercepted']) - success_count
        colors = ['#2ecc71', '#e74c3c']
        ax.pie([success_count, fail_count], labels=['Intercepted', 'Failed'],
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Intercept Success Rate', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = output_dir / 'success_rate.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plots['success_rate'] = plot_path
    plt.close()

    # 2. Final Distance Distribution (per-missile for volley mode)
    fig, ax = plt.subplots(figsize=(10, 6))
    if is_volley:
        # Volley mode: use per-missile minimum distances
        all_success_dists = data['all_success_distances']
        partial_dists = data['partial_distances']
        fail_dists = data['fail_distances']

        max_dist = max(data['all_missile_distances']) if data['all_missile_distances'] else 1000
        bins = np.linspace(0, max_dist, 30)
        ax.hist(all_success_dists, bins=bins, alpha=0.7, label='All Intercepted', color='#2ecc71', edgecolor='black')
        ax.hist(partial_dists, bins=bins, alpha=0.7, label='Partial', color='#f39c12', edgecolor='black')
        ax.hist(fail_dists, bins=bins, alpha=0.7, label='Failed', color='#e74c3c', edgecolor='black')
    else:
        # Standard mode: two outcome types
        success_distances = [d for d, i in zip(data['distances'], data['intercepted']) if i]
        fail_distances = [d for d, i in zip(data['distances'], data['intercepted']) if not i]

        bins = np.linspace(0, max(data['distances']) if data['distances'] else 1000, 30)
        ax.hist(success_distances, bins=bins, alpha=0.7, label='Intercepted', color='#2ecc71', edgecolor='black')
        ax.hist(fail_distances, bins=bins, alpha=0.7, label='Failed', color='#e74c3c', edgecolor='black')

    ax.axvline(200, color='orange', linestyle='--', linewidth=2, label='200m Threshold')
    ax.set_xlabel('Minimum Distance to Missile (m)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    title = 'Per-Missile Distance Distribution' if is_volley else 'Final Distance Distribution'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / 'distance_distribution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plots['distance_dist'] = plot_path
    plt.close()

    # 3. Distance Timeline (per-missile scatter for volley mode)
    fig, ax = plt.subplots(figsize=(12, 6))
    if is_volley:
        # Plot all missile minimum distances as individual points
        missile_idx = 0
        for ep, (all_int, part) in enumerate(zip(data['all_intercepted'], data['partial'])):
            # Determine color for this episode's missiles
            if all_int:
                color = '#2ecc71'
            elif part:
                color = '#f39c12'
            else:
                color = '#e74c3c'

            # Get missile distances for this episode
            num_missiles = data['volley_sizes'][ep]
            for i in range(num_missiles):
                if missile_idx < len(data['all_missile_distances']):
                    dist = data['all_missile_distances'][missile_idx]
                    ax.scatter(ep, dist, c=color, alpha=0.6, s=50)
                    missile_idx += 1
    else:
        episodes = range(len(data['distances']))
        colors_list = ['#2ecc71' if i else '#e74c3c' for i in data['intercepted']]
        ax.scatter(episodes, data['distances'], c=colors_list, alpha=0.6, s=50)

    ax.axhline(200, color='orange', linestyle='--', linewidth=2, label='200m Threshold')
    ax.set_xlabel('Episode', fontsize=12)
    ylabel = 'Minimum Distance to Each Missile (m)' if is_volley else 'Final Distance (m)'
    ax.set_ylabel(ylabel, fontsize=12)
    title = 'Per-Missile Minimum Distances' if is_volley else 'Final Distance per Episode'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / 'distance_timeline.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plots['distance_timeline'] = plot_path
    plt.close()

    # 4. Rewards Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    if is_volley:
        all_rewards = [r for r, a in zip(data['rewards'], data['all_intercepted']) if a]
        partial_rewards = [r for r, p in zip(data['rewards'], data['partial']) if p]
        fail_rewards = [r for r, f in zip(data['rewards'], data['failed']) if f]

        ax.hist(all_rewards, bins=20, alpha=0.7, label='All Intercepted', color='#2ecc71', edgecolor='black')
        ax.hist(partial_rewards, bins=20, alpha=0.7, label='Partial', color='#f39c12', edgecolor='black')
        ax.hist(fail_rewards, bins=20, alpha=0.7, label='Failed', color='#e74c3c', edgecolor='black')
    else:
        success_rewards = [r for r, i in zip(data['rewards'], data['intercepted']) if i]
        fail_rewards = [r for r, i in zip(data['rewards'], data['intercepted']) if not i]

        ax.hist(success_rewards, bins=20, alpha=0.7, label='Intercepted', color='#2ecc71', edgecolor='black')
        ax.hist(fail_rewards, bins=20, alpha=0.7, label='Failed', color='#e74c3c', edgecolor='black')

    ax.set_xlabel('Total Reward', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Reward Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / 'reward_distribution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plots['reward_dist'] = plot_path
    plt.close()

    # 5. Volley-specific: Missiles Intercepted per Episode (only for volley mode)
    if is_volley:
        fig, ax = plt.subplots(figsize=(12, 6))
        episodes = range(len(data['missiles_intercepted']))
        volley_sizes = data['volley_sizes']
        missiles_int = data['missiles_intercepted']

        # Bar chart showing missiles intercepted vs total
        x = np.arange(len(episodes))
        ax.bar(x, volley_sizes, label='Total Missiles', color='#95a5a6', alpha=0.5, edgecolor='black')
        ax.bar(x, missiles_int, label='Intercepted', color='#2ecc71', alpha=0.8, edgecolor='black')

        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Missiles', fontsize=12)
        ax.set_title('Missiles Intercepted per Episode', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plot_path = output_dir / 'volley_interception.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plots['volley_interception'] = plot_path
        plt.close()

    # 6. Fuel Usage
    fig, ax = plt.subplots(figsize=(10, 6))
    if is_volley:
        all_fuel = [f for f, a in zip(data['fuel_used'], data['all_intercepted']) if a]
        partial_fuel = [f for f, p in zip(data['fuel_used'], data['partial']) if p]
        fail_fuel = [f for f, f_flag in zip(data['fuel_used'], data['failed']) if f_flag]

        box_data = [all_fuel, partial_fuel, fail_fuel]
        bp = ax.boxplot(box_data, labels=['All Int.', 'Partial', 'Failed'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#f39c12')
        bp['boxes'][2].set_facecolor('#e74c3c')
    else:
        success_fuel = [f for f, i in zip(data['fuel_used'], data['intercepted']) if i]
        fail_fuel = [f for f, i in zip(data['fuel_used'], data['intercepted']) if not i]

        box_data = [success_fuel, fail_fuel]
        bp = ax.boxplot(box_data, labels=['Intercepted', 'Failed'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')

    ax.set_ylabel('Fuel Used (%)', fontsize=12)
    ax.set_title('Fuel Consumption Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plot_path = output_dir / 'fuel_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plots['fuel_comparison'] = plot_path
    plt.close()

    # 7. Episode Length
    fig, ax = plt.subplots(figsize=(10, 6))
    if is_volley:
        all_steps = [s for s, a in zip(data['steps'], data['all_intercepted']) if a]
        partial_steps = [s for s, p in zip(data['steps'], data['partial']) if p]
        fail_steps = [s for s, f in zip(data['steps'], data['failed']) if f]

        box_data = [all_steps, partial_steps, fail_steps]
        bp = ax.boxplot(box_data, labels=['All Int.', 'Partial', 'Failed'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#f39c12')
        bp['boxes'][2].set_facecolor('#e74c3c')
    else:
        success_steps = [s for s, i in zip(data['steps'], data['intercepted']) if i]
        fail_steps = [s for s, i in zip(data['steps'], data['intercepted']) if not i]

        box_data = [success_steps, fail_steps]
        bp = ax.boxplot(box_data, labels=['Intercepted', 'Failed'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')

    ax.set_ylabel('Steps', fontsize=12)
    ax.set_title('Episode Length Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plot_path = output_dir / 'steps_comparison.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plots['steps_comparison'] = plot_path
    plt.close()

    return plots


def generate_html_report(stats, plots, output_path, is_volley=False):
    """Generate HTML report."""
    # Generate volley-specific or standard summary
    if is_volley:
        mode_title = f"🚀 Missile Interception - Volley Mode Report (Size: {stats['volley_size']})"
        summary_stats = f"""
            <div class="stat-box">
                <div class="stat-label">Total Episodes</div>
                <div class="stat-value">{stats['total_episodes']}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Total Missiles</div>
                <div class="stat-value">{stats['total_missiles']}</div>
            </div>
            <div class="stat-box success">
                <div class="stat-label">Total Intercepted</div>
                <div class="stat-value">{stats['total_intercepted']}</div>
            </div>
            <div class="stat-box {'success' if stats['overall_interception_rate'] >= 70 else 'warning' if stats['overall_interception_rate'] >= 50 else 'danger'}">
                <div class="stat-label">Overall Interception Rate</div>
                <div class="stat-value">{stats['overall_interception_rate']:.1f}%</div>
            </div>
            <div class="stat-box success">
                <div class="stat-label">All Intercepted</div>
                <div class="stat-value">{stats['all_intercept_rate']:.1f}%</div>
            </div>
            <div class="stat-box warning">
                <div class="stat-label">Partial Interception</div>
                <div class="stat-value">{stats['partial_rate']:.1f}%</div>
            </div>
            <div class="stat-box danger">
                <div class="stat-label">Complete Failure</div>
                <div class="stat-value">{stats['fail_rate']:.1f}%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Avg Missiles/Episode</div>
                <div class="stat-value">{stats['avg_missiles_per_episode']:.2f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Average Final Distance</div>
                <div class="stat-value">{stats['avg_distance']:.1f}m</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Average Reward</div>
                <div class="stat-value">{stats['avg_reward']:.1f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Average Episode Length</div>
                <div class="stat-value">{stats['avg_steps']:.0f} steps</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Average Fuel Used</div>
                <div class="stat-value">{stats['avg_fuel_used']:.1f}%</div>
            </div>
        """
        breakdown_stats = f"""
            <div class="stat-box success">
                <div class="stat-label">Avg Distance (All Intercepted)</div>
                <div class="stat-value">{stats['all_success_avg_distance']:.1f}m</div>
            </div>
            <div class="stat-box warning">
                <div class="stat-label">Avg Distance (Partial)</div>
                <div class="stat-value">{stats['partial_avg_distance']:.1f}m</div>
            </div>
            <div class="stat-box danger">
                <div class="stat-label">Avg Distance (Failed)</div>
                <div class="stat-value">{stats['fail_avg_distance']:.1f}m</div>
            </div>
        """
        volley_plot = f"""
    <div class="plot">
        <h3>Missiles Intercepted per Episode</h3>
        <img src="{plots['volley_interception'].name}" alt="Volley Interception">
        <p style="color: #7f8c8d; text-align: center; margin-top: 10px;">
            Green bars show missiles intercepted, gray shows total volley size
        </p>
    </div>
        """ if 'volley_interception' in plots else ""
    else:
        mode_title = "🚀 Missile Interception - Inference Metrics Report"
        summary_stats = f"""
            <div class="stat-box">
                <div class="stat-label">Total Episodes</div>
                <div class="stat-value">{stats['total_episodes']}</div>
            </div>
            <div class="stat-box success">
                <div class="stat-label">Successful Intercepts</div>
                <div class="stat-value">{stats['intercept_count']}</div>
            </div>
            <div class="stat-box {'success' if stats['intercept_rate'] >= 50 else 'warning' if stats['intercept_rate'] >= 30 else 'danger'}">
                <div class="stat-label">Intercept Success Rate</div>
                <div class="stat-value">{stats['intercept_rate']:.1f}%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Average Final Distance</div>
                <div class="stat-value">{stats['avg_distance']:.1f}m</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Median Final Distance</div>
                <div class="stat-value">{stats['median_distance']:.1f}m</div>
            </div>
            <div class="stat-box success">
                <div class="stat-label">Best Intercept Distance</div>
                <div class="stat-value">{stats['min_distance']:.1f}m</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Average Reward</div>
                <div class="stat-value">{stats['avg_reward']:.1f}</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Average Episode Length</div>
                <div class="stat-value">{stats['avg_steps']:.0f} steps</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Average Fuel Used</div>
                <div class="stat-value">{stats['avg_fuel_used']:.1f}%</div>
            </div>
        """
        breakdown_stats = f"""
            <div class="stat-box success">
                <div class="stat-label">Avg Distance (Intercepted)</div>
                <div class="stat-value">{stats['success_avg_distance']:.1f}m</div>
            </div>
            <div class="stat-box danger">
                <div class="stat-label">Avg Distance (Failed)</div>
                <div class="stat-value">{stats['fail_avg_distance']:.1f}m</div>
            </div>
        """
        volley_plot = ""

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Inference Metrics Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-box {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #3498db;
        }}
        .stat-label {{
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .success {{
            border-left-color: #2ecc71;
        }}
        .warning {{
            border-left-color: #f39c12;
        }}
        .danger {{
            border-left-color: #e74c3c;
        }}
        .plot {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        .plot img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 14px;
            text-align: right;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <h1>{mode_title}</h1>

    <div class="summary">
        <h2>📊 Summary Statistics</h2>
        <div class="stat-grid">
            {summary_stats}
        </div>
    </div>

    <div class="summary">
        <h2>🎯 Performance Breakdown</h2>
        <div class="stat-grid">
            {breakdown_stats}
        </div>
    </div>

    <h2>📈 Visualizations</h2>

    <div class="plot">
        <h3>Intercept Success Rate</h3>
        <img src="{plots['success_rate'].name}" alt="Success Rate">
    </div>

    <div class="plot">
        <h3>Final Distance Distribution</h3>
        <img src="{plots['distance_dist'].name}" alt="Distance Distribution">
        <p style="color: #7f8c8d; text-align: center; margin-top: 10px;">
            Orange dashed line shows 200m intercept threshold (curriculum learning final radius)
        </p>
    </div>

    <div class="plot">
        <h3>Distance Over Episodes</h3>
        <img src="{plots['distance_timeline'].name}" alt="Distance Timeline">
        <p style="color: #7f8c8d; text-align: center; margin-top: 10px;">
            Green = Intercepted, Red = Failed
        </p>
    </div>

    <div class="plot">
        <h3>Reward Distribution</h3>
        <img src="{plots['reward_dist'].name}" alt="Reward Distribution">
    </div>

    {volley_plot}

    <div class="plot">
        <h3>Fuel Consumption</h3>
        <img src="{plots['fuel_comparison'].name}" alt="Fuel Comparison">
    </div>

    <div class="plot">
        <h3>Episode Length</h3>
        <img src="{plots['steps_comparison'].name}" alt="Steps Comparison">
        <p style="color: #7f8c8d; text-align: center; margin-top: 10px;">
            Shorter episodes often indicate successful early interception
        </p>
    </div>

    <div class="timestamp">
        Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_metrics.py <path_to_metrics.jsonl>")
        print("Example: python analyze_metrics.py inference_results/offline_run_*/*/metrics.jsonl")
        sys.exit(1)

    metrics_path = Path(sys.argv[1])

    if not metrics_path.exists():
        print(f"Error: File not found: {metrics_path}")
        sys.exit(1)

    print(f"Loading metrics from: {metrics_path}")
    metrics = load_metrics(metrics_path)
    print(f"✓ Loaded {len(metrics)} episodes")

    print("Analyzing metrics...")
    stats, data = analyze_metrics(metrics)
    is_volley = stats.get('volley_mode', False)

    print(f"\n{'='*60}")
    if is_volley:
        print(f"QUICK STATS (VOLLEY MODE - Size: {stats['volley_size']}):")
        print(f"{'='*60}")
        print(f"Total Episodes:         {stats['total_episodes']}")
        print(f"Total Missiles:         {stats['total_missiles']}")
        print(f"Missiles Intercepted:   {stats['total_intercepted']} ({stats['overall_interception_rate']:.1f}%)")
        print(f"All Intercepted Rate:   {stats['all_intercept_rate']:.1f}%")
        print(f"Partial Rate:           {stats['partial_rate']:.1f}%")
        print(f"Failure Rate:           {stats['fail_rate']:.1f}%")
        print(f"Avg Missiles/Episode:   {stats['avg_missiles_per_episode']:.2f}")
    else:
        print("QUICK STATS:")
        print(f"{'='*60}")
        print(f"Total Episodes:    {stats['total_episodes']}")
        print(f"Intercepts:        {stats['intercept_count']} ({stats['intercept_rate']:.1f}%)")
        print(f"Avg Distance:      {stats['avg_distance']:.1f}m")
        print(f"Best Distance:     {stats['min_distance']:.1f}m")
    print(f"Avg Fuel Used:     {stats['avg_fuel_used']:.1f}%")
    print(f"{'='*60}\n")

    output_dir = metrics_path.parent
    print(f"Generating plots in: {output_dir}")
    plots = generate_plots(data, output_dir, is_volley=is_volley)
    print(f"✓ Generated {len(plots)} plots")

    html_path = output_dir / 'report.html'
    print(f"Generating HTML report: {html_path}")
    generate_html_report(stats, plots, html_path, is_volley=is_volley)

    print(f"\n{'='*60}")
    print("✅ DONE!")
    print(f"{'='*60}")
    print(f"Report: {html_path}")
    print(f"\nOpen in browser:")
    print(f"  firefox {html_path}")
    print(f"  google-chrome {html_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
