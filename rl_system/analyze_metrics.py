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
    outcomes = [m['outcome'] for m in metrics]
    distances = [m['final_distance'] for m in metrics]
    rewards = [m['total_reward'] for m in metrics]
    steps = [m['steps'] for m in metrics]
    fuel_used = [m['fuel_used'] for m in metrics]

    intercepted = [o == 'intercepted' for o in outcomes]
    intercept_rate = np.mean(intercepted) * 100

    # Separate successful and failed
    success_distances = [d for d, i in zip(distances, intercepted) if i]
    fail_distances = [d for d, i in zip(distances, intercepted) if not i]

    stats = {
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


def generate_plots(data, output_dir):
    """Generate matplotlib plots."""
    plots = {}

    # 1. Intercept Success Rate
    fig, ax = plt.subplots(figsize=(8, 6))
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

    # 2. Final Distance Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    success_distances = [d for d, i in zip(data['distances'], data['intercepted']) if i]
    fail_distances = [d for d, i in zip(data['distances'], data['intercepted']) if not i]

    bins = np.linspace(0, max(data['distances']), 30)
    ax.hist(success_distances, bins=bins, alpha=0.7, label='Intercepted', color='#2ecc71', edgecolor='black')
    ax.hist(fail_distances, bins=bins, alpha=0.7, label='Failed', color='#e74c3c', edgecolor='black')
    ax.axvline(200, color='orange', linestyle='--', linewidth=2, label='200m Threshold')
    ax.set_xlabel('Final Distance (m)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Final Distance Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / 'distance_distribution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plots['distance_dist'] = plot_path
    plt.close()

    # 3. Distance Over Episodes
    fig, ax = plt.subplots(figsize=(12, 6))
    episodes = range(len(data['distances']))
    colors_list = ['#2ecc71' if i else '#e74c3c' for i in data['intercepted']]
    ax.scatter(episodes, data['distances'], c=colors_list, alpha=0.6, s=50)
    ax.axhline(200, color='orange', linestyle='--', linewidth=2, label='200m Threshold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Final Distance (m)', fontsize=12)
    ax.set_title('Final Distance per Episode', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / 'distance_timeline.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plots['distance_timeline'] = plot_path
    plt.close()

    # 4. Rewards Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
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

    # 5. Fuel Usage
    fig, ax = plt.subplots(figsize=(10, 6))
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

    # 6. Episode Length
    fig, ax = plt.subplots(figsize=(10, 6))
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


def generate_html_report(stats, plots, output_path):
    """Generate HTML report."""
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
    <h1>🚀 Missile Interception - Inference Metrics Report</h1>

    <div class="summary">
        <h2>📊 Summary Statistics</h2>
        <div class="stat-grid">
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
        </div>
    </div>

    <div class="summary">
        <h2>🎯 Performance Breakdown</h2>
        <div class="stat-grid">
            <div class="stat-box success">
                <div class="stat-label">Avg Distance (Intercepted)</div>
                <div class="stat-value">{stats['success_avg_distance']:.1f}m</div>
            </div>
            <div class="stat-box danger">
                <div class="stat-label">Avg Distance (Failed)</div>
                <div class="stat-value">{stats['fail_avg_distance']:.1f}m</div>
            </div>
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

    print(f"\n{'='*60}")
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
    plots = generate_plots(data, output_dir)
    print(f"✓ Generated {len(plots)} plots")

    html_path = output_dir / 'report.html'
    print(f"Generating HTML report: {html_path}")
    generate_html_report(stats, plots, html_path)

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
