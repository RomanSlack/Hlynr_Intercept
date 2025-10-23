#!/usr/bin/env python3
"""
Training Monitor for Hlynr Intercept RL System
Provides real-time training diagnostics and status with terminal graphs
"""

import json
import os
import glob
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import plotext as plt


def find_latest_training():
    """Find the most recent training run."""
    log_dirs = glob.glob("logs/training_*")
    if not log_dirs:
        return None
    latest = max(log_dirs, key=os.path.getmtime)
    return latest


def parse_metrics(log_dir):
    """Parse metrics.jsonl for success rate and reward data."""
    metrics_file = Path(log_dir) / "metrics.jsonl"
    if not metrics_file.exists():
        return []

    metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                metrics.append(data)
            except:
                continue
    return metrics


def parse_training_log(log_dir):
    """Parse training log for loss/value metrics."""
    log_files = glob.glob(str(Path(log_dir) / "*.log"))
    if not log_files:
        return None

    latest_log = max(log_files, key=os.path.getmtime)

    # Extract latest metrics from log
    latest_metrics = {}
    with open(latest_log, 'r') as f:
        lines = f.readlines()
        # Look for the last occurrence of training metrics
        for line in reversed(lines):
            if "Step" in line and "learning_rate" in line:
                # This is a step log line, extract step number
                try:
                    step = int(line.split("Step ")[1].split(":")[0])
                    latest_metrics['step'] = step
                except:
                    pass
                break

    return latest_metrics


def calculate_statistics(metrics):
    """Calculate statistics from metrics."""
    if not metrics:
        return {}

    # Get recent metrics (last 500 episodes or all if less)
    recent = metrics[-500:]

    # Success rate progression
    success_rates = [m.get('success_rate_pct', 0) for m in recent if 'success_rate_pct' in m]
    rewards = [m.get('mean_reward', 0) for m in recent if 'mean_reward' in m]
    lengths = [m.get('mean_length', 0) for m in recent if 'mean_length' in m]

    stats = {
        'total_episodes': len(metrics),
        'recent_episodes': len(recent),
        'current_success_rate': success_rates[-1] if success_rates else 0,
        'mean_success_rate': np.mean(success_rates) if success_rates else 0,
        'max_success_rate': max(success_rates) if success_rates else 0,
        'current_reward': rewards[-1] if rewards else 0,
        'mean_reward': np.mean(rewards) if rewards else 0,
        'current_episode_length': lengths[-1] if lengths else 0,
        'mean_episode_length': np.mean(lengths) if lengths else 0,
    }

    # Find peak performance
    if success_rates:
        peak_idx = success_rates.index(max(success_rates))
        peak_episode_idx = len(metrics) - len(success_rates) + peak_idx
        stats['peak_episode'] = metrics[peak_episode_idx].get('episode', 0)
        stats['peak_steps'] = metrics[peak_episode_idx].get('total_timesteps', 0)

    return stats


def format_time(seconds):
    """Format seconds into human-readable time."""
    return str(timedelta(seconds=int(seconds)))


def get_training_progress(metrics):
    """Estimate training progress and ETA."""
    if not metrics:
        return None

    latest = metrics[-1]
    current_steps = latest.get('total_timesteps', 0)
    target_steps = 10_000_000  # Default target

    progress_pct = (current_steps / target_steps) * 100 if target_steps > 0 else 0

    # Estimate time remaining
    if len(metrics) > 1:
        first = metrics[0]
        time_elapsed = latest.get('timestamp', 0) - first.get('timestamp', 0)
        steps_done = current_steps - first.get('total_timesteps', 0)

        if steps_done > 0:
            steps_per_second = steps_done / time_elapsed
            steps_remaining = target_steps - current_steps
            seconds_remaining = steps_remaining / steps_per_second if steps_per_second > 0 else 0

            return {
                'progress_pct': progress_pct,
                'current_steps': current_steps,
                'target_steps': target_steps,
                'time_elapsed': time_elapsed,
                'time_remaining': seconds_remaining,
                'steps_per_second': steps_per_second,
            }

    return {
        'progress_pct': progress_pct,
        'current_steps': current_steps,
        'target_steps': target_steps,
    }


def plot_success_rate_graph(metrics):
    """Plot success rate over training with plotext."""
    success_data = [(m.get('total_timesteps', 0), m.get('success_rate_pct', 0))
                    for m in metrics if 'success_rate_pct' in m]

    if not success_data:
        return

    steps, rates = zip(*success_data)

    # Convert steps to millions for readability
    steps_m = [s / 1_000_000 for s in steps]

    print(f"\n{'='*70}")
    print("SUCCESS RATE OVER TRAINING")
    print(f"{'='*70}")

    plt.clf()
    plt.plot(steps_m, rates, marker="braille")
    plt.title("Success Rate vs Training Steps")
    plt.xlabel("Training Steps (Millions)")
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, max(max(rates) * 1.2, 25))  # Give some headroom
    plt.theme("dark")
    plt.plotsize(60, 15)
    plt.show()


def plot_reward_graph(metrics):
    """Plot reward over training with plotext."""
    reward_data = [(m.get('total_timesteps', 0), m.get('mean_reward', 0))
                   for m in metrics if 'mean_reward' in m]

    if not reward_data:
        return

    steps, rewards = zip(*reward_data)

    # Convert steps to millions for readability
    steps_m = [s / 1_000_000 for s in steps]

    print(f"\n{'='*70}")
    print("MEAN REWARD OVER TRAINING")
    print(f"{'='*70}")

    plt.clf()
    plt.plot(steps_m, rewards, marker="braille")
    plt.title("Mean Episode Reward vs Training Steps")
    plt.xlabel("Training Steps (Millions)")
    plt.ylabel("Mean Reward")
    plt.theme("dark")
    plt.plotsize(60, 15)
    plt.show()


def print_success_rate_trend(metrics, window=10):
    """Print success rate trend for last N measurements."""
    success_data = [(m.get('episode', 0), m.get('success_rate_pct', 0), m.get('total_timesteps', 0))
                    for m in metrics if 'success_rate_pct' in m]

    if not success_data:
        return

    recent = success_data[-window:]

    print(f"\n{'='*70}")
    print(f"SUCCESS RATE TREND (Last {len(recent)} measurements)")
    print(f"{'='*70}")
    print(f"{'Episode':<12} {'Steps':<12} {'Success %':<12} {'Trend':<20}")
    print(f"{'-'*70}")

    for i, (ep, rate, steps) in enumerate(recent):
        # Simple trend indicator
        trend = ""
        if i > 0:
            prev_rate = recent[i-1][1]
            if rate > prev_rate:
                trend = "↑ IMPROVING"
            elif rate < prev_rate:
                trend = "↓ DECLINING"
            else:
                trend = "→ STABLE"

        print(f"{ep:<12} {steps:<12} {rate:<12.1f} {trend:<20}")


def print_diagnostic_alerts(stats, metrics):
    """Print diagnostic alerts based on training patterns."""
    print(f"\n{'='*70}")
    print("DIAGNOSTIC ALERTS")
    print(f"{'='*70}")

    alerts = []

    # Check for catastrophic forgetting
    if stats.get('max_success_rate', 0) > 15 and stats.get('current_success_rate', 0) < stats['max_success_rate'] * 0.5:
        alerts.append(("⚠️  CATASTROPHIC FORGETTING",
                      f"Success rate dropped from {stats['max_success_rate']:.1f}% to {stats['current_success_rate']:.1f}%"))

    # Check for poor learning
    if stats.get('total_episodes', 0) > 1000 and stats.get('mean_success_rate', 0) < 5:
        alerts.append(("❌ POOR LEARNING",
                      f"Mean success rate only {stats['mean_success_rate']:.1f}% after {stats['total_episodes']} episodes"))

    # Check for reward farming
    if stats.get('current_reward', 0) > 3000 and stats.get('current_success_rate', 0) < 10:
        alerts.append(("⚠️  POSSIBLE REWARD FARMING",
                      f"High reward ({stats['current_reward']:.0f}) but low success ({stats['current_success_rate']:.1f}%)"))

    # Check for good progress
    if stats.get('current_success_rate', 0) > 15:
        alerts.append(("✅ GOOD PROGRESS",
                      f"Success rate at {stats['current_success_rate']:.1f}%"))

    # Check for improvement trend
    recent_rates = [m.get('success_rate_pct', 0) for m in metrics[-10:] if 'success_rate_pct' in m]
    if len(recent_rates) >= 5:
        early_avg = np.mean(recent_rates[:len(recent_rates)//2])
        late_avg = np.mean(recent_rates[len(recent_rates)//2:])
        if late_avg > early_avg * 1.2:
            alerts.append(("📈 IMPROVING TREND",
                          f"Recent success rate trending up ({early_avg:.1f}% → {late_avg:.1f}%)"))
        elif late_avg < early_avg * 0.8:
            alerts.append(("📉 DECLINING TREND",
                          f"Recent success rate trending down ({early_avg:.1f}% → {late_avg:.1f}%)"))

    if alerts:
        for title, message in alerts:
            print(f"{title}")
            print(f"  {message}")
    else:
        print("No significant alerts")


def print_recommendations(stats, metrics):
    """Print recommendations based on training state."""
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")

    current_success = stats.get('current_success_rate', 0)
    max_success = stats.get('max_success_rate', 0)
    mean_reward = stats.get('mean_reward', 0)

    if current_success < 5 and stats.get('total_episodes', 0) > 1500:
        print("❌ Training is failing:")
        print("   - Consider checking if radar is still detecting targets")
        print("   - Verify curriculum settings aren't too aggressive")
        print("   - Sample some episodes to see what policy is doing")
        print("   - May need to adjust spawn ranges or reward structure")

    elif max_success > 15 and current_success < max_success * 0.6:
        print("⚠️  Performance degraded from peak:")
        print(f"   - Peak was {max_success:.1f}% at episode {stats.get('peak_episode', '?')}")
        print("   - Likely curriculum is progressing too fast")
        print("   - Consider slowing curriculum progression")
        print("   - May need to train longer at current difficulty")

    elif 10 <= current_success < 20:
        print("⚠️  Moderate success - room for improvement:")
        print("   - Training is working but could be better")
        print("   - Consider extending training beyond 10M steps")
        print("   - May benefit from 2-layer LSTM or larger network")
        print("   - Check if curriculum can be safely progressed")

    elif current_success >= 20:
        print("✅ Good progress:")
        print(f"   - Current success rate: {current_success:.1f}%")
        print("   - Continue training to convergence")
        print("   - Monitor for stability and continued improvement")
        print("   - Consider enabling harder curriculum stages if stable")

    else:
        print("ℹ️  Early training:")
        print("   - Too early to assess, continue monitoring")
        print("   - Success rate should appear by episode 500-1000")


def main():
    print("\n" + "="*70)
    print("HLYNR INTERCEPT RL TRAINING MONITOR")
    print("="*70)

    # Find latest training
    log_dir = find_latest_training()
    if not log_dir:
        print("❌ No training runs found in logs/")
        return

    print(f"\n📂 Monitoring: {log_dir}")

    # Parse metrics
    metrics = parse_metrics(log_dir)
    if not metrics:
        print("❌ No metrics found (training may have just started)")
        return

    # Get statistics
    stats = calculate_statistics(metrics)
    progress = get_training_progress(metrics)

    # Print overview
    print(f"\n{'='*70}")
    print("TRAINING OVERVIEW")
    print(f"{'='*70}")
    print(f"Total Episodes:        {stats.get('total_episodes', 0):,}")
    print(f"Training Progress:     {progress.get('progress_pct', 0):.1f}% ({progress.get('current_steps', 0):,} / {progress.get('target_steps', 0):,} steps)")

    if 'time_elapsed' in progress:
        print(f"Time Elapsed:          {format_time(progress['time_elapsed'])}")
        print(f"Time Remaining (est):  {format_time(progress.get('time_remaining', 0))}")
        print(f"Training Speed:        {progress.get('steps_per_second', 0):.1f} steps/sec")

    # Print current stats
    print(f"\n{'='*70}")
    print("CURRENT PERFORMANCE")
    print(f"{'='*70}")
    print(f"Current Success Rate:  {stats.get('current_success_rate', 0):.1f}%")
    print(f"Mean Success Rate:     {stats.get('mean_success_rate', 0):.1f}% (last {stats.get('recent_episodes', 0)} episodes)")
    print(f"Peak Success Rate:     {stats.get('max_success_rate', 0):.1f}%", end="")
    if 'peak_episode' in stats:
        print(f" (episode {stats['peak_episode']}, {stats['peak_steps']:,} steps)")
    else:
        print()

    print(f"\nCurrent Mean Reward:   {stats.get('current_reward', 0):.1f}")
    print(f"Average Episode Length: {stats.get('current_episode_length', 0):.0f} steps")

    # Plot graphs
    plot_success_rate_graph(metrics)
    plot_reward_graph(metrics)

    # Print success rate trend
    print_success_rate_trend(metrics, window=15)

    # Print diagnostic alerts
    print_diagnostic_alerts(stats, metrics)

    # Print recommendations
    print_recommendations(stats, metrics)

    print(f"\n{'='*70}")
    print("END OF REPORT")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
