#!/usr/bin/env python3
"""
Paper Benchmark Suite for Hlynr Intercept
==========================================

Comprehensive evaluation suite for generating paper-quality results.
Runs all baselines, ablations, and controlled experiments with statistical rigor.

Usage:
    # Full benchmark (takes ~2-3 hours)
    python scripts/paper_benchmark_suite.py --full

    # Quick sanity check (10 episodes per config)
    python scripts/paper_benchmark_suite.py --quick

    # Specific experiments
    python scripts/paper_benchmark_suite.py --experiments baselines ablations

    # Generate tables/figures from existing results
    python scripts/paper_benchmark_suite.py --analyze-only results/benchmark_20241214/

Authors: Roman Slack, Quinn Hasse
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import subprocess

sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import PercentFormatter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    n_episodes: int = 100          # Episodes per configuration
    n_seeds: int = 5               # Number of random seeds for statistical significance
    seed_base: int = 42            # Base seed (seeds will be seed_base + i)
    output_dir: str = "benchmark_results"

    # Model paths (to be filled in)
    flat_ppo_model: str = ""
    hrl_selector: str = ""
    hrl_search: str = ""
    hrl_track: str = ""
    hrl_terminal: str = ""

    # Scenarios to test
    scenarios: List[str] = field(default_factory=lambda: ["easy", "medium", "hard"])

    # Which experiments to run
    experiments: List[str] = field(default_factory=lambda: [
        "baselines",      # PN variants, flat PPO, full HRL
        "ablations",      # HRL without components
        "radar_degradation",  # Varying radar quality
        "approach_angles",    # 360-degree coverage
        "lock_loss_recovery", # Radar lock loss scenarios
    ])


# =============================================================================
# Baseline Policies
# =============================================================================

class ProportionalNavigation:
    """Classical Proportional Navigation guidance law."""

    def __init__(self, N: float = 3.0, use_true_state: bool = True):
        """
        Args:
            N: Navigation constant (typically 3-5)
            use_true_state: If True, use ground truth LOS rate (upper bound)
                           If False, use radar-derived estimates (realistic)
        """
        self.N = N
        self.use_true_state = use_true_state
        self.name = f"PN_N{N}" + ("_true" if use_true_state else "_radar")

    def get_action(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        """Compute PN guidance command."""
        # Check for radar lock
        if obs[14] < 0.3:  # Low lock quality
            # Coast - maintain current heading with reduced thrust
            return np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        if self.use_true_state and info is not None:
            # Use true LOS rate from info (upper bound performance)
            los_rate_az = info.get('true_los_rate_az', obs[2] if len(obs) > 2 else 0)
            los_rate_el = info.get('true_los_rate_el', obs[3] if len(obs) > 3 else 0)
        else:
            # Use radar-derived LOS rates (realistic)
            # obs[2], obs[3] are normalized LOS rates
            los_rate_az = obs[2] * 0.5  # Denormalize (max 0.5 rad/s)
            los_rate_el = obs[3] * 0.5

        # PN law: a_cmd = N * V_c * LOS_rate
        # Simplified: action is proportional to LOS rate
        action = np.array([
            1.0,                      # Full forward thrust (closing)
            self.N * los_rate_az,     # Lateral correction
            self.N * los_rate_el,     # Vertical correction
            0.0, 0.0, 0.0             # No angular rate commands
        ], dtype=np.float32)

        return np.clip(action, -1.0, 1.0)


class AugmentedPN:
    """
    Augmented Proportional Navigation with target acceleration compensation.
    Uses Kalman-filtered estimates when available.
    """

    def __init__(self, N: float = 4.0, K_acc: float = 0.5):
        self.N = N
        self.K_acc = K_acc  # Acceleration compensation gain
        self.name = f"AugPN_N{N}_K{K_acc}"
        self.prev_los_rate = np.zeros(2)

    def get_action(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        """Compute Augmented PN with target acceleration term."""
        if obs[14] < 0.3:  # Low lock quality
            return np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        los_rate_az = obs[2] * 0.5
        los_rate_el = obs[3] * 0.5

        # Estimate LOS rate derivative (target acceleration effect)
        los_rate = np.array([los_rate_az, los_rate_el])
        los_acc = los_rate - self.prev_los_rate
        self.prev_los_rate = los_rate.copy()

        # APN: a_cmd = N * V_c * LOS_rate + K * target_acc_estimate
        action = np.array([
            1.0,
            self.N * los_rate_az + self.K_acc * los_acc[0],
            self.N * los_rate_el + self.K_acc * los_acc[1],
            0.0, 0.0, 0.0
        ], dtype=np.float32)

        return np.clip(action, -1.0, 1.0)


class PurePursuit:
    """Pure pursuit - always fly directly toward target."""

    def __init__(self):
        self.name = "PurePursuit"

    def get_action(self, obs: np.ndarray, info: Dict = None) -> np.ndarray:
        """Fly directly toward target bearing."""
        if obs[14] < 0.3:
            return np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # obs[0:3] is relative position (normalized)
        rel_pos = obs[0:3]
        if np.linalg.norm(rel_pos) < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Point toward target
        bearing = rel_pos / np.linalg.norm(rel_pos)

        return np.array([
            1.0,           # Full thrust
            bearing[1],    # Lateral
            bearing[2],    # Vertical
            0.0, 0.0, 0.0
        ], dtype=np.float32)


# =============================================================================
# Evaluation Runner
# =============================================================================

class PolicyEvaluator:
    """Runs policy evaluation with comprehensive metrics."""

    def __init__(self, config_path: str, output_dir: Path):
        self.config_path = config_path
        self.output_dir = output_dir

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def evaluate_classical_policy(
        self,
        policy,
        n_episodes: int,
        seed: int,
        scenario_override: Dict = None
    ) -> Dict[str, Any]:
        """
        Evaluate a classical (non-learned) policy.

        Returns comprehensive metrics dict.
        """
        from environment import InterceptEnvironment

        env_config = self.config['environment'].copy()
        if scenario_override:
            env_config.update(scenario_override)

        env = InterceptEnvironment(env_config)
        np.random.seed(seed)

        results = {
            'policy_name': policy.name,
            'n_episodes': n_episodes,
            'seed': seed,
            'episodes': []
        }

        for ep in range(n_episodes):
            obs, info = env.reset()

            episode_data = {
                'episode_id': ep,
                'min_distance': float('inf'),
                'final_distance': None,
                'steps': 0,
                'total_reward': 0.0,
                'fuel_used': 0.0,
                'lock_losses': 0,
                'lock_recoveries': 0,
                'success': False,
            }

            initial_fuel = obs[12]
            was_locked = obs[14] > 0.5

            done = False
            while not done:
                action = policy.get_action(obs, info)
                obs, reward, done, truncated, info = env.step(action)
                done = done or truncated

                episode_data['steps'] += 1
                episode_data['total_reward'] += reward

                # Track distance
                if 'interceptor_pos' in info and 'missile_pos' in info:
                    dist = np.linalg.norm(
                        np.array(info['interceptor_pos']) -
                        np.array(info['missile_pos'])
                    )
                    episode_data['min_distance'] = min(episode_data['min_distance'], dist)
                    episode_data['final_distance'] = dist

                # Track lock loss/recovery
                is_locked = obs[14] > 0.5
                if was_locked and not is_locked:
                    episode_data['lock_losses'] += 1
                elif not was_locked and is_locked:
                    episode_data['lock_recoveries'] += 1
                was_locked = is_locked

            episode_data['fuel_used'] = initial_fuel - obs[12]
            episode_data['success'] = info.get('intercepted', False) or \
                                     episode_data['min_distance'] < env_config.get('proximity_kill_radius', 50)

            if episode_data['min_distance'] == float('inf'):
                episode_data['min_distance'] = None

            results['episodes'].append(episode_data)

        env.close()

        # Compute aggregate metrics
        results.update(self._compute_aggregate_metrics(results['episodes']))

        return results

    def evaluate_learned_policy(
        self,
        model_path: str,
        policy_type: str,  # 'ppo' or 'hrl'
        n_episodes: int,
        seed: int,
        hrl_paths: Dict = None,
        scenario_override: Dict = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a learned policy (PPO or HRL).

        Wraps the existing inference scripts.
        """
        if policy_type == 'ppo':
            # Use existing inference.py
            cmd = [
                "python", "inference.py",
                "--model", model_path,
                "--mode", "offline",
                "--episodes", str(n_episodes),
                "--seed", str(seed),
                "--config", self.config_path,
                "--output", str(self.output_dir / "temp_ppo")
            ]
        elif policy_type == 'hrl':
            # Use existing evaluate_hrl.py
            cmd = [
                "python", "scripts/evaluate_hrl.py",
                "--episodes", str(n_episodes),
                "--seed", str(seed),
                "--config", self.config_path,
                "--output", str(self.output_dir / "temp_hrl")
            ]
            if hrl_paths:
                if hrl_paths.get('selector'):
                    cmd.extend(["--selector", hrl_paths['selector']])
                if hrl_paths.get('search'):
                    cmd.extend(["--search", hrl_paths['search']])
                if hrl_paths.get('track'):
                    cmd.extend(["--track", hrl_paths['track']])
                if hrl_paths.get('terminal'):
                    cmd.extend(["--terminal", hrl_paths['terminal']])
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")

        # Run evaluation
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent.parent))

        if result.returncode != 0:
            print(f"Warning: Evaluation failed for {policy_type}")
            print(f"stderr: {result.stderr}")
            return None

        # Load results from the generated summary.json
        if policy_type == 'ppo':
            result_dirs = sorted((self.output_dir / "temp_ppo").glob("offline_run_*"))
        else:
            result_dirs = sorted((self.output_dir / "temp_hrl").glob("hrl_offline_run_*"))

        if not result_dirs:
            return None

        with open(result_dirs[-1] / "summary.json", 'r') as f:
            return json.load(f)

    def _compute_aggregate_metrics(self, episodes: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate statistics from episode list."""
        successes = [ep['success'] for ep in episodes]
        min_dists = [ep['min_distance'] for ep in episodes if ep['min_distance'] is not None]
        rewards = [ep['total_reward'] for ep in episodes]
        steps = [ep['steps'] for ep in episodes]
        fuel = [ep['fuel_used'] for ep in episodes]
        lock_losses = [ep['lock_losses'] for ep in episodes]
        lock_recoveries = [ep['lock_recoveries'] for ep in episodes]

        return {
            'success_rate': float(np.mean(successes)),
            'success_rate_std': float(np.std(successes) / np.sqrt(len(successes))),  # SEM

            'mean_min_distance': float(np.mean(min_dists)) if min_dists else None,
            'std_min_distance': float(np.std(min_dists)) if min_dists else None,
            'median_min_distance': float(np.median(min_dists)) if min_dists else None,
            'best_min_distance': float(np.min(min_dists)) if min_dists else None,

            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),

            'mean_steps': float(np.mean(steps)),
            'mean_fuel_used': float(np.mean(fuel)),

            'mean_lock_losses': float(np.mean(lock_losses)),
            'mean_lock_recoveries': float(np.mean(lock_recoveries)),
            'lock_recovery_rate': float(np.sum(lock_recoveries) / max(np.sum(lock_losses), 1)),

            # Precision brackets
            'sub_5m_rate': float(np.mean([d < 5 for d in min_dists])) if min_dists else 0,
            'sub_10m_rate': float(np.mean([d < 10 for d in min_dists])) if min_dists else 0,
            'sub_50m_rate': float(np.mean([d < 50 for d in min_dists])) if min_dists else 0,
            'sub_100m_rate': float(np.mean([d < 100 for d in min_dists])) if min_dists else 0,
        }


# =============================================================================
# Experiment Runners
# =============================================================================

def run_baseline_comparison(
    evaluator: PolicyEvaluator,
    config: BenchmarkConfig,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run baseline comparison: PN variants vs Flat PPO vs HRL.

    Returns results dict with all baselines.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: BASELINE COMPARISON")
    print("="*70)

    results = {'experiment': 'baselines', 'policies': {}}

    # Classical baselines
    classical_policies = [
        PurePursuit(),
        ProportionalNavigation(N=3.0, use_true_state=True),   # Upper bound
        ProportionalNavigation(N=3.0, use_true_state=False),  # Realistic
        ProportionalNavigation(N=4.0, use_true_state=False),
        ProportionalNavigation(N=5.0, use_true_state=False),
        AugmentedPN(N=4.0, K_acc=0.5),
    ]

    for policy in classical_policies:
        print(f"\nEvaluating {policy.name}...")

        seed_results = []
        for seed_idx in range(config.n_seeds):
            seed = config.seed_base + seed_idx
            res = evaluator.evaluate_classical_policy(
                policy,
                config.n_episodes,
                seed
            )
            seed_results.append(res)

        # Aggregate across seeds
        results['policies'][policy.name] = aggregate_seed_results(seed_results)
        print(f"  Success rate: {results['policies'][policy.name]['success_rate']*100:.1f}% "
              f"± {results['policies'][policy.name]['success_rate_ci']*100:.1f}%")

    # Learned policies
    if config.flat_ppo_model and Path(config.flat_ppo_model).exists():
        print(f"\nEvaluating Flat PPO...")
        seed_results = []
        for seed_idx in range(config.n_seeds):
            seed = config.seed_base + seed_idx
            res = evaluator.evaluate_learned_policy(
                config.flat_ppo_model,
                'ppo',
                config.n_episodes,
                seed
            )
            if res:
                seed_results.append(res)

        if seed_results:
            results['policies']['FlatPPO'] = aggregate_seed_results(seed_results)

    # HRL
    hrl_paths = {
        'selector': config.hrl_selector,
        'search': config.hrl_search,
        'track': config.hrl_track,
        'terminal': config.hrl_terminal,
    }

    if all(p and Path(p).exists() for p in hrl_paths.values() if p):
        print(f"\nEvaluating HRL...")
        seed_results = []
        for seed_idx in range(config.n_seeds):
            seed = config.seed_base + seed_idx
            res = evaluator.evaluate_learned_policy(
                None,
                'hrl',
                config.n_episodes,
                seed,
                hrl_paths=hrl_paths
            )
            if res:
                seed_results.append(res)

        if seed_results:
            results['policies']['HRL'] = aggregate_seed_results(seed_results)

    # Save results
    with open(output_dir / "baselines.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def run_ablation_studies(
    evaluator: PolicyEvaluator,
    config: BenchmarkConfig,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run HRL ablation studies.

    Tests:
    - HRL without forced transitions
    - Individual specialists only (no selector)
    - Selector with stub specialists
    """
    print("\n" + "="*70)
    print("EXPERIMENT: ABLATION STUDIES")
    print("="*70)

    results = {'experiment': 'ablations', 'variants': {}}

    ablation_configs = [
        {
            'name': 'HRL_NoForcedTransitions',
            'description': 'HRL with forced transitions disabled',
            'config_override': {'hrl': {'forced_transitions': False}}
        },
        {
            'name': 'HRL_SearchOnly',
            'description': 'Only Search specialist (no Track/Terminal)',
            'specialists': ['search']
        },
        {
            'name': 'HRL_TerminalOnly',
            'description': 'Only Terminal specialist',
            'specialists': ['terminal']
        },
        {
            'name': 'HRL_NoSelector',
            'description': 'Rule-based selector with trained specialists',
            'selector': None
        },
    ]

    for ablation in ablation_configs:
        print(f"\nEvaluating {ablation['name']}...")
        # Implementation would modify HRL evaluation based on ablation config
        # Placeholder for now
        results['variants'][ablation['name']] = {
            'description': ablation['description'],
            'status': 'not_implemented'
        }

    with open(output_dir / "ablations.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_radar_degradation_study(
    evaluator: PolicyEvaluator,
    config: BenchmarkConfig,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Test policy robustness to radar degradation.

    Varies:
    - Radar noise level
    - Beam width
    - Detection range
    - Lock loss probability
    """
    print("\n" + "="*70)
    print("EXPERIMENT: RADAR DEGRADATION")
    print("="*70)

    results = {'experiment': 'radar_degradation', 'conditions': {}}

    # Test conditions
    noise_levels = [0.05, 0.10, 0.15, 0.20, 0.30]  # radar_noise parameter
    beam_widths = [120, 90, 60, 45, 30]  # degrees

    # Evaluate HRL (or best policy) under each condition
    for noise in noise_levels:
        condition_name = f"noise_{noise}"
        print(f"\nEvaluating with radar_noise={noise}...")

        scenario_override = {'radar_noise': noise}

        # Test with best classical baseline (PN N=4)
        policy = ProportionalNavigation(N=4.0, use_true_state=False)
        res = evaluator.evaluate_classical_policy(
            policy,
            config.n_episodes,
            config.seed_base,
            scenario_override=scenario_override
        )

        results['conditions'][condition_name] = {
            'radar_noise': noise,
            'pn_success_rate': res['success_rate'],
            'pn_mean_min_distance': res['mean_min_distance'],
        }

    for beam in beam_widths:
        condition_name = f"beam_{beam}deg"
        print(f"\nEvaluating with beam_width={beam}°...")

        scenario_override = {'radar_beam_width': beam}

        policy = ProportionalNavigation(N=4.0, use_true_state=False)
        res = evaluator.evaluate_classical_policy(
            policy,
            config.n_episodes,
            config.seed_base,
            scenario_override=scenario_override
        )

        results['conditions'][condition_name] = {
            'beam_width': beam,
            'pn_success_rate': res['success_rate'],
            'pn_mean_min_distance': res['mean_min_distance'],
        }

    with open(output_dir / "radar_degradation.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def run_approach_angle_study(
    evaluator: PolicyEvaluator,
    config: BenchmarkConfig,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Test policy performance across 360° approach angles.

    Verifies LOS-frame invariance claim.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: 360° APPROACH ANGLES")
    print("="*70)

    results = {'experiment': 'approach_angles', 'angles': {}}

    # Test angles (degrees from north)
    angles = [0, 45, 90, 135, 180, 225, 270, 315]

    for angle in angles:
        print(f"\nEvaluating approach angle {angle}°...")

        # Compute spawn position for this angle
        # Missile spawns at angle relative to interceptor
        distance = 3000  # meters
        rad = np.radians(angle)
        missile_x = distance * np.sin(rad)
        missile_y = distance * np.cos(rad)

        scenario_override = {
            'missile_spawn': {
                'position': [[missile_x - 100, missile_y - 100, 2500],
                            [missile_x + 100, missile_y + 100, 3500]]
            }
        }

        policy = ProportionalNavigation(N=4.0, use_true_state=False)
        res = evaluator.evaluate_classical_policy(
            policy,
            config.n_episodes // 4,  # Fewer episodes per angle
            config.seed_base,
            scenario_override=scenario_override
        )

        results['angles'][f"{angle}deg"] = {
            'angle': angle,
            'success_rate': res['success_rate'],
            'mean_min_distance': res['mean_min_distance'],
        }

    # Check for directional bias
    success_rates = [r['success_rate'] for r in results['angles'].values()]
    results['directional_bias'] = {
        'mean_success': float(np.mean(success_rates)),
        'std_success': float(np.std(success_rates)),
        'max_deviation': float(max(success_rates) - min(success_rates)),
        'is_direction_invariant': float(np.std(success_rates)) < 0.1  # <10% variation
    }

    with open(output_dir / "approach_angles.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


# =============================================================================
# Results Aggregation and Visualization
# =============================================================================

def aggregate_seed_results(seed_results: List[Dict]) -> Dict[str, Any]:
    """Aggregate results across multiple random seeds."""
    if not seed_results:
        return {}

    # Keys to aggregate
    numeric_keys = [
        'success_rate', 'mean_min_distance', 'mean_reward',
        'mean_steps', 'mean_fuel_used', 'lock_recovery_rate',
        'sub_5m_rate', 'sub_10m_rate', 'sub_50m_rate', 'sub_100m_rate'
    ]

    aggregated = {'n_seeds': len(seed_results)}

    for key in numeric_keys:
        values = [r.get(key) for r in seed_results if r.get(key) is not None]
        if values:
            aggregated[key] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
            aggregated[f'{key}_ci'] = float(1.96 * np.std(values) / np.sqrt(len(values)))  # 95% CI

    return aggregated


def generate_latex_table(results: Dict[str, Any], output_path: Path):
    """Generate LaTeX table for paper."""

    table_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Baseline comparison across methods}",
        r"\label{tab:baselines}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Success (\%) & MID (m) & Time (s) & Fuel (\%) & Lock Recovery \\",
        r"\midrule",
    ]

    if 'policies' in results:
        for name, metrics in results['policies'].items():
            if not metrics or 'success_rate' not in metrics:
                continue

            sr = metrics.get('success_rate', 0) * 100
            sr_ci = metrics.get('success_rate_ci', 0) * 100
            mid = metrics.get('mean_min_distance', 0) or 0
            steps = metrics.get('mean_steps', 0) / 100  # Convert to seconds
            fuel = metrics.get('mean_fuel_used', 0) * 100
            lock = metrics.get('lock_recovery_rate', 0) * 100

            # Format with confidence intervals
            table_lines.append(
                f"{name} & {sr:.1f} $\\pm$ {sr_ci:.1f} & {mid:.1f} & {steps:.1f} & {fuel:.1f} & {lock:.0f} \\\\"
            )

    table_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(table_lines))

    print(f"LaTeX table saved to: {output_path}")


def generate_comparison_figure(results: Dict[str, Any], output_path: Path):
    """Generate bar chart comparing methods."""
    if not HAS_MATPLOTLIB:
        print("Skipping figure generation (matplotlib not available)")
        return

    if 'policies' not in results:
        return

    policies = results['policies']
    names = list(policies.keys())
    success_rates = [policies[n].get('success_rate', 0) * 100 for n in names]
    errors = [policies[n].get('success_rate_ci', 0) * 100 for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color code: classical=blue, learned=green
    colors = ['#2196F3' if 'PN' in n or 'Pursuit' in n else '#4CAF50' for n in names]

    x = np.arange(len(names))
    bars = ax.bar(x, success_rates, yerr=errors, capsize=5, color=colors, edgecolor='black')

    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Intercept Success Rate by Method', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 100)

    # Add value labels
    for bar, rate in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
               f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

    # Legend
    classical_patch = mpatches.Patch(color='#2196F3', label='Classical (PN)')
    learned_patch = mpatches.Patch(color='#4CAF50', label='Learned (RL)')
    ax.legend(handles=[classical_patch, learned_patch], loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Figure saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Paper Benchmark Suite for Hlynr Intercept",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--full", action="store_true",
                       help="Run full benchmark (100 episodes, 5 seeds)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test (10 episodes, 1 seed)")
    parser.add_argument("--experiments", nargs="+",
                       choices=['baselines', 'ablations', 'radar_degradation',
                               'approach_angles', 'lock_loss_recovery'],
                       help="Specific experiments to run")
    parser.add_argument("--analyze-only", type=str,
                       help="Only generate figures/tables from existing results")

    # Model paths
    parser.add_argument("--ppo-model", type=str, default="",
                       help="Path to flat PPO model")
    parser.add_argument("--hrl-selector", type=str, default="",
                       help="Path to HRL selector")
    parser.add_argument("--hrl-search", type=str, default="",
                       help="Path to HRL search specialist")
    parser.add_argument("--hrl-track", type=str, default="",
                       help="Path to HRL track specialist")
    parser.add_argument("--hrl-terminal", type=str, default="",
                       help="Path to HRL terminal specialist")

    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to environment config")
    parser.add_argument("--output", type=str, default="benchmark_results",
                       help="Output directory")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Episodes per configuration")
    parser.add_argument("--seeds", type=int, default=5,
                       help="Number of random seeds")

    args = parser.parse_args()

    # Set up configuration
    config = BenchmarkConfig(
        n_episodes=10 if args.quick else args.episodes,
        n_seeds=1 if args.quick else args.seeds,
        flat_ppo_model=args.ppo_model,
        hrl_selector=args.hrl_selector,
        hrl_search=args.hrl_search,
        hrl_track=args.hrl_track,
        hrl_terminal=args.hrl_terminal,
    )

    if args.experiments:
        config.experiments = args.experiments
    elif args.full:
        pass  # Use all default experiments
    elif args.quick:
        config.experiments = ['baselines']  # Just baselines for quick test

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / f"benchmark_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("HLYNR INTERCEPT - PAPER BENCHMARK SUITE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"Episodes per config: {config.n_episodes}")
    print(f"Random seeds: {config.n_seeds}")
    print(f"Experiments: {config.experiments}")
    print(f"{'='*70}\n")

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(asdict(config), f, indent=2)

    # Analysis-only mode
    if args.analyze_only:
        print(f"Analysis-only mode: loading results from {args.analyze_only}")
        results_dir = Path(args.analyze_only)

        if (results_dir / "baselines.json").exists():
            with open(results_dir / "baselines.json", 'r') as f:
                results = json.load(f)
            generate_latex_table(results, output_dir / "baselines_table.tex")
            generate_comparison_figure(results, output_dir / "baselines_figure.png")

        return 0

    # Create evaluator
    evaluator = PolicyEvaluator(args.config, output_dir)

    # Run experiments
    all_results = {}

    if 'baselines' in config.experiments:
        all_results['baselines'] = run_baseline_comparison(evaluator, config, output_dir)
        generate_latex_table(all_results['baselines'], output_dir / "baselines_table.tex")
        generate_comparison_figure(all_results['baselines'], output_dir / "baselines_figure.png")

    if 'ablations' in config.experiments:
        all_results['ablations'] = run_ablation_studies(evaluator, config, output_dir)

    if 'radar_degradation' in config.experiments:
        all_results['radar_degradation'] = run_radar_degradation_study(evaluator, config, output_dir)

    if 'approach_angles' in config.experiments:
        all_results['approach_angles'] = run_approach_angle_study(evaluator, config, output_dir)

    # Save all results
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"  - baselines.json + baselines_table.tex + baselines_figure.png")
    print(f"  - ablations.json")
    print(f"  - radar_degradation.json")
    print(f"  - approach_angles.json")
    print(f"  - all_results.json")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
