#!/usr/bin/env python3
"""
Policy Comparison Script

Compare flat PPO and HRL policies on the same episodes using identical seeds.
Perform statistical significance testing and generate comparison report.

Usage:
    # Compare flat PPO vs HRL
    python compare_policies.py --flat checkpoints/flat_ppo/best \\
                              --hrl-selector checkpoints/hrl/selector/best \\
                              --hrl-search checkpoints/hrl/specialists/search/best \\
                              --hrl-track checkpoints/hrl/specialists/track/best \\
                              --hrl-terminal checkpoints/hrl/specialists/terminal/best \\
                              --episodes 100 --config config.yaml

    # Quick comparison with 20 episodes
    python compare_policies.py --flat checkpoints/flat/model --hrl-selector checkpoints/hrl/selector --episodes 20

    # Save comparison report
    python compare_policies.py --flat ... --hrl-selector ... --output results/comparison.json

Example:
    python compare_policies.py --flat checkpoints/flat_ppo/best --hrl-selector checkpoints/hrl/selector/best --episodes 50
"""
import argparse
import json
import yaml
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from scipy import stats
import sys

sys.path.append(str(Path(__file__).parent.parent))

from environment import InterceptEnvironment
from hrl.hierarchical_env import make_hrl_env
from hrl.option_definitions import Option
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class PolicyComparator:
    """Compare flat PPO and HRL policies."""

    def __init__(
        self,
        flat_path: str,
        hrl_selector_path: Optional[str],
        hrl_specialist_paths: Dict[Option, str],
        config: Dict[str, Any],
        seed: Optional[int] = None,
    ):
        """
        Initialize policy comparator.

        Args:
            flat_path: Path to flat PPO checkpoint
            hrl_selector_path: Path to HRL selector (None for rule-based)
            hrl_specialist_paths: Dict mapping Option to specialist paths
            config: Configuration dictionary
            seed: Base random seed
        """
        self.logger = logging.getLogger("PolicyComparator")
        self.logger.setLevel(logging.INFO)
        self.config = config
        self.seed = seed if seed is not None else 42

        # Load flat PPO model
        self.logger.info(f"Loading flat PPO model from {flat_path}")
        self.flat_model = PPO.load(flat_path)

        # Create flat environment
        env_config = config['environment'].copy()
        env_config['dt'] = config.get('dt', 0.01)
        env_config['max_steps'] = config['environment']['max_steps']
        flat_env = InterceptEnvironment(env_config)
        self.flat_env = DummyVecEnv([lambda: flat_env])

        # Create HRL environment
        self.logger.info(f"Creating HRL environment")
        self.logger.info(f"  Selector: {hrl_selector_path or 'rule-based'}")
        self.logger.info(f"  Specialists: {hrl_specialist_paths}")

        hrl_base_env = InterceptEnvironment(env_config)
        self.hrl_env = make_hrl_env(
            base_env=hrl_base_env,
            cfg=config,
            selector_path=hrl_selector_path,
            specialist_paths=hrl_specialist_paths,
            mode='inference',
        )

        self.logger.info("Policy comparator initialized")

    def compare(self, n_episodes: int) -> Dict[str, Any]:
        """
        Run comparison on N episodes with same seeds.

        Args:
            n_episodes: Number of episodes to compare

        Returns:
            Comparison results dictionary
        """
        self.logger.info(f"Starting comparison: {n_episodes} episodes")

        flat_results = []
        hrl_results = []

        for episode in range(n_episodes):
            seed = self.seed + episode

            # Run flat PPO episode
            flat_result = self._run_flat_episode(episode, seed)
            flat_results.append(flat_result)

            # Run HRL episode
            hrl_result = self._run_hrl_episode(episode, seed)
            hrl_results.append(hrl_result)

            # Log progress
            if (episode + 1) % 10 == 0:
                self.logger.info(f"Completed {episode + 1}/{n_episodes} episodes")

        # Compute comparative statistics
        comparison = self._compute_comparison(flat_results, hrl_results)

        self.logger.info("Comparison complete")
        self._print_comparison_report(comparison)

        return comparison

    def _run_flat_episode(self, episode_id: int, seed: int) -> Dict[str, Any]:
        """
        Run episode with flat PPO policy.

        Args:
            episode_id: Episode number
            seed: Random seed

        Returns:
            Episode metrics
        """
        obs = self.flat_env.reset()
        # Note: VecEnv doesn't support seed parameter in reset, set manually
        self.flat_env.env_method('seed', seed)
        obs = self.flat_env.reset()

        episode_reward = 0.0
        episode_steps = 0
        min_distance = float('inf')
        final_distance = None
        fuel_used = 0.0
        initial_fuel = None

        done = False
        while not done:
            action, _ = self.flat_model.predict(obs, deterministic=True)
            obs, reward, done, info = self.flat_env.step(action)

            episode_reward += reward[0]
            episode_steps += 1

            # Extract metrics from observation
            obs_unwrapped = obs[0]  # VecEnv wraps observations
            rel_pos = obs_unwrapped[0:3]
            distance = np.linalg.norm(rel_pos)
            min_distance = min(min_distance, distance)
            final_distance = distance

            if initial_fuel is None:
                initial_fuel = obs_unwrapped[12]
            fuel_used = initial_fuel - obs_unwrapped[12]

            if done[0]:
                break

        success = final_distance < 50.0 if final_distance is not None else False

        return {
            'episode_id': episode_id,
            'seed': seed,
            'reward': float(episode_reward),
            'steps': episode_steps,
            'success': success,
            'miss_distance': float(final_distance) if final_distance is not None else None,
            'min_distance': float(min_distance) if min_distance != float('inf') else None,
            'fuel_used': float(fuel_used),
        }

    def _run_hrl_episode(self, episode_id: int, seed: int) -> Dict[str, Any]:
        """
        Run episode with HRL policy.

        Args:
            episode_id: Episode number
            seed: Random seed

        Returns:
            Episode metrics including option usage
        """
        obs, info = self.hrl_env.reset(seed=seed)

        episode_reward = 0.0
        episode_steps = 0
        min_distance = float('inf')
        final_distance = None
        fuel_used = 0.0
        initial_fuel = None

        option_usage = {option.name: 0 for option in Option}
        num_switches = 0

        done = False
        truncated = False

        while not (done or truncated):
            # HRL wrapper handles action selection
            action = self.hrl_env.action_space.sample()  # Dummy, overridden
            obs, reward, done, truncated, info = self.hrl_env.step(action)

            episode_reward += reward
            episode_steps += 1

            # Track option usage
            current_option = info['hrl/option']
            option_usage[current_option] += 1

            if info.get('hrl/option_switched', False):
                num_switches += 1

            # Extract metrics
            rel_pos = obs[0:3]
            distance = np.linalg.norm(rel_pos)
            min_distance = min(min_distance, distance)
            final_distance = distance

            if initial_fuel is None:
                initial_fuel = obs[12]
            fuel_used = initial_fuel - obs[12]

        success = final_distance < 50.0 if final_distance is not None else False

        return {
            'episode_id': episode_id,
            'seed': seed,
            'reward': float(episode_reward),
            'steps': episode_steps,
            'success': success,
            'miss_distance': float(final_distance) if final_distance is not None else None,
            'min_distance': float(min_distance) if min_distance != float('inf') else None,
            'fuel_used': float(fuel_used),
            'option_usage': option_usage,
            'num_switches': num_switches,
        }

    def _compute_comparison(
        self, flat_results: List[Dict[str, Any]], hrl_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute comparative statistics with significance testing.

        Args:
            flat_results: List of flat PPO episode results
            hrl_results: List of HRL episode results

        Returns:
            Comparison dictionary with statistical tests
        """
        n_episodes = len(flat_results)

        # Extract metrics
        flat_rewards = np.array([r['reward'] for r in flat_results])
        hrl_rewards = np.array([r['reward'] for r in hrl_results])

        flat_successes = np.array([r['success'] for r in flat_results])
        hrl_successes = np.array([r['success'] for r in hrl_results])

        flat_miss = np.array([r['miss_distance'] for r in flat_results if r['miss_distance'] is not None])
        hrl_miss = np.array([r['miss_distance'] for r in hrl_results if r['miss_distance'] is not None])

        flat_fuel = np.array([r['fuel_used'] for r in flat_results])
        hrl_fuel = np.array([r['fuel_used'] for r in hrl_results])

        flat_steps = np.array([r['steps'] for r in flat_results])
        hrl_steps = np.array([r['steps'] for r in hrl_results])

        # Statistical tests
        reward_ttest = stats.ttest_rel(flat_rewards, hrl_rewards)
        success_comparison = {
            'flat': float(flat_successes.mean()),
            'hrl': float(hrl_successes.mean()),
            'difference': float(hrl_successes.mean() - flat_successes.mean()),
        }

        miss_ttest = stats.ttest_ind(flat_miss, hrl_miss) if len(flat_miss) > 0 and len(hrl_miss) > 0 else None
        fuel_ttest = stats.ttest_rel(flat_fuel, hrl_fuel)
        steps_ttest = stats.ttest_rel(flat_steps, hrl_steps)

        # Build comparison
        comparison = {
            'n_episodes': n_episodes,
            'timestamp': datetime.now().isoformat(),

            # Reward comparison
            'reward': {
                'flat': {
                    'mean': float(flat_rewards.mean()),
                    'std': float(flat_rewards.std()),
                    'min': float(flat_rewards.min()),
                    'max': float(flat_rewards.max()),
                },
                'hrl': {
                    'mean': float(hrl_rewards.mean()),
                    'std': float(hrl_rewards.std()),
                    'min': float(hrl_rewards.min()),
                    'max': float(hrl_rewards.max()),
                },
                'difference': float(hrl_rewards.mean() - flat_rewards.mean()),
                'improvement_pct': float((hrl_rewards.mean() - flat_rewards.mean()) / abs(flat_rewards.mean()) * 100),
                'ttest': {
                    'statistic': float(reward_ttest.statistic),
                    'pvalue': float(reward_ttest.pvalue),
                    'significant': reward_ttest.pvalue < 0.05,
                },
            },

            # Success rate comparison
            'success_rate': success_comparison,

            # Miss distance comparison
            'miss_distance': {
                'flat': {
                    'mean': float(flat_miss.mean()) if len(flat_miss) > 0 else None,
                    'std': float(flat_miss.std()) if len(flat_miss) > 0 else None,
                },
                'hrl': {
                    'mean': float(hrl_miss.mean()) if len(hrl_miss) > 0 else None,
                    'std': float(hrl_miss.std()) if len(hrl_miss) > 0 else None,
                },
                'ttest': {
                    'statistic': float(miss_ttest.statistic) if miss_ttest else None,
                    'pvalue': float(miss_ttest.pvalue) if miss_ttest else None,
                    'significant': miss_ttest.pvalue < 0.05 if miss_ttest else None,
                } if miss_ttest else None,
            },

            # Fuel efficiency comparison
            'fuel_efficiency': {
                'flat': {
                    'mean': float(flat_fuel.mean()),
                    'std': float(flat_fuel.std()),
                },
                'hrl': {
                    'mean': float(hrl_fuel.mean()),
                    'std': float(hrl_fuel.std()),
                },
                'difference': float(hrl_fuel.mean() - flat_fuel.mean()),
                'ttest': {
                    'statistic': float(fuel_ttest.statistic),
                    'pvalue': float(fuel_ttest.pvalue),
                    'significant': fuel_ttest.pvalue < 0.05,
                },
            },

            # Episode length comparison
            'episode_length': {
                'flat': {
                    'mean': float(flat_steps.mean()),
                    'std': float(flat_steps.std()),
                },
                'hrl': {
                    'mean': float(hrl_steps.mean()),
                    'std': float(hrl_steps.std()),
                },
                'difference': float(hrl_steps.mean() - flat_steps.mean()),
                'ttest': {
                    'statistic': float(steps_ttest.statistic),
                    'pvalue': float(steps_ttest.pvalue),
                    'significant': steps_ttest.pvalue < 0.05,
                },
            },

            # HRL-specific metrics
            'hrl_metrics': {
                'mean_switches_per_episode': float(np.mean([r['num_switches'] for r in hrl_results])),
                'option_usage': self._aggregate_option_usage(hrl_results),
            },

            # Raw results
            'flat_results': flat_results,
            'hrl_results': hrl_results,
        }

        return comparison

    def _aggregate_option_usage(self, hrl_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate option usage across all HRL episodes."""
        total_usage = {option.name: 0 for option in Option}

        for result in hrl_results:
            for option_name, count in result['option_usage'].items():
                total_usage[option_name] += count

        total_steps = sum(total_usage.values())
        percentages = {
            name: float(count / total_steps * 100) if total_steps > 0 else 0.0
            for name, count in total_usage.items()
        }

        return {
            'total': total_usage,
            'percentages': percentages,
        }

    def _print_comparison_report(self, comparison: Dict[str, Any]):
        """Print detailed comparison report."""
        print("\n" + "=" * 80)
        print("POLICY COMPARISON REPORT")
        print("=" * 80)
        print(f"Episodes: {comparison['n_episodes']}")
        print(f"Timestamp: {comparison['timestamp']}")
        print()

        # Reward comparison
        print("REWARD COMPARISON:")
        flat_r = comparison['reward']['flat']
        hrl_r = comparison['reward']['hrl']
        print(f"  Flat PPO:  {flat_r['mean']:7.2f} ± {flat_r['std']:.2f}  [{flat_r['min']:.2f}, {flat_r['max']:.2f}]")
        print(f"  HRL:       {hrl_r['mean']:7.2f} ± {hrl_r['std']:.2f}  [{hrl_r['min']:.2f}, {hrl_r['max']:.2f}]")
        print(f"  Difference: {comparison['reward']['difference']:+.2f} ({comparison['reward']['improvement_pct']:+.1f}%)")
        ttest = comparison['reward']['ttest']
        print(f"  T-test: t={ttest['statistic']:.3f}, p={ttest['pvalue']:.4f}, significant={ttest['significant']}")
        print()

        # Success rate comparison
        print("SUCCESS RATE:")
        sr = comparison['success_rate']
        print(f"  Flat PPO:  {sr['flat'] * 100:.1f}%")
        print(f"  HRL:       {sr['hrl'] * 100:.1f}%")
        print(f"  Difference: {sr['difference'] * 100:+.1f}%")
        print()

        # Miss distance
        print("MISS DISTANCE:")
        md = comparison['miss_distance']
        if md['flat']['mean'] is not None and md['hrl']['mean'] is not None:
            print(f"  Flat PPO:  {md['flat']['mean']:.1f}m ± {md['flat']['std']:.1f}m")
            print(f"  HRL:       {md['hrl']['mean']:.1f}m ± {md['hrl']['std']:.1f}m")
            if md['ttest']:
                print(f"  T-test: p={md['ttest']['pvalue']:.4f}, significant={md['ttest']['significant']}")
        else:
            print("  Data unavailable")
        print()

        # Fuel efficiency
        print("FUEL EFFICIENCY:")
        fuel = comparison['fuel_efficiency']
        print(f"  Flat PPO:  {fuel['flat']['mean'] * 100:.1f}% ± {fuel['flat']['std'] * 100:.1f}%")
        print(f"  HRL:       {fuel['hrl']['mean'] * 100:.1f}% ± {fuel['hrl']['std'] * 100:.1f}%")
        print(f"  Difference: {fuel['difference'] * 100:+.1f}%")
        print(f"  T-test: p={fuel['ttest']['pvalue']:.4f}, significant={fuel['ttest']['significant']}")
        print()

        # Episode length
        print("EPISODE LENGTH:")
        length = comparison['episode_length']
        print(f"  Flat PPO:  {length['flat']['mean']:.1f} ± {length['flat']['std']:.1f} steps")
        print(f"  HRL:       {length['hrl']['mean']:.1f} ± {length['hrl']['std']:.1f} steps")
        print(f"  Difference: {length['difference']:+.1f} steps")
        print(f"  T-test: p={length['ttest']['pvalue']:.4f}, significant={length['ttest']['significant']}")
        print()

        # HRL metrics
        print("HRL-SPECIFIC METRICS:")
        hrl_m = comparison['hrl_metrics']
        print(f"  Mean Switches/Episode: {hrl_m['mean_switches_per_episode']:.1f}")
        print("  Option Usage:")
        for option_name in ['SEARCH', 'TRACK', 'TERMINAL']:
            pct = hrl_m['option_usage']['percentages'].get(option_name, 0.0)
            total = hrl_m['option_usage']['total'].get(option_name, 0)
            print(f"    {option_name:8s}: {pct:5.1f}% ({total:6d} steps)")
        print()

        # Overall assessment
        print("OVERALL ASSESSMENT:")
        if comparison['reward']['ttest']['significant']:
            if comparison['reward']['difference'] > 0:
                print("  ✓ HRL significantly outperforms Flat PPO (reward)")
            else:
                print("  ✗ Flat PPO significantly outperforms HRL (reward)")
        else:
            print("  ~ No significant difference in reward")
        print("=" * 80)


def main():
    """Main comparison script."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description="Compare flat PPO and HRL policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Flat model
    parser.add_argument(
        "--flat",
        type=str,
        required=True,
        help="Path to flat PPO checkpoint",
    )

    # HRL model
    parser.add_argument(
        "--hrl-selector",
        type=str,
        default=None,
        help="Path to HRL selector (None for rule-based)",
    )
    parser.add_argument(
        "--hrl-search",
        type=str,
        default=None,
        help="Path to search specialist",
    )
    parser.add_argument(
        "--hrl-track",
        type=str,
        default=None,
        help="Path to track specialist",
    )
    parser.add_argument(
        "--hrl-terminal",
        type=str,
        default=None,
        help="Path to terminal specialist",
    )

    # Comparison settings
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to compare (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: results/comparison_TIMESTAMP.json)",
    )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        return 1

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Prepare specialist paths
    specialist_paths = {}
    if args.hrl_search:
        specialist_paths[Option.SEARCH] = args.hrl_search
    if args.hrl_track:
        specialist_paths[Option.TRACK] = args.hrl_track
    if args.hrl_terminal:
        specialist_paths[Option.TERMINAL] = args.hrl_terminal

    # Create comparator
    comparator = PolicyComparator(
        flat_path=args.flat,
        hrl_selector_path=args.hrl_selector,
        hrl_specialist_paths=specialist_paths,
        config=config,
        seed=args.seed,
    )

    # Run comparison
    comparison = comparator.compare(args.episodes)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"comparison_{timestamp}.json"

    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
