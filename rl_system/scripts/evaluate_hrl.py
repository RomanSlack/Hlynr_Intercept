#!/usr/bin/env python3
"""
HRL Evaluation Script

Load trained HRL manager (selector + specialists) and run deterministic inference
for N episodes. Collect and report comprehensive metrics.

Usage:
    # Evaluate HRL policy for 100 episodes
    python evaluate_hrl.py --selector checkpoints/hrl/selector/model \\
                          --search checkpoints/hrl/specialists/search/model \\
                          --track checkpoints/hrl/specialists/track/model \\
                          --terminal checkpoints/hrl/specialists/terminal/model \\
                          --episodes 100 --config config.yaml

    # Evaluate with custom seed
    python evaluate_hrl.py --selector ... --episodes 50 --seed 42

    # Save results to custom file
    python evaluate_hrl.py --selector ... --output results/hrl_eval.json

Example:
    # Quick evaluation with default settings
    python evaluate_hrl.py --selector checkpoints/hrl/selector/best --episodes 10
"""
import argparse
import json
import yaml
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))

from environment import InterceptEnvironment
from hrl.hierarchical_env import make_hrl_env
from hrl.option_definitions import Option


class HRLEvaluator:
    """HRL policy evaluation with comprehensive metrics collection."""

    def __init__(
        self,
        selector_path: Optional[str],
        specialist_paths: Dict[Option, str],
        config: Dict[str, Any],
        seed: Optional[int] = None,
    ):
        """
        Initialize HRL evaluator.

        Args:
            selector_path: Path to trained selector model (None for rule-based)
            specialist_paths: Dict mapping Option to specialist checkpoint paths
            config: Configuration dictionary
            seed: Random seed for reproducibility
        """
        self.logger = logging.getLogger("HRLEvaluator")
        self.logger.setLevel(logging.INFO)
        self.config = config
        self.seed = seed

        # Create base environment
        env_config = config['environment']
        env_config['dt'] = config.get('dt', 0.01)
        env_config['max_steps'] = config['environment']['max_steps']
        base_env = InterceptEnvironment(env_config)

        # Create HRL environment
        self.env = make_hrl_env(
            base_env=base_env,
            cfg=config,
            selector_path=selector_path,
            specialist_paths=specialist_paths,
            mode='inference',
        )

        self.logger.info(f"HRL Evaluator initialized")
        self.logger.info(f"Selector: {selector_path or 'rule-based'}")
        self.logger.info(f"Specialists: {specialist_paths}")

    def evaluate_episodes(self, n_episodes: int) -> Dict[str, Any]:
        """
        Run evaluation for N episodes.

        Args:
            n_episodes: Number of episodes to evaluate

        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info(f"Starting evaluation: {n_episodes} episodes")

        episode_results = []
        option_usage = {option.name: 0 for option in Option}
        option_transitions = {option.name: 0 for option in Option}

        for episode in range(n_episodes):
            # Set seed if provided
            seed = self.seed + episode if self.seed is not None else None

            # Run episode
            result = self._run_episode(episode, seed)
            episode_results.append(result)

            # Aggregate option usage
            for option_name, count in result['option_usage'].items():
                option_usage[option_name] += count

            # Log progress
            if (episode + 1) % 10 == 0:
                self.logger.info(f"Completed {episode + 1}/{n_episodes} episodes")

        # Compute aggregate statistics
        metrics = self._compute_metrics(episode_results, option_usage)

        self.logger.info(f"Evaluation complete: {n_episodes} episodes")
        self._print_summary(metrics)

        return metrics

    def _run_episode(self, episode_id: int, seed: Optional[int]) -> Dict[str, Any]:
        """
        Run single evaluation episode.

        Args:
            episode_id: Episode number
            seed: Random seed for reproducibility

        Returns:
            Episode metrics dictionary
        """
        obs, info = self.env.reset(seed=seed)

        episode_reward = 0.0
        episode_steps = 0
        option_usage = {option.name: 0 for option in Option}
        option_switches = []
        forced_transitions = 0
        selector_decisions = 0

        # Track trajectory details
        min_distance = float('inf')
        final_distance = None
        fuel_used = 0.0
        initial_fuel = None

        done = False
        truncated = False
        prev_option = info.get('hrl/option', 0)  # Default to SEARCH (0) if not present

        while not (done or truncated):
            # Get action from HRL policy
            action, _ = self.env.action_space.sample(), None  # Dummy, overridden by wrapper
            obs, reward, done, truncated, info = self.env.step(action)

            episode_reward += reward
            episode_steps += 1

            # Track option usage
            current_option = info['hrl/option']
            option_usage[current_option] += 1

            # Track switches
            if info.get('hrl/option_switched', False):
                option_switches.append({
                    'step': episode_steps,
                    'from': info['hrl/prev_option'],
                    'to': info['hrl/new_option'],
                    'reason': info['hrl/switch_reason'],
                })

                if info['hrl/switch_reason'] == 'forced':
                    forced_transitions += 1
                elif info['hrl/switch_reason'] == 'selector':
                    selector_decisions += 1

            # Track trajectory metrics
            # Extract distance from observation (first 3 elements are relative position)
            rel_pos = obs[0:3]
            distance = np.linalg.norm(rel_pos)
            min_distance = min(min_distance, distance)
            final_distance = distance

            # Track fuel usage (observation[12] is fuel fraction)
            if initial_fuel is None:
                initial_fuel = obs[12]
            fuel_used = initial_fuel - obs[12]

            prev_option = current_option

        # Determine success (close intercept)
        success = final_distance < 50.0 if final_distance is not None else False

        return {
            'episode_id': episode_id,
            'seed': seed,
            'reward': float(episode_reward),
            'steps': episode_steps,
            'success': bool(success),
            'miss_distance': float(final_distance) if final_distance is not None else None,
            'min_distance': float(min_distance) if min_distance != float('inf') else None,
            'fuel_used': float(fuel_used),
            'option_usage': option_usage,
            'option_switches': option_switches,
            'num_switches': len(option_switches),
            'forced_transitions': forced_transitions,
            'selector_decisions': selector_decisions,
            'done': bool(done),
            'truncated': bool(truncated),
        }

    def _compute_metrics(
        self, episode_results: List[Dict[str, Any]], option_usage: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Compute aggregate metrics from episode results.

        Args:
            episode_results: List of episode result dictionaries
            option_usage: Aggregated option usage counts

        Returns:
            Aggregate metrics dictionary
        """
        n_episodes = len(episode_results)

        # Extract metrics
        rewards = [r['reward'] for r in episode_results]
        steps = [r['steps'] for r in episode_results]
        successes = [r['success'] for r in episode_results]
        miss_distances = [r['miss_distance'] for r in episode_results if r['miss_distance'] is not None]
        min_distances = [r['min_distance'] for r in episode_results if r['min_distance'] is not None]
        fuel_used = [r['fuel_used'] for r in episode_results]
        num_switches = [r['num_switches'] for r in episode_results]
        forced_transitions = [r['forced_transitions'] for r in episode_results]
        selector_decisions = [r['selector_decisions'] for r in episode_results]

        # Compute statistics
        metrics = {
            'n_episodes': n_episodes,
            'timestamp': datetime.now().isoformat(),

            # Performance metrics
            'success_rate': float(np.mean(successes)),
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),

            # Distance metrics
            'mean_miss_distance': float(np.mean(miss_distances)) if miss_distances else None,
            'std_miss_distance': float(np.std(miss_distances)) if miss_distances else None,
            'mean_min_distance': float(np.mean(min_distances)) if min_distances else None,

            # Efficiency metrics
            'mean_episode_length': float(np.mean(steps)),
            'std_episode_length': float(np.std(steps)),
            'mean_fuel_used': float(np.mean(fuel_used)),
            'std_fuel_used': float(np.std(fuel_used)),

            # Option usage metrics
            'option_usage': option_usage,
            'option_usage_percentages': {
                name: float(count / sum(option_usage.values()) * 100)
                for name, count in option_usage.items()
            },

            # Switching metrics
            'mean_switches_per_episode': float(np.mean(num_switches)),
            'std_switches_per_episode': float(np.std(num_switches)),
            'total_switches': int(np.sum(num_switches)),
            'mean_forced_transitions': float(np.mean(forced_transitions)),
            'mean_selector_decisions': float(np.mean(selector_decisions)),

            # Per-episode details
            'episodes': episode_results,
        }

        return metrics

    def _print_summary(self, metrics: Dict[str, Any]):
        """Print evaluation summary to console."""
        print("\n" + "=" * 80)
        print("HRL EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Episodes: {metrics['n_episodes']}")
        print(f"Timestamp: {metrics['timestamp']}")
        print()

        print("PERFORMANCE METRICS:")
        print(f"  Success Rate:      {metrics['success_rate'] * 100:.1f}%")
        print(f"  Mean Reward:       {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Reward Range:      [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
        print()

        print("DISTANCE METRICS:")
        if metrics['mean_miss_distance'] is not None:
            print(f"  Mean Miss Distance: {metrics['mean_miss_distance']:.1f}m ± {metrics['std_miss_distance']:.1f}m")
            print(f"  Mean Min Distance:  {metrics['mean_min_distance']:.1f}m")
        else:
            print("  Distance metrics unavailable")
        print()

        print("EFFICIENCY METRICS:")
        print(f"  Mean Episode Length: {metrics['mean_episode_length']:.1f} ± {metrics['std_episode_length']:.1f} steps")
        print(f"  Mean Fuel Used:      {metrics['mean_fuel_used'] * 100:.1f}% ± {metrics['std_fuel_used'] * 100:.1f}%")
        print()

        print("OPTION USAGE:")
        for option_name in ['SEARCH', 'TRACK', 'TERMINAL']:
            percentage = metrics['option_usage_percentages'].get(option_name, 0.0)
            count = metrics['option_usage'].get(option_name, 0)
            print(f"  {option_name:8s}: {percentage:5.1f}% ({count:6d} steps)")
        print()

        print("SWITCHING BEHAVIOR:")
        print(f"  Mean Switches/Episode:   {metrics['mean_switches_per_episode']:.1f} ± {metrics['std_switches_per_episode']:.1f}")
        print(f"  Total Switches:          {metrics['total_switches']}")
        print(f"  Forced Transitions:      {metrics['mean_forced_transitions']:.1f}/episode")
        print(f"  Selector Decisions:      {metrics['mean_selector_decisions']:.1f}/episode")
        print("=" * 80)


def main():
    """Main evaluation script."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description="Evaluate trained HRL policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model paths
    parser.add_argument(
        "--selector",
        type=str,
        default=None,
        help="Path to trained selector model (None for rule-based selector)",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Path to search specialist model (None for stub)",
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Path to track specialist model (None for stub)",
    )
    parser.add_argument(
        "--terminal",
        type=str,
        default=None,
        help="Path to terminal specialist model (None for stub)",
    )

    # Evaluation settings
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)",
    )

    # Output settings
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: results/hrl_eval_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Prepare specialist paths
    specialist_paths = {}
    if args.search:
        specialist_paths[Option.SEARCH] = args.search
    if args.track:
        specialist_paths[Option.TRACK] = args.track
    if args.terminal:
        specialist_paths[Option.TERMINAL] = args.terminal

    # Create evaluator
    evaluator = HRLEvaluator(
        selector_path=args.selector,
        specialist_paths=specialist_paths,
        config=config,
        seed=args.seed,
    )

    # Run evaluation
    metrics = evaluator.evaluate_episodes(args.episodes)

    # Save results
    if not args.no_save:
        if args.output:
            output_path = Path(args.output)
        else:
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = results_dir / f"hrl_eval_{timestamp}.json"

        output_path.parent.mkdir(exist_ok=True, parents=True)

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
