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
from logger import UnifiedLogger, make_json_serializable


class HRLEvaluator:
    """HRL policy evaluation with comprehensive metrics collection."""

    def __init__(
        self,
        selector_path: Optional[str],
        specialist_paths: Dict[Option, str],
        config: Dict[str, Any],
        output_dir: str = "inference_results",
        seed: Optional[int] = None,
    ):
        """
        Initialize HRL evaluator.

        Args:
            selector_path: Path to trained selector model (None for rule-based)
            specialist_paths: Dict mapping Option to specialist checkpoint paths
            config: Configuration dictionary
            output_dir: Directory to save results
            seed: Random seed for reproducibility
        """
        self.logger = logging.getLogger("HRLEvaluator")
        self.logger.setLevel(logging.INFO)
        self.config = config
        self.seed = seed

        # Create output directory
        output_path = Path(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = output_path / f"hrl_offline_run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize UnifiedLogger (same as PPO inference)
        self.unified_logger = UnifiedLogger(log_dir=str(self.run_dir), run_name="hrl_offline_inference")

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
        self.logger.info(f"Output directory: {self.run_dir}")

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
            outcome = result['outcome']
            reward = result['total_reward']
            steps = result['steps']
            print(f"Episode {episode+1}/{n_episodes}: {outcome}, reward={reward:.2f}, steps={steps}")

        # Compute aggregate statistics
        metrics = self._compute_metrics(episode_results, option_usage)

        # Save results in same format as PPO inference
        self._save_results(metrics, episode_results)

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
        # Initialize episode data structure (same as PPO inference)
        episode_data = {
            'episode_id': f"ep_{episode_id:04d}",
            'seed': seed,
            'states': [],
            'actions': [],
            'rewards': [],
            'info': [],
            # HRL-specific additions
            'hrl_options': [],
            'hrl_specialist_actions': [],
        }

        obs, info = self.env.reset(seed=seed)
        self.unified_logger.begin_episode(episode_data['episode_id'])

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
            # Get action from HRL policy (the wrapper handles this)
            action, _ = self.env.action_space.sample(), None  # Dummy, overridden by wrapper
            obs, reward, done, truncated, info = self.env.step(action)

            # Store step data (same format as PPO)
            episode_data['states'].append(obs.tolist() if isinstance(obs, np.ndarray) else obs)
            episode_data['actions'].append(action.tolist() if isinstance(action, np.ndarray) else action)
            episode_data['rewards'].append(float(reward))
            episode_data['info'].append(info)

            # Store HRL-specific data
            episode_data['hrl_options'].append(info.get('hrl/option', 'UNKNOWN'))
            episode_data['hrl_specialist_actions'].append(
                info.get('hrl/specialist_action', action).tolist()
                if isinstance(info.get('hrl/specialist_action', action), np.ndarray)
                else info.get('hrl/specialist_action', action)
            )

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

            # Track trajectory metrics using ACTUAL world positions (not normalized obs!)
            # The observation obs[0:3] is NORMALIZED (divided by max_range), NOT real meters!
            # We must use the actual world positions from the info dict
            if 'interceptor_pos' in info and 'missile_pos' in info:
                int_pos = np.array(info['interceptor_pos'])
                mis_pos = np.array(info['missile_pos'])
                distance = np.linalg.norm(int_pos - mis_pos)
                min_distance = min(min_distance, distance)
                final_distance = distance
            else:
                # Fallback: estimate from observation (but this is WRONG for precision!)
                # obs[0:3] is normalized, so multiply by max_range
                rel_pos = obs[0:3] * 10000.0  # max_range = 10000m
                distance = np.linalg.norm(rel_pos)
                min_distance = min(min_distance, distance)
                final_distance = distance

            # Track fuel usage (observation[12] is fuel fraction)
            if initial_fuel is None:
                initial_fuel = obs[12]
            fuel_used = initial_fuel - obs[12]

            # Log state with actions (same as PPO)
            if 'interceptor_pos' in info:
                self.unified_logger.log_state('interceptor', {
                    'position': info['interceptor_pos'].tolist() if isinstance(info['interceptor_pos'], np.ndarray) else info['interceptor_pos'],
                    'fuel': info.get('fuel_remaining', 0),
                    'action': action.tolist() if isinstance(action, np.ndarray) else action,
                    'hrl_option': info.get('hrl/option', 'UNKNOWN')
                })
            if 'missile_pos' in info:
                self.unified_logger.log_state('missile', {
                    'position': info['missile_pos'].tolist() if isinstance(info['missile_pos'], np.ndarray) else info['missile_pos']
                })

            prev_option = current_option

        # Determine success based on environment's actual intercept radius
        success = info.get('intercepted', False) if info else False
        outcome = "intercepted" if success else "failed"

        # Add summary metrics to episode_data (same as PPO)
        episode_data['outcome'] = outcome
        episode_data['total_reward'] = float(episode_reward)
        episode_data['steps'] = int(episode_steps)
        episode_data['final_distance'] = float(final_distance) if final_distance is not None else 0.0
        episode_data['min_distance'] = float(min_distance) if min_distance != float('inf') else None
        episode_data['fuel_used'] = float(fuel_used)

        # HRL-specific summary
        episode_data['option_usage'] = option_usage
        episode_data['option_switches'] = option_switches
        episode_data['num_switches'] = len(option_switches)
        episode_data['forced_transitions'] = forced_transitions
        episode_data['selector_decisions'] = selector_decisions

        # Log episode end (same as PPO)
        self.unified_logger.end_episode(outcome, {
            'total_reward': episode_reward,
            'steps': episode_steps,
            'final_distance': final_distance,
            'fuel_used': fuel_used,
            'option_usage': option_usage,
            'num_switches': len(option_switches)
        })

        return episode_data

    def _save_results(self, metrics: Dict[str, Any], episode_results: List[Dict[str, Any]]):
        """
        Save results in same format as PPO inference.

        Args:
            metrics: Aggregated metrics dictionary
            episode_results: List of episode result dictionaries
        """
        # Save summary.json (same as PPO)
        summary_file = self.run_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(make_json_serializable(metrics), f, indent=2)

        # Save episodes.jsonl (same as PPO)
        episodes_file = self.run_dir / "episodes.jsonl"
        with open(episodes_file, 'w') as f:
            for episode in episode_results:
                f.write(json.dumps(make_json_serializable(episode)) + '\n')

        # Create manifest
        self.unified_logger.create_manifest()

        self.logger.info(f"Results saved to: {self.run_dir}")
        self.logger.info(f"  - {summary_file}")
        self.logger.info(f"  - {episodes_file}")

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

        # Extract metrics (updated keys to match new episode_data structure)
        rewards = [r['total_reward'] for r in episode_results]
        steps = [r['steps'] for r in episode_results]
        successes = [r['outcome'] == 'intercepted' for r in episode_results]
        miss_distances = [r['final_distance'] for r in episode_results if r['final_distance'] is not None]
        min_distances = [r['min_distance'] for r in episode_results if r['min_distance'] is not None]
        fuel_used = [r['fuel_used'] for r in episode_results]
        num_switches = [r['num_switches'] for r in episode_results]
        forced_transitions = [r['forced_transitions'] for r in episode_results]
        selector_decisions = [r['selector_decisions'] for r in episode_results]

        # Compute statistics (matching PPO format where applicable)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sort min distances for percentile calculations
        sorted_min_distances = sorted(min_distances) if min_distances else []

        metrics = {
            # PPO-compatible fields
            'run_id': f"hrl_offline_{timestamp}",
            'num_episodes': n_episodes,
            'n_episodes': n_episodes,  # Keep both for compatibility
            'timestamp': datetime.now().isoformat(),
            'success_rate': float(np.mean(successes)),
            'avg_reward': float(np.mean(rewards)),  # PPO uses 'avg_reward'
            'avg_steps': float(np.mean(steps)),      # PPO uses 'avg_steps'
            'avg_final_distance': float(np.mean(miss_distances)) if miss_distances else None,

            # Additional performance metrics
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),

            # Distance metrics
            'mean_miss_distance': float(np.mean(miss_distances)) if miss_distances else None,
            'std_miss_distance': float(np.std(miss_distances)) if miss_distances else None,
            'mean_min_distance': float(np.mean(min_distances)) if min_distances else None,
            'median_min_distance': float(sorted_min_distances[len(sorted_min_distances)//2]) if sorted_min_distances else None,
            'best_min_distance': float(min(min_distances)) if min_distances else None,
            'worst_min_distance': float(max(min_distances)) if min_distances else None,

            # Efficiency metrics
            'mean_episode_length': float(np.mean(steps)),
            'std_episode_length': float(np.std(steps)),
            'mean_fuel_used': float(np.mean(fuel_used)),
            'std_fuel_used': float(np.std(fuel_used)),

            # HRL-specific: Option usage metrics
            'option_usage': option_usage,
            'option_usage_percentages': {
                name: float(count / sum(option_usage.values()) * 100)
                for name, count in option_usage.items()
            },

            # HRL-specific: Switching metrics
            'mean_switches_per_episode': float(np.mean(num_switches)),
            'std_switches_per_episode': float(np.std(num_switches)),
            'total_switches': int(np.sum(num_switches)),
            'mean_forced_transitions': float(np.mean(forced_transitions)),
            'mean_selector_decisions': float(np.mean(selector_decisions)),

            # Per-episode details (PPO compatible - episodes are saved separately in episodes.jsonl)
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

        # Calculate precision brackets
        min_distances = [ep['min_distance'] for ep in metrics['episodes'] if ep['min_distance'] is not None]
        sub_10cm = sum(1 for d in min_distances if d < 0.1)
        sub_50cm = sum(1 for d in min_distances if d < 0.5)
        sub_1m = sum(1 for d in min_distances if d < 1.0)

        print("🎯 PRECISION METRICS (Minimum Distance Achieved):")
        print(f"  Mean Min Distance:       {metrics['mean_min_distance']*100:.2f}cm ({metrics['mean_min_distance']:.4f}m)")
        print(f"  Median Min Distance:     {metrics.get('median_min_distance', 0)*100:.2f}cm")
        print(f"  Best (Closest):          {metrics.get('best_min_distance', 0)*100:.2f}cm")
        print(f"  Worst:                   {metrics.get('worst_min_distance', 0)*100:.2f}cm")
        print()
        print(f"  🎯 Sub-10cm Precision:   {sub_10cm}/{metrics['n_episodes']} ({sub_10cm/metrics['n_episodes']*100:.1f}%)")
        print(f"  🎯 Sub-50cm Precision:   {sub_50cm}/{metrics['n_episodes']} ({sub_50cm/metrics['n_episodes']*100:.1f}%)")
        print(f"  ✅ Sub-Meter Precision:  {sub_1m}/{metrics['n_episodes']} ({sub_1m/metrics['n_episodes']*100:.1f}%)")
        print()

        print("PERFORMANCE METRICS:")
        print(f"  Success Rate (at termination): {metrics['success_rate'] * 100:.1f}%")
        print(f"  Mean Reward:                   {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Reward Range:                  [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
        print()

        print("FINAL DISTANCE METRICS (at episode end):")
        if metrics['mean_miss_distance'] is not None:
            print(f"  Mean Final Distance: {metrics['mean_miss_distance']:.3f}m")
            print(f"  Std Final Distance:  {metrics['std_miss_distance']:.3f}m")
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

        # Print TOP 10 closest approaches
        print("\n🏆 TOP 10 CLOSEST APPROACHES:")
        print("-" * 80)
        sorted_episodes = sorted(metrics['episodes'], key=lambda x: x['min_distance'] if x['min_distance'] else float('inf'))
        for i, ep in enumerate(sorted_episodes[:10], 1):
            outcome_sym = "✅" if ep['outcome'] == 'intercepted' else "❌"
            min_cm = ep['min_distance'] * 100
            final_cm = ep['final_distance'] * 100
            print(f"{i:2d}. {outcome_sym} Episode {ep['episode_id']}: {min_cm:6.2f}cm min, {final_cm:6.2f}cm final")
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
        default="inference_results",
        help="Output directory for results (default: inference_results)",
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

    # Create evaluator (now handles output directory creation and saving internally)
    evaluator = HRLEvaluator(
        selector_path=args.selector,
        specialist_paths=specialist_paths,
        config=config,
        output_dir=args.output,
        seed=args.seed,
    )

    # Run evaluation (results are saved automatically)
    metrics = evaluator.evaluate_episodes(args.episodes)

    print(f"\nResults saved to: {evaluator.run_dir}")
    print(f"Success rate: {metrics['success_rate']:.2%}")
    print(f"Average reward: {metrics['avg_reward']:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
