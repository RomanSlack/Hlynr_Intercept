#!/usr/bin/env python3
"""
Test random baseline performance for comparison with trained models.

Enhanced version with multi-seed testing and statistical validation.
Tests multiple seeds to ensure statistical significance and requires
≥25% improvement over random baseline for trained models.
"""

import argparse
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from fast_sim_env import make_fast_sim_env


def test_random_baseline_single_seed(scenario_name: str = "easy", 
                                   num_episodes: int = 50, 
                                   seed: int = 0,
                                   verbose: bool = True) -> Dict:
    """
    Test random baseline performance for a single seed.
    
    Args:
        scenario_name: Scenario to test on
        num_episodes: Number of episodes to run
        seed: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        Dictionary with performance statistics
    """
    if verbose:
        print(f"Testing random baseline on {scenario_name} scenario (seed={seed}, {num_episodes} episodes)")
    
    # Create environment with deterministic seeding
    env = make_fast_sim_env(scenario_name)
    
    episode_rewards = []
    episode_lengths = []
    successes = 0
    
    # Set numpy random seed for reproducible action sampling
    np.random.seed(seed)
    
    for episode in range(num_episodes):
        # Use seed + episode for episode-specific seeding
        obs, _ = env.reset(seed=seed + episode * 1000)
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # Random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Simple success check (positive reward generally means good performance)
        if episode_reward > 0:
            successes += 1
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f}")
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = successes / num_episodes
    
    results = {
        'seed': seed,
        'scenario': scenario_name,
        'num_episodes': num_episodes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'statistics': {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'min_reward': min_reward,
            'max_reward': max_reward,
            'avg_length': avg_length,
            'success_rate': success_rate
        }
    }
    
    if verbose:
        print(f"\nRandom Baseline Results (seed={seed}, {scenario_name}):")
        print(f"  Episodes: {num_episodes}")
        print(f"  Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Min reward: {min_reward:.2f}")
        print(f"  Max reward: {max_reward:.2f}")
        print(f"  Average length: {avg_length:.1f}")
        print(f"  Success rate: {success_rate:.1%}")
    
    env.close()
    
    return results


def test_random_baseline_multi_seed(scenario_name: str = "easy", 
                                  num_episodes: int = 50, 
                                  seeds: List[int] = None,
                                  verbose: bool = True) -> Dict:
    """
    Test random baseline performance across multiple seeds.
    
    Args:
        scenario_name: Scenario to test on
        num_episodes: Number of episodes per seed
        seeds: List of seeds to test (default: [0, 1, 2])
        verbose: Whether to print progress
        
    Returns:
        Dictionary with aggregated multi-seed statistics
    """
    if seeds is None:
        seeds = [0, 1, 2]  # Default 3 seeds as required
    
    print(f"="*60)
    print(f"MULTI-SEED RANDOM BASELINE TEST")
    print(f"="*60)
    print(f"Scenario: {scenario_name}")
    print(f"Seeds: {seeds}")
    print(f"Episodes per seed: {num_episodes}")
    print(f"Total episodes: {len(seeds) * num_episodes}")
    print()
    
    # Run tests for each seed
    seed_results = []
    all_rewards = []
    
    for i, seed in enumerate(seeds):
        print(f"[{i+1}/{len(seeds)}] Testing seed {seed}...")
        result = test_random_baseline_single_seed(
            scenario_name=scenario_name,
            num_episodes=num_episodes,
            seed=seed,
            verbose=verbose
        )
        
        seed_results.append(result)
        all_rewards.extend(result['episode_rewards'])
        print()
    
    # Aggregate statistics across all seeds
    seed_means = [r['statistics']['avg_reward'] for r in seed_results]
    seed_stds = [r['statistics']['std_reward'] for r in seed_results]
    
    overall_stats = {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'min_reward': min(all_rewards),
        'max_reward': max(all_rewards),
        'seed_means': seed_means,
        'seed_mean_avg': np.mean(seed_means),
        'seed_mean_std': np.std(seed_means),
        'total_episodes': len(all_rewards),
        'seeds_tested': len(seeds)
    }
    
    # Statistical validation
    confidence_interval = 1.96 * (overall_stats['std_reward'] / np.sqrt(len(all_rewards)))
    
    results = {
        'scenario': scenario_name,
        'seeds': seeds,
        'episodes_per_seed': num_episodes,
        'seed_results': seed_results,
        'overall_statistics': overall_stats,
        'confidence_interval_95': confidence_interval,
        'all_rewards': all_rewards
    }
    
    # Print summary
    print(f"="*60)
    print(f"MULTI-SEED SUMMARY")
    print(f"="*60)
    print(f"Overall mean reward: {overall_stats['mean_reward']:.3f} ± {overall_stats['std_reward']:.3f}")
    print(f"95% confidence interval: ±{confidence_interval:.3f}")
    print(f"Seed means: {', '.join([f'{m:.3f}' for m in seed_means])}")
    print(f"Inter-seed variability: {overall_stats['seed_mean_std']:.3f}")
    print(f"Range: [{overall_stats['min_reward']:.3f}, {overall_stats['max_reward']:.3f}]")
    
    return results


def calculate_performance_improvement(trained_reward: float, random_reward: float) -> float:
    """
    Calculate percentage improvement over random baseline.
    
    Args:
        trained_reward: Average reward from trained model
        random_reward: Average reward from random baseline
        
    Returns:
        Percentage improvement (positive means trained is better)
    """
    if random_reward == 0:
        return float('inf') if trained_reward > 0 else 0
    
    improvement = ((trained_reward - random_reward) / abs(random_reward)) * 100
    return improvement


def validate_trained_model(trained_reward: float, 
                         baseline_results: Dict,
                         min_improvement: float = 25.0,
                         confidence_level: float = 0.95) -> Dict:
    """
    Validate that a trained model meets performance requirements.
    
    Args:
        trained_reward: Mean reward from trained model
        baseline_results: Results from multi-seed baseline test
        min_improvement: Minimum required improvement percentage
        confidence_level: Statistical confidence level
        
    Returns:
        Validation results with pass/fail status
    """
    random_reward = baseline_results['overall_statistics']['mean_reward']
    confidence_interval = baseline_results['confidence_interval_95']
    
    # Calculate improvement
    improvement = calculate_performance_improvement(trained_reward, random_reward)
    
    # Conservative estimate using confidence interval
    conservative_random = random_reward + confidence_interval
    conservative_improvement = calculate_performance_improvement(trained_reward, conservative_random)
    
    # Statistical significance test (simple)
    required_reward = random_reward * (1 + min_improvement / 100)
    
    validation_results = {
        'trained_reward': trained_reward,
        'random_baseline': random_reward,
        'confidence_interval': confidence_interval,
        'improvement_percent': improvement,
        'conservative_improvement': conservative_improvement,
        'required_improvement': min_improvement,
        'required_reward': required_reward,
        'passes_requirement': improvement >= min_improvement,
        'passes_conservative': conservative_improvement >= min_improvement,
        'statistical_significance': abs(trained_reward - random_reward) > confidence_interval
    }
    
    return validation_results


def save_baseline_results(results: Dict, output_file: str):
    """Save baseline results to JSON file for future comparisons."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if key == 'seed_results':
            json_results[key] = []
            for seed_result in value:
                json_seed = dict(seed_result)
                json_seed['episode_rewards'] = list(json_seed['episode_rewards'])
                json_seed['episode_lengths'] = list(json_seed['episode_lengths'])
                json_results[key].append(json_seed)
        elif key == 'all_rewards':
            json_results[key] = list(value)
        else:
            json_results[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"Baseline results saved to: {output_path}")


def load_baseline_results(input_file: str) -> Dict:
    """Load baseline results from JSON file."""
    with open(input_file, 'r') as f:
        return json.load(f)


def main():
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(description='Multi-Seed Random Baseline Testing')
    
    parser.add_argument(
        '--scenario',
        type=str,
        default='easy',
        choices=['easy', 'medium', 'hard', 'impossible'],
        help='Scenario to test'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=50,
        help='Number of episodes per seed'
    )
    
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[0, 1, 2],
        help='Seeds to test (default: 0 1 2)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )
    
    parser.add_argument(
        '--validate-model',
        type=float,
        help='Validate trained model reward against baseline'
    )
    
    parser.add_argument(
        '--load-baseline',
        type=str,
        help='Load baseline results from JSON file'
    )
    
    parser.add_argument(
        '--min-improvement',
        type=float,
        default=25.0,
        help='Minimum required improvement percentage (default: 25%%)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # Load existing baseline or run new test
    if args.load_baseline:
        print(f"Loading baseline results from: {args.load_baseline}")
        baseline_results = load_baseline_results(args.load_baseline)
        print(f"Loaded baseline: {baseline_results['overall_statistics']['mean_reward']:.3f} ± {baseline_results['confidence_interval_95']:.3f}")
    else:
        # Run multi-seed baseline test
        baseline_results = test_random_baseline_multi_seed(
            scenario_name=args.scenario,
            num_episodes=args.episodes,
            seeds=args.seeds,
            verbose=not args.quiet
        )
        
        # Save results if requested
        if args.output:
            save_baseline_results(baseline_results, args.output)
    
    # Validate trained model if provided
    if args.validate_model is not None:
        print(f"\n" + "="*60)
        print(f"TRAINED MODEL VALIDATION")
        print(f"="*60)
        
        validation = validate_trained_model(
            trained_reward=args.validate_model,
            baseline_results=baseline_results,
            min_improvement=args.min_improvement
        )
        
        print(f"Trained model reward: {validation['trained_reward']:.3f}")
        print(f"Random baseline: {validation['random_baseline']:.3f} ± {validation['confidence_interval']:.3f}")
        print(f"Improvement: {validation['improvement_percent']:.1f}%")
        print(f"Conservative improvement: {validation['conservative_improvement']:.1f}%")
        print(f"Required improvement: {validation['required_improvement']:.1f}%")
        print(f"Required reward threshold: {validation['required_reward']:.3f}")
        print()
        
        # Check requirements
        if validation['passes_requirement']:
            print(f"✅ PASSED: Trained model beats random baseline by {validation['improvement_percent']:.1f}% (≥ {args.min_improvement}% required)")
        else:
            print(f"❌ FAILED: Trained model only beats random baseline by {validation['improvement_percent']:.1f}% (< {args.min_improvement}% required)")
        
        if validation['statistical_significance']:
            print("✅ SIGNIFICANT: Difference is statistically significant")
        else:
            print("⚠️  WARNING: Difference may not be statistically significant")
        
        if validation['passes_conservative']:
            print("✅ ROBUST: Passes conservative estimate")
        else:
            print("⚠️  WARNING: May not pass conservative estimate")
        
        # Exit with appropriate code for CI/testing
        exit_code = 0 if validation['passes_requirement'] else 1
        return exit_code
    
    return 0


# Legacy function for backward compatibility
def test_random_baseline(scenario_name="easy", num_episodes=50):
    """Legacy function - use test_random_baseline_single_seed instead."""
    result = test_random_baseline_single_seed(scenario_name, num_episodes, seed=0)
    return result['statistics']['avg_reward']


if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        exit(exit_code)