"""
Enhanced inference runner for Phase 4 RL with scenario evaluation and diagnostics.

This script provides comprehensive inference capabilities with scenario management,
performance analysis, and integration with the diagnostics system.
"""

import argparse
import os
import sys
from pathlib import Path
import time
import json
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from .config import get_config, reset_config
    from .scenarios import get_scenario_loader, reset_scenario_loader
    from .radar_env import RadarEnv
    from .diagnostics import Logger, export_to_csv, export_to_json, plot_metrics
except ImportError:
    # Fallback for direct execution
    from config import get_config, reset_config
    from scenarios import get_scenario_loader, reset_scenario_loader
    from radar_env import RadarEnv
    from diagnostics import Logger, export_to_csv, export_to_json, plot_metrics


class Phase4InferenceRunner:
    """Enhanced inference runner for Phase 4 RL with scenario evaluation."""
    
    def __init__(self, 
                 checkpoint_path: str,
                 scenario_name: str = "easy",
                 config_path: Optional[str] = None,
                 output_dir: str = "inference_results"):
        """
        Initialize inference runner.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            scenario_name: Name of scenario to evaluate on
            config_path: Path to configuration file
            output_dir: Directory for saving results
        """
        # Reset global instances
        reset_config()
        reset_scenario_loader()
        
        # Load configuration
        self.config_loader = get_config(config_path)
        self.config = self.config_loader._config
        
        # Load scenario
        self.scenario_loader = get_scenario_loader()
        self.scenario_name = scenario_name
        self.scenario_config = self.scenario_loader.create_environment_config(
            scenario_name, self.config
        )
        
        # Setup paths
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inference configuration
        self.inference_config = self.config_loader.get_inference_config()
        
        # Model and environment
        self.model = None
        self.env = None
        self.vec_normalize = None
        
        # Logger for diagnostics
        self.logger = Logger()
        
    def load_model(self):
        """Load trained model and setup environment."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        print(f"Loading model from: {self.checkpoint_path}")
        
        # Create environment
        def make_env():
            return RadarEnv(
                config=self.scenario_config,
                scenario_name=self.scenario_name,
                render_mode=self.inference_config.get('render', False) and 'human' or None
            )
        
        self.env = DummyVecEnv([make_env])
        
        # Load normalization if available
        vec_normalize_path = self.checkpoint_path.parent / "vec_normalize.pkl"
        if vec_normalize_path.exists():
            print(f"Loading normalization parameters from: {vec_normalize_path}")
            self.env = VecNormalize.load(vec_normalize_path, self.env)
            self.env.training = False  # Set to evaluation mode
            self.env.norm_reward = False  # Don't normalize rewards during inference
        
        # Load model
        self.model = PPO.load(self.checkpoint_path, env=self.env)
        print("Model loaded successfully!")
        
    def run_episodes(self, 
                    num_episodes: int,
                    real_time: bool = False,
                    deterministic: bool = True,
                    render: bool = False) -> List[Dict[str, Any]]:
        """
        Run inference episodes.
        
        Args:
            num_episodes: Number of episodes to run
            real_time: Whether to run in real-time or accelerated
            deterministic: Whether to use deterministic policy
            render: Whether to render environment
            
        Returns:
            List of episode results
        """
        if self.model is None:
            self.load_model()
        
        print(f"Running {num_episodes} episodes:")
        print(f"  Scenario: {self.scenario_name}")
        print(f"  Real-time: {real_time}")
        print(f"  Deterministic: {deterministic}")
        print(f"  Render: {render}")
        print()
        
        episode_results = []
        
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}", end=" ")
            
            # Reset environment and logger
            obs = self.env.reset()
            self.logger.reset_episode()
            
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            step_start_time = time.time()
            
            while not done:
                # Get action from model
                action, _states = self.model.predict(obs, deterministic=deterministic)
                
                # Step environment
                obs, reward, done, info = self.env.step(action)
                
                # Log step data
                step_data = {
                    'step': episode_length,
                    'observation': obs[0] if isinstance(obs, (list, tuple)) else obs,
                    'action': action[0] if isinstance(action, (list, tuple)) else action,
                    'reward': reward[0] if isinstance(reward, (list, tuple)) else reward,
                    'done': done[0] if isinstance(done, (list, tuple)) else done,
                    'info': info[0] if isinstance(info, (list, tuple)) else info
                }
                self.logger.log_step(step_data)
                
                episode_reward += reward[0] if isinstance(reward, (list, tuple)) else reward
                episode_length += 1
                
                # Handle real-time execution
                if real_time:
                    elapsed = time.time() - step_start_time
                    target_dt = 0.1  # 10 Hz
                    if elapsed < target_dt:
                        time.sleep(target_dt - elapsed)
                    step_start_time = time.time()
                
                # Render if requested
                if render:
                    self.env.render()
            
            # Get episode metrics
            episode_metrics = self.logger.get_episode_metrics()
            
            episode_result = {
                'episode': episode,
                'scenario': self.scenario_name,
                'total_reward': episode_reward,
                'episode_length': episode_length,
                'success': self._evaluate_success(episode_metrics),
                'metrics': episode_metrics,
                'timestamp': time.time()
            }
            
            episode_results.append(episode_result)
            
            # Print episode summary
            success_str = "SUCCESS" if episode_result['success'] else "FAILURE"
            print(f"- Reward: {episode_reward:.2f}, Length: {episode_length}, {success_str}")
        
        return episode_results
    
    def _evaluate_success(self, metrics: Dict[str, Any]) -> bool:
        """
        Evaluate episode success based on scenario criteria.
        
        Args:
            metrics: Episode metrics from logger
            
        Returns:
            True if episode was successful
        """
        success_criteria = self.scenario_config.get('scenario', {}).get('success_criteria', {})
        
        # Check interception distance
        min_distances = metrics.get('min_interception_distances', [])
        if min_distances:
            min_distance = min(min_distances)
            threshold = success_criteria.get('interception_distance', 50.0)
            if min_distance <= threshold:
                return True
        
        # Additional success criteria can be added here
        
        return False
    
    def run_multi_scenario_evaluation(self, 
                                     num_episodes_per_scenario: int = 10,
                                     scenarios: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run evaluation across multiple scenarios.
        
        Args:
            num_episodes_per_scenario: Episodes to run per scenario
            scenarios: List of scenarios to evaluate. If None, uses all available.
            
        Returns:
            Dictionary mapping scenario names to episode results
        """
        if scenarios is None:
            scenarios = self.scenario_loader.list_scenarios()
        
        print(f"Running multi-scenario evaluation:")
        print(f"  Scenarios: {', '.join(scenarios)}")
        print(f"  Episodes per scenario: {num_episodes_per_scenario}")
        print()
        
        all_results = {}
        
        for scenario in scenarios:
            print(f"\n=== Evaluating scenario: {scenario} ===")
            
            # Update scenario configuration
            self.scenario_name = scenario
            self.scenario_config = self.scenario_loader.create_environment_config(
                scenario, self.config
            )
            
            # Reload environment with new scenario
            self.model = None  # Force model reload with new environment
            
            # Run episodes for this scenario
            results = self.run_episodes(
                num_episodes=num_episodes_per_scenario,
                real_time=False,
                deterministic=True,
                render=False
            )
            
            all_results[scenario] = results
            
            # Print scenario summary
            success_rate = sum(1 for r in results if r['success']) / len(results)
            avg_reward = np.mean([r['total_reward'] for r in results])
            avg_length = np.mean([r['episode_length'] for r in results])
            
            print(f"Scenario {scenario} summary:")
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Average reward: {avg_reward:.2f}")
            print(f"  Average episode length: {avg_length:.1f}")
        
        return all_results
    
    def export_results(self, 
                      results: Dict[str, List[Dict[str, Any]]], 
                      export_format: str = "both"):
        """
        Export results to CSV and/or JSON.
        
        Args:
            results: Results from inference runs
            export_format: Export format ("csv", "json", or "both")
        """
        timestamp = int(time.time())
        
        if export_format in ["csv", "both"]:
            csv_path = self.output_dir / f"inference_results_{timestamp}.csv"
            export_to_csv(results, csv_path)
            print(f"Results exported to CSV: {csv_path}")
        
        if export_format in ["json", "both"]:
            json_path = self.output_dir / f"inference_results_{timestamp}.json"
            export_to_json(results, json_path)
            print(f"Results exported to JSON: {json_path}")
    
    def generate_plots(self, results: Dict[str, List[Dict[str, Any]]]):
        """
        Generate visualization plots for results.
        
        Args:
            results: Results from inference runs
        """
        timestamp = int(time.time())
        plot_dir = self.output_dir / f"plots_{timestamp}"
        plot_dir.mkdir(exist_ok=True)
        
        plot_metrics(results, plot_dir)
        print(f"Plots generated in: {plot_dir}")
    
    def run_performance_analysis(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Analyze performance across scenarios.
        
        Args:
            results: Results from inference runs
            
        Returns:
            Performance analysis summary
        """
        analysis = {}
        
        for scenario, episode_results in results.items():
            success_rate = sum(1 for r in episode_results if r['success']) / len(episode_results)
            rewards = [r['total_reward'] for r in episode_results]
            lengths = [r['episode_length'] for r in episode_results]
            
            scenario_analysis = {
                'num_episodes': len(episode_results),
                'success_rate': success_rate,
                'reward_stats': {
                    'mean': np.mean(rewards),
                    'std': np.std(rewards),
                    'min': np.min(rewards),
                    'max': np.max(rewards)
                },
                'length_stats': {
                    'mean': np.mean(lengths),
                    'std': np.std(lengths),
                    'min': np.min(lengths),
                    'max': np.max(lengths)
                }
            }
            
            analysis[scenario] = scenario_analysis
        
        return analysis


def main():
    """Main inference script entry point."""
    parser = argparse.ArgumentParser(description='Phase 4 RL Inference with Scenario Evaluation')
    
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--scenario',
        type=str,
        default='easy',
        help='Scenario to evaluate on'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of episodes to run'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='inference_results',
        help='Directory for saving results'
    )
    
    parser.add_argument(
        '--real-time',
        action='store_true',
        help='Run in real-time mode'
    )
    
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render environment during inference'
    )
    
    parser.add_argument(
        '--multi-scenario',
        action='store_true',
        help='Evaluate on all available scenarios'
    )
    
    parser.add_argument(
        '--export-format',
        type=str,
        choices=['csv', 'json', 'both'],
        default='both',
        help='Export format for results'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )
    
    parser.add_argument(
        '--deterministic',
        action='store_true',
        default=True,
        help='Use deterministic policy'
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint path
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Create inference runner
    runner = Phase4InferenceRunner(
        checkpoint_path=args.checkpoint,
        scenario_name=args.scenario,
        config_path=args.config,
        output_dir=args.output_dir
    )
    
    try:
        # Run inference
        if args.multi_scenario:
            results = runner.run_multi_scenario_evaluation(
                num_episodes_per_scenario=args.episodes
            )
        else:
            episode_results = runner.run_episodes(
                num_episodes=args.episodes,
                real_time=args.real_time,
                deterministic=args.deterministic,
                render=args.render
            )
            results = {args.scenario: episode_results}
        
        # Export results
        runner.export_results(results, args.export_format)
        
        # Generate plots
        if not args.no_plots:
            runner.generate_plots(results)
        
        # Print performance analysis
        analysis = runner.run_performance_analysis(results)
        print(f"\n=== Performance Analysis ===")
        for scenario, stats in analysis.items():
            print(f"{scenario}:")
            print(f"  Success rate: {stats['success_rate']:.1%}")
            print(f"  Average reward: {stats['reward_stats']['mean']:.2f} ± {stats['reward_stats']['std']:.2f}")
            print(f"  Average length: {stats['length_stats']['mean']:.1f} ± {stats['length_stats']['std']:.1f}")
        
        print(f"\nInference completed! Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\nInference interrupted by user.")
    except Exception as e:
        print(f"\nInference failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()