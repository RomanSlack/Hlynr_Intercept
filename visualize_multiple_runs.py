#!/usr/bin/env python3
"""
Multi-Run Visualization Script for AegisIntercept Phase 3.

This script runs multiple training/demo sessions and creates comprehensive
visualizations comparing performance across runs.

Usage:
    python visualize_multiple_runs.py [options]

Examples:
    # Run 10 demo episodes with comparison
    python visualize_multiple_runs.py --num-runs 10 --episodes-per-run 5
    
    # Run with trained model
    python visualize_multiple_runs.py --model-path trained_model.zip --num-runs 10
    
    # Training comparison mode
    python visualize_multiple_runs.py --mode training --num-runs 5 --timesteps-per-run 10000
"""

import argparse
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Add project root to path
sys.path.append('.')

from aegis_intercept.envs.aegis_6dof_env import Aegis6DOFEnv
from aegis_intercept.demo.demo_6dof_system import DemoController
from aegis_intercept.curriculum import CurriculumManager

# Set up matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MultiRunVisualizer:
    """
    Multi-run visualization and analysis system.
    
    This class orchestrates multiple training/demo runs and creates
    comprehensive comparative visualizations.
    """
    
    def __init__(self, args: argparse.Namespace):
        """Initialize multi-run visualizer."""
        self.args = args
        self.runs_data = []
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis results
        self.summary_stats = {}
        self.comparison_plots = {}
        
        print("="*70)
        print("AegisIntercept Phase 3 - Multi-Run Visualization")
        print("="*70)
        print(f"Mode: {args.mode}")
        print(f"Number of runs: {args.num_runs}")
        if args.mode == 'demo':
            print(f"Episodes per run: {args.episodes_per_run}")
        elif args.mode == 'training':
            print(f"Timesteps per run: {args.timesteps_per_run}")
        print(f"Output directory: {self.output_dir}")
        print("="*70)
    
    def run_single_demo(self, run_id: int) -> Dict[str, Any]:
        """Run a single demo session."""
        print(f"Starting demo run {run_id + 1}/{self.args.num_runs}")
        
        # Create unique output directory for this run
        run_output_dir = self.output_dir / f"run_{run_id:03d}"
        run_output_dir.mkdir(exist_ok=True)
        
        # Configure demo arguments
        demo_args = argparse.Namespace(
            num_episodes=self.args.episodes_per_run,
            time_step=0.02,
            headless=True,
            fast_mode=True,
            model_path=self.args.model_path,
            scenario_file=None,
            output_dir=str(run_output_dir),
            export_unity=False,
            export_csv=True
        )
        
        try:
            # Run demo
            demo_controller = DemoController(demo_args)
            demo_controller.run_demo()
            
            # Collect results
            episode_summaries = demo_controller.episode_metrics
            
            # Calculate run statistics
            run_stats = self._analyze_run_data(episode_summaries, run_id)
            
            return {
                'run_id': run_id,
                'success': True,
                'episode_summaries': episode_summaries,
                'run_stats': run_stats,
                'output_dir': str(run_output_dir)
            }
            
        except Exception as e:
            print(f"Error in demo run {run_id}: {e}")
            return {
                'run_id': run_id,
                'success': False,
                'error': str(e)
            }
    
    def run_single_environment_test(self, run_id: int) -> Dict[str, Any]:
        """Run a single environment test session."""
        print(f"Starting environment test run {run_id + 1}/{self.args.num_runs}")
        
        try:
            # Create environment
            env = Aegis6DOFEnv(
                render_mode=None,
                max_episode_steps=self.args.episodes_per_run * 50,  # Longer episodes
                curriculum_level='medium'
            )
            
            episode_data = []
            
            # Run episodes
            for episode in range(self.args.episodes_per_run):
                obs, info = env.reset(seed=run_id * 1000 + episode)
                
                total_reward = 0.0
                steps = 0
                episode_start_time = time.time()
                
                while True:
                    # Random action
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    total_reward += reward
                    steps += 1
                    
                    if terminated or truncated:
                        break
                
                episode_time = time.time() - episode_start_time
                success = info.get('episode_metrics', {}).get('success', False)
                final_distance = info.get('intercept_distance', 0.0)
                
                episode_summary = {
                    'episode': episode,
                    'steps': steps,
                    'reward': total_reward,
                    'time': episode_time,
                    'success': success,
                    'final_distance': final_distance
                }
                
                episode_data.append(episode_summary)
            
            env.close()
            
            # Calculate run statistics
            run_stats = self._analyze_run_data(episode_data, run_id)
            
            return {
                'run_id': run_id,
                'success': True,
                'episode_summaries': episode_data,
                'run_stats': run_stats
            }
            
        except Exception as e:
            print(f"Error in environment test run {run_id}: {e}")
            return {
                'run_id': run_id,
                'success': False,
                'error': str(e)
            }
    
    def _analyze_run_data(self, episode_data: List[Dict], run_id: int) -> Dict[str, Any]:
        """Analyze data from a single run."""
        if not episode_data:
            return {}
        
        rewards = [ep['reward'] for ep in episode_data]
        steps = [ep['steps'] for ep in episode_data]
        times = [ep['time'] for ep in episode_data]
        successes = [ep['success'] for ep in episode_data]
        distances = [ep['final_distance'] for ep in episode_data]
        
        return {
            'run_id': int(run_id),
            'num_episodes': len(episode_data),
            'success_rate': float(np.mean(successes)),
            'avg_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'avg_steps': float(np.mean(steps)),
            'std_steps': float(np.std(steps)),
            'avg_time': float(np.mean(times)),
            'avg_final_distance': float(np.mean(distances)),
            'min_distance': float(np.min(distances)),
            'max_reward': float(np.max(rewards)),
            'min_reward': float(np.min(rewards)),
            'total_steps': int(np.sum(steps)),
            'total_time': float(np.sum(times))
        }
    
    def run_all_sessions(self):
        """Run all demo/training sessions."""
        print("\\nStarting all runs...")
        start_time = time.time()
        
        if self.args.parallel and self.args.num_runs > 1:
            # Parallel execution
            print(f"Running {self.args.num_runs} sessions in parallel...")
            
            # Limit to reasonable number of processes
            max_workers = min(self.args.num_runs, mp.cpu_count())
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                if self.args.mode == 'demo':
                    futures = [executor.submit(self.run_single_demo, i) 
                              for i in range(self.args.num_runs)]
                else:
                    futures = [executor.submit(self.run_single_environment_test, i) 
                              for i in range(self.args.num_runs)]
                
                # Collect results
                for future in futures:
                    result = future.result()
                    if result['success']:
                        self.runs_data.append(result)
                    else:
                        print(f"Run {result['run_id']} failed: {result.get('error', 'Unknown error')}")
        else:
            # Sequential execution
            print(f"Running {self.args.num_runs} sessions sequentially...")
            
            for i in range(self.args.num_runs):
                if self.args.mode == 'demo':
                    result = self.run_single_demo(i)
                else:
                    result = self.run_single_environment_test(i)
                
                if result['success']:
                    self.runs_data.append(result)
                else:
                    print(f"Run {i} failed: {result.get('error', 'Unknown error')}")
        
        total_time = time.time() - start_time
        print(f"\\nCompleted {len(self.runs_data)} successful runs in {total_time:.2f} seconds")
    
    def analyze_results(self):
        """Analyze results across all runs."""
        print("\\nAnalyzing results across all runs...")
        
        if not self.runs_data:
            print("No successful runs to analyze!")
            return
        
        # Extract run statistics
        run_stats = [run['run_stats'] for run in self.runs_data]
        
        # Calculate overall statistics
        success_rates = [stats['success_rate'] for stats in run_stats]
        avg_rewards = [stats['avg_reward'] for stats in run_stats]
        avg_steps = [stats['avg_steps'] for stats in run_stats]
        avg_distances = [stats['avg_final_distance'] for stats in run_stats]
        
        self.summary_stats = {
            'num_successful_runs': len(self.runs_data),
            'overall_success_rate': {
                'mean': np.mean(success_rates),
                'std': np.std(success_rates),
                'min': np.min(success_rates),
                'max': np.max(success_rates)
            },
            'average_reward': {
                'mean': np.mean(avg_rewards),
                'std': np.std(avg_rewards),
                'min': np.min(avg_rewards),
                'max': np.max(avg_rewards)
            },
            'average_steps': {
                'mean': np.mean(avg_steps),
                'std': np.std(avg_steps),
                'min': np.min(avg_steps),
                'max': np.max(avg_steps)
            },
            'average_final_distance': {
                'mean': np.mean(avg_distances),
                'std': np.std(avg_distances),
                'min': np.min(avg_distances),
                'max': np.max(avg_distances)
            }
        }
        
        # Print summary
        print(f"\\nSummary Statistics ({len(self.runs_data)} runs):")
        print(f"Success Rate: {self.summary_stats['overall_success_rate']['mean']:.2%} ± {self.summary_stats['overall_success_rate']['std']:.2%}")
        print(f"Average Reward: {self.summary_stats['average_reward']['mean']:.2f} ± {self.summary_stats['average_reward']['std']:.2f}")
        print(f"Average Steps: {self.summary_stats['average_steps']['mean']:.1f} ± {self.summary_stats['average_steps']['std']:.1f}")
        print(f"Average Final Distance: {self.summary_stats['average_final_distance']['mean']:.1f}m ± {self.summary_stats['average_final_distance']['std']:.1f}m")
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        print("\\nCreating visualizations...")
        
        if not self.runs_data:
            print("No data to visualize!")
            return
        
        # Set up the plotting style
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        
        # Create multiple visualization plots
        self._create_performance_comparison()
        self._create_episode_progression()
        self._create_statistical_summary()
        self._create_run_comparison_heatmap()
        
        print(f"Visualizations saved to: {self.output_dir}")
    
    def _create_performance_comparison(self):
        """Create performance comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Comparison Across Runs', fontsize=16, fontweight='bold')
        
        # Extract data
        run_stats = [run['run_stats'] for run in self.runs_data]
        run_ids = [stats['run_id'] for stats in run_stats]
        success_rates = [stats['success_rate'] for stats in run_stats]
        avg_rewards = [stats['avg_reward'] for stats in run_stats]
        avg_steps = [stats['avg_steps'] for stats in run_stats]
        avg_distances = [stats['avg_final_distance'] for stats in run_stats]
        
        # Success rates
        axes[0, 0].bar(run_ids, success_rates, alpha=0.7, color='green')
        axes[0, 0].set_title('Success Rate by Run')
        axes[0, 0].set_xlabel('Run ID')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Average rewards
        axes[0, 1].bar(run_ids, avg_rewards, alpha=0.7, color='blue')
        axes[0, 1].set_title('Average Reward by Run')
        axes[0, 1].set_xlabel('Run ID')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average steps
        axes[1, 0].bar(run_ids, avg_steps, alpha=0.7, color='orange')
        axes[1, 0].set_title('Average Steps by Run')
        axes[1, 0].set_xlabel('Run ID')
        axes[1, 0].set_ylabel('Average Steps')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Average final distances
        axes[1, 1].bar(run_ids, avg_distances, alpha=0.7, color='red')
        axes[1, 1].set_title('Average Final Distance by Run')
        axes[1, 1].set_xlabel('Run ID')
        axes[1, 1].set_ylabel('Final Distance (m)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_episode_progression(self):
        """Create episode progression plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Episode Progression Analysis', fontsize=16, fontweight='bold')
        
        # Collect all episode data
        all_episode_rewards = []
        all_episode_steps = []
        all_episode_successes = []
        all_episode_distances = []
        
        for run_idx, run_data in enumerate(self.runs_data):
            episodes = run_data['episode_summaries']
            
            rewards = [ep['reward'] for ep in episodes]
            steps = [ep['steps'] for ep in episodes]
            successes = [int(ep['success']) for ep in episodes]
            distances = [ep['final_distance'] for ep in episodes]
            
            episode_numbers = list(range(len(episodes)))
            
            # Plot individual run progressions
            alpha = 0.3 if len(self.runs_data) > 3 else 0.7
            
            axes[0, 0].plot(episode_numbers, rewards, alpha=alpha, marker='o', markersize=2)
            axes[0, 1].plot(episode_numbers, steps, alpha=alpha, marker='s', markersize=2)
            axes[1, 0].plot(episode_numbers, successes, alpha=alpha, marker='^', markersize=2)
            axes[1, 1].plot(episode_numbers, distances, alpha=alpha, marker='v', markersize=2)
            
            all_episode_rewards.extend(rewards)
            all_episode_steps.extend(steps)
            all_episode_successes.extend(successes)
            all_episode_distances.extend(distances)
        
        # Calculate and plot averages
        if len(self.runs_data) > 1:
            max_episodes = max(len(run['episode_summaries']) for run in self.runs_data)
            
            avg_rewards = []
            avg_steps = []
            avg_successes = []
            avg_distances = []
            
            for ep_idx in range(max_episodes):
                ep_rewards = []
                ep_steps = []
                ep_successes = []
                ep_distances = []
                
                for run_data in self.runs_data:
                    if ep_idx < len(run_data['episode_summaries']):
                        ep = run_data['episode_summaries'][ep_idx]
                        ep_rewards.append(ep['reward'])
                        ep_steps.append(ep['steps'])
                        ep_successes.append(int(ep['success']))
                        ep_distances.append(ep['final_distance'])
                
                if ep_rewards:
                    avg_rewards.append(np.mean(ep_rewards))
                    avg_steps.append(np.mean(ep_steps))
                    avg_successes.append(np.mean(ep_successes))
                    avg_distances.append(np.mean(ep_distances))
            
            episode_range = list(range(len(avg_rewards)))
            
            axes[0, 0].plot(episode_range, avg_rewards, 'k-', linewidth=3, label='Average')
            axes[0, 1].plot(episode_range, avg_steps, 'k-', linewidth=3, label='Average')
            axes[1, 0].plot(episode_range, avg_successes, 'k-', linewidth=3, label='Average')
            axes[1, 1].plot(episode_range, avg_distances, 'k-', linewidth=3, label='Average')
        
        # Configure axes
        axes[0, 0].set_title('Reward Progression')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        axes[0, 1].set_title('Steps Progression')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        axes[1, 0].set_title('Success Rate Progression')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Success (1=Success, 0=Failure)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        axes[1, 1].set_title('Final Distance Progression')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Final Distance (m)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'episode_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_statistical_summary(self):
        """Create statistical summary plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Statistical Summary', fontsize=16, fontweight='bold')
        
        # Extract all data
        run_stats = [run['run_stats'] for run in self.runs_data]
        
        success_rates = [stats['success_rate'] for stats in run_stats]
        avg_rewards = [stats['avg_reward'] for stats in run_stats]
        avg_steps = [stats['avg_steps'] for stats in run_stats]
        avg_distances = [stats['avg_final_distance'] for stats in run_stats]
        
        # Box plots
        data_to_plot = [success_rates, avg_rewards, avg_steps, avg_distances]
        labels = ['Success Rate', 'Avg Reward', 'Avg Steps', 'Avg Final Distance (m)']
        
        for i, (data, label) in enumerate(zip(data_to_plot, labels)):
            row, col = i // 2, i % 2
            
            # Box plot
            bp = axes[row, col].boxplot(data, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            
            # Add individual points
            y_pos = np.ones(len(data)) + np.random.normal(0, 0.02, len(data))
            axes[row, col].scatter(y_pos, data, alpha=0.7, s=30, color='red')
            
            axes[row, col].set_title(f'{label} Distribution')
            axes[row, col].set_ylabel(label)
            axes[row, col].grid(True, alpha=0.3)
            
            # Add statistics text
            mean_val = np.mean(data)
            std_val = np.std(data)
            axes[row, col].text(0.7, 0.95, f'μ = {mean_val:.3f}\\nσ = {std_val:.3f}', 
                              transform=axes[row, col].transAxes, 
                              verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_run_comparison_heatmap(self):
        """Create run comparison heatmap."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Prepare data for heatmap
        run_stats = [run['run_stats'] for run in self.runs_data]
        
        metrics = ['success_rate', 'avg_reward', 'avg_steps', 'avg_final_distance']
        metric_labels = ['Success Rate', 'Avg Reward', 'Avg Steps', 'Final Distance']
        
        data_matrix = []
        for stats in run_stats:
            row = [stats[metric] for metric in metrics]
            data_matrix.append(row)
        
        # Normalize data for better visualization
        data_matrix = np.array(data_matrix)
        std_dev = data_matrix.std(axis=0)
        std_dev[std_dev == 0] = 1  # Avoid division by zero
        normalized_data = (data_matrix - data_matrix.mean(axis=0)) / std_dev
        
        # Create heatmap
        im = ax.imshow(normalized_data, cmap='RdYlBu_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(metric_labels)))
        ax.set_xticklabels(metric_labels)
        ax.set_yticks(range(len(run_stats)))
        ax.set_yticklabels([f'Run {stats["run_id"]}' for stats in run_stats])
        
        # Add text annotations
        for i in range(len(run_stats)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Run Comparison Heatmap (Normalized Values)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Performance (σ)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'run_comparison_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save all results to files."""
        print("\\nSaving results...")
        
        # Save summary statistics
        with open(self.output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(self.summary_stats, f, indent=2)
        
        # Save detailed run data
        detailed_results = {
            'metadata': {
                'mode': self.args.mode,
                'num_runs': len(self.runs_data),
                'episodes_per_run': self.args.episodes_per_run,
                'timestamp': time.time()
            },
            'runs_data': self.runs_data,
            'summary_statistics': self.summary_stats
        }
        
        with open(self.output_dir / 'detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Save CSV summary for easy analysis
        import pandas as pd
        
        run_stats = [run['run_stats'] for run in self.runs_data]
        df = pd.DataFrame(run_stats)
        df.to_csv(self.output_dir / 'run_statistics.csv', index=False)
        
        print(f"Results saved to {self.output_dir}/")
        print("Files created:")
        print("  - summary_statistics.json")
        print("  - detailed_results.json") 
        print("  - run_statistics.csv")
        print("  - performance_comparison.png")
        print("  - episode_progression.png")
        print("  - statistical_summary.png")
        print("  - run_comparison_heatmap.png")
    
    def run_visualization(self):
        """Run the complete visualization pipeline."""
        try:
            # Run all sessions
            self.run_all_sessions()
            
            # Analyze results
            self.analyze_results()
            
            # Create visualizations
            self.create_visualizations()
            
            # Save results
            self.save_results()
            
            print("\\n" + "="*70)
            print("Multi-run visualization completed successfully!")
            print(f"Check {self.output_dir}/ for all results and visualizations.")
            print("="*70)
            
            return 0
            
        except Exception as e:
            print(f"\\nVisualization failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 1


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Multi-Run Visualization for AegisIntercept Phase 3')
    
    # Run configuration
    parser.add_argument('--mode', choices=['demo', 'env_test'], default='env_test',
                       help='Mode: demo (full demo system) or env_test (environment only)')
    parser.add_argument('--num-runs', type=int, default=10,
                       help='Number of runs to execute')
    parser.add_argument('--episodes-per-run', type=int, default=5,
                       help='Number of episodes per run')
    parser.add_argument('--timesteps-per-run', type=int, default=10000,
                       help='Timesteps per run (training mode)')
    
    # Model and scenario
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model (demo mode only)')
    
    # Execution options
    parser.add_argument('--parallel', action='store_true',
                       help='Run sessions in parallel')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='multi_run_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create and run visualizer
    visualizer = MultiRunVisualizer(args)
    exit_code = visualizer.run_visualization()
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)