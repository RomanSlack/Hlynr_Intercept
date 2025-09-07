"""
Multi-entity diagnostics system for Phase 4 RL training and inference.

This module provides comprehensive logging, analysis, and visualization capabilities
for multi-missile/interceptor scenarios with scenario-aware metrics.
"""

import csv
import json
import os
from pathlib import Path
import time
from typing import Dict, Any, List, Optional, Union, Tuple
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation


class Logger:
    """
    Multi-entity logger for capturing per-step metrics during training and inference.
    
    Tracks detailed metrics for each missile, interceptor, and overall system performance
    with scenario-specific analysis capabilities.
    """
    
    def __init__(self):
        """Initialize logger."""
        self.reset_episode()
        self.episode_data = []
    
    def reset_episode(self):
        """Reset logger for a new episode."""
        self.step_data = []
        self.episode_start_time = time.time()
        self.episode_metrics = {}
    
    def log_step(self, step_data: Dict[str, Any]):
        """
        Log data for a single step.
        
        Args:
            step_data: Dictionary containing step information including:
                - step: Step number
                - observation: Environment observation
                - action: Agent action
                - reward: Step reward
                - done: Episode termination flag
                - info: Additional information from environment
        """
        # Add timestamp
        step_data['timestamp'] = time.time()
        
        # Extract entity-specific information from info if available
        if 'info' in step_data and isinstance(step_data['info'], dict):
            info = step_data['info']
            
            # Extract positions and velocities
            if 'missile_positions' in info:
                step_data['missile_positions'] = info['missile_positions']
            if 'interceptor_positions' in info:
                step_data['interceptor_positions'] = info['interceptor_positions']
            if 'missile_velocities' in info:
                step_data['missile_velocities'] = info['missile_velocities']
            if 'interceptor_velocities' in info:
                step_data['interceptor_velocities'] = info['interceptor_velocities']
            if 'min_interception_distances' in info:
                step_data['min_interception_distances'] = info['min_interception_distances']
        
        self.step_data.append(step_data)
    
    def get_episode_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive episode metrics.
        
        Returns:
            Dictionary of episode metrics
        """
        if not self.step_data:
            return {}
        
        episode_duration = time.time() - self.episode_start_time
        total_steps = len(self.step_data)
        
        # Basic metrics
        rewards = [step['reward'] for step in self.step_data]
        total_reward = sum(rewards)
        
        # Trajectory analysis
        missile_trajectories = self._extract_trajectories('missile_positions')
        interceptor_trajectories = self._extract_trajectories('interceptor_positions')
        
        # Distance analysis
        min_distances_over_time = [
            step.get('min_interception_distances', [])
            for step in self.step_data
        ]
        
        # Calculate closest approach
        closest_approach = float('inf')
        if min_distances_over_time:
            for distances in min_distances_over_time:
                if distances:
                    closest_approach = min(closest_approach, min(distances))
        
        # Calculate interception success
        interception_threshold = 50.0  # Default threshold
        successful_interception = closest_approach <= interception_threshold
        
        # Calculate fuel efficiency (based on velocity changes)
        fuel_usage = self._calculate_fuel_usage()
        
        # Time-based metrics
        efficiency_score = total_reward / max(1, total_steps)  # Reward per step
        
        metrics = {
            'episode_duration': episode_duration,
            'total_steps': total_steps,
            'total_reward': total_reward,
            'average_reward': np.mean(rewards) if rewards else 0.0,
            'reward_std': np.std(rewards) if rewards else 0.0,
            'min_reward': min(rewards) if rewards else 0.0,
            'max_reward': max(rewards) if rewards else 0.0,
            'closest_approach': closest_approach,
            'successful_interception': successful_interception,
            'fuel_usage': fuel_usage,
            'efficiency_score': efficiency_score,
            'missile_trajectories': missile_trajectories,
            'interceptor_trajectories': interceptor_trajectories,
            'min_interception_distances': [
                min(distances) if distances else float('inf')
                for distances in min_distances_over_time
            ]
        }
        
        # Add scenario-specific metrics
        metrics.update(self._calculate_scenario_metrics())
        
        self.episode_metrics = metrics
        return metrics
    
    def _extract_trajectories(self, position_key: str) -> List[List[List[float]]]:
        """
        Extract position trajectories for entities.
        
        Args:
            position_key: Key for position data in step_data
            
        Returns:
            List of trajectories, one per entity
        """
        trajectories = []
        
        # Find maximum number of entities
        max_entities = 0
        for step in self.step_data:
            if position_key in step:
                positions = step[position_key]
                if isinstance(positions, (list, np.ndarray)):
                    max_entities = max(max_entities, len(positions))
        
        # Extract trajectory for each entity
        for entity_id in range(max_entities):
            trajectory = []
            for step in self.step_data:
                if position_key in step:
                    positions = step[position_key]
                    if isinstance(positions, (list, np.ndarray)) and entity_id < len(positions):
                        trajectory.append(positions[entity_id].tolist() if hasattr(positions[entity_id], 'tolist') else list(positions[entity_id]))
            trajectories.append(trajectory)
        
        return trajectories
    
    def _calculate_fuel_usage(self) -> Dict[str, float]:
        """Calculate fuel usage based on velocity changes."""
        fuel_metrics = {
            'total_interceptor_fuel': 0.0,
            'average_interceptor_fuel': 0.0,
            'max_interceptor_fuel': 0.0
        }
        
        interceptor_fuel_usage = []
        
        # Calculate fuel usage for each interceptor
        prev_velocities = None
        for step in self.step_data:
            if 'interceptor_velocities' in step:
                current_velocities = step['interceptor_velocities']
                if prev_velocities is not None:
                    # Calculate velocity changes (proxy for thrust/fuel usage)
                    for i, (prev_vel, curr_vel) in enumerate(zip(prev_velocities, current_velocities)):
                        if len(interceptor_fuel_usage) <= i:
                            interceptor_fuel_usage.append(0.0)
                        
                        # Fuel usage proportional to velocity change magnitude
                        vel_change = np.linalg.norm(np.array(curr_vel) - np.array(prev_vel))
                        interceptor_fuel_usage[i] += vel_change
                
                prev_velocities = current_velocities
        
        if interceptor_fuel_usage:
            fuel_metrics['total_interceptor_fuel'] = sum(interceptor_fuel_usage)
            fuel_metrics['average_interceptor_fuel'] = np.mean(interceptor_fuel_usage)
            fuel_metrics['max_interceptor_fuel'] = max(interceptor_fuel_usage)
        
        return fuel_metrics
    
    def _calculate_scenario_metrics(self) -> Dict[str, Any]:
        """Calculate scenario-specific performance metrics."""
        metrics = {}
        
        # Multi-target coordination (if multiple missiles/interceptors)
        missile_count = len(self._extract_trajectories('missile_positions'))
        interceptor_count = len(self._extract_trajectories('interceptor_positions'))
        
        metrics['entity_counts'] = {
            'missiles': missile_count,
            'interceptors': interceptor_count
        }
        
        # Coordination efficiency (how well interceptors coordinate)
        if interceptor_count > 1:
            coordination_score = self._calculate_coordination_score()
            metrics['coordination_score'] = coordination_score
        
        # Target assignment efficiency
        if missile_count > 1 and interceptor_count > 1:
            assignment_efficiency = self._calculate_assignment_efficiency()
            metrics['assignment_efficiency'] = assignment_efficiency
        
        return metrics
    
    def _calculate_coordination_score(self) -> float:
        """Calculate how well interceptors coordinate (avoid redundancy)."""
        # Simplified coordination metric based on interceptor separation
        # Higher score means better coordination (less redundancy)
        
        interceptor_positions = self._extract_trajectories('interceptor_positions')
        if len(interceptor_positions) < 2:
            return 1.0
        
        separation_scores = []
        for step_idx in range(len(self.step_data)):
            step_positions = []
            for traj in interceptor_positions:
                if step_idx < len(traj):
                    step_positions.append(traj[step_idx])
            
            if len(step_positions) >= 2:
                # Calculate minimum separation between interceptors
                min_separation = float('inf')
                for i in range(len(step_positions)):
                    for j in range(i + 1, len(step_positions)):
                        separation = np.linalg.norm(
                            np.array(step_positions[i]) - np.array(step_positions[j])
                        )
                        min_separation = min(min_separation, separation)
                
                # Higher separation indicates better coordination
                separation_scores.append(min_separation)
        
        return np.mean(separation_scores) if separation_scores else 0.0
    
    def _calculate_assignment_efficiency(self) -> float:
        """Calculate target assignment efficiency."""
        # Simplified metric: how well interceptors assign to different targets
        # This is a placeholder - real implementation would need more sophisticated analysis
        return 0.5  # Placeholder value
    
    def save_episode(self, filename: Optional[str] = None):
        """
        Save episode data to file.
        
        Args:
            filename: Output filename. If None, generates timestamp-based name.
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"episode_{timestamp}.json"
        
        episode_summary = {
            'schema_version': 1,
            'metrics': self.get_episode_metrics(),
            'step_data': self.step_data,
            'timestamp': time.time()
        }
        
        # Use centralized diagnostics logging
        try:
            from hlynr_bridge.paths import logs_diagnostics
            diagnostics_dir = logs_diagnostics()
            filename = diagnostics_dir / Path(filename).name
        except ImportError:
            pass  # Use original filename if centralized paths not available
        
        with open(filename, 'w') as f:
            json.dump(episode_summary, f, indent=2, default=str)


def export_to_csv(results: Dict[str, List[Dict[str, Any]]], output_path: Union[str, Path]):
    """
    Export inference results to CSV format.
    
    Args:
        results: Results dictionary from inference runs
        output_path: Path to output CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Flatten results for CSV export
    csv_rows = []
    
    for scenario_name, episode_results in results.items():
        for episode_result in episode_results:
            row = {
                'scenario': scenario_name,
                'episode': episode_result['episode'],
                'total_reward': episode_result['total_reward'],
                'episode_length': episode_result['episode_length'],
                'success': episode_result['success'],
                'timestamp': episode_result['timestamp']
            }
            
            # Add metrics if available
            if 'metrics' in episode_result:
                metrics = episode_result['metrics']
                
                # Add basic metrics
                for key in ['closest_approach', 'successful_interception', 'efficiency_score']:
                    if key in metrics:
                        row[key] = metrics[key]
                
                # Add fuel metrics
                if 'fuel_usage' in metrics:
                    fuel = metrics['fuel_usage']
                    for fuel_key, fuel_value in fuel.items():
                        row[f'fuel_{fuel_key}'] = fuel_value
                
                # Add entity counts
                if 'entity_counts' in metrics:
                    counts = metrics['entity_counts']
                    row['num_missiles'] = counts.get('missiles', 0)
                    row['num_interceptors'] = counts.get('interceptors', 0)
            
            csv_rows.append(row)
    
    # Write CSV
    if csv_rows:
        fieldnames = csv_rows[0].keys()
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)


def export_to_json(results: Dict[str, List[Dict[str, Any]]], output_path: Union[str, Path]):
    """
    Export inference results to JSON format.
    
    Args:
        results: Results dictionary from inference runs
        output_path: Path to output JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    export_data = {
        'schema_version': 1,
        'metadata': {
            'export_timestamp': time.time(),
            'total_scenarios': len(results),
            'total_episodes': sum(len(episodes) for episodes in results.values())
        },
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)


def load_episode_data(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load episode data with automatic schema migration.
    
    Args:
        filepath: Path to episode JSON file
        
    Returns:
        Migrated episode data dictionary
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Check schema version and migrate if necessary
    schema_version = data.get('schema_version', 0)  # Default to 0 for legacy files
    
    if schema_version == 0:
        # Migrate from legacy format (no schema version)
        data = _migrate_episode_data_v0_to_v1(data)
    elif schema_version == 1:
        # Current version, no migration needed
        pass
    else:
        # Future version - warn but try to load
        warnings.warn(f"Unknown schema version {schema_version}, attempting to load as-is")
    
    return data


def load_inference_results(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load inference results with automatic schema migration.
    
    Args:
        filepath: Path to inference results JSON file
        
    Returns:
        Migrated inference results dictionary
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Check schema version and migrate if necessary
    schema_version = data.get('schema_version', 0)  # Default to 0 for legacy files
    
    if schema_version == 0:
        # Migrate from legacy format (no schema version)
        data = _migrate_inference_results_v0_to_v1(data)
    elif schema_version == 1:
        # Current version, no migration needed
        pass
    else:
        # Future version - warn but try to load
        warnings.warn(f"Unknown schema version {schema_version}, attempting to load as-is")
    
    return data


def _migrate_episode_data_v0_to_v1(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate episode data from version 0 (legacy) to version 1.
    
    Args:
        data: Legacy episode data
        
    Returns:
        Migrated episode data with schema_version: 1
    """
    # Add schema version
    migrated_data = data.copy()
    migrated_data['schema_version'] = 1
    
    # Legacy format should have 'metrics', 'step_data', 'timestamp'
    # No structural changes needed for v0->v1, just add version
    
    return migrated_data


def _migrate_inference_results_v0_to_v1(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate inference results from version 0 (legacy) to version 1.
    
    Args:
        data: Legacy inference results data
        
    Returns:
        Migrated inference results with schema_version: 1
    """
    migrated_data = data.copy()
    migrated_data['schema_version'] = 1
    
    # Legacy format might not have metadata section
    if 'metadata' not in migrated_data:
        # Create metadata section for legacy files
        migrated_data['metadata'] = {
            'export_timestamp': migrated_data.get('timestamp', time.time()),
            'total_scenarios': len(migrated_data.get('results', {})),
            'total_episodes': sum(len(episodes) for episodes in migrated_data.get('results', {}).values())
        }
    
    return migrated_data


def plot_metrics(results: Dict[str, List[Dict[str, Any]]], output_dir: Union[str, Path]):
    """
    Generate comprehensive visualization plots for inference results.
    
    Args:
        results: Results dictionary from inference runs
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Suppress matplotlib warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    # 1. Success rate comparison across scenarios
    _plot_success_rates(results, output_dir)
    
    # 2. Reward distribution plots
    _plot_reward_distributions(results, output_dir)
    
    # 3. Episode length analysis
    _plot_episode_lengths(results, output_dir)
    
    # 4. Trajectory visualization (if trajectory data available)
    _plot_trajectories(results, output_dir)
    
    # 5. Performance correlation matrix
    _plot_performance_correlation(results, output_dir)
    
    print(f"Generated plots in: {output_dir}")


def _plot_success_rates(results: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """Plot success rates across scenarios."""
    scenarios = list(results.keys())
    success_rates = []
    
    for scenario, episodes in results.items():
        success_count = sum(1 for ep in episodes if ep['success'])
        success_rate = success_count / len(episodes) if episodes else 0
        success_rates.append(success_rate)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(scenarios, success_rates, color=['green', 'yellow', 'orange', 'red'][:len(scenarios)])
    plt.title('Success Rate by Scenario', fontsize=16)
    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rates.png', dpi=300, bbox_inches='tight')
    plt.close()


def _plot_reward_distributions(results: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """Plot reward distributions for each scenario."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (scenario, episodes) in enumerate(results.items()):
        if idx >= len(axes):
            break
        
        rewards = [ep['total_reward'] for ep in episodes]
        
        axes[idx].hist(rewards, bins=20, alpha=0.7, color=['green', 'yellow', 'orange', 'red'][idx])
        axes[idx].set_title(f'{scenario.title()} Scenario Rewards', fontsize=12)
        axes[idx].set_xlabel('Total Reward')
        axes[idx].set_ylabel('Frequency')
        axes[idx].axvline(np.mean(rewards), color='black', linestyle='--', 
                         label=f'Mean: {np.mean(rewards):.1f}')
        axes[idx].legend()
    
    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()


def _plot_episode_lengths(results: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """Plot episode length analysis."""
    scenarios = list(results.keys())
    length_data = []
    
    for scenario, episodes in results.items():
        lengths = [ep['episode_length'] for ep in episodes]
        length_data.append(lengths)
    
    plt.figure(figsize=(12, 6))
    box_plot = plt.boxplot(length_data, labels=scenarios, patch_artist=True)
    
    colors = ['lightgreen', 'lightyellow', 'lightcoral', 'lightblue']
    for patch, color in zip(box_plot['boxes'], colors[:len(scenarios)]):
        patch.set_facecolor(color)
    
    plt.title('Episode Length Distribution by Scenario', fontsize=16)
    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('Episode Length (steps)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'episode_lengths.png', dpi=300, bbox_inches='tight')
    plt.close()


def _plot_trajectories(results: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """Plot trajectory visualizations if trajectory data is available."""
    # This is a simplified implementation
    # In practice, you would need trajectory data from the metrics
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (scenario, episodes) in enumerate(results.items()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        ax.set_title(f'{scenario.title()} Scenario Trajectories', fontsize=12)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        
        # Plot sample trajectories if available
        for ep_idx, episode in enumerate(episodes[:3]):  # Show first 3 episodes
            if 'metrics' in episode and 'missile_trajectories' in episode['metrics']:
                trajectories = episode['metrics']['missile_trajectories']
                for traj_idx, trajectory in enumerate(trajectories):
                    if trajectory:
                        x_coords = [point[0] for point in trajectory]
                        y_coords = [point[1] for point in trajectory]
                        ax.plot(x_coords, y_coords, alpha=0.6, 
                               label=f'Missile {traj_idx}' if ep_idx == 0 else '')
        
        if idx == 0:  # Only show legend for first plot
            ax.legend()
    
    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()


def _plot_performance_correlation(results: Dict[str, List[Dict[str, Any]]], output_dir: Path):
    """Plot performance correlation matrix."""
    # Collect all performance metrics
    all_data = []
    
    for scenario, episodes in results.items():
        for episode in episodes:
            row = {
                'scenario_num': list(results.keys()).index(scenario),
                'total_reward': episode['total_reward'],
                'episode_length': episode['episode_length'],
                'success': int(episode['success'])
            }
            
            if 'metrics' in episode:
                metrics = episode['metrics']
                if 'closest_approach' in metrics:
                    row['closest_approach'] = metrics['closest_approach']
                if 'efficiency_score' in metrics:
                    row['efficiency_score'] = metrics['efficiency_score']
            
            all_data.append(row)
    
    if all_data:
        import pandas as pd
        df = pd.DataFrame(all_data)
        
        # Calculate correlation matrix
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Correlation Coefficient', fontsize=12)
        
        # Set labels
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha='right')
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        
        # Add correlation values
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                        ha='center', va='center', fontsize=10)
        
        plt.title('Performance Metrics Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()


# Legacy function names for backwards compatibility
def export(results: Dict[str, List[Dict[str, Any]]], 
          output_dir: Union[str, Path],
          formats: List[str] = ["csv", "json"]):
    """
    Legacy export function for backwards compatibility.
    
    Args:
        results: Results from inference runs
        output_dir: Output directory
        formats: List of export formats
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    
    if "csv" in formats:
        export_to_csv(results, output_dir / f"results_{timestamp}.csv")
    
    if "json" in formats:
        export_to_json(results, output_dir / f"results_{timestamp}.json")
    
    plot_metrics(results, output_dir / f"plots_{timestamp}")