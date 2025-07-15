#!/usr/bin/env python3
"""
Enhanced 3D Trajectory Visualization for AegisIntercept Phase 3.

This script creates high-quality 3D visualizations of missile intercept trajectories
similar to the reference image, with colored trajectory lines, grid overlay, and
proper markers for interceptor, adversary, and target.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import sys

# Add project root to path
sys.path.append('.')

from aegis_intercept.envs.aegis_6dof_env import Aegis6DOFEnv
from aegis_intercept.utils.maths import distance


class TrajectoryData:
    """Container for trajectory data."""
    
    def __init__(self):
        self.interceptor_positions = []
        self.adversary_positions = []
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.interceptor_velocities = []
        self.adversary_velocities = []
        self.times = []
        self.rewards = []
        self.distances = []
        self.fuel_levels = []
        self.success = False
        self.termination_reason = "unknown"
        self.episode_id = 0
        self.seed = None


class Trajectory3DVisualizer:
    """
    Enhanced 3D trajectory visualization system.
    
    Creates publication-quality 3D visualizations of missile intercept scenarios
    with proper styling, grid overlay, and trajectory analysis.
    """
    
    def __init__(self, 
                 world_scale: float = 3000.0,
                 grid_spacing: float = 500.0,
                 trajectory_alpha: float = 0.8,
                 figure_size: Tuple[int, int] = (14, 10)):
        """
        Initialize 3D trajectory visualizer.
        
        Args:
            world_scale: Scale of the visualization world (meters)
            grid_spacing: Spacing between grid lines (meters)
            trajectory_alpha: Transparency of trajectory lines
            figure_size: Figure size in inches
        """
        self.world_scale = world_scale
        self.grid_spacing = grid_spacing
        self.trajectory_alpha = trajectory_alpha
        self.figure_size = figure_size
        
        # Color schemes
        self.trajectory_colors = [
            '#FF6B6B',  # Red
            '#4ECDC4',  # Teal
            '#45B7D1',  # Blue
            '#FFA07A',  # Light Salmon
            '#98D8C8',  # Mint
            '#F7DC6F',  # Yellow
            '#BB8FCE',  # Purple
            '#85C1E9',  # Light Blue
            '#F8C471',  # Orange
            '#82E0AA'   # Light Green
        ]
        
        # Marker styles
        self.interceptor_marker = {
            'marker': '^',
            'size': 120,
            'color': '#2E86AB',
            'edgecolor': 'black',
            'linewidth': 2
        }
        
        self.adversary_marker = {
            'marker': 'o',
            'size': 100,
            'color': '#F24236',
            'edgecolor': 'black',
            'linewidth': 2
        }
        
        self.target_marker = {
            'marker': '*',
            'size': 200,
            'color': '#2ECC71',
            'edgecolor': 'black',
            'linewidth': 2
        }
        
        # Initialize matplotlib
        plt.style.use('seaborn-v0_8-white')
        
    def create_3d_grid(self, ax: plt.Axes):
        """Create a 3D grid overlay."""
        # Grid parameters
        limit = self.world_scale / 2
        grid_range = np.arange(-limit, limit + self.grid_spacing, self.grid_spacing)
        
        # Create grid lines
        for x in grid_range:
            # YZ plane lines
            ax.plot([x, x], [-limit, limit], [0, 0], 'k-', alpha=0.1, linewidth=0.5)
            ax.plot([x, x], [-limit, -limit], [0, limit], 'k-', alpha=0.1, linewidth=0.5)
            
        for y in grid_range:
            # XZ plane lines
            ax.plot([-limit, limit], [y, y], [0, 0], 'k-', alpha=0.1, linewidth=0.5)
            ax.plot([-limit, -limit], [y, y], [0, limit], 'k-', alpha=0.1, linewidth=0.5)
            
        # Vertical grid lines (every other line to avoid clutter)
        for i, z in enumerate(np.arange(0, limit, self.grid_spacing * 2)):
            ax.plot([-limit, limit], [-limit, -limit], [z, z], 'k-', alpha=0.1, linewidth=0.5)
            ax.plot([-limit, -limit], [-limit, limit], [z, z], 'k-', alpha=0.1, linewidth=0.5)
    
    def plot_trajectory_line(self, ax: plt.Axes, positions: np.ndarray, 
                           color: str, linewidth: float = 2.5, 
                           alpha: float = None, label: str = None):
        """Plot a 3D trajectory line with proper styling."""
        if alpha is None:
            alpha = self.trajectory_alpha
            
        if len(positions) < 2:
            return
            
        # Create line collection for better performance and styling
        lines = []
        colors = []
        
        for i in range(len(positions) - 1):
            lines.append([positions[i], positions[i + 1]])
            colors.append(color)
        
        # Create 3D line collection
        line_collection = Line3DCollection(lines, colors=colors, linewidths=linewidth, alpha=alpha)
        ax.add_collection3d(line_collection)
        
        # Add to legend if label provided
        if label:
            ax.plot([], [], [], color=color, linewidth=linewidth, alpha=alpha, label=label)
    
    def plot_single_trajectory(self, trajectory: TrajectoryData, 
                             color: str = None, show_markers: bool = True,
                             show_grid: bool = True, title: str = None) -> plt.Figure:
        """
        Plot a single trajectory in 3D space.
        
        Args:
            trajectory: Trajectory data to plot
            color: Color for the trajectory line
            show_markers: Whether to show position markers
            show_grid: Whether to show grid overlay
            title: Custom title for the plot
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=self.figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # Use default color if none provided
        if color is None:
            color = self.trajectory_colors[0]
        
        # Set up the plot
        limit = self.world_scale / 2
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([0, limit])
        
        # Labels and styling
        ax.set_xlabel('X (meters)', fontsize=12, labelpad=10)
        ax.set_ylabel('Y (meters)', fontsize=12, labelpad=10)
        ax.set_zlabel('Z (meters)', fontsize=12, labelpad=10)
        
        # Grid
        if show_grid:
            self.create_3d_grid(ax)
            
        # Plot trajectory lines
        if len(trajectory.interceptor_positions) > 1:
            interceptor_pos = np.array(trajectory.interceptor_positions)
            self.plot_trajectory_line(ax, interceptor_pos, color, 
                                    linewidth=3.0, label='Interceptor')
        
        if len(trajectory.adversary_positions) > 1:
            adversary_pos = np.array(trajectory.adversary_positions)
            self.plot_trajectory_line(ax, adversary_pos, '#FF4444', 
                                    linewidth=2.5, label='Adversary')
        
        # Plot markers
        if show_markers and len(trajectory.interceptor_positions) > 0:
            # Initial positions
            init_int_pos = trajectory.interceptor_positions[0]
            init_adv_pos = trajectory.adversary_positions[0]
            
            ax.scatter(init_int_pos[0], init_int_pos[1], init_int_pos[2],
                      s=self.interceptor_marker['size'], 
                      c=self.interceptor_marker['color'],
                      marker=self.interceptor_marker['marker'],
                      edgecolors=self.interceptor_marker['edgecolor'],
                      linewidths=self.interceptor_marker['linewidth'],
                      alpha=0.7, label='Interceptor Start')
            
            ax.scatter(init_adv_pos[0], init_adv_pos[1], init_adv_pos[2],
                      s=self.adversary_marker['size'],
                      c=self.adversary_marker['color'],
                      marker=self.adversary_marker['marker'],
                      edgecolors=self.adversary_marker['edgecolor'],
                      linewidths=self.adversary_marker['linewidth'],
                      alpha=0.7, label='Adversary Start')
            
            # Final positions
            final_int_pos = trajectory.interceptor_positions[-1]
            final_adv_pos = trajectory.adversary_positions[-1]
            
            ax.scatter(final_int_pos[0], final_int_pos[1], final_int_pos[2],
                      s=self.interceptor_marker['size'] * 0.8,
                      c='white',
                      marker=self.interceptor_marker['marker'],
                      edgecolors=self.interceptor_marker['color'],
                      linewidths=3,
                      alpha=0.9, label='Interceptor End')
            
            ax.scatter(final_adv_pos[0], final_adv_pos[1], final_adv_pos[2],
                      s=self.adversary_marker['size'] * 0.8,
                      c='white',
                      marker=self.adversary_marker['marker'],
                      edgecolors=self.adversary_marker['color'],
                      linewidths=3,
                      alpha=0.9, label='Adversary End')
        
        # Target position
        ax.scatter(trajectory.target_position[0], trajectory.target_position[1], trajectory.target_position[2],
                  s=self.target_marker['size'],
                  c=self.target_marker['color'],
                  marker=self.target_marker['marker'],
                  edgecolors=self.target_marker['edgecolor'],
                  linewidths=self.target_marker['linewidth'],
                  alpha=0.9, label='Target')
        
        # Title and info
        if title is None:
            status = "SUCCESS" if trajectory.success else "FAILED"
            final_distance = trajectory.distances[-1] if trajectory.distances else 0
            title = f"Episode {trajectory.episode_id}: {status} | Final Distance: {final_distance:.1f}m"
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Legend
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=True, 
                 fancybox=True, shadow=True, fontsize=10)
        
        # Set viewing angle
        ax.view_init(elev=25, azim=45)
        
        # Improve appearance
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_multiple_trajectories(self, trajectories: List[TrajectoryData],
                                 title: str = "Multiple Trajectory Comparison",
                                 show_grid: bool = True) -> plt.Figure:
        """
        Plot multiple trajectories in the same 3D space.
        
        Args:
            trajectories: List of trajectory data
            title: Title for the plot
            show_grid: Whether to show grid overlay
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=self.figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set up the plot
        limit = self.world_scale / 2
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([0, limit])
        
        # Labels and styling
        ax.set_xlabel('X (meters)', fontsize=12, labelpad=10)
        ax.set_ylabel('Y (meters)', fontsize=12, labelpad=10)
        ax.set_zlabel('Z (meters)', fontsize=12, labelpad=10)
        
        # Grid
        if show_grid:
            self.create_3d_grid(ax)
        
        # Plot each trajectory
        success_count = 0
        for i, trajectory in enumerate(trajectories):
            if trajectory.success:
                success_count += 1
                
            # Use different colors for each trajectory
            color = self.trajectory_colors[i % len(self.trajectory_colors)]
            
            # Plot interceptor trajectory
            if len(trajectory.interceptor_positions) > 1:
                interceptor_pos = np.array(trajectory.interceptor_positions)
                self.plot_trajectory_line(ax, interceptor_pos, color, 
                                        linewidth=2.0, alpha=0.7,
                                        label=f'Episode {trajectory.episode_id}')
            
            # Plot final interceptor position
            if len(trajectory.interceptor_positions) > 0:
                final_pos = trajectory.interceptor_positions[-1]
                marker_color = '#2ECC71' if trajectory.success else '#E74C3C'
                ax.scatter(final_pos[0], final_pos[1], final_pos[2],
                          s=80, c=marker_color, marker='o',
                          edgecolors='black', linewidths=1, alpha=0.8)
        
        # Plot target (shared by all trajectories)
        if trajectories:
            target_pos = trajectories[0].target_position
            ax.scatter(target_pos[0], target_pos[1], target_pos[2],
                      s=self.target_marker['size'],
                      c=self.target_marker['color'],
                      marker=self.target_marker['marker'],
                      edgecolors=self.target_marker['edgecolor'],
                      linewidths=self.target_marker['linewidth'],
                      alpha=0.9, label='Target')
        
        # Title with statistics
        success_rate = (success_count / len(trajectories)) * 100 if trajectories else 0
        title_with_stats = f"{title}\\n{len(trajectories)} Episodes | Success Rate: {success_rate:.1f}%"
        ax.set_title(title_with_stats, fontsize=14, fontweight='bold', pad=20)
        
        # Legend
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1), frameon=True, 
                 fancybox=True, shadow=True, fontsize=9)
        
        # Set viewing angle
        ax.view_init(elev=25, azim=45)
        
        # Improve appearance
        ax.grid(True, alpha=0.3)
        
        return fig


def generate_demo_trajectories(num_episodes: int = 5, seed: int = 42) -> List[TrajectoryData]:
    """
    Generate demo trajectories using the environment.
    
    Args:
        num_episodes: Number of episodes to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of trajectory data
    """
    print(f"Generating {num_episodes} demo trajectories...")
    
    # Create environment
    env = Aegis6DOFEnv(curriculum_level="easy")
    
    trajectories = []
    
    for episode in range(num_episodes):
        print(f"  Episode {episode + 1}/{num_episodes}")
        
        # Set seed for reproducibility
        np.random.seed(seed + episode)
        
        # Reset environment
        obs, info = env.reset()
        
        # Create trajectory container
        trajectory = TrajectoryData()
        trajectory.episode_id = episode + 1
        trajectory.seed = seed + episode
        trajectory.target_position = env.target_position.copy()
        
        # Run episode
        step = 0
        done = False
        total_reward = 0
        
        while not done and step < 200:  # Limit steps for demo
            # Random action (in a real scenario, you'd use a trained model)
            action = np.random.uniform(-0.3, 0.3, 6)  # Conservative actions
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            # Log trajectory data
            trajectory.times.append(step * env.time_step)
            trajectory.interceptor_positions.append(env.interceptor.get_position().copy())
            trajectory.adversary_positions.append(env.adversary.get_position().copy())
            trajectory.interceptor_velocities.append(env.interceptor.get_velocity().copy())
            trajectory.adversary_velocities.append(env.adversary.get_velocity().copy())
            trajectory.rewards.append(reward)
            trajectory.distances.append(info.get('intercept_distance', 0))
            trajectory.fuel_levels.append(info.get('fuel_remaining', 0))
            
            step += 1
        
        # Finalize trajectory
        trajectory.success = info.get('intercept_distance', float('inf')) < 20.0
        trajectory.termination_reason = info.get('termination_reason', 'max_steps')
        
        trajectories.append(trajectory)
        
        print(f"    Result: {'SUCCESS' if trajectory.success else 'FAILED'} | "
              f"Steps: {step} | Distance: {trajectory.distances[-1]:.1f}m")
    
    env.close()
    print(f"Generated {len(trajectories)} trajectories")
    return trajectories


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='3D Trajectory Visualization for AegisIntercept')
    parser.add_argument('--num-episodes', type=int, default=5, 
                       help='Number of episodes to generate (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output-dir', type=str, default='trajectory_3d_output',
                       help='Output directory for visualizations (default: trajectory_3d_output)')
    parser.add_argument('--world-scale', type=float, default=3000.0,
                       help='World scale in meters (default: 3000.0)')
    parser.add_argument('--show-individual', action='store_true',
                       help='Show individual trajectory plots')
    parser.add_argument('--show-combined', action='store_true', default=True,
                       help='Show combined trajectory plot (default: True)')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to files')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("AegisIntercept Phase 3 - 3D Trajectory Visualization")
    print("="*70)
    print(f"Episodes: {args.num_episodes}")
    print(f"Seed: {args.seed}")
    print(f"World Scale: {args.world_scale}m")
    print(f"Output Directory: {output_dir}")
    print("="*70)
    
    # Generate trajectories
    trajectories = generate_demo_trajectories(args.num_episodes, args.seed)
    
    # Create visualizer
    visualizer = Trajectory3DVisualizer(world_scale=args.world_scale)
    
    # Individual trajectory plots
    if args.show_individual:
        print("\\nCreating individual trajectory plots...")
        for i, trajectory in enumerate(trajectories):
            fig = visualizer.plot_single_trajectory(trajectory)
            
            if args.save_plots:
                filename = output_dir / f"trajectory_episode_{trajectory.episode_id:03d}.png"
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"  Saved: {filename}")
            
            plt.show()
    
    # Combined trajectory plot
    if args.show_combined:
        print("\\nCreating combined trajectory plot...")
        fig = visualizer.plot_multiple_trajectories(trajectories)
        
        if args.save_plots:
            filename = output_dir / "trajectories_combined.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Saved: {filename}")
        
        plt.show()
    
    # Summary statistics
    success_count = sum(1 for t in trajectories if t.success)
    success_rate = (success_count / len(trajectories)) * 100
    avg_distance = np.mean([t.distances[-1] for t in trajectories if t.distances])
    
    print("\\n" + "="*70)
    print("TRAJECTORY ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total Episodes: {len(trajectories)}")
    print(f"Successful Intercepts: {success_count}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Final Distance: {avg_distance:.1f}m")
    print("="*70)
    
    if args.save_plots:
        print(f"\\nAll plots saved to: {output_dir}/")
    
    return 0


if __name__ == "__main__":
    exit(main())