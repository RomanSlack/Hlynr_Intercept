#!/usr/bin/env python3
"""
Demo 3D Visualization for AegisIntercept Phase 3.

This script demonstrates the 3D trajectory visualization capabilities
without requiring trained models, showing different trajectory patterns
and visualization styles.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import argparse
from pathlib import Path
from typing import List, Tuple
import sys

# Add project root to path
sys.path.append('.')

from visualize_3d_trajectories import TrajectoryData, Trajectory3DVisualizer


def create_intercept_trajectory(scenario: str = "head_on") -> TrajectoryData:
    """
    Create a realistic intercept trajectory based on different scenarios.
    
    Args:
        scenario: Type of intercept scenario
        
    Returns:
        Trajectory data
    """
    trajectory = TrajectoryData()
    trajectory.target_position = np.array([0.0, 0.0, 0.0])
    
    # Time parameters
    dt = 0.05  # 50ms timesteps
    max_time = 15.0  # 15 seconds
    times = np.arange(0, max_time, dt)
    
    if scenario == "head_on":
        # Head-on intercept scenario
        # Adversary starts far away, moves toward target
        adv_start = np.array([2000.0, 0.0, 1000.0])
        adv_vel = np.array([-200.0, 0.0, -50.0])  # Moving toward target
        
        # Interceptor starts at favorable position
        int_start = np.array([-1000.0, 500.0, 800.0])
        int_target_vel = np.array([250.0, -50.0, 0.0])  # Intercept course
        
        # Success at t=8s
        success_time = 8.0
        success_step = int(success_time / dt)
        
    elif scenario == "pursuit":
        # Pursuit scenario - interceptor chases adversary
        adv_start = np.array([1500.0, 1000.0, 1200.0])
        adv_vel = np.array([-150.0, -100.0, -80.0])
        
        int_start = np.array([0.0, 0.0, 1000.0])
        int_target_vel = np.array([200.0, 120.0, 20.0])
        
        success_time = 12.0
        success_step = int(success_time / dt)
        
    elif scenario == "evasion":
        # Evasion scenario - adversary tries to evade
        adv_start = np.array([1800.0, -500.0, 1500.0])
        adv_vel = np.array([-100.0, 50.0, -100.0])
        
        int_start = np.array([-800.0, 800.0, 1000.0])
        int_target_vel = np.array([180.0, -80.0, 40.0])
        
        success_time = 10.0
        success_step = int(success_time / dt)
        
    elif scenario == "miss":
        # Miss scenario - interceptor fails to intercept
        adv_start = np.array([2200.0, 300.0, 1800.0])
        adv_vel = np.array([-180.0, -30.0, -120.0])
        
        int_start = np.array([-1200.0, -600.0, 1200.0])
        int_target_vel = np.array([200.0, 100.0, -50.0])
        
        success_time = None  # No intercept
        success_step = None
        
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # Generate trajectories
    adv_pos = adv_start.copy()
    int_pos = int_start.copy()
    
    # Add some realistic control adjustments
    np.random.seed(42)  # For reproducibility
    
    for i, t in enumerate(times):
        # Adversary motion (mostly straight with some evasion)
        if i > 0:
            # Add some evasion maneuvers
            evasion_factor = np.sin(t * 0.5) * 0.3 if scenario == "evasion" else 0.0
            adv_accel = np.array([0.0, evasion_factor * 20.0, evasion_factor * 10.0])
            adv_vel += adv_accel * dt
            adv_pos += adv_vel * dt
        
        # Interceptor motion (guidance toward intercept)
        if i > 0:
            # Proportional navigation guidance
            relative_pos = adv_pos - int_pos
            relative_vel = adv_vel - int_target_vel
            
            # Simple proportional navigation
            if np.linalg.norm(relative_pos) > 10.0:
                desired_vel = normalize_vector(relative_pos) * np.linalg.norm(int_target_vel)
                vel_error = desired_vel - int_target_vel
                int_accel = vel_error * 0.5  # Proportional control
                int_target_vel += int_accel * dt
            
            int_pos += int_target_vel * dt
        
        # Store trajectory points
        trajectory.times.append(t)
        trajectory.interceptor_positions.append(int_pos.copy())
        trajectory.adversary_positions.append(adv_pos.copy())
        trajectory.interceptor_velocities.append(int_target_vel.copy())
        trajectory.adversary_velocities.append(adv_vel.copy())
        
        # Calculate distance and other metrics
        distance = np.linalg.norm(adv_pos - int_pos)
        trajectory.distances.append(distance)
        trajectory.rewards.append(max(0, 1000 - distance))  # Simple reward
        trajectory.fuel_levels.append(max(0, 1.0 - t / max_time))  # Fuel consumption
        
        # Check for intercept
        if success_step is not None and i >= success_step:
            trajectory.success = True
            trajectory.termination_reason = "intercept_success"
            break
        
        # Check if adversary reached target
        if np.linalg.norm(adv_pos - trajectory.target_position) < 50.0:
            trajectory.success = False
            trajectory.termination_reason = "adversary_reached_target"
            break
    
    # Finalize trajectory
    if not trajectory.success and trajectory.termination_reason == "unknown":
        trajectory.success = False
        trajectory.termination_reason = "max_time_reached"
    
    return trajectory


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return np.zeros_like(vector)
    return vector / norm


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Demo 3D Trajectory Visualization')
    parser.add_argument('--scenario', type=str, default='all',
                       choices=['head_on', 'pursuit', 'evasion', 'miss', 'all'],
                       help='Scenario to visualize (default: all)')
    parser.add_argument('--output-dir', type=str, default='demo_3d_output',
                       help='Output directory for visualizations')
    parser.add_argument('--world-scale', type=float, default=3000.0,
                       help='World scale in meters (default: 3000.0)')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to files')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("AegisIntercept Phase 3 - Demo 3D Trajectory Visualization")
    print("="*70)
    print(f"Scenario: {args.scenario}")
    print(f"World Scale: {args.world_scale}m")
    print(f"Output Directory: {output_dir}")
    print("="*70)
    
    # Create visualizer
    visualizer = Trajectory3DVisualizer(world_scale=args.world_scale)
    
    # Define scenarios
    if args.scenario == 'all':
        scenarios = ['head_on', 'pursuit', 'evasion', 'miss']
    else:
        scenarios = [args.scenario]
    
    # Create trajectories for each scenario
    trajectories = []
    for i, scenario in enumerate(scenarios):
        print(f"\\nGenerating {scenario} scenario...")
        trajectory = create_intercept_trajectory(scenario)
        trajectory.episode_id = i + 1
        trajectories.append(trajectory)
        
        status = "SUCCESS" if trajectory.success else "FAILED"
        final_distance = trajectory.distances[-1] if trajectory.distances else 0
        print(f"  Result: {status} | Steps: {len(trajectory.times)} | Distance: {final_distance:.1f}m")
        
        # Create individual plot
        fig = visualizer.plot_single_trajectory(
            trajectory, 
            title=f"{scenario.replace('_', ' ').title()} Scenario: {status}"
        )
        
        if args.save_plots:
            filename = output_dir / f"scenario_{scenario}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Saved: {filename}")
        
        plt.show()
    
    # Create combined plot if multiple scenarios
    if len(trajectories) > 1:
        print("\\nCreating combined scenario comparison...")
        fig = visualizer.plot_multiple_trajectories(
            trajectories, 
            title="Missile Intercept Scenarios Comparison"
        )
        
        if args.save_plots:
            filename = output_dir / "scenarios_combined.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Saved: {filename}")
        
        plt.show()
    
    # Summary statistics
    success_count = sum(1 for t in trajectories if t.success)
    success_rate = (success_count / len(trajectories)) * 100
    avg_distance = np.mean([t.distances[-1] for t in trajectories if t.distances])
    
    print("\\n" + "="*70)
    print("DEMO TRAJECTORY ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total Scenarios: {len(trajectories)}")
    print(f"Successful Intercepts: {success_count}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Final Distance: {avg_distance:.1f}m")
    print("="*70)
    
    if args.save_plots:
        print(f"\\nAll plots saved to: {output_dir}/")
    
    print("\\nDemo completed! This shows the 3D visualization capabilities")
    print("for different missile intercept scenarios.")
    
    return 0


if __name__ == "__main__":
    exit(main())