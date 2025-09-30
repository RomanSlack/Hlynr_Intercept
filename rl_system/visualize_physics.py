#!/usr/bin/env python3
"""
Physics-based trajectory visualizer - simulates actual physics with logged actions.
Shows what SHOULD happen when thrust commands are applied.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import re


def load_episode_actions(jsonl_path):
    """Load episode data including actions."""
    interceptor_data = []
    missile_data = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)

            if data['type'] == 'state':
                pos = data['state']['position']
                timestamp = data['timestamp']

                if data['entity_id'] == 'interceptor':
                    action = data['state'].get('action', None)
                    fuel = data['state'].get('fuel', 100.0)
                    interceptor_data.append({
                        'time': timestamp,
                        'pos': np.array(pos),
                        'fuel': fuel,
                        'action': np.array(action) if action else np.zeros(6)
                    })
                elif data['entity_id'] == 'missile':
                    missile_data.append({
                        'time': timestamp,
                        'pos': np.array(pos)
                    })

    return interceptor_data, missile_data


def simulate_physics(interceptor_data, dt=0.01):
    """Re-simulate interceptor physics with logged actions."""
    if not interceptor_data or interceptor_data[0]['action'] is None:
        return None

    # Initial state
    initial_pos = interceptor_data[0]['pos']

    # Estimate initial velocity from position change
    if len(interceptor_data) > 1:
        initial_vel = (interceptor_data[1]['pos'] - interceptor_data[0]['pos']) / dt
    else:
        initial_vel = np.array([100.0, 100.0, 50.0])  # Default upward velocity

    # Physics parameters
    mass = 500.0  # kg
    gravity = np.array([0.0, 0.0, -9.81])  # m/s^2
    drag_coef = 0.47
    cross_section = 0.2  # m^2
    air_density = 1.225  # kg/m^3

    # Simulate
    positions = [initial_pos]
    velocities = [initial_vel]

    pos = initial_pos.copy()
    vel = initial_vel.copy()

    # Get total simulation time from logged data
    t0 = interceptor_data[0]['time']
    total_time = interceptor_data[-1]['time'] - t0

    # Simulate with actions
    for i, state in enumerate(interceptor_data[:-1]):
        action = state['action']

        # Extract thrust command (first 3 components, normalized -1 to 1)
        thrust_normalized = action[0:3]

        # Scale to actual thrust force (500N max per component)
        thrust_force = thrust_normalized * 500.0

        # Calculate drag
        vel_mag = np.linalg.norm(vel)
        if vel_mag > 1e-6:
            drag_force = -0.5 * air_density * drag_coef * cross_section * vel_mag**2 * (vel / vel_mag)
        else:
            drag_force = np.zeros(3)

        # Total acceleration
        accel = (thrust_force + drag_force) / mass + gravity

        # Euler integration
        vel = vel + accel * dt
        pos = pos + vel * dt

        positions.append(pos.copy())
        velocities.append(vel.copy())

    return np.array(positions)


def visualize_comparison(jsonl_path):
    """Visualize logged trajectory vs physics simulation."""
    print(f"Loading episode: {jsonl_path}")
    interceptor_data, missile_data = load_episode_actions(jsonl_path)

    if not interceptor_data or not missile_data:
        print("Error: No trajectory data found")
        return

    # Extract logged positions
    logged_positions = np.array([s['pos'] for s in interceptor_data])
    missile_positions = np.array([s['pos'] for s in missile_data])

    # Simulate physics with actions
    print("Simulating physics with logged actions...")
    simulated_positions = simulate_physics(interceptor_data)

    if simulated_positions is None:
        print("Error: No action data found for physics simulation")
        return

    print(f"Logged trajectory: {len(logged_positions)} points")
    print(f"Simulated trajectory: {len(simulated_positions)} points")
    print(f"Missile trajectory: {len(missile_positions)} points")

    # Setup figure
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.18)

    # Calculate bounds
    all_positions = np.vstack([logged_positions, simulated_positions, missile_positions])
    mins = all_positions.min(axis=0)
    maxs = all_positions.max(axis=0)
    ranges = maxs - mins
    padding = 0.2

    ax.set_xlim(mins[0] - padding * ranges[0], maxs[0] + padding * ranges[0])
    ax.set_ylim(mins[1] - padding * ranges[1], maxs[1] + padding * ranges[1])
    ax.set_zlim(0, maxs[2] + padding * ranges[2])

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Altitude (m)', fontsize=12)
    ax.set_title('Physics Simulation Comparison\nBlue=Logged | Green=Simulated | Red=Missile',
                 fontsize=14, weight='bold')

    # Ground plane
    xx, yy = np.meshgrid(
        np.linspace(mins[0] - padding * ranges[0], maxs[0] + padding * ranges[0], 10),
        np.linspace(mins[1] - padding * ranges[1], maxs[1] + padding * ranges[1], 10)
    )
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')

    # Plot trajectories
    ax.plot(logged_positions[:, 0], logged_positions[:, 1], logged_positions[:, 2],
            'b-', linewidth=2, alpha=0.7, label='Logged (from env)')

    ax.plot(simulated_positions[:, 0], simulated_positions[:, 1], simulated_positions[:, 2],
            'g-', linewidth=2, alpha=0.7, label='Simulated (with actions)')

    ax.plot(missile_positions[:, 0], missile_positions[:, 1], missile_positions[:, 2],
            'r-', linewidth=2, alpha=0.7, label='Missile')

    # Starting positions
    ax.plot([logged_positions[0, 0]], [logged_positions[0, 1]], [logged_positions[0, 2]],
            'b^', markersize=12, label='Logged start')
    ax.plot([simulated_positions[0, 0]], [simulated_positions[0, 1]], [simulated_positions[0, 2]],
            'g^', markersize=12, label='Sim start')
    ax.plot([missile_positions[0, 0]], [missile_positions[0, 1]], [missile_positions[0, 2]],
            'rx', markersize=12, label='Missile start')

    # Ending positions
    ax.plot([logged_positions[-1, 0]], [logged_positions[-1, 1]], [logged_positions[-1, 2]],
            'bo', markersize=12, label='Logged end')
    ax.plot([simulated_positions[-1, 0]], [simulated_positions[-1, 1]], [simulated_positions[-1, 2]],
            'go', markersize=12, label='Sim end')

    # Calculate divergence
    min_len = min(len(logged_positions), len(simulated_positions))
    divergence = np.linalg.norm(logged_positions[:min_len] - simulated_positions[:min_len], axis=1)
    max_divergence = np.max(divergence)
    avg_divergence = np.mean(divergence)

    # Info text
    info_text = (
        f"Trajectory Divergence:\n"
        f"Max: {max_divergence:.1f}m\n"
        f"Avg: {avg_divergence:.1f}m\n"
        f"\n"
        f"If divergence is HIGH:\n"
        f"→ Actions aren't being applied correctly\n"
        f"If divergence is LOW:\n"
        f"→ Physics is working, policy is bad"
    )

    ax.text2D(0.02, 0.95, info_text, transform=ax.transAxes,
              fontsize=10, verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    ax.legend(loc='upper right', fontsize=9)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize physics simulation vs logged trajectory')
    parser.add_argument('episode', type=str, help='Path to episode JSONL file')

    args = parser.parse_args()

    episode_path = Path(args.episode)
    if not episode_path.exists():
        print(f"Error: Episode file not found: {episode_path}")
        return

    visualize_comparison(episode_path)


if __name__ == '__main__':
    main()