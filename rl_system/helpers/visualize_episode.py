#!/usr/bin/env python3
"""
3D trajectory visualizer for intercept episodes.
Replay missile and interceptor trajectories from episode JSONL files.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import re


def load_episode(jsonl_path):
    """Load episode data from JSONL file."""
    interceptor_states = []
    missile_states = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)

            if data['type'] == 'state':
                pos = data['state']['position']
                timestamp = data['timestamp']

                if data['entity_id'] == 'interceptor':
                    action = data['state'].get('action', None)
                    interceptor_states.append({
                        'time': timestamp,
                        'pos': np.array(pos),
                        'action': np.array(action) if action else None
                    })
                elif data['entity_id'] == 'missile':
                    missile_states.append({
                        'time': timestamp,
                        'pos': np.array(pos)
                    })

    return interceptor_states, missile_states


def get_episode_number(path):
    """Extract episode number from path."""
    match = re.search(r'ep_(\d+)', str(path))
    if match:
        return int(match.group(1))
    return None


def build_episode_path(base_path, episode_num):
    """Build episode path with new episode number."""
    return Path(str(base_path).replace(re.search(r'ep_\d+', str(base_path)).group(), f'ep_{episode_num:04d}'))


def simulate_interceptor_physics(interceptor_states, dt=0.01):
    """Re-simulate interceptor trajectory with physics and thrust."""
    if not interceptor_states or interceptor_states[0]['action'] is None:
        # No actions, return original positions
        return np.array([s['pos'] for s in interceptor_states])

    # Physics parameters
    mass = 500.0  # kg
    gravity = np.array([0.0, 0.0, -9.81])  # m/s^2
    drag_coef = 0.47
    cross_section = 0.2  # m^2
    air_density = 1.225  # kg/m^3

    # Initial state
    pos = interceptor_states[0]['pos'].copy()

    # Estimate initial velocity
    if len(interceptor_states) > 1:
        vel = (interceptor_states[1]['pos'] - interceptor_states[0]['pos']) / dt
    else:
        vel = np.array([100.0, 100.0, 50.0])

    positions = [pos.copy()]

    # Simulate each step
    for i in range(len(interceptor_states) - 1):
        action = interceptor_states[i]['action']

        # Extract thrust command (normalized -1 to 1)
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

        # Keep above ground
        if pos[2] < 0:
            pos[2] = 0
            vel[2] = max(0, vel[2])

        positions.append(pos.copy())

    return np.array(positions)


def visualize_episode(jsonl_path):
    """Create interactive 3D visualization of episode with time scrubbing."""
    jsonl_path = Path(jsonl_path)
    current_episode = [get_episode_number(jsonl_path)]
    base_path = [jsonl_path]

    print(f"Loading episode: {jsonl_path}")
    interceptor_states, missile_states = load_episode(jsonl_path)

    if not interceptor_states or not missile_states:
        print("Error: No trajectory data found")
        return

    # Simulate interceptor physics with actions
    print("Simulating physics with thrust actions...")
    int_positions = simulate_interceptor_physics(interceptor_states)
    mis_positions = np.array([s['pos'] for s in missile_states])

    print(f"Simulated {len(int_positions)} interceptor positions")

    # Normalize timestamps
    t0 = min(interceptor_states[0]['time'], missile_states[0]['time'])
    int_times = np.array([s['time'] - t0 for s in interceptor_states])
    mis_times = np.array([s['time'] - t0 for s in missile_states])

    print(f"Interceptor: {len(int_positions)} states over {int_times[-1]:.2f}s")
    print(f"Missile: {len(mis_positions)} states over {mis_times[-1]:.2f}s")

    # Calculate final distance
    final_distance = np.linalg.norm(int_positions[-1] - mis_positions[-1])
    intercepted = final_distance < 20.0

    print(f"Final distance: {final_distance:.1f}m")
    print(f"Outcome: {'INTERCEPTED' if intercepted else 'MISSED'}")

    # Setup figure with slider
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.18)

    # Calculate bounds with padding
    all_positions = np.vstack([int_positions, mis_positions])
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

    # Ground plane
    xx, yy = np.meshgrid(
        np.linspace(mins[0] - padding * ranges[0], maxs[0] + padding * ranges[0], 10),
        np.linspace(mins[1] - padding * ranges[1], maxs[1] + padding * ranges[1], 10)
    )
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')

    # Plot full trajectories (faded)
    ax.plot(int_positions[:, 0], int_positions[:, 1], int_positions[:, 2],
            'b-', alpha=0.3, linewidth=1, label='Interceptor path')
    ax.plot(mis_positions[:, 0], mis_positions[:, 1], mis_positions[:, 2],
            'r-', alpha=0.3, linewidth=1, label='Missile path')

    # Current positions (will be updated)
    int_current = ax.plot([], [], [], 'bo', markersize=12, label='Interceptor')[0]
    mis_current = ax.plot([], [], [], 'rs', markersize=12, label='Missile')[0]

    # Trailing path (last 50 steps)
    int_trail = ax.plot([], [], [], 'b-', linewidth=3, alpha=0.8)[0]
    mis_trail = ax.plot([], [], [], 'r-', linewidth=3, alpha=0.8)[0]

    # Starting positions
    ax.plot([int_positions[0, 0]], [int_positions[0, 1]], [int_positions[0, 2]],
            'g^', markersize=10, label='Interceptor start')
    ax.plot([mis_positions[0, 0]], [mis_positions[0, 1]], [mis_positions[0, 2]],
            'mx', markersize=10, label='Missile start')

    # Distance line
    dist_line = ax.plot([], [], [], 'k--', linewidth=1, alpha=0.5)[0]

    # Thrust vector arrow
    thrust_arrow = None

    # Title
    outcome_text = 'INTERCEPTED ✓' if intercepted else 'MISSED ✗'
    outcome_color = 'green' if intercepted else 'red'
    title = ax.text2D(0.5, 0.95, '', transform=ax.transAxes,
                      ha='center', fontsize=14, weight='bold')

    # Action display
    action_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes,
                           ha='left', fontsize=10, family='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.legend(loc='upper left', fontsize=10)

    # Create time slider
    num_frames = max(len(int_positions), len(mis_positions))
    trail_length = 50

    ax_slider = plt.axes([0.15, 0.08, 0.6, 0.03])
    time_slider = Slider(
        ax_slider, 'Time',
        0, num_frames - 1,
        valinit=0,
        valstep=1
    )

    # Create play/pause button
    ax_button = plt.axes([0.77, 0.075, 0.08, 0.04])
    play_button = Button(ax_button, 'Play')

    # Episode navigation
    ax_prev = plt.axes([0.15, 0.02, 0.08, 0.04])
    prev_button = Button(ax_prev, '◀ Prev')

    ax_next = plt.axes([0.25, 0.02, 0.08, 0.04])
    next_button = Button(ax_next, 'Next ▶')

    ax_textbox = plt.axes([0.35, 0.02, 0.15, 0.04])
    episode_textbox = TextBox(ax_textbox, 'Episode:', initial=str(current_episode[0]) if current_episode[0] is not None else '0')

    # Animation state
    is_playing = [False]
    timer = [None]

    def update(frame):
        nonlocal thrust_arrow
        frame = int(frame)

        # Map frame to actual indices
        int_idx = min(frame, len(int_positions) - 1)
        mis_idx = min(frame, len(mis_positions) - 1)

        # Update current positions
        int_pos = int_positions[int_idx]
        mis_pos = mis_positions[mis_idx]

        int_current.set_data([int_pos[0]], [int_pos[1]])
        int_current.set_3d_properties([int_pos[2]])

        mis_current.set_data([mis_pos[0]], [mis_pos[1]])
        mis_current.set_3d_properties([mis_pos[2]])

        # Update trails
        trail_start_int = max(0, int_idx - trail_length)
        trail_start_mis = max(0, mis_idx - trail_length)

        int_trail_data = int_positions[trail_start_int:int_idx+1]
        mis_trail_data = mis_positions[trail_start_mis:mis_idx+1]

        int_trail.set_data(int_trail_data[:, 0], int_trail_data[:, 1])
        int_trail.set_3d_properties(int_trail_data[:, 2])

        mis_trail.set_data(mis_trail_data[:, 0], mis_trail_data[:, 1])
        mis_trail.set_3d_properties(mis_trail_data[:, 2])

        # Update distance line
        dist_line.set_data([int_pos[0], mis_pos[0]], [int_pos[1], mis_pos[1]])
        dist_line.set_3d_properties([int_pos[2], mis_pos[2]])

        # Update thrust vector arrow
        if thrust_arrow is not None:
            thrust_arrow.remove()
            thrust_arrow = None

        if int_idx < len(interceptor_states) and interceptor_states[int_idx]['action'] is not None:
            action = interceptor_states[int_idx]['action']
            thrust = action[0:3] * 100.0  # Scale for visibility (action is -1 to 1, scale to ~100m vectors)

            # Draw thrust vector as arrow from interceptor position
            thrust_arrow = ax.quiver(
                int_pos[0], int_pos[1], int_pos[2],
                thrust[0], thrust[1], thrust[2],
                color='orange', arrow_length_ratio=0.3, linewidth=3, alpha=0.9,
                label='Thrust Vector'
            )

        # Update title
        current_time = int_times[int_idx]
        distance = np.linalg.norm(int_pos - mis_pos)
        title.set_text(f'Time: {current_time:.2f}s | Distance: {distance:.1f}m | {outcome_text}')
        title.set_color(outcome_color if frame >= num_frames - 1 else 'black')

        # Update action display
        if int_idx < len(interceptor_states) and interceptor_states[int_idx]['action'] is not None:
            action = interceptor_states[int_idx]['action']
            thrust = action[0:3]
            angular = action[3:6]
            action_str = (
                f"Policy Action:\n"
                f"Thrust:  [{thrust[0]:+.2f}, {thrust[1]:+.2f}, {thrust[2]:+.2f}]\n"
                f"Angular: [{angular[0]:+.2f}, {angular[1]:+.2f}, {angular[2]:+.2f}]"
            )
            action_text.set_text(action_str)
        else:
            action_text.set_text("No action data")

        fig.canvas.draw_idle()

    def animate():
        """Advance to next frame during playback."""
        if is_playing[0]:
            current_frame = time_slider.val
            if current_frame < num_frames - 1:
                time_slider.set_val(current_frame + 1)
            else:
                # Loop back to start
                time_slider.set_val(0)

    def toggle_play(event):
        """Toggle play/pause."""
        is_playing[0] = not is_playing[0]

        if is_playing[0]:
            play_button.label.set_text('Pause')
            # Start timer
            timer[0] = fig.canvas.new_timer(interval=50)
            timer[0].add_callback(animate)
            timer[0].start()
        else:
            play_button.label.set_text('Play')
            # Stop timer
            if timer[0]:
                timer[0].stop()

    def load_new_episode(episode_num):
        """Load a new episode and refresh the visualization."""
        nonlocal interceptor_states, missile_states, int_positions, mis_positions
        nonlocal int_times, mis_times, final_distance, intercepted, outcome_text, outcome_color
        nonlocal num_frames, int_current, mis_current, int_trail, mis_trail, dist_line

        try:
            new_path = build_episode_path(base_path[0], episode_num)

            if not new_path.exists():
                print(f"Episode {episode_num} not found at {new_path}")
                return

            print(f"Loading episode {episode_num}: {new_path}")

            # Stop playback
            if is_playing[0]:
                toggle_play(None)

            # Load new data
            interceptor_states, missile_states = load_episode(new_path)

            if not interceptor_states or not missile_states:
                print("Error: No trajectory data found")
                return

            # Update paths
            base_path[0] = new_path
            current_episode[0] = episode_num

            # Recalculate everything with physics simulation
            int_positions = simulate_interceptor_physics(interceptor_states)
            mis_positions = np.array([s['pos'] for s in missile_states])

            t0 = min(interceptor_states[0]['time'], missile_states[0]['time'])
            int_times = np.array([s['time'] - t0 for s in interceptor_states])
            mis_times = np.array([s['time'] - t0 for s in missile_states])

            final_distance = np.linalg.norm(int_positions[-1] - mis_positions[-1])
            intercepted = final_distance < 20.0
            outcome_text = 'INTERCEPTED ✓' if intercepted else 'MISSED ✗'
            outcome_color = 'green' if intercepted else 'red'

            print(f"Final distance: {final_distance:.1f}m")
            print(f"Outcome: {outcome_text}")

            # Clear and recreate the plot
            ax.clear()

            # Recalculate bounds
            all_positions = np.vstack([int_positions, mis_positions])
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

            # Redraw ground plane
            xx, yy = np.meshgrid(
                np.linspace(mins[0] - padding * ranges[0], maxs[0] + padding * ranges[0], 10),
                np.linspace(mins[1] - padding * ranges[1], maxs[1] + padding * ranges[1], 10)
            )
            zz = np.zeros_like(xx)
            ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')

            # Redraw trajectories
            ax.plot(int_positions[:, 0], int_positions[:, 1], int_positions[:, 2],
                    'b-', alpha=0.3, linewidth=1, label='Interceptor path')
            ax.plot(mis_positions[:, 0], mis_positions[:, 1], mis_positions[:, 2],
                    'r-', alpha=0.3, linewidth=1, label='Missile path')

            # Recreate line objects
            int_current = ax.plot([], [], [], 'bo', markersize=12, label='Interceptor')[0]
            mis_current = ax.plot([], [], [], 'rs', markersize=12, label='Missile')[0]
            int_trail = ax.plot([], [], [], 'b-', linewidth=3, alpha=0.8)[0]
            mis_trail = ax.plot([], [], [], 'r-', linewidth=3, alpha=0.8)[0]
            dist_line = ax.plot([], [], [], 'k--', linewidth=1, alpha=0.5)[0]

            # Starting positions
            ax.plot([int_positions[0, 0]], [int_positions[0, 1]], [int_positions[0, 2]],
                    'g^', markersize=10, label='Interceptor start')
            ax.plot([mis_positions[0, 0]], [mis_positions[0, 1]], [mis_positions[0, 2]],
                    'mx', markersize=10, label='Missile start')

            ax.legend(loc='upper left', fontsize=10)

            # Update slider range
            num_frames = max(len(int_positions), len(mis_positions))
            time_slider.valmax = num_frames - 1
            time_slider.ax.set_xlim(0, num_frames - 1)
            time_slider.set_val(0)

            # Update textbox
            episode_textbox.set_val(str(episode_num))

            update(0)

        except Exception as e:
            print(f"Error loading episode {episode_num}: {e}")

    def on_prev(event):
        """Load previous episode."""
        if current_episode[0] is not None and current_episode[0] > 0:
            load_new_episode(current_episode[0] - 1)

    def on_next(event):
        """Load next episode."""
        if current_episode[0] is not None:
            load_new_episode(current_episode[0] + 1)

    def on_episode_submit(text):
        """Load episode from text input."""
        try:
            episode_num = int(text)
            if episode_num >= 0:
                load_new_episode(episode_num)
        except ValueError:
            print(f"Invalid episode number: {text}")

    time_slider.on_changed(update)
    play_button.on_clicked(toggle_play)
    prev_button.on_clicked(on_prev)
    next_button.on_clicked(on_next)
    episode_textbox.on_submit(on_episode_submit)

    # Initialize at frame 0
    update(0)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize missile intercept episode')
    parser.add_argument('episode', type=str, help='Path to episode JSONL file')

    args = parser.parse_args()

    episode_path = Path(args.episode)
    if not episode_path.exists():
        print(f"Error: Episode file not found: {episode_path}")
        return

    visualize_episode(episode_path)


if __name__ == '__main__':
    main()