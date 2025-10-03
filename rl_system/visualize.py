"""
Real-time 3D visualization of trained interceptor policy.

Shows live missile interception with:
- 3D trajectories (interceptor and missile)
- Thrust vector visualization
- Radar coverage cones (onboard + ground)
- Sensor fusion status
- Real-time metrics

Usage:
    python visualize.py --model checkpoints/best --episodes 5
    python visualize.py --model checkpoints/model_3000000_steps.zip --speed 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
from pathlib import Path
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment import InterceptEnvironment
from core import quaternion_to_euler, get_forward_vector


class InterceptVisualizer:
    """Real-time 3D visualization of missile interception."""

    def __init__(self, model_path: str, config_path: str = "config.yaml",
                 speed: float = 1.0, save_video: bool = False):
        """
        Initialize visualizer.

        Args:
            model_path: Path to trained model
            config_path: Path to config file
            speed: Playback speed (1.0 = real-time, 0.5 = slow-mo, 2.0 = fast)
            save_video: Whether to save animation to file
        """
        self.speed = speed
        self.save_video = save_video

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load model
        print(f"Loading model from {model_path}...")
        model_path = Path(model_path)

        # Create environment
        env = InterceptEnvironment(self.config)
        env = DummyVecEnv([lambda: env])

        # Load VecNormalize if exists
        vec_normalize_path = model_path.parent / "vec_normalize.pkl"
        self.has_vecnormalize = False
        if vec_normalize_path.exists():
            env = VecNormalize.load(str(vec_normalize_path), env)
            env.training = False
            env.norm_reward = False
            self.has_vecnormalize = True
            print("✓ Loaded VecNormalize statistics")
        else:
            print("⚠️  VecNormalize not found - using observation clipping fallback")

        self.env = env

        # Load model
        if model_path.is_dir():
            model_file = model_path / "best_model.zip"
            if not model_file.exists():
                model_file = model_path / "model.zip"
        else:
            model_file = model_path

        self.model = PPO.load(str(model_file), env=env)
        print(f"✓ Loaded model from {model_file}")

        # Episode data storage
        self.reset_episode_data()

        # Setup figure
        self.setup_figure()

    def reset_episode_data(self):
        """Reset data for new episode."""
        self.interceptor_trajectory = []
        self.missile_trajectory = []
        self.thrust_vectors = []
        self.onboard_detections = []
        self.ground_detections = []
        self.fusion_confidences = []
        self.episode_done = False
        self.step_count = 0

    def setup_figure(self):
        """Setup matplotlib figure with subplots."""
        self.fig = plt.figure(figsize=(16, 9))

        # Main 3D plot
        self.ax_3d = self.fig.add_subplot(2, 3, (1, 4), projection='3d')
        self.ax_3d.set_xlabel('East (m)')
        self.ax_3d.set_ylabel('North (m)')
        self.ax_3d.set_zlabel('Up (m)')
        self.ax_3d.set_title('Missile Interception - Live View', fontsize=14, fontweight='bold')

        # Metrics subplots
        self.ax_distance = self.fig.add_subplot(2, 3, 2)
        self.ax_distance.set_title('Distance to Target')
        self.ax_distance.set_xlabel('Time (s)')
        self.ax_distance.set_ylabel('Distance (m)')
        self.ax_distance.grid(True, alpha=0.3)

        self.ax_radar = self.fig.add_subplot(2, 3, 3)
        self.ax_radar.set_title('Radar & Sensor Status')
        self.ax_radar.set_xlim(0, 1)
        self.ax_radar.set_ylim(0, 4)
        self.ax_radar.axis('off')

        self.ax_fuel = self.fig.add_subplot(2, 3, 5)
        self.ax_fuel.set_title('Fuel & Thrust')
        self.ax_fuel.set_xlabel('Time (s)')
        self.ax_fuel.set_ylabel('Fuel %')
        self.ax_fuel.grid(True, alpha=0.3)

        self.ax_actions = self.fig.add_subplot(2, 3, 6)
        self.ax_actions.set_title('Control Actions')
        self.ax_actions.set_xlabel('Time (s)')
        self.ax_actions.set_ylabel('Action Magnitude')
        self.ax_actions.grid(True, alpha=0.3)

        plt.tight_layout()

    def run_episode(self):
        """Run one episode and collect data."""
        self.reset_episode_data()

        obs = self.env.reset()
        done = False

        # Get initial state
        env_state = self.env.get_attr('interceptor_state')[0]
        missile_state = self.env.get_attr('missile_state')[0]

        while not done and self.step_count < 2000:
            # Apply observation clipping fallback if VecNormalize not available
            if not self.has_vecnormalize:
                obs = np.clip(obs, -1.0, 1.0)

            # Get action from model
            action, _states = self.model.predict(obs, deterministic=True)

            # Store pre-step data
            self.interceptor_trajectory.append(env_state['position'].copy())
            self.missile_trajectory.append(missile_state['position'].copy())

            # Calculate thrust vector (action [0:3] is linear acceleration)
            thrust_direction = action[0][0:3]
            thrust_magnitude = np.linalg.norm(thrust_direction)
            self.thrust_vectors.append({
                'pos': env_state['position'].copy(),
                'dir': thrust_direction.copy(),
                'mag': thrust_magnitude
            })

            # Extract radar data from observation (26D)
            onboard_detected = obs[0][14] > 0.1  # [14] = onboard radar quality
            ground_detected = obs[0][23] > 0.1   # [23] = ground radar quality
            fusion_confidence = obs[0][25]        # [25] = fusion confidence

            self.onboard_detections.append(onboard_detected)
            self.ground_detections.append(ground_detected)
            self.fusion_confidences.append(fusion_confidence)

            # Step environment
            obs, reward, done, info = self.env.step(action)

            # Update states
            env_state = self.env.get_attr('interceptor_state')[0]
            missile_state = self.env.get_attr('missile_state')[0]

            self.step_count += 1

        # Final positions
        self.interceptor_trajectory.append(env_state['position'].copy())
        self.missile_trajectory.append(missile_state['position'].copy())

        # Get final info
        self.final_info = self.env.get_attr('interceptor_state')[0]
        self.final_distance = info[0].get('distance', 0)
        self.intercepted = info[0].get('intercepted', False)

        print(f"\n{'='*60}")
        print(f"Episode Complete:")
        print(f"  Steps: {self.step_count}")
        print(f"  Final Distance: {self.final_distance:.1f}m")
        print(f"  Intercepted: {'✓ YES' if self.intercepted else '✗ NO'}")
        print(f"  Fuel Remaining: {self.final_info['fuel']:.1f}%")
        print(f"{'='*60}\n")

    def animate(self):
        """Create animated visualization."""
        self.run_episode()

        # Convert to numpy arrays
        interceptor_traj = np.array(self.interceptor_trajectory)
        missile_traj = np.array(self.missile_trajectory)

        # Animation state
        self.current_frame = 0
        max_frames = len(interceptor_traj) - 1

        # Initialize plot elements
        self.int_line, = self.ax_3d.plot([], [], [], 'b-', linewidth=2, label='Interceptor', alpha=0.7)
        self.mis_line, = self.ax_3d.plot([], [], [], 'r-', linewidth=2, label='Missile', alpha=0.7)
        self.int_point, = self.ax_3d.plot([], [], [], 'bo', markersize=10, label='Interceptor')
        self.mis_point, = self.ax_3d.plot([], [], [], 'rs', markersize=10, label='Missile')
        self.thrust_arrow = None

        # Ground radar position
        ground_radar_pos = np.array(self.config['environment']['ground_radar']['position'])
        self.ax_3d.plot([ground_radar_pos[0]], [ground_radar_pos[1]], [ground_radar_pos[2]],
                       'g^', markersize=15, label='Ground Radar', zorder=10)

        # Target position
        target_pos = np.array(self.config['environment']['target_position'])
        self.ax_3d.plot([target_pos[0]], [target_pos[1]], [target_pos[2]],
                       'kX', markersize=15, label='Defended Target', zorder=10)

        # Set axis limits
        all_points = np.vstack([interceptor_traj, missile_traj, ground_radar_pos.reshape(1, 3)])
        max_range = np.max(np.abs(all_points)) * 1.2
        self.ax_3d.set_xlim(-max_range, max_range)
        self.ax_3d.set_ylim(-max_range, max_range)
        self.ax_3d.set_zlim(0, max_range * 1.5)

        self.ax_3d.legend(loc='upper right')

        # Distance data for time plot
        self.distance_history = []
        self.time_history = []
        dt = self.config['environment']['dt']

        def update(frame):
            """Update function for animation."""
            if frame >= max_frames:
                return

            self.current_frame = frame
            current_time = frame * dt

            # Update trajectories
            self.int_line.set_data(interceptor_traj[:frame+1, 0], interceptor_traj[:frame+1, 1])
            self.int_line.set_3d_properties(interceptor_traj[:frame+1, 2])

            self.mis_line.set_data(missile_traj[:frame+1, 0], missile_traj[:frame+1, 1])
            self.mis_line.set_3d_properties(missile_traj[:frame+1, 2])

            # Update current positions
            self.int_point.set_data([interceptor_traj[frame, 0]], [interceptor_traj[frame, 1]])
            self.int_point.set_3d_properties([interceptor_traj[frame, 2]])

            self.mis_point.set_data([missile_traj[frame, 0]], [missile_traj[frame, 1]])
            self.mis_point.set_3d_properties([missile_traj[frame, 2]])

            # Update thrust vector
            if frame < len(self.thrust_vectors):
                thrust = self.thrust_vectors[frame]
                if self.thrust_arrow:
                    self.thrust_arrow.remove()

                # Scale thrust vector for visibility
                thrust_scale = 100.0
                thrust_end = thrust['pos'] + thrust['dir'] * thrust_scale

                self.thrust_arrow = self.ax_3d.quiver(
                    thrust['pos'][0], thrust['pos'][1], thrust['pos'][2],
                    thrust['dir'][0] * thrust_scale, thrust['dir'][1] * thrust_scale, thrust['dir'][2] * thrust_scale,
                    color='orange', arrow_length_ratio=0.3, linewidth=2, alpha=0.8
                )

            # Update distance plot
            distance = np.linalg.norm(interceptor_traj[frame] - missile_traj[frame])
            self.distance_history.append(distance)
            self.time_history.append(current_time)

            self.ax_distance.clear()
            self.ax_distance.plot(self.time_history, self.distance_history, 'b-', linewidth=2)
            self.ax_distance.axhline(y=20, color='r', linestyle='--', label='Final Radius (20m)', alpha=0.5)
            self.ax_distance.set_xlabel('Time (s)')
            self.ax_distance.set_ylabel('Distance (m)')
            self.ax_distance.set_title('Distance to Target')
            self.ax_distance.grid(True, alpha=0.3)
            self.ax_distance.legend()

            # Update radar status
            self.ax_radar.clear()
            self.ax_radar.set_xlim(0, 1)
            self.ax_radar.set_ylim(0, 4)
            self.ax_radar.axis('off')

            y_pos = 3.5
            self.ax_radar.text(0.05, y_pos, f"Time: {current_time:.2f}s", fontsize=11, fontweight='bold')
            y_pos -= 0.6
            self.ax_radar.text(0.05, y_pos, f"Distance: {distance:.1f}m", fontsize=10)
            y_pos -= 0.6

            # Radar status
            if frame < len(self.onboard_detections):
                onboard_status = "🟢 LOCK" if self.onboard_detections[frame] else "🔴 NO LOCK"
                ground_status = "🟢 TRACKING" if self.ground_detections[frame] else "🔴 NO TRACK"
                fusion = self.fusion_confidences[frame]

                self.ax_radar.text(0.05, y_pos, f"Onboard Radar: {onboard_status}", fontsize=9)
                y_pos -= 0.5
                self.ax_radar.text(0.05, y_pos, f"Ground Radar: {ground_status}", fontsize=9)
                y_pos -= 0.5
                self.ax_radar.text(0.05, y_pos, f"Fusion: {fusion:.0%}", fontsize=9)

            # Title update
            status = "INTERCEPTED!" if frame == max_frames - 1 and self.intercepted else "TRACKING"
            self.ax_3d.set_title(f'Missile Interception - {status} (t={current_time:.2f}s)',
                                fontsize=14, fontweight='bold')

            return self.int_line, self.mis_line, self.int_point, self.mis_point

        # Create animation
        interval = (dt / self.speed) * 1000  # ms
        anim = FuncAnimation(self.fig, update, frames=max_frames,
                           interval=interval, blit=False, repeat=True)

        if self.save_video:
            print("Saving animation to video...")
            anim.save('interception.mp4', writer='ffmpeg', fps=30, dpi=100)
            print("✓ Saved to interception.mp4")

        plt.show()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Visualize trained missile interception policy')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed (1.0 = real-time)')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes to visualize')
    parser.add_argument('--save', action='store_true', help='Save animation to video file')

    args = parser.parse_args()

    print("=" * 60)
    print("REAL-TIME INTERCEPTION VISUALIZER")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Speed: {args.speed}x")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)
    print()

    for ep in range(args.episodes):
        if args.episodes > 1:
            print(f"\n>>> Episode {ep + 1}/{args.episodes} <<<\n")

        visualizer = InterceptVisualizer(
            args.model,
            args.config,
            speed=args.speed,
            save_video=args.save and ep == 0  # Only save first episode
        )
        visualizer.animate()


if __name__ == '__main__':
    main()
