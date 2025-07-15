#!/usr/bin/env python3
"""
Real-Time 3D Missile Animation System for AegisIntercept Phase 3.

This system provides live animation of missile intercept scenarios with missiles
moving through 3D space in real-time, showing the actual intercept attempt as it unfolds.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider, CheckButtons
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys
from collections import deque

# Add project root to path
sys.path.append('.')

from aegis_intercept.envs.aegis_6dof_env import Aegis6DOFEnv


class RealTime3DVisualizer:
    """
    Real-time 3D missile animation system.
    
    Shows missiles moving through 3D space in real-time with smooth animation,
    interactive controls, and professional visualization.
    """
    
    def __init__(self, 
                 world_scale: float = 3000.0,
                 update_rate: int = 50,  # milliseconds between updates
                 trail_length: int = 100,
                 grid_spacing: float = 500.0):
        """
        Initialize real-time 3D visualizer.
        
        Args:
            world_scale: Scale of the visualization world (meters)
            update_rate: Update rate in milliseconds
            trail_length: Number of trail points to keep
            grid_spacing: Spacing between grid lines
        """
        self.world_scale = world_scale
        self.update_rate = update_rate
        self.trail_length = trail_length
        self.grid_spacing = grid_spacing
        
        # Animation state
        self.is_running = False
        self.is_paused = False
        self.speed_multiplier = 1.0
        self.follow_interceptor = False
        
        # Data storage
        self.interceptor_trail = deque(maxlen=trail_length)
        self.adversary_trail = deque(maxlen=trail_length)
        self.performance_data = {
            'times': deque(maxlen=200),
            'distances': deque(maxlen=200),
            'rewards': deque(maxlen=200),
            'fuel_levels': deque(maxlen=200)
        }
        
        # Episode state
        self.current_episode = 0
        self.episode_running = False
        self.episode_start_time = 0
        self.simulation_time = 0
        
        # Environment and model
        self.env = None
        self.model = None
        self.current_obs = None
        self.current_info = None
        
        # Visual elements
        self.fig = None
        self.ax = None
        self.interceptor_marker = None
        self.adversary_marker = None
        self.target_marker = None
        self.interceptor_trail_line = None
        self.adversary_trail_line = None
        self.velocity_arrows = []
        self.grid_lines = []
        
        # UI elements
        self.btn_pause = None
        self.btn_reset = None
        self.btn_new_episode = None
        self.speed_slider = None
        self.checkboxes = None
        
        # Performance plot elements
        self.ax_distance = None
        self.ax_metrics = None
        self.distance_line = None
        self.reward_line = None
        self.fuel_line = None
        
        # Animation object
        self.animation = None
        
        # Initialize visualization
        self._setup_visualization()
    
    def _setup_visualization(self):
        """Setup the matplotlib visualization."""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('AegisIntercept Phase 3 - Real-Time 3D Animation', fontsize=16)
        
        # Main 3D plot (left side)
        self.ax = self.fig.add_subplot(121, projection='3d')
        
        # Performance plots (right side)
        self.ax_distance = self.fig.add_subplot(222)
        self.ax_metrics = self.fig.add_subplot(224)
        
        # Setup 3D plot
        self._setup_3d_plot()
        
        # Setup performance plots
        self._setup_performance_plots()
        
        # Setup UI controls
        self._setup_ui_controls()
        
        # Setup grid
        self._create_3d_grid()
        
        plt.tight_layout()
    
    def _setup_3d_plot(self):
        """Setup the main 3D plot."""
        # Set plot limits
        limit = self.world_scale / 2
        self.ax.set_xlim([-limit, limit])
        self.ax.set_ylim([-limit, limit])
        self.ax.set_zlim([0, limit])
        
        # Labels and styling
        self.ax.set_xlabel('X (meters)', fontsize=12)
        self.ax.set_ylabel('Y (meters)', fontsize=12)
        self.ax.set_zlabel('Z (meters)', fontsize=12)
        self.ax.set_title('Real-Time Missile Animation', fontsize=14)
        
        # Initialize plot elements
        self.interceptor_marker = self.ax.scatter([], [], [], s=150, c='blue', 
                                                marker='^', edgecolors='black', 
                                                linewidths=2, label='Interceptor')
        
        self.adversary_marker = self.ax.scatter([], [], [], s=120, c='red', 
                                              marker='o', edgecolors='black', 
                                              linewidths=2, label='Adversary')
        
        self.target_marker = self.ax.scatter([0], [0], [0], s=200, c='green', 
                                           marker='*', edgecolors='black', 
                                           linewidths=2, label='Target')
        
        # Trail lines
        self.interceptor_trail_line, = self.ax.plot([], [], [], 'b-', alpha=0.7, 
                                                  linewidth=2.5, label='Interceptor Trail')
        
        self.adversary_trail_line, = self.ax.plot([], [], [], 'r-', alpha=0.7, 
                                                linewidth=2.0, label='Adversary Trail')
        
        # Legend
        self.ax.legend(loc='upper right', fontsize=10)
        
        # Set viewing angle
        self.ax.view_init(elev=25, azim=45)
        
        # Enable grid
        self.ax.grid(True, alpha=0.3)
    
    def _setup_performance_plots(self):
        """Setup performance monitoring plots."""
        # Distance plot
        self.ax_distance.set_title('Intercept Distance')
        self.ax_distance.set_xlabel('Time (s)')
        self.ax_distance.set_ylabel('Distance (m)')
        self.ax_distance.grid(True, alpha=0.3)
        self.distance_line, = self.ax_distance.plot([], [], 'b-', linewidth=2)
        
        # Metrics plot
        self.ax_metrics.set_title('Performance Metrics')
        self.ax_metrics.set_xlabel('Time (s)')
        self.ax_metrics.set_ylabel('Value')
        self.ax_metrics.grid(True, alpha=0.3)
        self.reward_line, = self.ax_metrics.plot([], [], 'g-', linewidth=2, label='Reward')
        self.fuel_line, = self.ax_metrics.plot([], [], 'orange', linewidth=2, label='Fuel')
        self.ax_metrics.legend()
    
    def _setup_ui_controls(self):
        """Setup UI controls."""
        # Control buttons
        ax_pause = plt.axes([0.02, 0.95, 0.08, 0.04])
        self.btn_pause = Button(ax_pause, 'Pause')
        self.btn_pause.on_clicked(self._toggle_pause)
        
        ax_reset = plt.axes([0.12, 0.95, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._reset_view)
        
        ax_episode = plt.axes([0.22, 0.95, 0.12, 0.04])
        self.btn_new_episode = Button(ax_episode, 'New Episode')
        self.btn_new_episode.on_clicked(self._start_new_episode)
        
        # Speed control
        ax_speed = plt.axes([0.02, 0.90, 0.25, 0.03])
        self.speed_slider = Slider(ax_speed, 'Speed', 0.1, 5.0, valinit=1.0)
        self.speed_slider.on_changed(self._update_speed)
        
        # Display options
        ax_options = plt.axes([0.02, 0.70, 0.12, 0.15])
        options_labels = ['Follow Interceptor', 'Show Trails', 'Show Velocity', 'Show Grid']
        options_states = [False, True, True, True]
        self.checkboxes = CheckButtons(ax_options, options_labels, options_states)
        self.checkboxes.on_clicked(self._toggle_option)
    
    def _create_3d_grid(self):
        """Create 3D grid lines."""
        limit = self.world_scale / 2
        grid_range = np.arange(-limit, limit + self.grid_spacing, self.grid_spacing)
        
        # Clear existing grid lines
        for line in self.grid_lines:
            line.remove()
        self.grid_lines.clear()
        
        # Create grid lines
        for x in grid_range[::2]:  # Every other line to avoid clutter
            line, = self.ax.plot([x, x], [-limit, limit], [0, 0], 'k-', alpha=0.2, linewidth=0.5)
            self.grid_lines.append(line)
            
        for y in grid_range[::2]:
            line, = self.ax.plot([-limit, limit], [y, y], [0, 0], 'k-', alpha=0.2, linewidth=0.5)
            self.grid_lines.append(line)
            
        # Vertical lines (sparser)
        for z in np.arange(0, limit, self.grid_spacing * 2):
            line, = self.ax.plot([-limit, limit], [-limit, -limit], [z, z], 'k-', alpha=0.1, linewidth=0.5)
            self.grid_lines.append(line)
    
    def setup_environment(self, checkpoint_dir: str = None):
        """Setup environment and optionally load trained model."""
        print("Setting up environment...")
        
        # Create environment
        self.env = Aegis6DOFEnv(curriculum_level="easy")
        
        # Try to load model if checkpoint provided
        if checkpoint_dir:
            try:
                self._load_checkpoint(checkpoint_dir)
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Using random policy instead.")
                self.model = None
        
        print("Environment setup complete")
    
    def _load_checkpoint(self, checkpoint_dir: str):
        """Load trained model from checkpoint."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        
        checkpoint_path = Path(checkpoint_dir)
        model_path = checkpoint_path / "model.zip"
        vec_normalize_path = checkpoint_path / "vec_normalize.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        self.model = PPO.load(model_path)
        
        # Load VecNormalize if available
        if vec_normalize_path.exists():
            self.env = DummyVecEnv([lambda: self.env])
            self.env = VecNormalize.load(vec_normalize_path, self.env)
            self.env.training = False
            self.env.norm_reward = False
        
        print(f"Loaded model from: {checkpoint_path.name}")
    
    def _get_action(self, obs):
        """Get action from model or random policy."""
        if self.model:
            try:
                action, _ = self.model.predict(obs, deterministic=True)
                return action
            except Exception as e:
                print(f"Model prediction error: {e}")
        
        # Random policy fallback
        return self.env.action_space.sample()
    
    def _get_positions(self):
        """Get current positions from environment."""
        try:
            # Handle different environment wrapper types
            env = self.env
            while hasattr(env, 'env') and not hasattr(env, 'interceptor'):
                env = env.env
            
            if hasattr(env, 'envs'):
                env = env.envs[0]
                while hasattr(env, 'env') and not hasattr(env, 'interceptor'):
                    env = env.env
            
            if hasattr(env, 'venv'):
                env = env.venv.envs[0]
                while hasattr(env, 'env') and not hasattr(env, 'interceptor'):
                    env = env.env
            
            interceptor_pos = env.interceptor.get_position()
            adversary_pos = env.adversary.get_position()
            target_pos = env.target_position
            
            return interceptor_pos, adversary_pos, target_pos
            
        except Exception as e:
            print(f"Error getting positions: {e}")
            return np.array([0, 0, 1000]), np.array([1000, 1000, 1000]), np.array([0, 0, 0])
    
    def start_episode(self):
        """Start a new episode."""
        if not self.env:
            print("Environment not setup. Call setup_environment() first.")
            return
        
        print(f"Starting episode {self.current_episode + 1}")
        
        # Reset environment
        self.current_obs, self.current_info = self._dynamic_reset()
        
        # Reset episode state
        self.episode_running = True
        self.episode_start_time = time.time()
        self.simulation_time = 0
        
        # Clear trails and data
        self.interceptor_trail.clear()
        self.adversary_trail.clear()
        for data_list in self.performance_data.values():
            data_list.clear()
        
        # Clear velocity arrows
        for arrow in self.velocity_arrows:
            arrow.remove()
        self.velocity_arrows.clear()
        
        # Update display
        self._update_display()
        
        # Start animation if not running
        if not self.is_running:
            self.is_running = True
            self.animation = animation.FuncAnimation(
                self.fig, self._animate_frame, interval=self.update_rate, 
                blit=False, repeat=True
            )
        
        self.current_episode += 1
    
    def _dynamic_reset(self):
        """Handle different gym API versions for reset."""
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            return reset_result
        else:
            return reset_result, {}
    
    def _dynamic_step(self, action):
        """Handle different gym API versions for step."""
        step_result = self.env.step(action)
        
        def process_info(info):
            if isinstance(info, list):
                return info[0] if len(info) > 0 else {}
            return info
        
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            info = process_info(info)
            return obs, reward, done, done, info
        elif len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            info = process_info(info)
            return obs, reward, terminated, truncated, info
        else:
            raise ValueError(f"Unexpected step return format: {len(step_result)} values")
    
    def _animate_frame(self, frame):
        """Animation frame update function."""
        if not self.episode_running or self.is_paused:
            return
        
        # Get action and step environment
        action = self._get_action(self.current_obs)
        
        # Step with speed multiplier
        for _ in range(int(self.speed_multiplier)):
            if not self.episode_running:
                break
                
            self.current_obs, reward, terminated, truncated, self.current_info = self._dynamic_step(action)
            self.simulation_time += 0.01  # Assuming 0.01s timestep
            
            # Check for episode end
            if terminated or truncated:
                self.episode_running = False
                print(f"Episode ended: {self.current_info.get('termination_reason', 'unknown')}")
                break
        
        # Update visualization
        self._update_display()
        
        return self._get_artists()
    
    def _update_display(self):
        """Update all visual elements."""
        if not self.episode_running:
            return
        
        # Get current positions
        interceptor_pos, adversary_pos, target_pos = self._get_positions()
        
        # Update marker positions
        self.interceptor_marker._offsets3d = ([interceptor_pos[0]], [interceptor_pos[1]], [interceptor_pos[2]])
        self.adversary_marker._offsets3d = ([adversary_pos[0]], [adversary_pos[1]], [adversary_pos[2]])
        
        # Update trails
        self.interceptor_trail.append(interceptor_pos.copy())
        self.adversary_trail.append(adversary_pos.copy())
        
        if len(self.interceptor_trail) > 1:
            trail_array = np.array(self.interceptor_trail)
            self.interceptor_trail_line.set_data_3d(
                trail_array[:, 0], trail_array[:, 1], trail_array[:, 2]
            )
        
        if len(self.adversary_trail) > 1:
            trail_array = np.array(self.adversary_trail)
            self.adversary_trail_line.set_data_3d(
                trail_array[:, 0], trail_array[:, 1], trail_array[:, 2]
            )
        
        # Update performance data
        distance = self.current_info.get('intercept_distance', 0)
        reward = self.current_info.get('step_reward', 0)
        fuel = self.current_info.get('fuel_remaining', 1.0)
        
        self.performance_data['times'].append(self.simulation_time)
        self.performance_data['distances'].append(distance)
        self.performance_data['rewards'].append(reward)
        self.performance_data['fuel_levels'].append(fuel)
        
        # Update performance plots
        self._update_performance_plots()
        
        # Update camera if following interceptor
        if self.follow_interceptor:
            self._update_camera_follow(interceptor_pos)
    
    def _update_performance_plots(self):
        """Update performance plots."""
        times = list(self.performance_data['times'])
        distances = list(self.performance_data['distances'])
        rewards = list(self.performance_data['rewards'])
        fuel_levels = list(self.performance_data['fuel_levels'])
        
        if len(times) > 1:
            # Distance plot
            self.distance_line.set_data(times, distances)
            self.ax_distance.set_xlim([min(times), max(times)])
            self.ax_distance.set_ylim([0, max(distances) * 1.1 if distances else 100])
            
            # Metrics plot
            self.reward_line.set_data(times, rewards)
            self.fuel_line.set_data(times, fuel_levels)
            self.ax_metrics.set_xlim([min(times), max(times)])
            
            if rewards and fuel_levels:
                y_min = min(min(rewards), 0)
                y_max = max(max(rewards), 1)
                self.ax_metrics.set_ylim([y_min, y_max])
    
    def _update_camera_follow(self, interceptor_pos):
        """Update camera to follow interceptor."""
        offset = 500.0
        self.ax.set_xlim([interceptor_pos[0] - offset, interceptor_pos[0] + offset])
        self.ax.set_ylim([interceptor_pos[1] - offset, interceptor_pos[1] + offset])
        self.ax.set_zlim([max(0, interceptor_pos[2] - offset), interceptor_pos[2] + offset])
    
    def _get_artists(self):
        """Get all artists for blitting."""
        return [
            self.interceptor_marker, self.adversary_marker,
            self.interceptor_trail_line, self.adversary_trail_line,
            self.distance_line, self.reward_line, self.fuel_line
        ]
    
    def _toggle_pause(self, event):
        """Toggle pause state."""
        self.is_paused = not self.is_paused
        self.btn_pause.label.set_text('Resume' if self.is_paused else 'Pause')
    
    def _reset_view(self, event):
        """Reset view to default."""
        limit = self.world_scale / 2
        self.ax.set_xlim([-limit, limit])
        self.ax.set_ylim([-limit, limit])
        self.ax.set_zlim([0, limit])
        self.ax.view_init(elev=25, azim=45)
    
    def _start_new_episode(self, event):
        """Start a new episode."""
        self.episode_running = False
        time.sleep(0.1)  # Brief pause
        self.start_episode()
    
    def _update_speed(self, val):
        """Update animation speed."""
        self.speed_multiplier = val
    
    def _toggle_option(self, label):
        """Toggle display options."""
        if label == 'Follow Interceptor':
            self.follow_interceptor = not self.follow_interceptor
        elif label == 'Show Trails':
            alpha = 0.7 if self.interceptor_trail_line.get_alpha() == 0 else 0
            self.interceptor_trail_line.set_alpha(alpha)
            self.adversary_trail_line.set_alpha(alpha)
        elif label == 'Show Grid':
            alpha = 0.2 if self.grid_lines[0].get_alpha() == 0 else 0
            for line in self.grid_lines:
                line.set_alpha(alpha)
    
    def show(self):
        """Show the visualization."""
        plt.show()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Real-Time 3D Missile Animation')
    parser.add_argument('--checkpoint-dir', type=str, 
                       help='Path to trained model checkpoint directory')
    parser.add_argument('--world-scale', type=float, default=3000.0,
                       help='World scale in meters (default: 3000.0)')
    parser.add_argument('--update-rate', type=int, default=50,
                       help='Update rate in milliseconds (default: 50)')
    parser.add_argument('--trail-length', type=int, default=100,
                       help='Trail length in points (default: 100)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("AegisIntercept Phase 3 - Real-Time 3D Missile Animation")
    print("="*70)
    print(f"World Scale: {args.world_scale}m")
    print(f"Update Rate: {args.update_rate}ms")
    print(f"Trail Length: {args.trail_length} points")
    if args.checkpoint_dir:
        print(f"Checkpoint: {args.checkpoint_dir}")
    else:
        print("Model: Random Policy")
    print("="*70)
    
    # Create visualizer
    visualizer = RealTime3DVisualizer(
        world_scale=args.world_scale,
        update_rate=args.update_rate,
        trail_length=args.trail_length
    )
    
    # Setup environment
    visualizer.setup_environment(args.checkpoint_dir)
    
    # Start first episode
    visualizer.start_episode()
    
    print("\\nControls:")
    print("  - Pause/Resume: Click 'Pause' button")
    print("  - New Episode: Click 'New Episode' button")
    print("  - Speed Control: Use speed slider")
    print("  - Follow Interceptor: Check 'Follow Interceptor' option")
    print("  - Reset View: Click 'Reset' button")
    print("\\nStarting real-time animation...")
    
    # Show visualization
    visualizer.show()
    
    return 0


if __name__ == "__main__":
    exit(main())