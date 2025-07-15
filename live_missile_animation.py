#!/usr/bin/env python3
"""
Live Missile Animation Demo for AegisIntercept Phase 3.

This script provides a simplified real-time animation system that works
without trained models, showing missiles moving through 3D space.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import argparse
import time
from typing import Dict, List, Any, Optional, Tuple
import sys
from collections import deque

# Add project root to path
sys.path.append('.')

from aegis_intercept.envs.aegis_6dof_env import Aegis6DOFEnv


class LiveMissileAnimation:
    """
    Live missile animation system showing missiles moving in real-time.
    
    This system runs episodes and shows the missiles moving through 3D space
    with smooth animation and real-time updates.
    """
    
    def __init__(self, world_scale: float = 3000.0, fps: int = 20):
        """
        Initialize live missile animation.
        
        Args:
            world_scale: Scale of the world in meters
            fps: Frames per second for animation
        """
        self.world_scale = world_scale
        self.fps = fps
        self.update_interval = 1000 // fps  # milliseconds
        
        # Animation state
        self.is_paused = False
        self.speed_multiplier = 1.0
        self.follow_interceptor = False
        
        # Episode state
        self.env = None
        self.current_obs = None
        self.current_info = None
        self.episode_running = False
        self.episode_count = 0
        
        # Data storage
        self.interceptor_positions = deque(maxlen=200)
        self.adversary_positions = deque(maxlen=200)
        self.times = deque(maxlen=200)
        self.distances = deque(maxlen=200)
        self.rewards = deque(maxlen=200)
        
        # Visualization elements
        self.fig = None
        self.ax = None
        self.ax_dist = None
        self.interceptor_point = None
        self.adversary_point = None
        self.target_point = None
        self.interceptor_trail = None
        self.adversary_trail = None
        self.distance_line = None
        
        # Controls
        self.btn_pause = None
        self.btn_new_episode = None
        self.speed_slider = None
        
        # Animation
        self.animation = None
        self.simulation_time = 0
        
        # Setup visualization
        self._setup_visualization()
    
    def _setup_visualization(self):
        """Setup the matplotlib visualization."""
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.suptitle('AegisIntercept - Live Missile Animation', fontsize=16)
        
        # 3D plot
        self.ax = self.fig.add_subplot(121, projection='3d')
        self._setup_3d_plot()
        
        # Distance plot
        self.ax_dist = self.fig.add_subplot(122)
        self._setup_distance_plot()
        
        # Controls
        self._setup_controls()
        
        plt.tight_layout()
    
    def _setup_3d_plot(self):
        """Setup 3D plot."""
        # Set limits
        limit = self.world_scale / 2
        self.ax.set_xlim([-limit, limit])
        self.ax.set_ylim([-limit, limit])
        self.ax.set_zlim([0, limit])
        
        # Labels
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_zlabel('Z (meters)')
        self.ax.set_title('3D Missile Animation')
        
        # Grid
        self._create_grid()
        
        # Initialize markers
        self.interceptor_point = self.ax.scatter([], [], [], s=150, c='blue', 
                                               marker='^', edgecolors='black', 
                                               linewidths=2, label='Interceptor')
        
        self.adversary_point = self.ax.scatter([], [], [], s=120, c='red', 
                                             marker='o', edgecolors='black', 
                                             linewidths=2, label='Adversary')
        
        self.target_point = self.ax.scatter([0], [0], [0], s=200, c='green', 
                                          marker='*', edgecolors='black', 
                                          linewidths=2, label='Target')
        
        # Trail lines
        self.interceptor_trail, = self.ax.plot([], [], [], 'b-', alpha=0.7, 
                                             linewidth=2.5, label='Interceptor Path')
        
        self.adversary_trail, = self.ax.plot([], [], [], 'r-', alpha=0.6, 
                                           linewidth=2.0, label='Adversary Path')
        
        # Legend
        self.ax.legend(loc='upper right')
        
        # View angle
        self.ax.view_init(elev=25, azim=45)
    
    def _create_grid(self):
        """Create 3D reference grid."""
        limit = self.world_scale / 2
        spacing = 500.0
        
        # Grid lines
        for x in np.arange(-limit, limit + spacing, spacing):
            self.ax.plot([x, x], [-limit, limit], [0, 0], 'k-', alpha=0.1, linewidth=0.5)
            
        for y in np.arange(-limit, limit + spacing, spacing):
            self.ax.plot([-limit, limit], [y, y], [0, 0], 'k-', alpha=0.1, linewidth=0.5)
            
        for z in np.arange(0, limit, spacing * 2):
            self.ax.plot([-limit, limit], [-limit, -limit], [z, z], 'k-', alpha=0.05, linewidth=0.5)
    
    def _setup_distance_plot(self):
        """Setup distance monitoring plot."""
        self.ax_dist.set_title('Intercept Distance')
        self.ax_dist.set_xlabel('Time (s)')
        self.ax_dist.set_ylabel('Distance (m)')
        self.ax_dist.grid(True, alpha=0.3)
        
        self.distance_line, = self.ax_dist.plot([], [], 'b-', linewidth=2)
    
    def _setup_controls(self):
        """Setup control buttons."""
        # Pause button
        ax_pause = plt.axes([0.02, 0.02, 0.08, 0.04])
        self.btn_pause = Button(ax_pause, 'Pause')
        self.btn_pause.on_clicked(self._toggle_pause)
        
        # New episode button
        ax_new = plt.axes([0.12, 0.02, 0.12, 0.04])
        self.btn_new_episode = Button(ax_new, 'New Episode')
        self.btn_new_episode.on_clicked(self._start_new_episode)
        
        # Speed slider
        ax_speed = plt.axes([0.3, 0.02, 0.2, 0.04])
        self.speed_slider = Slider(ax_speed, 'Speed', 0.1, 3.0, valinit=1.0)
        self.speed_slider.on_changed(self._update_speed)
    
    def setup_environment(self):
        """Setup the environment."""
        print("Setting up environment...")
        self.env = Aegis6DOFEnv(curriculum_level="easy")
        print("Environment ready!")
    
    def start_episode(self):
        """Start a new episode."""
        if not self.env:
            self.setup_environment()
        
        self.episode_count += 1
        print(f"Starting episode {self.episode_count}")
        
        # Reset environment
        self.current_obs, self.current_info = self._reset_env()
        
        # Clear data
        self.interceptor_positions.clear()
        self.adversary_positions.clear()
        self.times.clear()
        self.distances.clear()
        self.rewards.clear()
        
        # Reset simulation time
        self.simulation_time = 0
        
        # Start episode
        self.episode_running = True
        
        # Start animation
        if self.animation is None:
            self.animation = animation.FuncAnimation(
                self.fig, self._animate_frame, interval=self.update_interval, 
                blit=False, repeat=True
            )
    
    def _reset_env(self):
        """Reset environment with compatibility."""
        result = self.env.reset()
        if isinstance(result, tuple):
            return result
        return result, {}
    
    def _step_env(self, action):
        """Step environment with compatibility."""
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, done, info
        return result
    
    def _get_positions(self):
        """Get missile positions."""
        try:
            interceptor_pos = self.env.interceptor.get_position()
            adversary_pos = self.env.adversary.get_position()
            target_pos = self.env.target_position
            return interceptor_pos, adversary_pos, target_pos
        except:
            return np.array([0, 0, 1000]), np.array([1000, 1000, 1000]), np.array([0, 0, 0])
    
    def _animate_frame(self, frame):
        """Animation frame update."""
        if not self.episode_running or self.is_paused:
            return
        
        # Get action (random policy for demo)
        action = self.env.action_space.sample() * 0.3  # Reduced action for stability
        
        # Step environment
        try:
            self.current_obs, reward, terminated, truncated, self.current_info = self._step_env(action)
            
            # Update simulation time
            self.simulation_time += 0.05  # 50ms timestep
            
            # Check for episode end
            if terminated or truncated:
                self.episode_running = False
                print(f"Episode {self.episode_count} ended: {self.current_info.get('termination_reason', 'unknown')}")
                
                # Auto-start new episode after brief pause
                self.fig.canvas.draw_idle()
                plt.pause(2.0)  # 2 second pause
                self.start_episode()
                return
            
            # Update visualization
            self._update_visualization(reward)
            
        except Exception as e:
            print(f"Animation error: {e}")
            self.episode_running = False
    
    def _update_visualization(self, reward):
        """Update all visualization elements."""
        # Get positions
        interceptor_pos, adversary_pos, target_pos = self._get_positions()
        
        # Store data
        self.interceptor_positions.append(interceptor_pos.copy())
        self.adversary_positions.append(adversary_pos.copy())
        self.times.append(self.simulation_time)
        
        distance = np.linalg.norm(interceptor_pos - adversary_pos)
        self.distances.append(distance)
        self.rewards.append(reward)
        
        # Update markers
        self.interceptor_point._offsets3d = ([interceptor_pos[0]], [interceptor_pos[1]], [interceptor_pos[2]])
        self.adversary_point._offsets3d = ([adversary_pos[0]], [adversary_pos[1]], [adversary_pos[2]])
        
        # Update trails
        if len(self.interceptor_positions) > 1:
            int_trail = np.array(self.interceptor_positions)
            self.interceptor_trail.set_data_3d(int_trail[:, 0], int_trail[:, 1], int_trail[:, 2])
        
        if len(self.adversary_positions) > 1:
            adv_trail = np.array(self.adversary_positions)
            self.adversary_trail.set_data_3d(adv_trail[:, 0], adv_trail[:, 1], adv_trail[:, 2])
        
        # Update distance plot
        if len(self.times) > 1:
            self.distance_line.set_data(list(self.times), list(self.distances))
            self.ax_dist.set_xlim([min(self.times), max(self.times)])
            self.ax_dist.set_ylim([0, max(self.distances) * 1.1])
        
        # Update title with current info
        status = f"Episode {self.episode_count} | Time: {self.simulation_time:.1f}s | Distance: {distance:.1f}m"
        self.ax.set_title(status)
        
        # Camera following
        if self.follow_interceptor:
            self._follow_interceptor(interceptor_pos)
        
        # Force redraw
        self.fig.canvas.draw_idle()
    
    def _follow_interceptor(self, interceptor_pos):
        """Update camera to follow interceptor."""
        offset = 800.0
        self.ax.set_xlim([interceptor_pos[0] - offset, interceptor_pos[0] + offset])
        self.ax.set_ylim([interceptor_pos[1] - offset, interceptor_pos[1] + offset])
        self.ax.set_zlim([max(0, interceptor_pos[2] - offset), interceptor_pos[2] + offset])
    
    def _toggle_pause(self, event):
        """Toggle pause state."""
        self.is_paused = not self.is_paused
        self.btn_pause.label.set_text('Resume' if self.is_paused else 'Pause')
        print(f"Animation {'paused' if self.is_paused else 'resumed'}")
    
    def _start_new_episode(self, event):
        """Start new episode."""
        self.episode_running = False
        plt.pause(0.1)  # Brief pause
        self.start_episode()
    
    def _update_speed(self, val):
        """Update animation speed."""
        self.speed_multiplier = val
        # Update animation interval
        if self.animation:
            self.animation.event_source.interval = self.update_interval / val
    
    def show(self):
        """Show the animation."""
        plt.show()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Live Missile Animation Demo')
    parser.add_argument('--world-scale', type=float, default=3000.0,
                       help='World scale in meters (default: 3000.0)')
    parser.add_argument('--fps', type=int, default=20,
                       help='Animation frames per second (default: 20)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("AegisIntercept Phase 3 - Live Missile Animation Demo")
    print("="*70)
    print(f"World Scale: {args.world_scale}m")
    print(f"Animation FPS: {args.fps}")
    print("="*70)
    
    # Create animation system
    animation_system = LiveMissileAnimation(
        world_scale=args.world_scale,
        fps=args.fps
    )
    
    # Start first episode
    animation_system.start_episode()
    
    print("\\nControls:")
    print("  - Pause/Resume: Click 'Pause' button")
    print("  - New Episode: Click 'New Episode' button")
    print("  - Speed Control: Use 'Speed' slider")
    print("\\nWatch the missiles move in real-time!")
    print("Episodes will automatically restart when they end.")
    
    # Show animation
    animation_system.show()
    
    return 0


if __name__ == "__main__":
    exit(main())