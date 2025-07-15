"""
Matplotlib 3D Viewer for AegisIntercept Phase 3.

This module provides real-time 3D visualization of the missile intercept
simulation using matplotlib with interactive controls and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider, CheckButtons
from typing import Dict, List, Optional, Any, Tuple
import time
from collections import deque

from ..physics import WindField
from ..utils.maths import normalize_vector


class MatplotlibViewer:
    """
    Real-time 3D visualization using matplotlib.
    
    This class provides an interactive 3D view of the missile intercept
    simulation with customizable display options, real-time updates,
    and performance monitoring.
    """
    
    def __init__(self,
                 world_scale: float = 2000.0,
                 update_rate: float = 20.0,
                 trail_length: int = 50,
                 enable_wind_vectors: bool = True,
                 enable_performance_plots: bool = True):
        """
        Initialize matplotlib viewer.
        
        Args:
            world_scale: World scale for visualization bounds (meters)
            update_rate: Update frequency (Hz)
            trail_length: Number of trail points to display
            enable_wind_vectors: Show wind vector field
            enable_performance_plots: Show performance metrics
        """
        self.world_scale = world_scale
        self.update_rate = update_rate
        self.trail_length = trail_length
        self.enable_wind_vectors = enable_wind_vectors
        self.enable_performance_plots = enable_performance_plots
        
        # Visualization state
        self.is_running = False
        self.is_paused = False
        self.current_time = 0.0
        self.time_scale = 1.0
        
        # Data storage
        self.interceptor_trail = deque(maxlen=trail_length)
        self.adversary_trail = deque(maxlen=trail_length)
        self.performance_data = {
            'times': deque(maxlen=200),
            'distances': deque(maxlen=200),
            'rewards': deque(maxlen=200),
            'fuel_levels': deque(maxlen=200)
        }
        
        # Visualization options
        self.display_options = {
            'show_trails': True,
            'show_velocity_vectors': True,
            'show_wind_vectors': enable_wind_vectors,
            'show_target': True,
            'show_grid': True,
            'show_axes': True,
            'follow_interceptor': False,
            'show_performance': enable_performance_plots
        }
        
        # Initialize matplotlib components
        self._setup_figure()
        self._setup_3d_plot()
        self._setup_ui_controls()
        if enable_performance_plots:
            self._setup_performance_plots()
    
    def _setup_figure(self):
        """Setup main matplotlib figure."""
        # Create figure with subplots
        if self.enable_performance_plots:
            self.fig = plt.figure(figsize=(16, 10))
            # 3D plot takes up left 2/3 of figure
            self.ax_3d = self.fig.add_subplot(121, projection='3d')
            # Performance plots on the right
            self.ax_perf = self.fig.add_subplot(222)
            self.ax_reward = self.fig.add_subplot(224)
        else:
            self.fig = plt.figure(figsize=(12, 9))
            self.ax_3d = self.fig.add_subplot(111, projection='3d')
        
        self.fig.suptitle('AegisIntercept Phase 3 - 6DOF Simulation', fontsize=16)
        
        # Set up close event
        self.fig.canvas.mpl_connect('close_event', self._on_close)
    
    def _setup_3d_plot(self):
        """Setup 3D visualization plot."""
        # Set plot limits
        limit = self.world_scale / 2
        self.ax_3d.set_xlim([-limit, limit])
        self.ax_3d.set_ylim([-limit, limit])
        self.ax_3d.set_zlim([0, limit])
        
        # Labels and title
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('3D Trajectory View')
        
        # Initialize plot elements
        self.interceptor_point, = self.ax_3d.plot([], [], [], 'bo', markersize=8, label='Interceptor')
        self.adversary_point, = self.ax_3d.plot([], [], [], 'ro', markersize=8, label='Adversary')
        self.target_point, = self.ax_3d.plot([0], [0], [0], 'g^', markersize=12, label='Target')
        
        # Trail lines
        self.interceptor_trail_line, = self.ax_3d.plot([], [], [], 'b-', alpha=0.6, linewidth=2)
        self.adversary_trail_line, = self.ax_3d.plot([], [], [], 'r-', alpha=0.6, linewidth=2)
        
        # Velocity vectors
        self.interceptor_vel_arrow = None
        self.adversary_vel_arrow = None
        
        # Wind vectors
        self.wind_arrows = []
        if self.enable_wind_vectors:
            self._setup_wind_grid()
        
        # Legend
        self.ax_3d.legend()
        
        # Set view angle
        self.ax_3d.view_init(elev=20, azim=45)
    
    def _setup_wind_grid(self):
        """Setup wind vector visualization grid."""
        # Create a sparse grid for wind vectors
        grid_spacing = self.world_scale / 10
        x_range = np.arange(-self.world_scale/2, self.world_scale/2, grid_spacing)
        y_range = np.arange(-self.world_scale/2, self.world_scale/2, grid_spacing)
        z_range = np.arange(100, self.world_scale/2, grid_spacing * 2)
        
        self.wind_grid_points = []
        for x in x_range[::2]:  # Sparse sampling
            for y in y_range[::2]:
                for z in z_range[::2]:
                    self.wind_grid_points.append(np.array([x, y, z]))
    
    def _setup_ui_controls(self):
        """Setup UI controls and buttons."""
        # Create control panel axes
        ax_pause = plt.axes([0.02, 0.95, 0.08, 0.04])
        ax_reset = plt.axes([0.12, 0.95, 0.08, 0.04])
        ax_speed = plt.axes([0.22, 0.95, 0.15, 0.04])
        
        # Create buttons
        self.btn_pause = Button(ax_pause, 'Pause')
        self.btn_pause.on_clicked(self._toggle_pause)
        
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._reset_view)
        
        # Speed control slider
        self.slider_speed = Slider(ax_speed, 'Speed', 0.1, 5.0, valinit=1.0)
        self.slider_speed.on_changed(self._update_speed)
        
        # Display options checkboxes
        if self.enable_performance_plots:
            checkbox_ax = plt.axes([0.02, 0.05, 0.15, 0.25])
        else:
            checkbox_ax = plt.axes([0.02, 0.05, 0.15, 0.4])
        
        checkbox_labels = [
            'Trails', 'Velocity Vectors', 'Wind Vectors',
            'Target', 'Grid', 'Follow Interceptor'
        ]
        checkbox_states = [
            self.display_options['show_trails'],
            self.display_options['show_velocity_vectors'],
            self.display_options['show_wind_vectors'],
            self.display_options['show_target'],
            self.display_options['show_grid'],
            self.display_options['follow_interceptor']
        ]
        
        self.checkboxes = CheckButtons(checkbox_ax, checkbox_labels, checkbox_states)
        self.checkboxes.on_clicked(self._toggle_display_option)
    
    def _setup_performance_plots(self):
        """Setup performance monitoring plots."""
        # Distance plot
        self.ax_perf.set_title('Intercept Distance')
        self.ax_perf.set_xlabel('Time (s)')
        self.ax_perf.set_ylabel('Distance (m)')
        self.distance_line, = self.ax_perf.plot([], [], 'b-', linewidth=2)
        self.ax_perf.grid(True)
        
        # Reward and fuel plot
        self.ax_reward.set_title('Reward & Fuel')
        self.ax_reward.set_xlabel('Time (s)')
        self.ax_reward.set_ylabel('Value')
        self.reward_line, = self.ax_reward.plot([], [], 'g-', linewidth=2, label='Reward')
        self.fuel_line, = self.ax_reward.plot([], [], 'orange', linewidth=2, label='Fuel')
        self.ax_reward.legend()
        self.ax_reward.grid(True)
    
    def update_visualization(self,
                           interceptor_pos: np.ndarray,
                           interceptor_vel: np.ndarray,
                           adversary_pos: np.ndarray,
                           adversary_vel: np.ndarray,
                           target_pos: np.ndarray,
                           wind_field: Optional[WindField] = None,
                           performance_metrics: Optional[Dict[str, float]] = None,
                           simulation_time: float = 0.0):
        """
        Update visualization with current simulation state.
        
        Args:
            interceptor_pos: Interceptor position
            interceptor_vel: Interceptor velocity
            adversary_pos: Adversary position
            adversary_vel: Adversary velocity
            target_pos: Target position
            wind_field: Wind field for visualization
            performance_metrics: Current performance metrics
            simulation_time: Current simulation time
        """
        if self.is_paused:
            return
        
        self.current_time = simulation_time
        
        # Update position markers
        self.interceptor_point.set_data_3d([interceptor_pos[0]], [interceptor_pos[1]], [interceptor_pos[2]])
        self.adversary_point.set_data_3d([adversary_pos[0]], [adversary_pos[1]], [adversary_pos[2]])
        
        if self.display_options['show_target']:
            self.target_point.set_data_3d([target_pos[0]], [target_pos[1]], [target_pos[2]])
        
        # Update trails
        if self.display_options['show_trails']:
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
        
        # Update velocity vectors
        if self.display_options['show_velocity_vectors']:
            self._update_velocity_vectors(interceptor_pos, interceptor_vel, 
                                        adversary_pos, adversary_vel)
        
        # Update wind vectors
        if self.display_options['show_wind_vectors'] and wind_field:
            self._update_wind_vectors(wind_field, simulation_time)
        
        # Update camera if following interceptor
        if self.display_options['follow_interceptor']:
            self._update_camera_follow(interceptor_pos)
        
        # Update performance plots
        if self.enable_performance_plots and performance_metrics:
            self._update_performance_plots(performance_metrics, simulation_time)
        
        # Redraw
        if self.is_running:
            self.fig.canvas.draw_idle()
    
    def _update_velocity_vectors(self, int_pos: np.ndarray, int_vel: np.ndarray,
                               adv_pos: np.ndarray, adv_vel: np.ndarray):
        """Update velocity vector arrows."""
        # Remove old arrows
        if self.interceptor_vel_arrow:
            self.interceptor_vel_arrow.remove()
        if self.adversary_vel_arrow:
            self.adversary_vel_arrow.remove()
        
        # Scale velocity vectors for visibility
        vel_scale = 50.0
        
        # Interceptor velocity vector
        int_vel_scaled = int_vel / np.linalg.norm(int_vel) * vel_scale if np.linalg.norm(int_vel) > 0 else np.zeros(3)
        self.interceptor_vel_arrow = self.ax_3d.quiver(
            int_pos[0], int_pos[1], int_pos[2],
            int_vel_scaled[0], int_vel_scaled[1], int_vel_scaled[2],
            color='blue', alpha=0.7, arrow_length_ratio=0.1
        )
        
        # Adversary velocity vector
        adv_vel_scaled = adv_vel / np.linalg.norm(adv_vel) * vel_scale if np.linalg.norm(adv_vel) > 0 else np.zeros(3)
        self.adversary_vel_arrow = self.ax_3d.quiver(
            adv_pos[0], adv_pos[1], adv_pos[2],
            adv_vel_scaled[0], adv_vel_scaled[1], adv_vel_scaled[2],
            color='red', alpha=0.7, arrow_length_ratio=0.1
        )
    
    def _update_wind_vectors(self, wind_field: WindField, simulation_time: float):
        """Update wind vector field visualization."""
        # Remove old wind arrows
        for arrow in self.wind_arrows:
            arrow.remove()
        self.wind_arrows.clear()
        
        # Sample wind at grid points and create arrows
        wind_scale = 20.0
        
        for point in self.wind_grid_points[::4]:  # Further reduce for performance
            wind_vel = wind_field.get_wind_velocity(point, simulation_time)
            wind_magnitude = np.linalg.norm(wind_vel)
            
            if wind_magnitude > 1.0:  # Only show significant wind
                wind_scaled = wind_vel / wind_magnitude * wind_scale
                
                arrow = self.ax_3d.quiver(
                    point[0], point[1], point[2],
                    wind_scaled[0], wind_scaled[1], wind_scaled[2],
                    color='green', alpha=0.3, arrow_length_ratio=0.1
                )
                self.wind_arrows.append(arrow)
    
    def _update_camera_follow(self, interceptor_pos: np.ndarray):
        """Update camera to follow interceptor."""
        # Simple following: center view on interceptor
        offset = 200.0
        
        # Calculate new view center
        center_x = interceptor_pos[0]
        center_y = interceptor_pos[1]
        center_z = interceptor_pos[2]
        
        # Update plot limits
        self.ax_3d.set_xlim([center_x - offset, center_x + offset])
        self.ax_3d.set_ylim([center_y - offset, center_y + offset])
        self.ax_3d.set_zlim([max(0, center_z - offset), center_z + offset])
    
    def _update_performance_plots(self, metrics: Dict[str, float], simulation_time: float):
        """Update performance monitoring plots."""
        # Add new data points
        self.performance_data['times'].append(simulation_time)
        self.performance_data['distances'].append(metrics.get('intercept_distance', 0.0))
        self.performance_data['rewards'].append(metrics.get('step_reward', 0.0))
        self.performance_data['fuel_levels'].append(metrics.get('fuel_remaining', 1.0))
        
        # Update distance plot
        times = list(self.performance_data['times'])
        distances = list(self.performance_data['distances'])
        
        self.distance_line.set_data(times, distances)
        if times:
            self.ax_perf.set_xlim([min(times), max(times)])
            self.ax_perf.set_ylim([0, max(distances) * 1.1 if distances else 100])
        
        # Update reward and fuel plot
        rewards = list(self.performance_data['rewards'])
        fuel_levels = list(self.performance_data['fuel_levels'])
        
        self.reward_line.set_data(times, rewards)
        self.fuel_line.set_data(times, fuel_levels)
        
        if times:
            self.ax_reward.set_xlim([min(times), max(times)])
            if rewards and fuel_levels:
                y_min = min(min(rewards), 0)
                y_max = max(max(rewards), 1)
                self.ax_reward.set_ylim([y_min, y_max])
    
    def _toggle_pause(self, event):
        """Toggle pause state."""
        self.is_paused = not self.is_paused
        self.btn_pause.label.set_text('Resume' if self.is_paused else 'Pause')
    
    def _reset_view(self, event):
        """Reset view to default."""
        limit = self.world_scale / 2
        self.ax_3d.set_xlim([-limit, limit])
        self.ax_3d.set_ylim([-limit, limit])
        self.ax_3d.set_zlim([0, limit])
        self.ax_3d.view_init(elev=20, azim=45)
        
        # Clear trails
        self.interceptor_trail.clear()
        self.adversary_trail.clear()
        
        # Clear performance data
        for key in self.performance_data:
            self.performance_data[key].clear()
    
    def _update_speed(self, val):
        """Update time scale."""
        self.time_scale = val
    
    def _toggle_display_option(self, label):
        """Toggle display options."""
        option_map = {
            'Trails': 'show_trails',
            'Velocity Vectors': 'show_velocity_vectors',
            'Wind Vectors': 'show_wind_vectors',
            'Target': 'show_target',
            'Grid': 'show_grid',
            'Follow Interceptor': 'follow_interceptor'
        }
        
        if label in option_map:
            option_key = option_map[label]
            self.display_options[option_key] = not self.display_options[option_key]
            
            # Apply immediate changes
            if option_key == 'show_target':
                self.target_point.set_visible(self.display_options[option_key])
            elif option_key == 'show_trails':
                self.interceptor_trail_line.set_visible(self.display_options[option_key])
                self.adversary_trail_line.set_visible(self.display_options[option_key])
            elif option_key == 'show_grid':
                self.ax_3d.grid(self.display_options[option_key])
    
    def _on_close(self, event):
        """Handle figure close event."""
        self.is_running = False
    
    def start(self):
        """Start the visualization."""
        self.is_running = True
        plt.show(block=False)
    
    def stop(self):
        """Stop the visualization."""
        self.is_running = False
        plt.close(self.fig)
    
    def save_screenshot(self, filename: str):
        """Save current view as screenshot."""
        try:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Screenshot saved: {filename}")
        except Exception as e:
            print(f"Error saving screenshot: {e}")
    
    def export_trajectory_plot(self, filename: str):
        """Export trajectory plot as image."""
        try:
            # Create a clean plot with just trajectories
            fig_export = plt.figure(figsize=(12, 9))
            ax_export = fig_export.add_subplot(111, projection='3d')
            
            # Plot trails
            if len(self.interceptor_trail) > 1:
                trail_array = np.array(self.interceptor_trail)
                ax_export.plot(trail_array[:, 0], trail_array[:, 1], trail_array[:, 2], 
                              'b-', linewidth=3, label='Interceptor', alpha=0.8)
            
            if len(self.adversary_trail) > 1:
                trail_array = np.array(self.adversary_trail)
                ax_export.plot(trail_array[:, 0], trail_array[:, 1], trail_array[:, 2], 
                              'r-', linewidth=3, label='Adversary', alpha=0.8)
            
            # Mark target
            ax_export.plot([0], [0], [0], 'g^', markersize=15, label='Target')
            
            # Set labels and limits
            ax_export.set_xlabel('X (m)')
            ax_export.set_ylabel('Y (m)')
            ax_export.set_zlabel('Z (m)')
            ax_export.set_title('AegisIntercept Phase 3 - Mission Trajectory')
            ax_export.legend()
            
            # Set equal aspect ratio
            limit = self.world_scale / 2
            ax_export.set_xlim([-limit, limit])
            ax_export.set_ylim([-limit, limit])
            ax_export.set_zlim([0, limit])
            
            # Save and close
            fig_export.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig_export)
            
            print(f"Trajectory plot exported: {filename}")
            
        except Exception as e:
            print(f"Error exporting trajectory plot: {e}")
    
    def get_display_options(self) -> Dict[str, bool]:
        """Get current display options."""
        return self.display_options.copy()
    
    def set_display_option(self, option: str, value: bool):
        """Set a display option."""
        if option in self.display_options:
            self.display_options[option] = value