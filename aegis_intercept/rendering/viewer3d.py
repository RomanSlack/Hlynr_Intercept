"""3D Viewer for AegisIntercept using Matplotlib."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from collections import deque

class Viewer3D:
    def __init__(self, world_size: float, speed_multiplier: float = 5.0):
        self.world_size = world_size
        self.speed_multiplier = speed_multiplier  # Speed up visualization
        self.frame_skip = max(1, int(speed_multiplier))  # Skip frames for faster rendering
        self.frame_count = 0
        
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=(10, 7))  # Smaller window for faster rendering
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set limits
        self.ax.set_xlim([0, world_size * 2])
        self.ax.set_ylim([0, world_size * 2])
        self.ax.set_zlim([0, world_size * 2])
        self.ax.set_xlabel("X (East)")
        self.ax.set_ylabel("Y (North)")
        self.ax.set_zlabel("Z (Altitude)")
        self.ax.set_title(f"AegisIntercept 3D - Missile Defense Training (Speed: {speed_multiplier}x)")
        
        # Optimize rendering performance
        plt.tight_layout()
        plt.show(block=False)
        
        # Performance optimizations
        self.ax.mouse_init()  # Disable mouse interaction for speed
        self.fig.canvas.draw_idle()  # Use idle drawing
        
        # Disable some visual elements for speed
        self.ax.grid(False)
        self.ax.set_facecolor('black')  # Dark background is faster to render
        
        # Track intercept status for visual feedback
        self.last_intercept = False
        
        # Ghost trail history - reduced for performance
        self.trail_length = max(20, 100 // speed_multiplier)  # Fewer trail points for speed
        self.interceptor_trail = deque(maxlen=self.trail_length)
        self.missile_trail = deque(maxlen=self.trail_length)
        
        # Create custom colormaps for speed visualization
        # Blue to cyan to yellow to red for interceptor (cool to warm)
        interceptor_colors = ['#000080', '#0080FF', '#00FFFF', '#80FF00', '#FFFF00', '#FF8000', '#FF0000']
        self.interceptor_cmap = LinearSegmentedColormap.from_list('interceptor_speed', interceptor_colors, N=256)
        
        # Dark red to red to orange to yellow for missile (threat colors)
        missile_colors = ['#800000', '#FF0000', '#FF4000', '#FF8000', '#FFC000', '#FFFF00']
        self.missile_cmap = LinearSegmentedColormap.from_list('missile_speed', missile_colors, N=256)
        
        # Previous positions for velocity calculation
        self.prev_interceptor_pos = None
        self.prev_missile_pos = None

    def render(self, interceptor_pos: np.ndarray, missile_pos: np.ndarray, target_pos: np.ndarray, intercepted: bool = False, popup_info: dict = None):
        # Frame skipping for performance
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0 and not intercepted:
            return  # Skip this frame unless it's an intercept event
        # Clear previous frame efficiently
        self.ax.clear()
        
        # Re-apply optimized settings after clear
        self.ax.set_xlim([0, self.world_size * 2])
        self.ax.set_ylim([0, self.world_size * 2])
        self.ax.set_zlim([0, self.world_size * 2])
        self.ax.set_facecolor('black')
        self.ax.grid(False)
        
        # Calculate velocities if we have previous positions
        interceptor_velocity = 0
        missile_velocity = 0
        
        if self.prev_interceptor_pos is not None:
            interceptor_velocity = np.linalg.norm(interceptor_pos - self.prev_interceptor_pos)
        if self.prev_missile_pos is not None:
            missile_velocity = np.linalg.norm(missile_pos - self.prev_missile_pos)
        
        # Store current positions and velocities in trails
        self.interceptor_trail.append({
            'pos': interceptor_pos.copy(),
            'velocity': interceptor_velocity
        })
        self.missile_trail.append({
            'pos': missile_pos.copy(),
            'velocity': missile_velocity
        })
        
        # Update previous positions
        self.prev_interceptor_pos = interceptor_pos.copy()
        self.prev_missile_pos = missile_pos.copy()
        
        # Clear on intercept or episode reset
        if intercepted or (self.last_intercept and not intercepted):
            self.interceptor_trail.clear()
            self.missile_trail.clear()
            self.prev_interceptor_pos = None
            self.prev_missile_pos = None
        
        self.ax.cla()
        # New coordinate system: 0 to 600
        self.ax.set_xlim([0, self.world_size * 2])
        self.ax.set_ylim([0, self.world_size * 2])
        self.ax.set_zlim([0, self.world_size * 2])
        self.ax.set_xlabel("X (East)")
        self.ax.set_ylabel("Y (North)")
        self.ax.set_zlabel("Z (Altitude)")
        
        # Update title based on intercept status
        if intercepted:
            self.ax.set_title("AegisIntercept 3D - 🎯 INTERCEPT SUCCESS! 🎯", color='green')
            self.last_intercept = True
        elif self.last_intercept:
            self.ax.set_title("AegisIntercept 3D - Episode Resetting...", color='orange')
            self.last_intercept = False
        else:
            self.ax.set_title("AegisIntercept 3D - Missile Defense Training")

        # Draw ground plane reference
        ground_x, ground_y = np.meshgrid(np.linspace(0, self.world_size * 2, 10), 
                                        np.linspace(0, self.world_size * 2, 10))
        ground_z = np.zeros_like(ground_x)
        self.ax.plot_surface(ground_x, ground_y, ground_z, alpha=0.1, color='brown')

        # Draw ghost trails with speed-based coloring
        self._draw_ghost_trails()

        # Draw target at ground level
        self.ax.scatter(target_pos[0], target_pos[1], target_pos[2], c='green', marker='o', s=200, label="Target (Ground)", edgecolors='darkgreen', linewidth=2)

        # Draw interceptor with different colors based on status
        if intercepted:
            self.ax.scatter(interceptor_pos[0], interceptor_pos[1], interceptor_pos[2], c='gold', marker='^', s=150, label="Interceptor (SUCCESS!)", edgecolors='orange', linewidth=2)
        else:
            self.ax.scatter(interceptor_pos[0], interceptor_pos[1], interceptor_pos[2], c='blue', marker='^', s=120, label="Interceptor", edgecolors='darkblue', linewidth=1)

        # Draw missile with threat indicator
        distance_to_target = np.linalg.norm(missile_pos - target_pos)
        if distance_to_target < 50:
            missile_color = 'red'
            missile_label = "Incoming Missile (CRITICAL)"
            missile_size = 150
        else:
            missile_color = 'darkred'
            missile_label = "Incoming Missile"
            missile_size = 120
            
        self.ax.scatter(missile_pos[0], missile_pos[1], missile_pos[2], c=missile_color, marker='X', s=missile_size, label=missile_label, edgecolors='black', linewidth=1)

        self.ax.legend(loc='upper right')
        
        # Add popup message if provided
        if popup_info:
            message = popup_info['message']
            color = popup_info['color']
            timer = popup_info['timer']
            
            # Convert RGB to matplotlib format (0-1 range)
            color_normalized = tuple(c/255.0 for c in color)
            
            # Calculate alpha based on timer (fade out effect)
            alpha = min(1.0, timer / 60.0)  # Full opacity for first second, then fade
            
            # Add large text overlay
            self.fig.suptitle(message, fontsize=20, fontweight='bold', 
                            color=color_normalized, alpha=alpha, y=0.95)
            
            # Add a background box for better visibility
            bbox_props = dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8)
            self.ax.text2D(0.5, 0.95, message, transform=self.ax.transAxes, 
                          fontsize=16, fontweight='bold', color=color_normalized,
                          horizontalalignment='center', verticalalignment='top',
                          bbox=bbox_props, alpha=alpha)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)  # Small pause to ensure rendering

    def _draw_ghost_trails(self):
        """Draw ghost trails with speed-based color coding"""
        # Draw interceptor trail
        if len(self.interceptor_trail) > 1:
            self._draw_trail(self.interceptor_trail, self.interceptor_cmap, 'Interceptor Trail')
        
        # Draw missile trail
        if len(self.missile_trail) > 1:
            self._draw_trail(self.missile_trail, self.missile_cmap, 'Missile Trail')
    
    def _draw_trail(self, trail, colormap, label):
        """Draw a single trail with speed-based coloring"""
        if len(trail) < 2:
            return
            
        # Extract positions and velocities
        positions = np.array([point['pos'] for point in trail])
        velocities = np.array([point['velocity'] for point in trail])
        
        # Normalize velocities for color mapping (0-1 range)
        if velocities.max() > 0:
            normalized_velocities = velocities / velocities.max()
        else:
            normalized_velocities = velocities
        
        # Draw trail segments with speed-based colors
        for i in range(len(positions) - 1):
            # Calculate segment color based on speed
            speed_color = colormap(normalized_velocities[i])
            
            # Calculate alpha based on age (newer = more opaque)
            age_factor = (i + 1) / len(positions)
            alpha = 0.3 + 0.7 * age_factor  # Range from 0.3 to 1.0
            
            # Draw line segment
            self.ax.plot([positions[i][0], positions[i+1][0]], 
                        [positions[i][1], positions[i+1][1]], 
                        [positions[i][2], positions[i+1][2]], 
                        color=speed_color, alpha=alpha, linewidth=2)
        
        # Add speed indicators along the trail (every 10th point)
        sample_indices = range(0, len(positions), 10)
        for i in sample_indices:
            if i < len(positions):
                speed_color = colormap(normalized_velocities[i])
                # Draw small spheres to show speed at key points
                self.ax.scatter(positions[i][0], positions[i][1], positions[i][2], 
                              c=[speed_color], s=20, alpha=0.6, edgecolors='none')

    def close(self):
        plt.ioff()
        plt.close()
