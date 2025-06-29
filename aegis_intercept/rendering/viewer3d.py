"""3D Viewer for AegisIntercept using Matplotlib."""

import matplotlib.pyplot as plt
import numpy as np

class Viewer3D:
    def __init__(self, world_size: float):
        self.world_size = world_size
        plt.ion()  # Enable interactive mode
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        # New coordinate system: 0 to 600
        self.ax.set_xlim([0, world_size * 2])
        self.ax.set_ylim([0, world_size * 2])
        self.ax.set_zlim([0, world_size * 2])
        self.ax.set_xlabel("X (East)")
        self.ax.set_ylabel("Y (North)")
        self.ax.set_zlabel("Z (Altitude)")
        self.ax.set_title("AegisIntercept 3D - Missile Defense Training")
        plt.show(block=False)  # Show the window without blocking
        
        # Track intercept status for visual feedback
        self.last_intercept = False

    def render(self, interceptor_pos: np.ndarray, missile_pos: np.ndarray, target_pos: np.ndarray, intercepted: bool = False, popup_info: dict = None):
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

        # Draw trajectory lines
        self.ax.plot([target_pos[0], interceptor_pos[0]], [target_pos[1], interceptor_pos[1]], [target_pos[2], interceptor_pos[2]], 'b--', alpha=0.5, linewidth=1)
        self.ax.plot([target_pos[0], missile_pos[0]], [target_pos[1], missile_pos[1]], [target_pos[2], missile_pos[2]], 'r--', alpha=0.5, linewidth=1)

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

    def close(self):
        plt.ioff()
        plt.close()
