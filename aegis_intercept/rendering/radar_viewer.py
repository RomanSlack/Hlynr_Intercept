"""
Radar visualization system for AegisIntercept Phase 3.

This module provides radar-style visualization capabilities for tracking
and displaying targets, tracks, and sensor coverage areas.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Tuple, Optional, Any
import math


class RadarViewer:
    """
    Radar-style visualization for missile interception simulation.
    
    This class provides a radar display similar to real air traffic control
    or military radar systems, showing targets, tracks, and coverage areas
    in a polar coordinate system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize radar viewer.
        
        Args:
            config: Configuration dictionary with display parameters
        """
        # Default configuration
        self.config = {
            'max_range': 50000.0,           # Maximum display range (m)
            'range_rings': [10000, 20000, 30000, 40000, 50000],  # Range ring positions
            'update_rate': 10.0,            # Display update rate (Hz)
            'trail_length': 50,             # Length of target trails
            'grid_color': 'green',          # Grid color
            'background_color': 'black',    # Background color
            'target_color': 'red',          # Target color
            'track_color': 'yellow',        # Track color
            'interceptor_color': 'blue',    # Interceptor color
            'coverage_color': 'lightblue',  # Coverage area color
            'text_color': 'white',          # Text color
            'show_coverage': True,          # Show sensor coverage
            'show_trails': True,            # Show target trails
            'show_predictions': True,       # Show predicted positions
            'polar_display': True,          # Use polar coordinate display
            'center_position': [0, 0, 0],   # Center position for display
        }
        
        # Update with provided configuration
        if config:
            self.config.update(config)
        
        # Initialize display state
        self.fig = None
        self.ax = None
        self.target_trails = {}
        self.track_trails = {}
        self.last_update_time = 0.0
        
        # Initialize display
        self._initialize_display()
    
    def _initialize_display(self):
        """Initialize the radar display."""
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 12), 
                                        facecolor=self.config['background_color'])
        
        if self.config['polar_display']:
            # Create polar plot
            self.ax.remove()
            self.ax = self.fig.add_subplot(111, projection='polar')
            self.ax.set_theta_zero_location('N')
            self.ax.set_theta_direction(-1)
            self.ax.set_ylim(0, self.config['max_range'])
            self.ax.set_facecolor(self.config['background_color'])
            
            # Set range rings
            self.ax.set_rgrids(self.config['range_rings'], 
                             labels=[f'{r/1000:.0f}km' for r in self.config['range_rings']],
                             color=self.config['grid_color'], alpha=0.5)
            
            # Set angular grid
            self.ax.set_thetagrids(range(0, 360, 30), 
                                 color=self.config['grid_color'], alpha=0.5)
        else:
            # Create Cartesian plot
            self.ax.set_xlim(-self.config['max_range'], self.config['max_range'])
            self.ax.set_ylim(-self.config['max_range'], self.config['max_range'])
            self.ax.set_aspect('equal')
            self.ax.set_facecolor(self.config['background_color'])
            self.ax.grid(True, color=self.config['grid_color'], alpha=0.3)
            
            # Add range rings
            for ring_range in self.config['range_rings']:
                circle = plt.Circle((0, 0), ring_range, 
                                  fill=False, color=self.config['grid_color'], 
                                  alpha=0.5, linestyle='--')
                self.ax.add_patch(circle)
        
        # Set title and labels
        self.ax.set_title('AegisIntercept Radar Display', 
                         color=self.config['text_color'], fontsize=16)
        
        plt.tight_layout()
    
    def update_display(self, 
                      targets: List[Dict[str, Any]], 
                      tracks: List[Dict[str, Any]], 
                      interceptors: List[Dict[str, Any]], 
                      sensor_coverage: Optional[Dict[str, Any]] = None,
                      current_time: float = 0.0) -> None:
        """
        Update the radar display with new data.
        
        Args:
            targets: List of target dictionaries
            tracks: List of track dictionaries
            interceptors: List of interceptor dictionaries
            sensor_coverage: Sensor coverage information
            current_time: Current simulation time
        """
        # Check if enough time has passed for update
        if current_time - self.last_update_time < 1.0 / self.config['update_rate']:
            return
        
        # Clear previous frame
        self.ax.clear()
        self._initialize_display()
        
        # Draw sensor coverage
        if self.config['show_coverage'] and sensor_coverage:
            self._draw_coverage(sensor_coverage)
        
        # Draw targets
        self._draw_targets(targets, current_time)
        
        # Draw tracks
        self._draw_tracks(tracks, current_time)
        
        # Draw interceptors
        self._draw_interceptors(interceptors, current_time)
        
        # Draw predictions
        if self.config['show_predictions']:
            self._draw_predictions(targets, tracks, current_time)
        
        # Update display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        self.last_update_time = current_time
    
    def _draw_targets(self, targets: List[Dict[str, Any]], current_time: float) -> None:
        """Draw targets on the radar display."""
        for target in targets:
            if 'position' not in target:
                continue
            
            position = target['position']
            target_id = target.get('id', 'unknown')
            
            # Convert to display coordinates
            if self.config['polar_display']:
                r, theta = self._cartesian_to_polar(position)
                if r > self.config['max_range']:
                    continue
                
                # Draw target symbol
                self.ax.plot(theta, r, 'o', color=self.config['target_color'], 
                           markersize=8, markeredgecolor='white', markeredgewidth=1)
                
                # Add target label
                self.ax.text(theta, r + 1000, f'T{target_id}', 
                           color=self.config['text_color'], fontsize=8, 
                           ha='center', va='bottom')
            else:
                x, y = position[0], position[1]
                if abs(x) > self.config['max_range'] or abs(y) > self.config['max_range']:
                    continue
                
                # Draw target symbol
                self.ax.plot(x, y, 'o', color=self.config['target_color'], 
                           markersize=8, markeredgecolor='white', markeredgewidth=1)
                
                # Add target label
                self.ax.text(x, y + 1000, f'T{target_id}', 
                           color=self.config['text_color'], fontsize=8, 
                           ha='center', va='bottom')
            
            # Update target trail
            if self.config['show_trails']:
                self._update_trail(target_id, position, current_time, 'target')
    
    def _draw_tracks(self, tracks: List[Dict[str, Any]], current_time: float) -> None:
        """Draw tracks on the radar display."""
        for track in tracks:
            if 'state' not in track:
                continue
            
            state = track['state']
            position = state[:3]  # First 3 elements are position
            track_id = track.get('track_id', 'unknown')
            quality = track.get('quality', 0.0)
            
            # Convert to display coordinates
            if self.config['polar_display']:
                r, theta = self._cartesian_to_polar(position)
                if r > self.config['max_range']:
                    continue
                
                # Draw track symbol (size based on quality)
                markersize = 6 + 4 * quality
                self.ax.plot(theta, r, 's', color=self.config['track_color'], 
                           markersize=markersize, markeredgecolor='white', 
                           markeredgewidth=1, alpha=0.7)
                
                # Add track label
                self.ax.text(theta, r - 1500, f'TR{track_id}', 
                           color=self.config['track_color'], fontsize=8, 
                           ha='center', va='top')
            else:
                x, y = position[0], position[1]
                if abs(x) > self.config['max_range'] or abs(y) > self.config['max_range']:
                    continue
                
                # Draw track symbol
                markersize = 6 + 4 * quality
                self.ax.plot(x, y, 's', color=self.config['track_color'], 
                           markersize=markersize, markeredgecolor='white', 
                           markeredgewidth=1, alpha=0.7)
                
                # Add track label
                self.ax.text(x, y - 1500, f'TR{track_id}', 
                           color=self.config['track_color'], fontsize=8, 
                           ha='center', va='top')
            
            # Update track trail
            if self.config['show_trails']:
                self._update_trail(track_id, position, current_time, 'track')
    
    def _draw_interceptors(self, interceptors: List[Dict[str, Any]], current_time: float) -> None:
        """Draw interceptors on the radar display."""
        for interceptor in interceptors:
            if 'position' not in interceptor:
                continue
            
            position = interceptor['position']
            interceptor_id = interceptor.get('id', 'unknown')
            
            # Convert to display coordinates
            if self.config['polar_display']:
                r, theta = self._cartesian_to_polar(position)
                if r > self.config['max_range']:
                    continue
                
                # Draw interceptor symbol
                self.ax.plot(theta, r, '^', color=self.config['interceptor_color'], 
                           markersize=10, markeredgecolor='white', markeredgewidth=1)
                
                # Add interceptor label
                self.ax.text(theta, r + 2000, f'I{interceptor_id}', 
                           color=self.config['interceptor_color'], fontsize=8, 
                           ha='center', va='bottom')
            else:
                x, y = position[0], position[1]
                if abs(x) > self.config['max_range'] or abs(y) > self.config['max_range']:
                    continue
                
                # Draw interceptor symbol
                self.ax.plot(x, y, '^', color=self.config['interceptor_color'], 
                           markersize=10, markeredgecolor='white', markeredgewidth=1)
                
                # Add interceptor label
                self.ax.text(x, y + 2000, f'I{interceptor_id}', 
                           color=self.config['interceptor_color'], fontsize=8, 
                           ha='center', va='bottom')
    
    def _draw_coverage(self, coverage: Dict[str, Any]) -> None:
        """Draw sensor coverage area."""
        if 'range' not in coverage:
            return
        
        max_range = coverage['range']
        azimuth_fov = coverage.get('azimuth_fov', 360)
        elevation_fov = coverage.get('elevation_fov', 90)
        
        if self.config['polar_display']:
            # Draw coverage sector
            theta_start = -math.radians(azimuth_fov / 2)
            theta_end = math.radians(azimuth_fov / 2)
            
            # Create coverage wedge
            wedge = patches.Wedge((0, 0), max_range, 
                                math.degrees(theta_start), math.degrees(theta_end),
                                facecolor=self.config['coverage_color'], 
                                alpha=0.2, edgecolor=self.config['coverage_color'])
            self.ax.add_patch(wedge)
        else:
            # Draw coverage circle
            circle = plt.Circle((0, 0), max_range, 
                              fill=True, facecolor=self.config['coverage_color'], 
                              alpha=0.1, edgecolor=self.config['coverage_color'])
            self.ax.add_patch(circle)
    
    def _draw_predictions(self, targets: List[Dict[str, Any]], 
                         tracks: List[Dict[str, Any]], 
                         current_time: float) -> None:
        """Draw predicted positions."""
        prediction_time = 5.0  # Predict 5 seconds ahead
        
        # Draw target predictions
        for target in targets:
            if 'position' not in target or 'velocity' not in target:
                continue
            
            position = target['position']
            velocity = target['velocity']
            
            # Simple linear prediction
            predicted_position = position + velocity * prediction_time
            
            if self.config['polar_display']:
                r, theta = self._cartesian_to_polar(predicted_position)
                if r <= self.config['max_range']:
                    self.ax.plot(theta, r, 'x', color=self.config['target_color'], 
                               markersize=6, alpha=0.5)
                    
                    # Draw prediction line
                    r_current, theta_current = self._cartesian_to_polar(position)
                    self.ax.plot([theta_current, theta], [r_current, r], 
                               '--', color=self.config['target_color'], alpha=0.3)
            else:
                x, y = predicted_position[0], predicted_position[1]
                if abs(x) <= self.config['max_range'] and abs(y) <= self.config['max_range']:
                    self.ax.plot(x, y, 'x', color=self.config['target_color'], 
                               markersize=6, alpha=0.5)
                    
                    # Draw prediction line
                    self.ax.plot([position[0], x], [position[1], y], 
                               '--', color=self.config['target_color'], alpha=0.3)
    
    def _cartesian_to_polar(self, position: np.ndarray) -> Tuple[float, float]:
        """Convert Cartesian coordinates to polar."""
        x, y, z = position
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta
    
    def _update_trail(self, object_id: str, position: np.ndarray, 
                     current_time: float, object_type: str) -> None:
        """Update position trail for an object."""
        if object_type == 'target':
            trails = self.target_trails
        else:
            trails = self.track_trails
        
        if object_id not in trails:
            trails[object_id] = []
        
        # Add current position to trail
        trails[object_id].append({
            'position': position.copy(),
            'time': current_time
        })
        
        # Limit trail length
        if len(trails[object_id]) > self.config['trail_length']:
            trails[object_id].pop(0)
        
        # Draw trail
        if len(trails[object_id]) > 1:
            positions = [entry['position'] for entry in trails[object_id]]
            
            if self.config['polar_display']:
                rs, thetas = [], []
                for pos in positions:
                    r, theta = self._cartesian_to_polar(pos)
                    if r <= self.config['max_range']:
                        rs.append(r)
                        thetas.append(theta)
                
                if len(rs) > 1:
                    color = self.config['target_color'] if object_type == 'target' else self.config['track_color']
                    self.ax.plot(thetas, rs, '-', color=color, alpha=0.3, linewidth=1)
            else:
                xs = [pos[0] for pos in positions]
                ys = [pos[1] for pos in positions]
                
                color = self.config['target_color'] if object_type == 'target' else self.config['track_color']
                self.ax.plot(xs, ys, '-', color=color, alpha=0.3, linewidth=1)
    
    def set_center_position(self, position: np.ndarray) -> None:
        """Set the center position for the display."""
        self.config['center_position'] = position.copy()
    
    def clear_trails(self) -> None:
        """Clear all position trails."""
        self.target_trails = {}
        self.track_trails = {}
    
    def save_frame(self, filename: str) -> None:
        """Save current frame to file."""
        self.fig.savefig(filename, facecolor=self.config['background_color'], 
                        dpi=150, bbox_inches='tight')
    
    def get_config(self) -> Dict[str, Any]:
        """Get current viewer configuration."""
        return self.config.copy()
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Update viewer configuration."""
        self.config.update(config)
        
        # Reinitialize display if necessary
        if any(key in config for key in ['polar_display', 'max_range', 'background_color']):
            self._initialize_display()
    
    def close(self) -> None:
        """Close the radar display."""
        if self.fig:
            plt.close(self.fig)