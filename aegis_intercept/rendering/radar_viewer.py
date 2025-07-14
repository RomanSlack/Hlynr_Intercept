"""
Realistic Radar Visualization System

Displays authentic radar PPI (Plan Position Indicator) and RHI (Range Height Indicator)
scopes showing only sensor-derived information, just like real radar operators see.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from collections import deque
import time
import math

class RadarPPI:
    """Plan Position Indicator - Traditional circular radar scope"""
    
    def __init__(self, max_range: float = 400000, update_rate: float = 6.0):
        """
        Initialize PPI display
        
        Args:
            max_range: Maximum radar range in meters
            update_rate: Radar sweep rate in seconds per rotation
        """
        self.max_range = max_range
        self.update_rate = update_rate
        self.current_azimuth = 0.0  # Current sweep beam position
        
        # Create figure with dark radar-style theme
        plt.ion()
        self.fig, (self.ax_ppi, self.ax_rhi) = plt.subplots(1, 2, figsize=(16, 8), 
                                                           facecolor='black')
        
        # PPI Setup (left panel)
        self.ax_ppi.set_xlim([-max_range/1000, max_range/1000])  # Convert to km
        self.ax_ppi.set_ylim([-max_range/1000, max_range/1000])
        self.ax_ppi.set_aspect('equal')
        self.ax_ppi.set_facecolor('black')
        self.ax_ppi.tick_params(colors='green')
        self.ax_ppi.set_xlabel('East (km)', color='green')
        self.ax_ppi.set_ylabel('North (km)', color='green')
        self.ax_ppi.set_title('Ground Radar PPI - Plan Position Indicator', color='green')
        
        # RHI Setup (right panel)
        self.ax_rhi.set_xlim([0, max_range/1000])
        self.ax_rhi.set_ylim([0, 100])  # Altitude in km
        self.ax_rhi.set_facecolor('black')
        self.ax_rhi.tick_params(colors='green')
        self.ax_rhi.set_xlabel('Range (km)', color='green')
        self.ax_rhi.set_ylabel('Altitude (km)', color='green')
        self.ax_rhi.set_title('Range Height Indicator', color='green')
        
        # Draw range rings and bearing lines
        self._draw_radar_grid()
        
        # Track storage for persistence
        self.track_history = {}  # track_id -> deque of positions
        self.max_history = 50
        
        # Detection storage for sweep painting
        self.detections = deque(maxlen=500)  # Raw detections with timestamp
        self.last_sweep_time = time.time()
        
        # Color maps for radar
        self.track_colors = ['lime', 'cyan', 'yellow', 'orange', 'red']
        
        plt.tight_layout()
        plt.show(block=False)
        
    def _draw_radar_grid(self):
        """Draw authentic radar grid with range rings and bearing lines"""
        
        # PPI Range rings (km)
        for range_km in [50, 100, 200, 300, 400]:
            if range_km <= self.max_range/1000:
                circle = plt.Circle((0, 0), range_km, fill=False, 
                                  color='darkgreen', alpha=0.5, linewidth=0.5)
                self.ax_ppi.add_patch(circle)
                # Range labels
                self.ax_ppi.text(range_km*0.707, range_km*0.707, f'{range_km}', 
                               color='darkgreen', fontsize=8)
        
        # Bearing lines every 30 degrees
        for bearing in range(0, 360, 30):
            angle_rad = math.radians(bearing)
            x_end = (self.max_range/1000) * math.cos(angle_rad)
            y_end = (self.max_range/1000) * math.sin(angle_rad)
            self.ax_ppi.plot([0, x_end], [0, y_end], 'darkgreen', alpha=0.3, linewidth=0.5)
            
            # Bearing labels
            label_dist = (self.max_range/1000) * 0.9
            label_x = label_dist * math.cos(angle_rad)
            label_y = label_dist * math.sin(angle_rad)
            self.ax_ppi.text(label_x, label_y, f'{bearing:03d}°', 
                           color='darkgreen', fontsize=8, ha='center')
        
        # RHI Grid lines
        for range_km in [50, 100, 200, 300, 400]:
            if range_km <= self.max_range/1000:
                self.ax_rhi.axvline(x=range_km, color='darkgreen', alpha=0.3, linewidth=0.5)
                
        for alt_km in [10, 20, 50, 80]:
            self.ax_rhi.axhline(y=alt_km, color='darkgreen', alpha=0.3, linewidth=0.5)
    
    def update_sweep(self, dt: float):
        """Update radar sweep beam position"""
        # Rotate sweep beam
        sweep_rate = 360.0 / self.update_rate  # degrees per second
        self.current_azimuth = (self.current_azimuth + sweep_rate * dt) % 360.0
        
    def add_detection(self, position: np.ndarray, snr: float, timestamp: float,
                     detection_type: str = 'target', track_id: int = None):
        """
        Add a radar detection (only available when beam sweeps over target)
        
        Args:
            position: 3D position in global coordinates
            snr: Signal-to-noise ratio in dB
            timestamp: Detection timestamp
            detection_type: 'target', 'false_alarm', 'clutter'
            track_id: Associated track ID if available
        """
        # Convert to radar-relative coordinates (assuming radar at origin)
        range_m = np.linalg.norm(position[:2])  # Horizontal range
        azimuth = math.degrees(math.atan2(position[1], position[0]))
        if azimuth < 0:
            azimuth += 360
        elevation = math.degrees(math.atan2(position[2], range_m))
        
        detection = {
            'range': range_m,
            'azimuth': azimuth,
            'elevation': elevation,
            'altitude': position[2],
            'snr': snr,
            'timestamp': timestamp,
            'type': detection_type,
            'track_id': track_id,
            'position': position.copy()
        }
        
        self.detections.append(detection)
        
        # Update track history if this is a confirmed track
        if track_id is not None and detection_type == 'target':
            if track_id not in self.track_history:
                self.track_history[track_id] = deque(maxlen=self.max_history)
            self.track_history[track_id].append({
                'position': position.copy(),
                'timestamp': timestamp,
                'snr': snr
            })
    
    def render(self, radar_position: np.ndarray = None, sensor_info: dict = None):
        """
        Render realistic radar display
        
        Args:
            radar_position: Position of radar system
            sensor_info: Additional sensor status information
        """
        current_time = time.time()
        
        # Clear both displays
        self.ax_ppi.clear()
        self.ax_rhi.clear()
        
        # Redraw grids
        self._setup_axes()
        self._draw_radar_grid()
        
        # Draw sweep beam on PPI
        self._draw_sweep_beam()
        
        # Plot detections with age-based fading
        self._plot_detections(current_time)
        
        # Plot confirmed tracks
        self._plot_tracks(current_time)
        
        # Draw sensor status
        self._draw_sensor_status(sensor_info)
        
        # Update display
        self.fig.canvas.draw_idle()
        
    def _setup_axes(self):
        """Reset axis properties after clearing"""
        # PPI
        self.ax_ppi.set_xlim([-self.max_range/1000, self.max_range/1000])
        self.ax_ppi.set_ylim([-self.max_range/1000, self.max_range/1000])
        self.ax_ppi.set_aspect('equal')
        self.ax_ppi.set_facecolor('black')
        self.ax_ppi.tick_params(colors='green')
        self.ax_ppi.set_xlabel('East (km)', color='green')
        self.ax_ppi.set_ylabel('North (km)', color='green')
        self.ax_ppi.set_title('Ground Radar PPI - Plan Position Indicator', color='green')
        
        # RHI
        self.ax_rhi.set_xlim([0, self.max_range/1000])
        self.ax_rhi.set_ylim([0, 100])
        self.ax_rhi.set_facecolor('black')
        self.ax_rhi.tick_params(colors='green')
        self.ax_rhi.set_xlabel('Range (km)', color='green')
        self.ax_rhi.set_ylabel('Altitude (km)', color='green')
        self.ax_rhi.set_title('Range Height Indicator', color='green')
    
    def _draw_sweep_beam(self):
        """Draw the rotating radar beam"""
        beam_angle = math.radians(self.current_azimuth)
        beam_length = self.max_range / 1000
        
        # Main beam line
        x_beam = beam_length * math.cos(beam_angle)
        y_beam = beam_length * math.sin(beam_angle)
        self.ax_ppi.plot([0, x_beam], [0, y_beam], 'lime', linewidth=2, alpha=0.8)
        
        # Beam width (approximate 3-degree beamwidth)
        beam_width = math.radians(1.5)  # Half-width
        for offset in [-beam_width, beam_width]:
            offset_angle = beam_angle + offset
            x_edge = beam_length * math.cos(offset_angle)
            y_edge = beam_length * math.sin(offset_angle)
            self.ax_ppi.plot([0, x_edge], [0, y_edge], 'lime', linewidth=0.5, alpha=0.3)
    
    def _plot_detections(self, current_time: float):
        """Plot raw radar detections with realistic fading"""
        detection_age_limit = 30.0  # Seconds
        
        for detection in self.detections:
            age = current_time - detection['timestamp']
            if age > detection_age_limit:
                continue
                
            # Age-based alpha (fade out over time)
            alpha = max(0.1, 1.0 - age / detection_age_limit)
            
            # Position on displays
            pos = detection['position']
            x_km = pos[0] / 1000
            y_km = pos[1] / 1000
            range_km = detection['range'] / 1000
            alt_km = pos[2] / 1000
            
            # Color and size based on detection type and SNR
            if detection['type'] == 'target':
                color = 'yellow'
                size = max(10, min(50, detection['snr']))  # Size based on SNR
            elif detection['type'] == 'false_alarm':
                color = 'red'
                size = 15
            else:  # clutter
                color = 'orange'
                size = 8
                
            # Plot on PPI
            self.ax_ppi.scatter(x_km, y_km, c=color, s=size, alpha=alpha, marker='o')
            
            # Plot on RHI
            self.ax_rhi.scatter(range_km, alt_km, c=color, s=size, alpha=alpha, marker='o')
    
    def _plot_tracks(self, current_time: float):
        """Plot confirmed tracks with trails"""
        track_age_limit = 60.0  # Seconds
        
        for track_id, track_history in self.track_history.items():
            if not track_history:
                continue
                
            # Check if track is still recent
            latest = track_history[-1]
            age = current_time - latest['timestamp']
            if age > track_age_limit:
                continue
                
            color = self.track_colors[track_id % len(self.track_colors)]
            
            # Plot track trail
            positions = [point['position'] for point in track_history]
            if len(positions) > 1:
                x_trail = [pos[0]/1000 for pos in positions]
                y_trail = [pos[1]/1000 for pos in positions]
                range_trail = [np.linalg.norm(pos[:2])/1000 for pos in positions]
                alt_trail = [pos[2]/1000 for pos in positions]
                
                # PPI trail
                self.ax_ppi.plot(x_trail, y_trail, color=color, alpha=0.6, linewidth=2)
                
                # RHI trail
                self.ax_rhi.plot(range_trail, alt_trail, color=color, alpha=0.6, linewidth=2)
            
            # Current position (larger, brighter)
            current_pos = latest['position']
            x_km = current_pos[0] / 1000
            y_km = current_pos[1] / 1000
            range_km = np.linalg.norm(current_pos[:2]) / 1000
            alt_km = current_pos[2] / 1000
            
            # Track symbol (diamond for confirmed tracks)
            self.ax_ppi.scatter(x_km, y_km, c=color, s=100, marker='D', 
                              edgecolors='white', linewidth=1)
            self.ax_rhi.scatter(range_km, alt_km, c=color, s=100, marker='D',
                              edgecolors='white', linewidth=1)
            
            # Track ID label
            self.ax_ppi.text(x_km + 5, y_km + 5, f'T{track_id}', color=color, fontsize=10)
    
    def _draw_sensor_status(self, sensor_info: dict):
        """Draw sensor status information"""
        if not sensor_info:
            return
            
        # Status text in upper corners
        status_text = []
        
        if 'radar_active' in sensor_info:
            status = 'ACTIVE' if sensor_info['radar_active'] else 'STANDBY'
            status_text.append(f'Radar: {status}')
            
        if 'detection_rate' in sensor_info:
            status_text.append(f'Det Rate: {sensor_info["detection_rate"]:.1%}')
            
        if 'false_alarm_rate' in sensor_info:
            status_text.append(f'FA Rate: {sensor_info["false_alarm_rate"]:.2e}')
            
        if 'active_tracks' in sensor_info:
            status_text.append(f'Tracks: {sensor_info["active_tracks"]}')
            
        if 'weather' in sensor_info:
            status_text.append(f'Weather: {sensor_info["weather"].upper()}')
        
        # Display status text
        y_pos = 0.95
        for line in status_text:
            self.ax_ppi.text(0.02, y_pos, line, transform=self.ax_ppi.transAxes,
                           color='cyan', fontsize=10, verticalalignment='top')
            y_pos -= 0.05
            
        # Sweep indicator
        self.ax_ppi.text(0.98, 0.95, f'Az: {self.current_azimuth:03.0f}°',
                       transform=self.ax_ppi.transAxes, color='lime', fontsize=10,
                       verticalalignment='top', horizontalalignment='right')
    
    def close(self):
        """Close the radar display"""
        plt.close(self.fig)


class RadarViewer3D:
    """3D visualization showing radar coverage and sensor-based tracking"""
    
    def __init__(self, world_size: float, radar_position: np.ndarray):
        """
        Initialize 3D radar visualization
        
        Args:
            world_size: World size for display bounds
            radar_position: Position of ground radar
        """
        self.world_size = world_size
        self.radar_position = radar_position
        
        plt.ion()
        self.fig = plt.figure(figsize=(14, 10), facecolor='black')
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Dark theme
        self.ax.set_facecolor('black')
        self.ax.tick_params(colors='green')
        self.ax.set_xlabel('East (km)', color='green')
        self.ax.set_ylabel('North (km)', color='green')
        self.ax.set_zlabel('Altitude (km)', color='green')
        self.ax.set_title('3D Radar Coverage and Tracking View', color='green')
        
        # Set limits in km
        self.ax.set_xlim([0, world_size*2/1000])
        self.ax.set_ylim([0, world_size*2/1000])
        self.ax.set_zlim([0, world_size/1000])
        
        # Draw radar coverage volume
        self._draw_radar_coverage()
        
        plt.tight_layout()
        plt.show(block=False)
        
    def _draw_radar_coverage(self):
        """Draw 3D radar coverage cone"""
        # Radar position in km
        radar_km = self.radar_position / 1000
        
        # Draw radar site
        self.ax.scatter(*radar_km, c='red', s=200, marker='^', 
                       label='Ground Radar', edgecolors='white', linewidth=2)
        
        # Radar coverage cone (simplified)
        max_range_km = 400  # km
        elevation_angles = [0, 30, 60, 90]  # degrees
        
        for elev in elevation_angles:
            # Create circle at this elevation
            angles = np.linspace(0, 2*np.pi, 36)
            
            if elev == 90:  # Zenith
                continue
                
            # Range at this elevation (accounting for Earth curvature)
            horizon_range = min(max_range_km, 
                              4.12 * math.sqrt(radar_km[2] + 50*math.sin(math.radians(elev))))
            
            x_circle = radar_km[0] + horizon_range * np.cos(angles)
            y_circle = radar_km[1] + horizon_range * np.sin(angles)
            z_circle = np.full_like(x_circle, radar_km[2] + horizon_range * math.tan(math.radians(elev)))
            
            # Only plot if within world bounds
            valid_mask = (z_circle >= 0) & (z_circle <= self.world_size/1000)
            if np.any(valid_mask):
                self.ax.plot(x_circle[valid_mask], y_circle[valid_mask], z_circle[valid_mask], 
                           'darkgreen', alpha=0.3, linewidth=1)
    
    def update(self, estimated_tracks: list, own_position: np.ndarray, 
               sensor_detections: list = None):
        """
        Update 3D display with sensor-based information only
        
        Args:
            estimated_tracks: List of estimated track states
            own_position: Own platform position
            sensor_detections: Raw sensor detections
        """
        # Clear and redraw static elements
        self.ax.clear()
        self._setup_axes()
        self._draw_radar_coverage()
        
        # Plot own platform
        own_km = own_position / 1000
        self.ax.scatter(*own_km, c='blue', s=150, marker='o', 
                       label='Interceptor', edgecolors='white', linewidth=2)
        
        # Plot estimated tracks (only what sensors can see)
        for i, track in enumerate(estimated_tracks):
            if track is None:
                continue
                
            pos_km = track.position / 1000
            
            # Color based on track quality
            if track.quality.value == 'confirmed':
                color = 'lime'
                size = 100
            elif track.quality.value == 'tentative':
                color = 'yellow'
                size = 80
            else:  # coasting or lost
                color = 'orange'
                size = 60
                
            # Plot with uncertainty ellipsoid (simplified as sphere)
            pos_uncertainty = np.sqrt(np.trace(track.covariance[0:3, 0:3])) / 1000  # km
            
            self.ax.scatter(*pos_km, c=color, s=size, marker='D',
                          alpha=0.8, edgecolors='white', linewidth=1)
            
            # Uncertainty sphere (wireframe)
            if pos_uncertainty < 10:  # Only show if reasonable
                u = np.linspace(0, 2 * np.pi, 10)
                v = np.linspace(0, np.pi, 10)
                x_sphere = pos_km[0] + pos_uncertainty * np.outer(np.cos(u), np.sin(v))
                y_sphere = pos_km[1] + pos_uncertainty * np.outer(np.sin(u), np.sin(v))
                z_sphere = pos_km[2] + pos_uncertainty * np.outer(np.ones(np.size(u)), np.cos(v))
                
                self.ax.plot_wireframe(x_sphere, y_sphere, z_sphere, 
                                     color=color, alpha=0.2, linewidth=0.5)
        
        # Plot raw detections
        if sensor_detections:
            for detection in sensor_detections:
                pos_km = detection.get('position', np.zeros(3)) / 1000
                det_type = detection.get('type', 'unknown')
                
                if det_type == 'target':
                    color = 'white'
                    marker = '.'
                elif det_type == 'false_alarm':
                    color = 'red'
                    marker = 'x'
                else:
                    color = 'gray'
                    marker = '.'
                    
                self.ax.scatter(*pos_km, c=color, s=20, marker=marker, alpha=0.6)
        
        self.ax.legend()
        self.fig.canvas.draw_idle()
        
    def _setup_axes(self):
        """Reset axis properties after clearing"""
        self.ax.set_facecolor('black')
        self.ax.tick_params(colors='green')
        self.ax.set_xlabel('East (km)', color='green')
        self.ax.set_ylabel('North (km)', color='green')
        self.ax.set_zlabel('Altitude (km)', color='green')
        self.ax.set_title('3D Radar Coverage and Tracking View', color='green')
        
        self.ax.set_xlim([0, self.world_size*2/1000])
        self.ax.set_ylim([0, self.world_size*2/1000])
        self.ax.set_zlim([0, self.world_size/1000])
    
    def close(self):
        """Close the 3D display"""
        plt.close(self.fig)