import numpy as np
import pyglet
import math
from typing import List, Optional, Tuple, Dict

from .radar import RadarContact, GroundRadar, InterceptorRadar


class RadarScope:
    """Radar scope display overlay for visual debugging - SIMPLIFIED VERSION"""
    
    def __init__(self, x: int, y: int, width: int = 300, height: int = 300):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        # Scope parameters
        self.max_range = 50000  # 50km max display range
        
        # Display state
        self.show_ground_radar = True
        self.show_interceptor_radar = True
        
        # Contact display
        self.contact_fade_time = 5.0  # Seconds to fade old contacts
        self.persistent_contacts = []  # List of (contact, timestamp) tuples
        
        # Text labels for debugging
        self.radar_labels = []
        self.last_debug_print = 0
        
    def _print_radar_debug(self, current_time: float):
        """Print radar contacts to console for debugging"""
        if current_time - self.last_debug_print < 2.0:  # Print every 2 seconds
            return
            
        self.last_debug_print = current_time
        
        print("\n=== RADAR SCOPE DEBUG ===")
        ground_contacts = 0
        interceptor_contacts = 0
        
        for contact, timestamp in self.persistent_contacts:
            if current_time - timestamp < self.contact_fade_time:
                contact_type = getattr(contact, 'radar_type', 'unknown')
                if hasattr(contact, 'range') and hasattr(contact, 'bearing'):
                    range_km = contact.range / 1000.0
                    bearing_deg = math.degrees(contact.bearing)
                    print(f"  {contact_type.upper()}: Range {range_km:.1f}km, Bearing {bearing_deg:.0f}°")
                    
                    if contact_type == 'ground':
                        ground_contacts += 1
                    elif contact_type == 'interceptor':
                        interceptor_contacts += 1
                        
        print(f"Total contacts: Ground={ground_contacts}, Interceptor={interceptor_contacts}")
        print("==========================")

    def polar_to_screen(self, range_m: float, bearing_rad: float) -> Tuple[float, float]:
        """Convert polar coordinates to screen position"""
        center_x = self.x + self.width // 2
        center_y = self.y + self.height // 2
        
        # Normalize range to screen radius
        max_screen_radius = min(self.width, self.height) // 2 - 20
        screen_radius = max_screen_radius * (range_m / self.max_range)
        
        # Convert to screen coordinates (bearing 0 = north/up)
        screen_x = center_x + screen_radius * math.sin(bearing_rad)
        screen_y = center_y + screen_radius * math.cos(bearing_rad)
        
        return screen_x, screen_y
        
    def update_ground_radar(self, ground_radars: List[GroundRadar], current_time: float):
        """Update scope with ground radar contacts"""
        if not self.show_ground_radar:
            return
            
        for radar in ground_radars:
            for contact in radar.contacts:
                # Add to persistent contacts with metadata
                contact_data = (contact, current_time)
                contact.radar_type = 'ground'  # Tag for identification
                self.persistent_contacts.append(contact_data)
                
        # Clean old contacts
        self.persistent_contacts = [
            (c, t) for c, t in self.persistent_contacts 
            if current_time - t < self.contact_fade_time
        ]
        
    def update_interceptor_radar(self, interceptor_radars: List[InterceptorRadar], current_time: float):
        """Update scope with interceptor radar contacts"""
        if not self.show_interceptor_radar:
            return
            
        for radar in interceptor_radars:
            if radar.missile.active:
                for contact in radar.contacts:
                    # Add to persistent contacts with metadata
                    contact_data = (contact, current_time)
                    contact.radar_type = 'interceptor'  # Tag for identification
                    self.persistent_contacts.append(contact_data)
                    
        # Clean old contacts
        self.persistent_contacts = [
            (c, t) for c, t in self.persistent_contacts 
            if current_time - t < self.contact_fade_time
        ]
        
    def clear_contacts(self):
        """Clear all radar contacts"""
        self.persistent_contacts.clear()
        
    def render(self, current_time: float):
        """Render complete radar scope - simplified text-only version"""
        # For now, just print debug information to console
        self._print_radar_debug(current_time)
        
        # TODO: Implement proper 2D graphics overlay using pyglet shapes
        # This avoids mixing legacy OpenGL with ModernGL