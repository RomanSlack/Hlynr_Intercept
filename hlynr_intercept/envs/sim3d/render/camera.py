import numpy as np
import math
from typing import Tuple


class Camera:
    """3D camera with orbital controls and tracking capabilities"""
    
    def __init__(self, width: int = 1200, height: int = 800):
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        
        # Camera position and orientation
        self.position = np.array([0.0, -3000.0, 2000.0])   # Start 3km behind, 2km up
        self.target = np.array([0.0, 0.0, 500.0])          # Look at 500m altitude  
        self.up = np.array([0.0, 0.0, 1.0])                # Z-up coordinate system
        
        # Projection parameters
        self.fov = math.radians(60)  # 60 degree field of view
        self.near_clip = 10.0        # 10m near clip
        self.far_clip = 50000.0      # 50km far clip (reduced from 100km)
        
        # Orbital control parameters
        self.orbital_distance = 10000.0  # 10km default distance (reduced from 20km)
        self.orbital_azimuth = 0.0       # Rotation around Z axis
        self.orbital_elevation = 30.0    # Degrees above horizon
        
        # Tracking parameters
        self.tracking_enabled = False
        self.tracking_target = None
        self.tracking_offset = np.array([0.0, -5000.0, 2000.0])  # Offset from target
        
        # Movement parameters
        self.move_speed = 1000.0     # m/s
        self.rotation_speed = 45.0   # degrees/s
        self.zoom_speed = 0.1        # zoom factor per scroll
        
    def set_orbital_position(self, azimuth: float, elevation: float, distance: float):
        """Set camera to orbital position around target"""
        self.orbital_azimuth = azimuth
        self.orbital_elevation = np.clip(elevation, -89, 89)
        self.orbital_distance = max(100, distance)
        
        # Convert to cartesian coordinates
        elev_rad = math.radians(self.orbital_elevation)
        azim_rad = math.radians(self.orbital_azimuth)
        
        # Position relative to target
        x = self.orbital_distance * math.cos(elev_rad) * math.cos(azim_rad)
        y = self.orbital_distance * math.cos(elev_rad) * math.sin(azim_rad)
        z = self.orbital_distance * math.sin(elev_rad)
        
        self.position = self.target + np.array([x, y, z])
        
    def set_tracking_target(self, target_position: np.ndarray, offset: np.ndarray = None):
        """Enable camera tracking of moving target"""
        self.tracking_enabled = True
        self.tracking_target = target_position.copy()
        if offset is not None:
            self.tracking_offset = offset.copy()
            
    def disable_tracking(self):
        """Disable target tracking"""
        self.tracking_enabled = False
        self.tracking_target = None
        
    def update_tracking(self, target_position: np.ndarray):
        """Update camera position to track target"""
        if self.tracking_enabled and target_position is not None:
            self.tracking_target = target_position.copy()
            self.target = self.tracking_target.copy()
            self.position = self.tracking_target + self.tracking_offset
            
    def move_orbital(self, d_azimuth: float, d_elevation: float, d_distance: float):
        """Move camera in orbital coordinates"""
        self.orbital_azimuth += d_azimuth
        self.orbital_elevation = np.clip(self.orbital_elevation + d_elevation, -89, 89)
        self.orbital_distance = max(100, self.orbital_distance + d_distance)
        
        self.set_orbital_position(self.orbital_azimuth, self.orbital_elevation, self.orbital_distance)
        
    def move_free(self, forward: float, right: float, up: float):
        """Move camera in world coordinates"""
        if self.tracking_enabled:
            return  # Don't allow free movement while tracking
            
        # Camera coordinate system
        view_dir = self.target - self.position
        view_dir = view_dir / np.linalg.norm(view_dir)
        right_dir = np.cross(view_dir, self.up)
        right_dir = right_dir / np.linalg.norm(right_dir)
        up_dir = np.cross(right_dir, view_dir)
        
        # Apply movement
        movement = (forward * view_dir + 
                   right * right_dir + 
                   up * up_dir) * self.move_speed
                   
        self.position += movement
        self.target += movement
        
    def look_at(self, eye: np.ndarray, target: np.ndarray, up: np.ndarray):
        """Set camera to look at target from eye position"""
        self.position = eye.copy()
        self.target = target.copy()
        self.up = up.copy()
        
    def get_view_matrix(self) -> np.ndarray:
        """Calculate view matrix (world to camera transform)"""
        # Camera coordinate system
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # View matrix
        view = np.eye(4)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = [-np.dot(right, self.position),
                       -np.dot(up, self.position), 
                       np.dot(forward, self.position)]
        
        return view.astype(np.float32)
        
    def get_projection_matrix(self) -> np.ndarray:
        """Calculate perspective projection matrix"""
        f = 1.0 / math.tan(self.fov / 2.0)
        
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / self.aspect_ratio
        proj[1, 1] = f
        proj[2, 2] = (self.far_clip + self.near_clip) / (self.near_clip - self.far_clip)
        proj[2, 3] = (2 * self.far_clip * self.near_clip) / (self.near_clip - self.far_clip)
        proj[3, 2] = -1.0
        
        return proj
        
    def get_view_projection_matrix(self) -> np.ndarray:
        """Get combined view-projection matrix"""
        return self.get_projection_matrix() @ self.get_view_matrix()
        
    def world_to_screen(self, world_pos: np.ndarray) -> Tuple[float, float, bool]:
        """Convert world position to screen coordinates"""
        # Homogeneous world position
        world_homo = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0])
        
        # Transform to clip space
        clip_pos = self.get_view_projection_matrix() @ world_homo
        
        # Check if behind camera
        if clip_pos[3] <= 0:
            return 0, 0, False
            
        # Perspective divide
        ndc_pos = clip_pos[:3] / clip_pos[3]
        
        # Check if outside view frustum
        if abs(ndc_pos[0]) > 1 or abs(ndc_pos[1]) > 1 or abs(ndc_pos[2]) > 1:
            return 0, 0, False
            
        # Convert to screen coordinates
        screen_x = (ndc_pos[0] + 1) * 0.5 * self.width
        screen_y = (1 - ndc_pos[1]) * 0.5 * self.height  # Flip Y
        
        return screen_x, screen_y, True
        
    def screen_to_ray(self, screen_x: float, screen_y: float) -> Tuple[np.ndarray, np.ndarray]:
        """Convert screen coordinates to world ray (origin, direction)"""
        # Normalize screen coordinates to [-1, 1]
        ndc_x = (screen_x / self.width) * 2 - 1
        ndc_y = 1 - (screen_y / self.height) * 2
        
        # Unproject to world space
        inv_proj = np.linalg.inv(self.get_projection_matrix())
        inv_view = np.linalg.inv(self.get_view_matrix())
        
        # Near and far points in clip space
        near_clip = np.array([ndc_x, ndc_y, -1, 1])
        far_clip = np.array([ndc_x, ndc_y, 1, 1])
        
        # Transform to world space
        near_world = inv_view @ (inv_proj @ near_clip)
        far_world = inv_view @ (inv_proj @ far_clip)
        
        near_world /= near_world[3]
        far_world /= far_world[3]
        
        # Ray origin and direction
        ray_origin = near_world[:3]
        ray_direction = far_world[:3] - near_world[:3]
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        
        return ray_origin, ray_direction
        
    def get_distance_to_target(self) -> float:
        """Get current distance from camera to target"""
        return np.linalg.norm(self.position - self.target)