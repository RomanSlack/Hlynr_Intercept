import numpy as np
import pyglet
from pyglet.gl import *
import moderngl as mgl
from typing import List, Dict, Any, Optional, Tuple
import math

from .camera import Camera
from ..core.world import World


class RenderEngine:
    """High-performance 3D rendering engine using ModernGL and pyglet"""
    
    def __init__(self, width: int = 1200, height: int = 800, title: str = "Hlynr Intercept 3D"):
        self.width = width
        self.height = height
        
        # Create pyglet window
        config = pyglet.gl.Config(
            double_buffer=True,
            depth_size=24,
            samples=4  # 4x MSAA
        )
        self.window = pyglet.window.Window(
            width=width, height=height, 
            config=config, caption=title,
            resizable=True
        )
        
        # Ensure window context is current before creating ModernGL context
        self.window.switch_to()
        
        # Initialize ModernGL context with compatibility
        try:
            self.ctx = mgl.create_context(require=330)
        except Exception as e:
            print(f"Failed to create OpenGL 3.3 context: {e}")
            # Fallback to default context detection
            self.ctx = mgl.create_context()
            
        print(f"OpenGL Version: {self.ctx.version_code}")
        self.ctx.enable(mgl.DEPTH_TEST)
        self.ctx.enable(mgl.CULL_FACE)
        self.ctx.front_face = 'ccw'
        
        # Camera
        self.camera = Camera(width, height)
        
        # Rendering resources
        self.shaders = {}
        self.vertex_buffers = {}
        self.vertex_arrays = {}
        self.textures = {}
        
        # Scene objects
        self.missile_models = {}
        self.terrain_model = None
        self.skybox_model = None
        
        # UI overlay context
        self.ui_batch = pyglet.graphics.Batch()
        self.ui_labels = []
        
        # Performance tracking
        self.frame_time = 0.0
        self.fps_counter = 0
        self.fps_display = 0
        
        # Initialize rendering resources
        self._init_shaders()
        self._init_models()
        self._setup_lighting()
        
        # Event handlers
        self.setup_event_handlers()
        
    def setup_event_handlers(self):
        """Setup pyglet event handlers"""
        @self.window.event
        def on_resize(width, height):
            self.width = width
            self.height = height
            self.camera.width = width
            self.camera.height = height
            self.camera.aspect_ratio = width / height
            self.ctx.viewport = (0, 0, width, height)
            
        @self.window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            if buttons & pyglet.window.mouse.LEFT:
                # Orbital camera control
                azimuth_delta = -dx * 0.5
                elevation_delta = dy * 0.5
                self.camera.move_orbital(azimuth_delta, elevation_delta, 0)
                
        @self.window.event  
        def on_mouse_scroll(x, y, scroll_x, scroll_y):
            # Zoom control
            zoom_factor = 1000 * scroll_y
            self.camera.move_orbital(0, 0, -zoom_factor)
            
    def _init_shaders(self):
        """Initialize shader programs"""
        # Basic 3D shader (simplified without texcoords for now)
        vertex_shader_3d = '''
        #version 330 core
        in vec3 position;
        in vec3 normal;
        
        uniform mat4 mvp_matrix;
        uniform mat4 model_matrix;
        uniform mat3 normal_matrix;
        
        out vec3 world_pos;
        out vec3 world_normal;
        
        void main() {
            world_pos = (model_matrix * vec4(position, 1.0)).xyz;
            world_normal = normalize(normal_matrix * normal);
            gl_Position = mvp_matrix * vec4(position, 1.0);
        }
        '''
        
        fragment_shader_3d = '''
        #version 330 core
        in vec3 world_pos;
        in vec3 world_normal;
        
        uniform vec3 light_dir;
        uniform vec3 light_color;
        uniform vec3 ambient_color;
        uniform vec3 object_color;
        uniform vec3 camera_pos;
        
        out vec4 fragment_color;
        
        void main() {
            // Diffuse lighting
            float diff = max(dot(world_normal, -light_dir), 0.0);
            vec3 diffuse = diff * light_color;
            
            // Specular lighting
            vec3 view_dir = normalize(camera_pos - world_pos);
            vec3 reflect_dir = reflect(light_dir, world_normal);
            float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
            vec3 specular = spec * light_color * 0.3;
            
            // Combine lighting
            vec3 result = (ambient_color + diffuse + specular) * object_color;
            fragment_color = vec4(result, 1.0);
        }
        '''
        
        self.shaders['basic_3d'] = self.ctx.program(
            vertex_shader=vertex_shader_3d,
            fragment_shader=fragment_shader_3d
        )
        
        # Skybox shader
        skybox_vertex = '''
        #version 330 core
        in vec3 position;
        uniform mat4 vp_matrix;
        out vec3 tex_coords;
        
        void main() {
            tex_coords = position;
            vec4 pos = vp_matrix * vec4(position, 1.0);
            gl_Position = pos.xyww;  // Ensure skybox is always at far plane
        }
        '''
        
        skybox_fragment = '''
        #version 330 core
        in vec3 tex_coords;
        out vec4 fragment_color;
        
        void main() {
            // Simple gradient sky
            float height = normalize(tex_coords).z;
            vec3 sky_color = mix(vec3(0.5, 0.7, 1.0), vec3(0.1, 0.3, 0.8), height);
            fragment_color = vec4(sky_color, 1.0);
        }
        '''
        
        self.shaders['skybox'] = self.ctx.program(
            vertex_shader=skybox_vertex,
            fragment_shader=skybox_fragment
        )
        
    def _init_models(self):
        """Initialize 3D models and geometry"""
        # Missile model (simple cylinder with cone)
        self._create_missile_model()
        
        # Terrain grid
        self._create_terrain_model()
        
        # Skybox
        self._create_skybox_model()
        
        # Ground installations (cubes) - temporarily disabled
        # self._create_ground_cube_model()
        
    def _create_missile_model(self):
        """Create missile geometry"""
        # Simple missile shape: cylinder body + cone nose
        segments = 12
        length = 1.0  # Will be scaled per missile
        radius = 0.05
        
        vertices = []
        normals = []
        indices = []
        
        # Body cylinder
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            # Back end
            vertices.extend([x, y, 0])
            normals.extend([x/radius, y/radius, 0])
            
            # Front end  
            vertices.extend([x, y, length * 0.8])
            normals.extend([x/radius, y/radius, 0])
            
        # Nose cone tip
        vertices.extend([0, 0, length])
        normals.extend([0, 0, 1])
        
        # Generate indices for cylinder and cone
        for i in range(segments):
            # Cylinder body quads
            base = i * 2
            indices.extend([
                base, base + 1, base + 3,
                base, base + 3, base + 2
            ])
            
            # Cone triangles
            tip_index = len(vertices) // 3 - 1
            front_base = i * 2 + 1
            next_front = ((i + 1) % segments) * 2 + 1
            indices.extend([tip_index, next_front, front_base])
            
        vertices = np.array(vertices, dtype=np.float32)
        normals = np.array(normals, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        # Create buffers
        vbo = self.ctx.buffer(vertices.tobytes())
        nbo = self.ctx.buffer(normals.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        
        # Create vertex array with position and normal only
        self.vertex_arrays['missile'] = self.ctx.vertex_array(
            self.shaders['basic_3d'],
            [(vbo, '3f', 'position'), (nbo, '3f', 'normal')],
            ibo
        )
        
    def _create_terrain_model(self):
        """Create simple terrain grid"""
        size = 20000  # 20km (reduced from 50km for better precision)
        divisions = 40  # Fewer divisions for better performance
        step = size / divisions
        
        vertices = []
        indices = []
        
        # Generate grid vertices
        for i in range(divisions + 1):
            for j in range(divisions + 1):
                x = -size/2 + i * step
                y = -size/2 + j * step
                z = 0  # Flat terrain for now
                vertices.extend([x, y, z])
                
        # Generate indices for grid lines
        for i in range(divisions + 1):
            for j in range(divisions):
                # Horizontal lines
                base = i * (divisions + 1) + j
                indices.extend([base, base + 1])
                
        for i in range(divisions):
            for j in range(divisions + 1):
                # Vertical lines  
                base = i * (divisions + 1) + j
                indices.extend([base, base + divisions + 1])
                
        vertices = np.array(vertices, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        
        # Create dummy normals for terrain
        normals = np.tile([0, 0, 1], len(vertices) // 3).astype(np.float32)  # All point up
        
        vbo = self.ctx.buffer(vertices.tobytes())
        nbo = self.ctx.buffer(normals.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        
        self.vertex_arrays['terrain'] = self.ctx.vertex_array(
            self.shaders['basic_3d'],
            [(vbo, '3f', 'position'), (nbo, '3f', 'normal')],
            ibo
        )
        
    def _create_skybox_model(self):
        """Create skybox cube"""
        vertices = np.array([
            # Front face
            -1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1,
            # Back face  
            -1, -1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,
            # Top face
            -1,  1, -1, -1,  1,  1,  1,  1,  1,  1,  1, -1,
            # Bottom face
            -1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,
            # Right face
             1, -1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1,
            # Left face
            -1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1, -1
        ], dtype=np.float32)
        
        indices = np.array([
            0, 1, 2, 2, 3, 0,      # Front
            4, 5, 6, 6, 7, 4,      # Back  
            8, 9, 10, 10, 11, 8,   # Top
            12, 13, 14, 14, 15, 12, # Bottom
            16, 17, 18, 18, 19, 16, # Right
            20, 21, 22, 22, 23, 20  # Left
        ], dtype=np.uint32)
        
        vbo = self.ctx.buffer(vertices.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        
        self.vertex_arrays['skybox'] = self.ctx.vertex_array(
            self.shaders['skybox'],
            [(vbo, '3f', 'position')],
            ibo
        )
        
    def _create_ground_cube_model(self):
        """Create simple cube for ground installations"""
        # Cube vertices (50m cube)
        s = 1.0  # Unit cube, will be scaled in model matrix
        vertices = np.array([
            # Front face
            -s, -s,  s,  s, -s,  s,  s,  s,  s, -s,  s,  s,
            # Back face
            -s, -s, -s, -s,  s, -s,  s,  s, -s,  s, -s, -s,
            # Top face
            -s,  s, -s, -s,  s,  s,  s,  s,  s,  s,  s, -s,
            # Bottom face
            -s, -s, -s,  s, -s, -s,  s, -s,  s, -s, -s,  s,
            # Right face
             s, -s, -s,  s,  s, -s,  s,  s,  s,  s, -s,  s,
            # Left face
            -s, -s, -s, -s, -s,  s, -s,  s,  s, -s,  s, -s
        ], dtype=np.float32)
        
        # Normals for each face
        normals = np.array([
            # Front face (z+)
            0, 0, 1,  0, 0, 1,  0, 0, 1,  0, 0, 1,
            # Back face (z-)
            0, 0, -1,  0, 0, -1,  0, 0, -1,  0, 0, -1,
            # Top face (y+)
            0, 1, 0,  0, 1, 0,  0, 1, 0,  0, 1, 0,
            # Bottom face (y-)
            0, -1, 0,  0, -1, 0,  0, -1, 0,  0, -1, 0,
            # Right face (x+)
            1, 0, 0,  1, 0, 0,  1, 0, 0,  1, 0, 0,
            # Left face (x-)
            -1, 0, 0,  -1, 0, 0,  -1, 0, 0,  -1, 0, 0
        ], dtype=np.float32)
        
        indices = np.array([
            0, 1, 2, 2, 3, 0,      # Front
            4, 5, 6, 6, 7, 4,      # Back
            8, 9, 10, 10, 11, 8,   # Top
            12, 13, 14, 14, 15, 12, # Bottom
            16, 17, 18, 18, 19, 16, # Right
            20, 21, 22, 22, 23, 20  # Left
        ], dtype=np.uint32)
        
        # Create buffers
        vbo = self.ctx.buffer(vertices.tobytes())
        nbo = self.ctx.buffer(normals.tobytes())
        ibo = self.ctx.buffer(indices.tobytes())
        
        # Create vertex array
        self.vertex_arrays['ground_cube'] = self.ctx.vertex_array(
            self.shaders['basic_3d'],
            [(vbo, '3f', 'position'), (nbo, '3f', 'normal')],
            ibo
        )
        
    def _setup_lighting(self):
        """Setup scene lighting parameters"""
        self.light_direction = np.array([0.3, -0.5, -0.8])  # Directional light
        self.light_direction = self.light_direction / np.linalg.norm(self.light_direction)
        self.light_color = np.array([1.0, 0.95, 0.8])       # Warm sunlight
        self.ambient_color = np.array([0.2, 0.2, 0.25])     # Cool ambient
        
    def render_frame(self, world: World):
        """Render complete frame"""
        # Clear buffers
        self.ctx.clear(0.1, 0.1, 0.15)  # Dark blue background
        
        # Update camera if tracking
        missiles = world.get_active_missiles()
        if missiles and self.camera.tracking_enabled:
            # Track first active missile
            self.camera.update_tracking(missiles[0].position)
            
        # Get camera matrices
        view_matrix = self.camera.get_view_matrix()
        proj_matrix = self.camera.get_projection_matrix()
        vp_matrix = proj_matrix @ view_matrix
        
        # Render skybox first
        self._render_skybox(vp_matrix)
        
        # Render terrain (temporarily disabled for debugging)
        # self._render_terrain(vp_matrix)
        
        # Render missiles
        for missile in missiles:
            self._render_missile(missile, vp_matrix)
            
        # Render ground sites (temporarily disabled for debugging)
        # for site in world.ground_sites:
        #     if site['active']:
        #         self._render_ground_site(site, vp_matrix)
                
        # Update UI
        self._update_ui(world)
        
        # Swap buffers
        self.window.flip()
        
    def _render_skybox(self, vp_matrix: np.ndarray):
        """Render skybox"""
        self.ctx.disable(mgl.DEPTH_TEST)
        
        shader = self.shaders['skybox']
        shader['vp_matrix'].write(vp_matrix.tobytes())
        
        self.vertex_arrays['skybox'].render()
        self.ctx.enable(mgl.DEPTH_TEST)
        
    def _render_terrain(self, vp_matrix: np.ndarray):
        """Render terrain grid"""
        shader = self.shaders['basic_3d']
        
        # Identity model matrix
        model_matrix = np.eye(4, dtype=np.float32)
        mvp_matrix = vp_matrix @ model_matrix
        normal_matrix = np.eye(3, dtype=np.float32)
        
        # Set uniforms
        shader['mvp_matrix'].write(mvp_matrix.tobytes())
        shader['model_matrix'].write(model_matrix.tobytes())
        shader['normal_matrix'].write(normal_matrix.tobytes())
        shader['light_dir'].write(self.light_direction.astype(np.float32).tobytes())
        shader['light_color'].write(self.light_color.astype(np.float32).tobytes())
        shader['ambient_color'].write(self.ambient_color.astype(np.float32).tobytes())
        shader['object_color'].write(np.array([0.3, 0.7, 0.3], dtype=np.float32).tobytes())
        shader['camera_pos'].write(self.camera.position.astype(np.float32).tobytes())
        
        # Render as wireframe
        self.ctx.wireframe = True
        self.vertex_arrays['terrain'].render(mgl.LINES)
        self.ctx.wireframe = False
        
    def _render_missile(self, missile, vp_matrix: np.ndarray):
        """Render individual missile"""
        if not missile.active:
            return
            
        shader = self.shaders['basic_3d']
        
        # Create model matrix from missile state
        pos = missile.position.astype(np.float32)
        orient = missile.orientation
        
        # Get full 3-axis rotation matrix from physics
        if not hasattr(self, '_physics_helper'):
            from ..core.physics import Physics6DOF
            self._physics_helper = Physics6DOF()
            
        # Full rotation matrix from roll, pitch, yaw
        rotation_3x3 = self._physics_helper.rotation_matrix_from_euler(
            orient[0], orient[1], orient[2]  # roll, pitch, yaw
        ).astype(np.float32)
        
        # Scale matrix for missile geometry
        scale_matrix = np.diag([
            missile.length * 0.05,  # radius scale
            missile.length * 0.05,  # radius scale  
            missile.length          # length scale
        ]).astype(np.float32)
        
        # Combined rotation and scale
        rotation_scale = rotation_3x3 @ scale_matrix
        
        # Build 4x4 model matrix
        model_matrix = np.eye(4, dtype=np.float32)
        model_matrix[:3, :3] = rotation_scale
        model_matrix[:3, 3] = pos
        
        mvp_matrix = vp_matrix @ model_matrix
        # Proper normal matrix: inverse transpose of the rotation part only (no scale)
        rotation_only = rotation_3x3
        normal_matrix = np.linalg.inv(rotation_only).T
        
        # Set uniforms
        shader['mvp_matrix'].write(mvp_matrix.tobytes())
        shader['model_matrix'].write(model_matrix.tobytes()) 
        shader['normal_matrix'].write(normal_matrix.tobytes())
        shader['light_dir'].write(self.light_direction.astype(np.float32).tobytes())
        shader['light_color'].write(self.light_color.astype(np.float32).tobytes())
        shader['ambient_color'].write(self.ambient_color.astype(np.float32).tobytes())
        shader['camera_pos'].write(self.camera.position.astype(np.float32).tobytes())
        
        # Color based on missile type
        if missile.type == "attacker":
            color = np.array([1.0, 0.3, 0.3], dtype=np.float32)  # Red
        else:
            color = np.array([0.3, 0.3, 1.0], dtype=np.float32)  # Blue
            
        shader['object_color'].write(color.tobytes())
        
        # Render missile
        self.vertex_arrays['missile'].render()
        
    def _render_ground_site(self, site: Dict, vp_matrix: np.ndarray):
        """Render ground installation"""
        if not site.get('active', True):
            return
            
        shader = self.shaders['basic_3d']
        pos = site['position'].astype(np.float32)
        
        # Create model matrix (50m cube)
        model_matrix = np.eye(4, dtype=np.float32)
        model_matrix[:3, 3] = pos
        model_matrix[:3, :3] *= 50.0  # 50m cube scale
        
        mvp_matrix = vp_matrix @ model_matrix
        normal_matrix = np.eye(3, dtype=np.float32)  # No rotation, so identity is fine
        
        # Set uniforms
        shader['mvp_matrix'].write(mvp_matrix.tobytes())
        shader['model_matrix'].write(model_matrix.tobytes())
        shader['normal_matrix'].write(normal_matrix.tobytes())
        shader['light_dir'].write(self.light_direction.astype(np.float32).tobytes())
        shader['light_color'].write(self.light_color.astype(np.float32).tobytes())
        shader['ambient_color'].write(self.ambient_color.astype(np.float32).tobytes())
        shader['camera_pos'].write(self.camera.position.astype(np.float32).tobytes())
        
        # Color based on site type
        if site.get('type') == 'radar':
            color = np.array([0.2, 1.0, 0.2], dtype=np.float32)  # Bright green for radar
        else:
            color = np.array([0.8, 0.8, 0.8], dtype=np.float32)  # Gray for other
            
        shader['object_color'].write(color.tobytes())
        
        # Render cube
        self.vertex_arrays['ground_cube'].render()
        
    def _update_ui(self, world: World):
        """Update UI overlay"""
        # Clear previous labels
        for label in self.ui_labels:
            label.delete()
        self.ui_labels.clear()
        
        # Status information
        active_missiles = world.get_active_missiles()
        attackers = [m for m in active_missiles if m.type == "attacker"]
        interceptors = [m for m in active_missiles if m.type == "interceptor"]
        
        y_pos = self.height - 30
        
        # Simulation info
        info_text = f"Time: {world.time:.1f}s | Attackers: {len(attackers)} | Interceptors: {len(interceptors)}"
        label = pyglet.text.Label(
            info_text, font_name='Arial', font_size=14,
            x=10, y=y_pos, color=(255, 255, 255, 255)
        )
        self.ui_labels.append(label)
        y_pos -= 25
        
        # Camera info
        cam_dist = self.camera.get_distance_to_target() / 1000  # km
        cam_text = f"Camera: {cam_dist:.1f}km | Elev: {self.camera.orbital_elevation:.0f}° | Az: {self.camera.orbital_azimuth:.0f}°"
        label = pyglet.text.Label(
            cam_text, font_name='Arial', font_size=12,
            x=10, y=y_pos, color=(200, 200, 200, 255)
        )
        self.ui_labels.append(label)
        
        # Draw labels
        for label in self.ui_labels:
            label.draw()
            
    def set_camera_tracking(self, enabled: bool, target_pos: Optional[np.ndarray] = None):
        """Enable/disable camera tracking"""
        if enabled and target_pos is not None:
            self.camera.set_tracking_target(target_pos)
        else:
            self.camera.disable_tracking()
            
    def handle_keyboard_input(self, keys_pressed: Dict[int, bool], dt: float):
        """Handle camera movement from keyboard input"""
        # Calculate normalized movement directions (-1 to 1)
        forward = 0
        right = 0  
        up = 0
        
        if keys_pressed.get(pyglet.window.key.W, False):
            forward += 1
        if keys_pressed.get(pyglet.window.key.S, False):
            forward -= 1
        if keys_pressed.get(pyglet.window.key.A, False):
            right -= 1
        if keys_pressed.get(pyglet.window.key.D, False):
            right += 1
        if keys_pressed.get(pyglet.window.key.Q, False):
            up += 1
        if keys_pressed.get(pyglet.window.key.E, False):
            up -= 1
            
        if forward != 0 or right != 0 or up != 0:
            # Pass normalized direction and let camera scale by dt and move_speed
            self.camera.move_free(forward * dt, right * dt, up * dt)
            
        # Arrow key camera rotation
        rotation_speed = self.camera.rotation_speed * dt
        
        if keys_pressed.get(pyglet.window.key.UP, False):
            self.camera.move_orbital(0, rotation_speed, 0)
        if keys_pressed.get(pyglet.window.key.DOWN, False):
            self.camera.move_orbital(0, -rotation_speed, 0)
        if keys_pressed.get(pyglet.window.key.LEFT, False):
            self.camera.move_orbital(-rotation_speed, 0, 0)
        if keys_pressed.get(pyglet.window.key.RIGHT, False):
            self.camera.move_orbital(rotation_speed, 0, 0)
            
    def cleanup(self):
        """Cleanup rendering resources"""
        for vao in self.vertex_arrays.values():
            vao.release()
        for shader in self.shaders.values():
            shader.release()
        self.ctx.release()