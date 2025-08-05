import pyglet
from pyglet.window import key
import numpy as np
from typing import Dict, Set, Optional

from .api import MissileInterceptSim


class ManualControls:
    """Manual keyboard controls for missile intercept simulation"""
    
    def __init__(self, sim: MissileInterceptSim):
        self.sim = sim
        
        # Current key states
        self.keys_pressed: Set[int] = set()
        
        # Control parameters
        self.thrust_increment = 0.05     # 5% thrust per key press
        self.control_increment = 0.1     # 10% control surface per key press
        self.camera_speed = 1000.0       # m/s
        self.camera_rotation_speed = 45.0  # deg/s
        
        # Current control state
        self.thrust_fraction = 0.0
        self.pitch_control = 0.0
        self.yaw_control = 0.0
        self.roll_control = 0.0
        
        # Control mode
        self.control_mode = "interceptor"  # "interceptor" or "camera"
        
        # Auto-pilot modes
        self.autopilot_enabled = False
        self.tracking_target = False
        
        # Setup event handlers if render engine exists
        if self.sim.render_engine:
            self._setup_event_handlers()
            
    def _setup_event_handlers(self):
        """Setup keyboard event handlers"""
        window = self.sim.render_engine.window
        
        @window.event
        def on_key_press(symbol, modifiers):
            self.keys_pressed.add(symbol)
            self._handle_key_press(symbol, modifiers)
            
        @window.event
        def on_key_release(symbol, modifiers):
            self.keys_pressed.discard(symbol)
            self._handle_key_release(symbol, modifiers)
            
    def _handle_key_press(self, symbol: int, modifiers: int):
        """Handle individual key press events"""
        
        # Mode switching
        if symbol == key.TAB:
            self.control_mode = "camera" if self.control_mode == "interceptor" else "interceptor"
            print(f"Control mode: {self.control_mode}")
            
        # Interceptor controls
        elif self.control_mode == "interceptor":
            
            # Thrust controls
            if symbol == key.SPACE:  # Main engine
                self.thrust_fraction = min(1.0, self.thrust_fraction + self.thrust_increment)
            elif symbol == key.X:    # Reduce thrust
                self.thrust_fraction = max(0.0, self.thrust_fraction - self.thrust_increment)
            elif symbol == key.Z:    # Cut engines
                self.thrust_fraction = 0.0
                
            # Control surface inputs (immediate response)
            elif symbol == key.UP:     # Pitch up
                self.pitch_control = min(1.0, self.pitch_control + self.control_increment)
            elif symbol == key.DOWN:   # Pitch down
                self.pitch_control = max(-1.0, self.pitch_control - self.control_increment)
            elif symbol == key.LEFT:   # Yaw left
                self.yaw_control = max(-1.0, self.yaw_control - self.control_increment)
            elif symbol == key.RIGHT:  # Yaw right
                self.yaw_control = min(1.0, self.yaw_control + self.control_increment)
            elif symbol == key.Q:      # Roll left
                self.roll_control = max(-1.0, self.roll_control - self.control_increment)
            elif symbol == key.E:      # Roll right
                self.roll_control = min(1.0, self.roll_control + self.control_increment)
                
            # Stage/fire control
            elif symbol == key.F:
                interceptors = self.sim.world.get_missiles_by_type("interceptor")
                if interceptors:
                    interceptors[0].stage_fire()
                    print("Stage fired!")
                    
            # Spawn attacker missile
            elif symbol == key.G:
                # Check if attacker already exists
                attackers = self.sim.world.get_missiles_by_type("attacker")
                if not attackers:
                    self.sim.world.spawn_attacker_missile()
                else:
                    print("Attacker already exists!")
                    
        # Camera controls (handled in render engine)
        elif self.control_mode == "camera":
            # Camera controls are handled by render engine
            pass
            
        # Global controls
        if symbol == key.R:  # Reset simulation
            print("Resetting simulation...")
            self.sim.reset()
            self._reset_controls()
            
        elif symbol == key.P:  # Pause/unpause
            # Toggle pause (would need pause functionality in sim)
            pass
            
        elif symbol == key.T:  # Toggle camera tracking
            if self.sim.render_engine:
                self.tracking_target = not self.tracking_target
                if self.tracking_target:
                    # Track first active missile
                    missiles = self.sim.world.get_active_missiles()
                    if missiles:
                        self.sim.render_engine.set_camera_tracking(True, missiles[0].position)
                        print("Camera tracking enabled")
                else:
                    self.sim.render_engine.set_camera_tracking(False)
                    print("Camera tracking disabled")
                    
        elif symbol == key.C:  # Cycle camera views
            if self.sim.render_engine:
                camera = self.sim.render_engine.camera
                # Predefined camera positions
                views = [
                    (0, 30, 15000),      # Behind interceptor
                    (90, 45, 20000),     # Side view
                    (180, 60, 10000),    # Front view
                    (-90, 20, 25000)     # Other side
                ]
                # Cycle through views (simplified)
                camera.set_orbital_position(*views[0])
                
        elif symbol == key.H:  # Show help
            self._show_help()
            
    def _handle_key_release(self, symbol: int, modifiers: int):
        """Handle key release events"""
        # Control surfaces return to center when keys released
        if self.control_mode == "interceptor":
            if symbol in [key.UP, key.DOWN]:
                self.pitch_control = 0.0
            elif symbol in [key.LEFT, key.RIGHT]:
                self.yaw_control = 0.0
            elif symbol in [key.Q, key.E]:
                self.roll_control = 0.0
                
    def update(self, dt: float):
        """Update control state and apply to simulation"""
        if self.control_mode == "interceptor":
            # Apply current control inputs to interceptor
            action = np.array([
                self.thrust_fraction,
                self.pitch_control,
                self.yaw_control,
                self.roll_control
            ])
            
            # Apply action through simulation API
            # Note: We don't call sim.step() here to avoid double-stepping
            self.sim._apply_action(action)
            
        # Handle continuous camera movement
        if self.control_mode == "camera" and self.sim.render_engine:
            self.sim.render_engine.handle_keyboard_input(
                {k: True for k in self.keys_pressed}, dt
            )
            
        # Update camera tracking
        if self.tracking_target and self.sim.render_engine:
            missiles = self.sim.world.get_active_missiles()
            if missiles:
                self.sim.render_engine.camera.update_tracking(missiles[0].position)
                
    def _reset_controls(self):
        """Reset all control inputs to neutral"""
        self.thrust_fraction = 0.0
        self.pitch_control = 0.0
        self.yaw_control = 0.0
        self.roll_control = 0.0
        self.keys_pressed.clear()
        
    def _show_help(self):
        """Display control help"""
        help_text = """
MANUAL CONTROL HELP:

INTERCEPTOR CONTROLS (Tab to switch modes):
  SPACE    - Increase thrust
  X        - Decrease thrust  
  Z        - Cut engines
  
  UP/DOWN  - Pitch control
  LEFT/RIGHT - Yaw control
  Q/E      - Roll control
  
  F        - Fire stage/weapon
  G        - Spawn attacker missile
  
CAMERA CONTROLS (Tab to switch modes):
  WASD     - Move camera
  Q/E      - Up/down
  Mouse    - Look around
  Scroll   - Zoom
  
GLOBAL CONTROLS:
  R        - Reset simulation
  T        - Toggle camera tracking
  C        - Cycle camera views
  H        - Show this help
  TAB      - Switch control mode
  ESC      - Exit
"""
        print(help_text)
        
    def get_status_text(self) -> str:
        """Get current control status as text"""
        if self.control_mode == "interceptor":
            return f"""INTERCEPTOR CONTROL
Thrust: {self.thrust_fraction:.2f}
Pitch: {self.pitch_control:+.2f}
Yaw: {self.yaw_control:+.2f}  
Roll: {self.roll_control:+.2f}
Mode: {self.control_mode.upper()}"""
        else:
            return f"""CAMERA CONTROL
Mode: {self.control_mode.upper()}
Tracking: {'ON' if self.tracking_target else 'OFF'}"""
            
    def enable_autopilot(self, enabled: bool):
        """Enable/disable simple autopilot"""
        self.autopilot_enabled = enabled
        if enabled:
            print("Autopilot enabled - basic guidance active")
        else:
            print("Autopilot disabled - manual control active")
            
    def update_autopilot(self):
        """Simple autopilot logic"""
        if not self.autopilot_enabled:
            return
            
        # Get current situation
        interceptors = self.sim.world.get_missiles_by_type("interceptor")
        attackers = self.sim.world.get_missiles_by_type("attacker")
        
        if not interceptors or not attackers:
            return
            
        interceptor = interceptors[0]
        attacker = attackers[0]
        
        if not interceptor.active or not attacker.active:
            return
            
        # Simple proportional navigation guidance
        rel_pos = attacker.position - interceptor.position
        rel_vel = attacker.velocity - interceptor.velocity
        
        # Calculate required acceleration for intercept
        range_to_target = np.linalg.norm(rel_pos)
        if range_to_target < 100:
            return
            
        # Time to intercept estimate
        closing_speed = -np.dot(rel_vel, rel_pos) / range_to_target
        if closing_speed <= 0:
            closing_speed = 1.0  # Avoid division by zero
            
        time_to_intercept = range_to_target / closing_speed
        
        # Required acceleration
        required_accel = 2 * rel_pos / (time_to_intercept ** 2)
        
        # Convert to control inputs (simplified)
        # This is a very basic implementation
        thrust_needed = np.linalg.norm(required_accel) / 100.0  # Scale appropriately
        self.thrust_fraction = min(1.0, thrust_needed)
        
        # Pointing control (very simplified)
        forward_dir = np.array([
            np.cos(interceptor.orientation[2]),
            np.sin(interceptor.orientation[2]),
            0
        ])
        
        target_dir = rel_pos / range_to_target
        cross_product = np.cross(forward_dir, target_dir[:2])
        
        # Simple proportional control
        self.yaw_control = np.clip(cross_product * 5.0, -1.0, 1.0)
        
    def process_events(self):
        """Process any pending window events - handled by pyglet loop now"""
        # No longer needed - pyglet handles this automatically in pyglet.app.run()
        pass
            
    def is_exit_requested(self) -> bool:
        """Check if user requested exit"""
        if self.sim.render_engine and self.sim.render_engine.window:
            return self.sim.render_engine.window.has_exit
        return key.ESCAPE in self.keys_pressed