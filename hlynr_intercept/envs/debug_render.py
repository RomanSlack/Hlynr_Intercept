#!/usr/bin/env python3
"""
Debug rendering to isolate visual issues
"""

import pyglet
import numpy as np
from sim3d import MissileInterceptSim
from sim3d.interface.controls import ManualControls

def minimal_render_test():
    """Test rendering with minimal elements"""
    print("Starting minimal render test...")
    
    # Create simulation
    sim = MissileInterceptSim(
        render_enabled=True,
        render_width=800,
        render_height=600,
        dt=1/60
    )
    
    # Disable everything except basic rendering
    print("Disabling complex rendering elements...")
    
    # Reset to get basic setup
    obs = sim.reset("standard")
    print(f"Setup complete - {len(obs['missiles'])} missiles")
    
    controls = ManualControls(sim)
    frame_count = 0
    
    def update_minimal(dt):
        """Minimal update function"""
        nonlocal frame_count
        
        try:
            # Just step simulation without complex logic
            obs, reward, done, info = sim.step()
            
            # Basic render
            sim.render()
            
            frame_count += 1
            
            if frame_count % 60 == 0:  # Every second
                print(f"Frame {frame_count}: {info.get('active_missiles', 0)} missiles active")
                
            if frame_count > 300:  # Stop after 5 seconds
                print("Test complete - closing")
                pyglet.app.exit()
                
        except Exception as e:
            print(f"Render error: {e}")
            pyglet.app.exit()
    
    @sim.render_engine.window.event
    def on_close():
        pyglet.app.exit()
        
    @sim.render_engine.window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            pyglet.app.exit()
    
    try:
        pyglet.clock.schedule_interval(update_minimal, sim.dt)
        pyglet.app.run()
    except Exception as e:
        print(f"App error: {e}")
    finally:
        sim.close()

if __name__ == "__main__":
    minimal_render_test()