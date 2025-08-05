#!/usr/bin/env python3
"""
Hlynr Intercept 3D Simulation Demo

Simple demo script showing manual control and API usage.
"""

import time
import numpy as np
import pyglet
from sim3d import MissileInterceptSim
from sim3d.interface.controls import ManualControls


def run_manual_demo():
    """Run interactive manual control demo with pyglet event loop"""
    print("Starting Hlynr Intercept 3D Simulation - Manual Control Demo")
    print("=" * 60)
    
    # Create simulation
    sim = MissileInterceptSim(
        render_enabled=True,
        render_width=1400,
        render_height=900,
        dt=1/60
    )
    
    # Setup manual controls
    controls = ManualControls(sim)
    
    # Reset to standard scenario
    print("Resetting to standard intercept scenario...")
    obs = sim.reset("standard")
    print(f"Initial observation: {len(obs['missiles'])} missiles, {len(obs['ground_radar_contacts'])} radar contacts")
    
    # Show initial help
    controls._show_help()
    
    print("\nStarting simulation loop...")
    print("Press TAB to switch between interceptor and camera control")
    print("Press H for help, R to reset, ESC to exit")
    
    # State for pyglet loop
    frame_count = 0
    
    def update_simulation(dt):
        """Update function called by pyglet scheduler"""
        nonlocal frame_count
        
        try:
            # Update controls
            controls.update(dt)
            
            # Update autopilot if enabled
            controls.update_autopilot()
            
            # Step simulation
            obs, reward, done, info = sim.step()
            
            # Render frame
            sim.render()
            
            frame_count += 1
            
            # Print status every 2 seconds (120 frames at 60fps)
            if frame_count % 120 == 0:
                active_missiles = info.get('active_missiles', 0)
                sim_time = info.get('time', 0)
                print(f"Time: {sim_time:.1f}s | Active missiles: {active_missiles}")
                
            # Check if scenario is complete
            if done:
                print(f"\nScenario complete after {info.get('time', 0):.1f} seconds!")
                print("Active missiles:", info.get('active_missiles', 0))
                
                # Reset automatically and continue
                sim.reset("standard")
                print("Scenario reset!")
                frame_count = 0
                        
        except Exception as e:
            print(f"Update error: {e}")
            pyglet.app.exit()
    
    # Set up exit handler for window close
    @sim.render_engine.window.event
    def on_close():
        print("Window close requested")
        pyglet.app.exit()
        
    # Add ESC key handler to main window
    @sim.render_engine.window.event 
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            print("ESC pressed - exiting")
            pyglet.app.exit()
    
    try:
        # Schedule the update function
        pyglet.clock.schedule_interval(update_simulation, sim.dt)
        
        # Run the pyglet application loop
        print("Starting pyglet app loop...")
        pyglet.app.run()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        
    finally:
        print("Cleaning up...")
        sim.close()
        print("Demo complete!")


def run_api_demo():
    """Run headless API demo (no rendering)"""
    print("Starting Hlynr Intercept 3D Simulation - API Demo")
    print("=" * 50)
    
    # Create headless simulation
    sim = MissileInterceptSim(render_enabled=False)
    
    # Reset simulation
    obs = sim.reset("standard")
    print(f"Initial state: {len(obs['missiles'])} missiles")
    
    # Run simple random control demo
    max_steps = 1000
    step_count = 0
    
    print("Running random control demo...")
    
    while step_count < max_steps:
        # Random action
        action = np.random.uniform(-1, 1, size=4)
        action[0] = np.random.uniform(0, 1)  # Thrust always positive
        
        # Step simulation
        obs, reward, done, info = sim.step(action)
        step_count += 1
        
        # Print progress
        if step_count % 100 == 0:
            active_missiles = info.get('active_missiles', 0)
            sim_time = info.get('time', 0)
            print(f"Step {step_count}: Time={sim_time:.1f}s, Missiles={active_missiles}, Reward={reward:.2f}")
            
        if done:
            print(f"Episode finished after {step_count} steps")
            break
            
    # Get final debug info
    debug_info = sim.get_debug_info()
    print("\nFinal state:")
    print(f"  Simulation time: {debug_info['world_state']['time']:.1f}s")
    print(f"  Active missiles: {debug_info['world_state']['active_missiles']}")
    print(f"  Episode length: {debug_info['episode_info']['length']}")
    
    sim.close()
    print("API demo complete!")


def run_rl_example():
    """Example of RL-style usage"""
    print("RL Integration Example")
    print("=" * 30)
    
    sim = MissileInterceptSim(render_enabled=False)
    
    # Example training loop structure
    episodes = 3
    
    for episode in range(episodes):
        obs = sim.reset("standard")
        episode_reward = 0.0
        step_count = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        while True:
            # Get flattened observation for RL algorithm
            obs_vector = sim.get_flattened_observation()
            
            # Dummy policy (replace with actual RL algorithm)
            action = np.random.uniform(-1, 1, size=4)
            action[0] = max(0, action[0])  # Thrust positive
            
            # Step environment
            obs, reward, done, info = sim.step(action)
            episode_reward += reward
            step_count += 1
            
            if done or step_count > 500:
                break
                
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Steps: {step_count}")
        print(f"  Final time: {info.get('time', 0):.1f}s")
        
    sim.close()
    print("RL example complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "manual":
            run_manual_demo()
        elif mode == "api":
            run_api_demo()
        elif mode == "rl":
            run_rl_example()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python demo.py [manual|api|rl]")
            sys.exit(1)
    else:
        print("Hlynr Intercept 3D Simulation")
        print("Usage: python demo.py [mode]")
        print("\nAvailable modes:")
        print("  manual  - Interactive manual control demo")
        print("  api     - Headless API usage demo") 
        print("  rl      - RL integration example")
        print("\nDefault: running manual demo...")
        run_manual_demo()