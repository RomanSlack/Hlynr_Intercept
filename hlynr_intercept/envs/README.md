# Hlynr Intercept 3D Simulation

A production-ready 3D missile intercept simulation with GPU-accelerated rendering, realistic 6-DOF physics, and radar sensor models.

## Features

- **Real-time 3D rendering** using ModernGL and pyglet with GPU acceleration
- **6-degree-of-freedom physics** with atmospheric drag, gravity, wind, and fuel consumption
- **Realistic radar systems** including ground-based arrays and interceptor-mounted radars
- **Clean Python API** ready for RL integration with `reset()`, `step()`, `get_obs()`, `render()`
- **Manual flight control** with keyboard input for validation and testing
- **Radar scope overlay** for visual debugging of sensor data

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import sim3d; print('Installation successful!')"
```

### Basic Usage

```python
from sim3d import MissileInterceptSim

# Create simulation
sim = MissileInterceptSim(render_enabled=True)

# Reset to standard intercept scenario
obs = sim.reset("standard")

# Main simulation loop
while True:
    # Manual control or RL action
    action = [0.8, 0.1, -0.2, 0.0]  # [thrust, pitch, yaw, roll]
    
    # Step simulation
    obs, reward, done, info = sim.step(action)
    
    # Render frame
    sim.render()
    
    if done:
        break

sim.close()
```

### Manual Control Demo

```bash
# Interactive demo with keyboard controls
python demo.py manual

# API usage demo (headless)
python demo.py api

# RL integration example
python demo.py rl
```

## Control Reference

### Manual Flight Controls
- **SPACE** - Increase thrust
- **X** - Decrease thrust
- **Z** - Cut engines
- **Arrow Keys** - Pitch/Yaw control
- **Q/E** - Roll control
- **F** - Fire/Stage
- **TAB** - Switch control mode (interceptor/camera)

### Camera Controls
- **WASD** - Move camera
- **Mouse drag** - Look around
- **Mouse scroll** - Zoom
- **T** - Toggle target tracking
- **C** - Cycle camera views

### Global Controls
- **R** - Reset simulation
- **H** - Show help
- **ESC** - Exit

## API Reference

### MissileInterceptSim

```python
sim = MissileInterceptSim(
    render_enabled=True,    # Enable 3D rendering
    render_width=1200,      # Window width
    render_height=800,      # Window height
    dt=1/60                 # Physics timestep
)

# Reset to scenario
obs = sim.reset(scenario="standard")  # or "multiple_attackers"

# Step simulation
obs, reward, done, info = sim.step(action)

# Render frame
sim.render()

# Get debug info
debug_info = sim.get_debug_info()

# Cleanup
sim.close()
```

### Observation Structure

```python
obs = {
    "time": 15.3,                          # Simulation time (s)
    "missiles": [...],                     # List of missile states
    "ground_radar_contacts": [...],        # Ground radar detections
    "interceptor_radar_contacts": [...],   # Interceptor radar detections
    "threat_assessment": {...},            # Threat analysis
    "intercept_geometry": {...}            # Intercept parameters
}

# For RL algorithms, use flattened observation
obs_vector = sim.get_flattened_observation()  # numpy array
```

## Scenarios

### Standard
- Single attacker missile incoming from 30km
- Single interceptor launched from defender site
- 3 ground radar sites providing tracking

### Multiple Attackers
- 3 attacker missiles with staggered launches
- Single interceptor
- 5 ground radar sites in defensive array

## Physics Model

- **Coordinate system**: Z-up, X-forward, Y-right
- **Atmosphere**: Exponential density model with 8.4km scale height
- **Gravity**: Altitude-dependent with Earth curvature
- **Wind**: Configurable uniform wind field
- **Drag**: Speed and altitude dependent with realistic coefficients
- **Fuel**: Finite burn time with mass-dependent dynamics

## Radar Models

### Ground Radar
- **Range**: 100km maximum
- **Beam width**: 2 degrees
- **Scan rate**: 6 RPM (10 second full scan)
- **Frequency**: 3 GHz S-band
- **Detection**: SNR-based with atmospheric losses

### Interceptor Radar
- **Range**: 20km maximum  
- **Field of view**: 60 degrees
- **Update rate**: 10 Hz
- **Gimbal limits**: ±45 degrees
- **Detection**: Range and angle dependent SNR

## System Requirements

- **OS**: Ubuntu 24.04 (tested) or compatible Linux
- **GPU**: CUDA-capable RTX 40-series recommended
- **Python**: 3.8+
- **Memory**: 4GB RAM minimum
- **Storage**: 100MB for package

## Limitations

- Simplified atmospheric model (no weather effects)
- Basic terrain (flat grid)
- No electromagnetic interference
- Single-threaded physics simulation

## Integration with RL

```python
import gymnasium as gym
from stable_baselines3 import PPO

# Wrap simulation for RL training
class InterceptEnv(gym.Env):
    def __init__(self):
        self.sim = MissileInterceptSim(render_enabled=False)
        self.action_space = gym.spaces.Box(-1, 1, shape=(4,))
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(64,)
        )
    
    def reset(self):
        obs = self.sim.reset()
        return self.sim.get_flattened_observation()
    
    def step(self, action):
        obs, reward, done, info = self.sim.step(action)
        return self.sim.get_flattened_observation(), reward, done, info

# Train agent
env = InterceptEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```