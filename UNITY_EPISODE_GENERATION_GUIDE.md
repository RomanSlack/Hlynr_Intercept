# Unity Episode Generation Guide

## Overview

This guide explains how to generate JSONL episode files for Unity visualization without needing the `/v1/inference` API endpoint. The Python simulation generates complete episodes with both interceptor and threat data that Unity can directly read and replay.

## Current System Architecture

The inference system is **self-contained** and works as follows:

1. **Python Side (FastSimEnv)** generates the full simulation:
   - Runs 6DOF physics for both interceptor and threat
   - Feeds states into the RL policy
   - Gets control commands back
   - Applies commands to interceptor
   - Logs everything to JSONL

2. **The `/v1/inference` endpoint** was designed to receive external states BUT currently the server actually:
   - Creates its own `FastSimEnv` environment internally
   - Runs the simulation loop itself
   - The request data would transform external states to RL observations
   - But the actual simulation happens Python-side

## ✅ How to Generate Unity Episodes

### 1. Use Existing Episodes (Quickest)

You already have 21 episodes ready to use:

```bash
# Episodes are here:
src/phase4_rl/my_runs/run_2025-08-04-201430/

# Each episode has ~1000 timesteps (10 seconds at 100Hz)
# Contains both interceptor and threat data
```

### 2. Generate New Episodes

#### Option A: Use the demo script
```bash
cd src/phase4_rl
python demo_episode_logging.py --section inference
```

#### Option B: Use run_inference.py
```bash
cd src/phase4_rl

# Generate 5 new episodes
python run_inference.py \
  --checkpoint checkpoints/phase4_easy_final.zip \
  --scenario easy \
  --episodes 5 \
  --enable-episode-logging
```

#### Option C: Use the simple generation script
```bash
# From the root directory
python generate_unity_episodes.py \
  --checkpoint src/phase4_rl/checkpoints/phase4_easy_final.zip \
  --scenario easy \
  --episodes 10 \
  --output-dir unity_episodes
```

### 3. Configuration Details

- **Sampling Rate**: 100Hz (0.01s timesteps) - configurable
- **Episode Length**: ~10-15 seconds (1000-1500 timesteps)
- **Coordinate System**: ENU Right-handed (needs Unity conversion)
- **Data Per Timestep**:
  - **Interceptor**: Position, quaternion, velocity, angular velocity, fuel, RL action
  - **Threat**: Position, quaternion, velocity, angular velocity

### 4. View Episode Data

```bash
# Examine a specific episode
python generate_unity_episodes.py \
  --examine src/phase4_rl/my_runs/run-2025-08-04-201430/ep_000001.jsonl

# Or manually:
head -1 ep_000001.jsonl | python -m json.tool  # Header
sed -n '2p' ep_000001.jsonl | python -m json.tool  # First timestep
tail -1 ep_000001.jsonl | python -m json.tool  # Summary
```

## How Unity Gets Both Interceptor & Attacker Data

The **JSONL episode files** contain BOTH entities' states at every timestep:

```json
{
  "t": 1.23,
  "agents": {
    "interceptor_0": {
      "p": [100.5, 200.3, 50.0],         // Position
      "q": [1.0, 0.0, 0.0, 0.0],         // Quaternion  
      "v": [10.2, 5.1, -2.0],            // Velocity
      "w": [0.0, 0.0, 0.1],              // Angular velocity
      "u": [0.5, -0.2, 0.1, 0.8, 0, 0], // RL action/commands
      "fuel_kg": 85.3,
      "status": "active"
    },
    "threat_0": {
      "p": [500.1, 600.2, 100.0],        // Position
      "q": [0.924, 0.0, 0.0, 0.383],     // Quaternion
      "v": [-50.0, -40.0, -10.0],        // Velocity  
      "w": [0.0, 0.0, 0.0],              // Angular velocity
      "status": "active"
    }
  }
}
```

## Episode File Structure

### File Organization
```
runs/
└── run_2025-08-04-HHMMSS/
    ├── manifest.json          # Episode index and metadata
    ├── ep_000001.jsonl        # Episode 1 timestep data
    ├── ep_000002.jsonl        # Episode 2 timestep data
    └── ...
```

### Manifest.json Schema
```json
{
  "schema_version": "1.0",
  "coord_frame": "ENU_RH",
  "units": {"pos": "m", "vel": "m/s", "ang": "rad", "time": "s"},
  "dt_nominal": 0.01,
  "gravity": [0.0, 0.0, -9.81],
  "episodes": [
    {
      "id": "ep_000001",
      "file": "ep_000001.jsonl",
      "seed": 12345,
      "scenario": "easy",
      "notes": "FastSimEnv training episode"
    }
  ]
}
```

### Episode File Format (JSONL)

**Line 1: Header**
```json
{
  "t": 0.0,
  "meta": {
    "ep_id": "ep_000001",
    "seed": 12345,
    "coord_frame": "ENU_RH",
    "dt_nominal": 0.01,
    "scenario": "easy"
  },
  "scene": {
    "interceptor_0": {
      "mass_kg": 10.0,
      "max_torque": [8000, 8000, 2000],
      "sensor_fov_deg": 30.0,
      "max_thrust_n": 1000.0
    },
    "threat_0": {
      "type": "ballistic",
      "mass_kg": 100.0,
      "aim_point": [0, 0, 0]
    }
  }
}
```

**Lines 2-N: Timestep Data**
```json
{
  "t": 1.23,
  "agents": {
    "interceptor_0": {
      "p": [100.5, 200.3, 0.0],         // Position [x, y, z] in meters
      "q": [1.0, 0.0, 0.0, 0.0],        // Quaternion [w, x, y, z] 
      "v": [10.2, 5.1, 0.0],            // Velocity [vx, vy, vz] in m/s
      "w": [0.0, 0.0, 0.1],             // Angular velocity [wx, wy, wz] in rad/s
      "u": [0.5, 0.3, 0.0, 0.0, 0.0, 0.8], // Action vector (6 elements)
      "fuel_kg": 85.3,                 // Fuel remaining in kg
      "status": "active"                // "active" | "destroyed" | "finished"
    },
    "threat_0": {
      "p": [500.1, 600.2, 0.0],
      "q": [0.924, 0.0, 0.0, 0.383],
      "v": [-50.0, -40.0, 0.0],
      "w": [0.0, 0.0, 0.0],
      "status": "active"
    }
  },
  "events": []                          // Optional discrete events
}
```

**Final Line: Summary**
```json
{
  "t": 12.37,
  "summary": {
    "outcome": "hit",                   // "hit" | "miss" | "timeout"
    "miss_distance_m": 2.1,            // Closest approach distance
    "impact_time_s": 11.94,            // Time of impact/closest approach
    "episode_duration": 12.37,         // Total episode time
    "notes": "Episode ended after 124 steps"
  }
}
```

## Unity Integration Steps

### 1. Read the Data
1. **Read the manifest.json** to get episode list
2. **Load an episode JSONL file** line by line
3. **Parse each timestep** to get both interceptor and threat states

### 2. Coordinate Transformation
Apply ENU to Unity coordinate conversion:

```csharp
Vector3 ENUToUnity(Vector3 enu) {
    return new Vector3(
        enu.x,    // X stays the same (East)
        enu.z,    // Y = ENU.Z (Up)
        -enu.y    // Z = -ENU.Y (North with handedness flip)
    );
}

Quaternion ENUToUnity(Quaternion enuQuat) {
    // Convert quaternion from ENU to Unity coordinates
    // Note: Quaternion order in JSON is [w,x,y,z]
    return new Quaternion(enuQuat.x, enuQuat.z, -enuQuat.y, enuQuat.w);
}
```

### 3. Unity Implementation Example

```csharp
public class EpisodeVisualizer : MonoBehaviour {
    void LoadEpisode(string jsonlPath) {
        foreach (string line in File.ReadLines(jsonlPath)) {
            var timestep = JsonConvert.DeserializeObject<Timestep>(line);
            
            if (timestep.agents != null) {
                // Update interceptor
                if (timestep.agents.interceptor_0 != null) {
                    var interceptorData = timestep.agents.interceptor_0;
                    interceptor.transform.position = ENUToUnity(new Vector3(
                        interceptorData.p[0], interceptorData.p[1], interceptorData.p[2]
                    ));
                    interceptor.transform.rotation = ENUToUnity(new Quaternion(
                        interceptorData.q[1], interceptorData.q[2], 
                        interceptorData.q[3], interceptorData.q[0]
                    ));
                    
                    // Show RL commands
                    ShowCommands(interceptorData.u);
                    UpdateFuelDisplay(interceptorData.fuel_kg);
                }
                
                // Update threat/attacker  
                if (timestep.agents.threat_0 != null) {
                    var threatData = timestep.agents.threat_0;
                    threat.transform.position = ENUToUnity(new Vector3(
                        threatData.p[0], threatData.p[1], threatData.p[2]
                    ));
                    threat.transform.rotation = ENUToUnity(new Quaternion(
                        threatData.q[1], threatData.q[2], 
                        threatData.q[3], threatData.q[0]
                    ));
                }
            }
        }
    }
}
```

### 4. Visualization Components
- **Trajectory Renderer**: Display missile paths using position history
- **HUD Display**: Show fuel, thrust, angular rates from episode data
- **Target Indicators**: Visualize threat positions and status
- **Control Visualizer**: Display RL policy commands from action vectors
- **Timeline Scrubber**: Allow user to jump to different points in episode

## Key Benefits

### ✅ Complete Data Available
The episodes contain everything Unity needs:
- ✅ Interceptor trajectory (controlled by RL policy)
- ✅ Threat trajectory (ballistic/predetermined)  
- ✅ Fuel consumption over time
- ✅ RL actions/commands at each timestep
- ✅ Episode outcomes (hit/miss/timeout)
- ✅ Physics-accurate 6DOF motion

### ✅ No API Required for Visualization
- **The `/v1/inference` endpoint** isn't needed for visualization
- Python simulation generates complete episodes offline
- Unity reads pre-computed trajectories
- Allows for pause, rewind, fast-forward functionality
- No network latency or connection issues

### ✅ Performance Optimized
- Episodes are pre-computed (no real-time inference needed)
- Unity can focus on visualization and user interface
- Smooth playback with interpolation between timesteps
- Multiple episodes can be loaded and compared

## Configuration Options

### Scenario Types
- `easy`: Basic intercept scenarios
- `medium`: More challenging scenarios  
- `hard`: Difficult intercept conditions

### Episode Parameters
- **Duration**: Configurable episode length
- **Sampling Rate**: Default 100Hz, adjustable
- **Multiple Interceptors**: Support for swarm scenarios (future)
- **Environmental Conditions**: Wind, noise, obstacles

### Output Customization
- **Coordinate Frames**: ENU, NED, or custom
- **Unit Systems**: SI units (meters/seconds) or custom
- **Data Filtering**: Choose which fields to log
- **Compression**: Optional data compression for large datasets

## Summary

The episode generation system provides a complete solution for Unity visualization:

1. **Self-contained simulation** generates both interceptor and threat trajectories
2. **Rich episode data** includes all necessary information for replay
3. **Standard format** (JSONL) is easy to parse and process
4. **No real-time dependencies** - episodes can be generated offline and played back
5. **Scalable architecture** supports multiple scenarios and difficulty levels

Unity's role is to read these pre-computed episodes and provide an engaging visual experience, with full control over playback speed, camera angles, and display options.