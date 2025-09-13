# Hlynr Intercept RL Inference API Documentation

## Executive Summary

The Hlynr Intercept system provides a **reinforcement learning (RL) inference API** at `/v1/inference` that enables real-time missile guidance using trained PPO (Proximal Policy Optimization) neural network policies. This API serves as the critical bridge between the Python-based RL models and external visualization/simulation systems like Unity.

**Key Insight**: While the API was designed to receive state data from Unity and return control commands, the current architecture suggests Unity will **only receive and visualize** the RL policy outputs, not send data to the endpoint. The simulation state generation happens Python-side within the FastSimEnv environment.

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Python RL Environment                      │
│  ┌──────────────┐        ┌──────────────┐                   │
│  │  FastSimEnv  │───────▶│ Bridge Server│                   │
│  │  (6DOF Sim)  │        │ (/v1/inference)                  │
│  └──────────────┘        └──────────────┘                   │
│         │                        │                           │
│         │                        ▼                           │
│  ┌──────────────┐        ┌──────────────┐                   │
│  │Trained Model │◀───────│   RL Policy  │                   │
│  │(PPO Network) │        │  Inference   │                   │
│  └──────────────┘        └──────────────┘                   │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  Episode Logger   │
                    │  (JSONL Output)   │
                    └──────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │   Unity Client    │
                    │  (Visualization)  │
                    └──────────────────┘
```

## API Endpoint: `/v1/inference`

### Endpoint Details
- **URL**: `POST /v1/inference`
- **Protocol**: HTTP/JSON
- **Purpose**: Converts missile/target states into RL policy control commands
- **Latency**: ~3-5ms inference time (p50 < 20ms, p95 < 35ms target)
- **Deterministic**: Same input always produces identical output

### Request Schema (InferenceRequest)

```json
{
  "meta": {
    "episode_id": "ep_000001",      // Unique episode identifier
    "t": 1.23,                       // Simulation time (seconds)
    "dt": 0.01,                      // Timestep duration (seconds)
    "sim_tick": 123                  // Simulation tick counter
  },
  "frames": {
    "frame": "ENU",                  // Coordinate frame (always ENU)
    "unity_lh": true                 // Unity uses left-handed coords
  },
  "blue": {                          // Interceptor state
    "pos_m": [100.0, 200.0, 50.0],  // Position [x,y,z] meters
    "vel_mps": [150.0, 10.0, -5.0], // Velocity [vx,vy,vz] m/s
    "quat_wxyz": [0.995, 0.0, 0.1, 0.0], // Quaternion [w,x,y,z]
    "ang_vel_radps": [0.1, 0.2, 0.05],   // Angular velocity rad/s
    "fuel_frac": 0.75                // Fuel fraction [0..1]
  },
  "red": {                           // Threat/target state
    "pos_m": [500.0, 600.0, 100.0],
    "vel_mps": [-50.0, -40.0, -10.0],
    "quat_wxyz": [0.924, 0.0, 0.0, 0.383]
  },
  "guidance": {                      // Guidance information
    "los_unit": [0.8, 0.6, 0.0],    // Line-of-sight unit vector
    "los_rate_radps": [0.01, -0.02, 0.0], // LOS rate rad/s
    "range_m": 500.0,                // Range to target (meters)
    "closing_speed_mps": 200.0,      // Closing speed (m/s)
    "fov_ok": true,                  // Target in field of view
    "g_limit_ok": true               // Within G-force limits
  },
  "env": {                           // Environment conditions
    "wind_mps": [2.0, 1.0, 0.0],    // Wind velocity (optional)
    "noise_std": 0.01,               // Observation noise
    "episode_step": 123,             // Current step number
    "max_steps": 1000                // Maximum episode steps
  },
  "normalization": {
    "obs_version": "obs_v1.0",       // Observation format version
    "vecnorm_stats_id": "vecnorm_baseline_001" // Statistics ID
  }
}
```

### Response Schema (InferenceResponse)

```json
{
  "action": {
    "rate_cmd_radps": {              // Body rate commands
      "pitch": 0.5,                  // Pitch rate (rad/s)
      "yaw": -0.2,                   // Yaw rate (rad/s)
      "roll": 0.1                    // Roll rate (rad/s)
    },
    "thrust_cmd": 0.8,               // Thrust command [0..1]
    "aux": [0.0, 0.0]                // Auxiliary outputs
  },
  "diagnostics": {
    "policy_latency_ms": 15.3,       // Policy inference time
    "obs_clip_fractions": {
      "low": 0.02,                   // Fraction clipped below
      "high": 0.01                   // Fraction clipped above
    },
    "value_estimate": 0.75           // Optional value estimate
  },
  "safety": {
    "clamped": false,                // Were outputs clamped?
    "clamp_reason": null             // Reason if clamped
  },
  "timestamp": 1753297939.125,      // Response timestamp
  "success": true,                  // Request success
  "error": null                      // Error message if failed
}
```

## Data Flow and Processing Pipeline

### 1. State Transformation (Unity → RL)
The server transforms incoming Unity coordinates to the RL observation space:

```python
# Unity (left-handed) to ENU (right-handed) transformation
Unity.X = Python.X  # East
Unity.Y = Python.Z  # Up  
Unity.Z = Python.Y  # North

# The observation is then converted to radar-only format (17 dims):
- Interceptor essentials (4): fuel, status, target_id, distance
- Ground radar returns (4): detected_x, detected_y, confidence, range
- Onboard radar returns (3): detections, bearing, range
- Environmental data (6): wind, noise, episode progress
```

### 2. Policy Inference
The trained PPO model processes the observation:

```python
# Deterministic inference with VecNormalize
normalized_obs = vec_normalize.normalize_obs(observation)
action = model.predict(normalized_obs, deterministic=True)

# Action space (6 dimensions):
action[0] → pitch rate command
action[1] → yaw rate command  
action[2] → roll rate command
action[3] → thrust command
action[4:6] → auxiliary/reserved
```

### 3. Safety Clamping
Post-policy safety limits are applied:

```python
# Angular rate limits (default ±10 rad/s)
rate_cmd.pitch = clamp(action[0], -rate_max, rate_max)
rate_cmd.yaw = clamp(action[1], -rate_max, rate_max)
rate_cmd.roll = clamp(action[2], -rate_max, rate_max)

# Thrust limits
thrust_cmd = clamp(action[3], 0.0, 1.0)
```

### 4. Episode Logging
All inference steps are logged to JSONL format:

```json
{
  "t": 1.23,
  "request": { /* full request data */ },
  "response": { /* full response data */ },
  "inference": {
    "latency_ms": 3.4,
    "obs_version": "obs_v1.0",
    "vecnorm_stats_id": "vecnorm_baseline_001",
    "transform_version": "tfm_v1.0",
    "policy_id": "policy_123"
  }
}
```

## Unity Integration Strategy

### Current Implementation Reality
Based on the codebase analysis, the actual data flow is:

1. **Python Side**: FastSimEnv generates the full 6DOF simulation
2. **RL Policy**: Processes states and generates control commands
3. **Episode Logger**: Saves all timesteps to JSONL files
4. **Unity Side**: Reads JSONL files for visualization/replay

### Unity as Pure Visualization Client
Unity would:
1. **Read JSONL episode files** from `inference_episodes/` directory
2. **Parse timestep data** including positions, quaternions, velocities
3. **Visualize missile trajectories** using the logged states
4. **Display control commands** from the RL policy
5. **Show diagnostics** like fuel, thrust, angular rates

### JSONL Episode Format for Unity
```json
// Header (first line)
{
  "t": 0.0,
  "meta": {
    "ep_id": "ep_000001",
    "seed": 12345,
    "coord_frame": "ENU_RH",
    "scenario": "easy"
  }
}

// Timestep data
{
  "t": 1.23,
  "agents": {
    "interceptor_0": {
      "p": [100.5, 200.3, 50.0],      // Position
      "q": [1.0, 0.0, 0.0, 0.0],      // Quaternion
      "v": [10.2, 5.1, -2.0],         // Velocity
      "w": [0.0, 0.0, 0.1],           // Angular velocity
      "u": [0.5, -0.2, 0.1, 0.8, 0, 0], // Action vector
      "fuel_kg": 85.3,
      "status": "active"
    },
    "threat_0": {
      "p": [500.1, 600.2, 100.0],
      "q": [0.924, 0.0, 0.0, 0.383],
      "v": [-50.0, -40.0, -10.0],
      "status": "active"
    }
  }
}

// Summary (last line)
{
  "t": 12.37,
  "summary": {
    "outcome": "hit",
    "miss_distance_m": 2.1,
    "impact_time_s": 11.94
  }
}
```

## Key Technical Features

### 1. Determinism
- **Seeded random numbers**: Reproducible across runs
- **Fixed VecNormalize stats**: Versioned normalization parameters
- **Single-thread inference**: Prevents non-deterministic parallelism
- **Versioned transforms**: Consistent coordinate conversions

### 2. Performance Optimization
- **FastAPI ASGI server**: High-performance async framework
- **Pre-loaded models**: No loading overhead per request
- **Efficient JSON parsing**: Pydantic validation
- **Connection pooling**: For multi-missile scenarios

### 3. Safety & Validation
- **Input validation**: Pydantic schemas with bounds checking
- **Safety clamps**: Hard limits on control outputs
- **Error handling**: Graceful degradation on failures
- **Health monitoring**: `/healthz` endpoint for status

### 4. Observability
- **Prometheus metrics**: Request counts, latencies, percentiles
- **Episode logging**: Complete timestep history
- **Manifest generation**: Index of all episodes
- **Debug mode**: Verbose logging when enabled

## Supporting Endpoints

### Health Check: `GET /healthz`
```json
{
  "ok": true,
  "policy_loaded": true,
  "policy_id": "policy_123",
  "vecnorm_stats_id": "vecnorm_001",
  "obs_version": "obs_v1.0",
  "transform_version": "tfm_v1.0",
  "seed": 42
}
```

### Metrics: `GET /metrics`
```json
{
  "requests_served": 10000,
  "latency_p50_ms": 3.4,
  "latency_p95_ms": 8.2,
  "safety_clamps_total": 125,
  "requests_per_second": 115.3
}
```

## Running the Server

### Start Command
```bash
# Using uvicorn (recommended)
uvicorn hlynr_bridge.server:app --host 0.0.0.0 --port 5000

# Or with environment variables
MODEL_CHECKPOINT=checkpoints/best_model.zip \
VECNORM_STATS_ID=vecnorm_baseline_001 \
OBS_VERSION=obs_v1.0 \
TRANSFORM_VERSION=tfm_v1.0 \
uvicorn hlynr_bridge.server:app
```

### Configuration
The server uses environment variables or command-line arguments:
- `MODEL_CHECKPOINT`: Path to trained PPO model
- `VECNORM_STATS_ID`: VecNormalize statistics identifier
- `OBS_VERSION`: Observation format version
- `TRANSFORM_VERSION`: Coordinate transform version
- `RATE_MAX_RADPS`: Maximum angular rate limit (default: 10.0)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 5000)

## Unity Implementation Recommendations

### 1. Episode Reader Component
```csharp
public class RLEpisodeReader : MonoBehaviour {
    // Read JSONL files from inference_episodes/
    // Parse timestep data into Unity GameObjects
    // Apply ENU→Unity coordinate transformation
    // Interpolate between timesteps for smooth playback
}
```

### 2. Visualization Components
- **Trajectory Renderer**: Display missile paths
- **HUD Display**: Show fuel, thrust, angular rates
- **Target Indicators**: Visualize threat positions
- **Control Visualizer**: Display RL policy commands

### 3. Coordinate Transformation
```csharp
// ENU to Unity transformation
Vector3 ENUToUnity(Vector3 enuPos) {
    return new Vector3(
        enuPos.x,    // X stays the same
        enuPos.z,    // Y = ENU.Z (up)
        -enuPos.y    // Z = -ENU.Y (north with handedness flip)
    );
}
```

## Performance Characteristics

### Measured Performance
- **Inference latency**: 3.4ms average (measured)
- **JSON parsing**: <1ms overhead
- **Coordinate transform**: ~0.5ms
- **Total round-trip**: ~8ms (within 10ms target)

### Scalability
- **Single missile**: 125 Hz sustained rate
- **4 missiles**: 31 Hz per missile
- **Memory limit**: ~16 concurrent missiles

## Summary

The `/v1/inference` endpoint provides a production-ready RL inference API that:

1. **Processes missile states** into control commands using trained PPO policies
2. **Ensures deterministic** and reproducible inference results
3. **Logs complete episodes** to JSONL for Unity visualization
4. **Maintains safety** through validation and output clamping
5. **Achieves low latency** (<8ms round-trip) for real-time control

Unity's role is to **visualize the RL policy outputs** by reading the JSONL episode files generated by the Python simulation, rather than sending live state data to the API. This architecture separates the compute-intensive RL inference from the visualization layer, allowing each system to operate at its optimal performance level.