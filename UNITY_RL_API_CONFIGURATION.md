# Unity RL Inference API Configuration Guide

## Overview

This document provides comprehensive guidance for configuring Unity to visualize the RL policy inference outputs from the Hlynr Intercept system. The API provides real-time control commands and diagnostics for both interceptor (blue) and threat (red) entities.

## API Architecture

### Server Endpoint
- **Base URL**: `http://localhost:5000` (configurable)
- **Main Inference Endpoint**: `POST /v1/inference`
- **Health Check**: `GET /healthz`
- **Metrics**: `GET /metrics`

### Communication Flow
1. Unity sends state data (positions, velocities, orientations) to Python server
2. Python RL policy processes state and returns control commands
3. Unity applies commands and visualizes results
4. Episode data is logged in JSONL format for replay

## API Request Format (Unity → Python)

### POST /v1/inference Request Structure
```json
{
  "meta": {
    "episode_id": "ep_000001",
    "t": 1.23,                  // Simulation time (seconds)
    "dt": 0.01,                 // Time step (seconds)
    "sim_tick": 123             // Simulation tick counter
  },
  "frames": {
    "frame": "ENU",             // Always "ENU"
    "unity_lh": true            // Set true if Unity uses left-handed coords
  },
  "blue": {                     // Interceptor state
    "pos_m": [100.0, 200.0, 50.0],        // Position [x,y,z] meters
    "vel_mps": [150.0, 10.0, -5.0],       // Velocity [vx,vy,vz] m/s
    "quat_wxyz": [0.995, 0.0, 0.1, 0.0],  // Quaternion [w,x,y,z]
    "ang_vel_radps": [0.1, 0.2, 0.05],    // Angular vel [wx,wy,wz] rad/s
    "fuel_frac": 0.75                      // Fuel remaining [0..1]
  },
  "red": {                      // Threat state
    "pos_m": [500.0, 600.0, 100.0],
    "vel_mps": [-50.0, -40.0, -10.0],
    "quat_wxyz": [0.924, 0.0, 0.0, 0.383]
  },
  "guidance": {
    "los_unit": [0.8, 0.6, 0.0],          // Line-of-sight unit vector
    "los_rate_radps": [0.01, -0.02, 0.0], // LOS rate rad/s
    "range_m": 500.0,                      // Distance to target
    "closing_speed_mps": 200.0,            // Closing speed m/s
    "fov_ok": true,                        // Target in field of view
    "g_limit_ok": true                     // Within G-force limits
  },
  "env": {
    "wind_mps": [2.0, 1.0, 0.0],          // Wind velocity (optional)
    "noise_std": 0.01,                     // Observation noise (optional)
    "episode_step": 123,                   // Current step number
    "max_steps": 1000                      // Maximum episode steps
  },
  "normalization": {
    "obs_version": "obs_v1.0",             // Observation version
    "vecnorm_stats_id": "vecnorm_baseline_001"  // Normalization ID
  }
}
```

## API Response Format (Python → Unity)

### Inference Response Structure
```json
{
  "action": {
    "rate_cmd_radps": {
      "pitch": 0.5,      // Pitch rate command (rad/s)
      "yaw": -0.2,       // Yaw rate command (rad/s)
      "roll": 0.1        // Roll rate command (rad/s)
    },
    "thrust_cmd": 0.8,   // Thrust command [0..1]
    "aux": [0.0, 0.0]    // Auxiliary outputs (optional)
  },
  "diagnostics": {
    "policy_latency_ms": 15.3,     // Policy inference time
    "obs_clip_fractions": {
      "low": 0.02,                  // Fraction clipped below
      "high": 0.01                  // Fraction clipped above
    },
    "value_estimate": 0.75          // Optional value estimate
  },
  "safety": {
    "clamped": false,               // Whether outputs were clamped
    "clamp_reason": null            // Reason if clamped
  },
  "timestamp": 1692889200.456,
  "success": true,
  "error": null
}
```

### Control Command Interpretation

#### Angular Rate Commands (Body Frame)
- **pitch**: Rotation around lateral axis (rad/s)
- **yaw**: Rotation around vertical axis (rad/s)  
- **roll**: Rotation around longitudinal axis (rad/s)
- **Range**: Typically ±10 rad/s (configurable via `rate_max_radps`)

#### Thrust Command
- **Range**: [0.0, 1.0] normalized
- **0.0**: No thrust
- **1.0**: Maximum thrust
- Maps to actual thrust via: `actual_thrust_N = thrust_cmd * max_thrust_N`

#### Safety Clamps
The system applies safety limits to prevent dangerous commands:
- Angular rates clamped to ±`rate_max_radps` (default: 10 rad/s)
- Thrust clamped to [0, 1] range
- Clamp reasons: `"rate_limit_pitch"`, `"rate_limit_yaw"`, `"rate_limit_roll"`, `"thrust_floor"`, `"thrust_ceiling"`

## Coordinate System Transformations

### ENU (Python/RL Side) - Right-Handed
- **X**: East (positive = right)
- **Y**: North (positive = forward)
- **Z**: Up (positive = upward)

### Unity - Left-Handed
- **X**: Right (corresponds to ENU X)
- **Y**: Up (corresponds to ENU Z)
- **Z**: Forward (corresponds to ENU Y)

### Transformation Matrix
```
Unity to ENU Position/Velocity:
[x_enu]   [1  0  0] [x_unity]
[y_enu] = [0  0 -1] [y_unity]
[z_enu]   [0  1  0] [z_unity]

ENU to Unity Position/Velocity:
[x_unity]   [1  0  0] [x_enu]
[y_unity] = [0  0  1] [y_enu]
[z_unity]   [0 -1  0] [z_enu]
```

**Note**: When `unity_lh: true` is set in the request, the server automatically handles coordinate transformations.

## Unity Implementation Guide

### 1. HTTP Client Setup
```csharp
public class RLAPIClient : MonoBehaviour
{
    private string apiUrl = "http://localhost:5000/v1/inference";
    private HttpClient httpClient = new HttpClient();
    
    async Task<InferenceResponse> GetRLCommand(InferenceRequest request)
    {
        var json = JsonSerializer.Serialize(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        var response = await httpClient.PostAsync(apiUrl, content);
        var responseJson = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<InferenceResponse>(responseJson);
    }
}
```

### 2. State Collection
```csharp
private InferenceRequest BuildRequest()
{
    return new InferenceRequest
    {
        Meta = new MetaInfo
        {
            EpisodeId = currentEpisodeId,
            T = Time.time,
            Dt = Time.fixedDeltaTime,
            SimTick = Time.frameCount
        },
        Frames = new FrameInfo
        {
            Frame = "ENU",
            UnityLh = true  // Unity uses left-handed coords
        },
        Blue = GetInterceptorState(),
        Red = GetThreatState(),
        Guidance = GetGuidanceInfo(),
        Env = GetEnvironmentInfo(),
        Normalization = new NormalizationInfo
        {
            ObsVersion = "obs_v1.0",
            VecnormStatsId = "vecnorm_baseline_001"
        }
    };
}
```

### 3. Command Application
```csharp
private void ApplyRLCommand(InferenceResponse response)
{
    // Apply angular rate commands
    var rateCmd = response.Action.RateCmdRadps;
    pidController.ApplyRateCommand(
        new Vector3(rateCmd.Pitch, rateCmd.Yaw, rateCmd.Roll)
    );
    
    // Apply thrust command
    thrustModel.SetThrottle(response.Action.ThrustCmd);
    
    // Check safety status
    if (response.Safety.Clamped)
    {
        Debug.LogWarning($"Commands clamped: {response.Safety.ClampReason}");
    }
    
    // Display diagnostics
    UpdateDiagnosticsUI(response.Diagnostics);
}
```

## Visualization Components

### 1. Trajectory Visualization
- Display predicted vs actual flight paths
- Show intercept point predictions
- Color-code by confidence/value estimate

### 2. Control Surface Indicators
```csharp
// Visualize control commands
void UpdateControlIndicators(ActionCommand action)
{
    pitchIndicator.value = action.RateCmdRadps.Pitch / maxRate;
    yawIndicator.value = action.RateCmdRadps.Yaw / maxRate;
    rollIndicator.value = action.RateCmdRadps.Roll / maxRate;
    thrustBar.value = action.ThrustCmd;
}
```

### 3. Performance Metrics Display
- **Latency**: Show `policy_latency_ms` in real-time
- **Clipping**: Display `obs_clip_fractions` as warning indicators
- **Value Estimate**: Show policy confidence/expected return
- **Safety Status**: Highlight when commands are clamped

### 4. Episode Replay System
```csharp
// Load and replay JSONL episode files
public class EpisodeReplay : MonoBehaviour
{
    public void LoadEpisode(string jsonlPath)
    {
        var lines = File.ReadAllLines(jsonlPath);
        foreach (var line in lines)
        {
            var frame = JsonSerializer.Deserialize<EpisodeFrame>(line);
            replayFrames.Add(frame);
        }
    }
    
    public void PlaybackFrame(int frameIndex)
    {
        var frame = replayFrames[frameIndex];
        UpdateEntityPositions(frame.Agents);
        UpdateControlVisuals(frame.Inference);
    }
}
```

## Health Monitoring

### Health Check Response
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

### Unity Health Monitor
```csharp
IEnumerator HealthCheck()
{
    while (true)
    {
        var response = await httpClient.GetAsync(healthUrl);
        var health = JsonSerializer.Deserialize<HealthResponse>(
            await response.Content.ReadAsStringAsync()
        );
        
        UpdateHealthStatus(health);
        yield return new WaitForSeconds(5.0f);
    }
}
```

## Performance Considerations

### Latency Requirements
- **Target**: < 20ms p50, < 35ms p95
- **Unity Frame Rate**: 60 Hz (16.7ms per frame)
- **Network Overhead**: ~2-5ms typical
- **Policy Inference**: ~10-15ms typical

### Optimization Tips
1. **Batch Requests**: Send multiple entity states in one request
2. **Async Operations**: Use async/await for network calls
3. **Connection Pooling**: Reuse HTTP connections
4. **Local Caching**: Cache normalization parameters
5. **Frame Skipping**: Apply commands over multiple frames if needed

## Debugging and Monitoring

### Enable Debug Logging
```bash
# Set environment variable
export HLYNR_DEBUG=true
```

### Monitor Metrics Endpoint
```bash
# Check performance metrics
curl http://localhost:5000/metrics
```

### View Episode Logs
```bash
# Episode files are in JSONL format
cat inference_episodes/ep_000001.jsonl | jq '.'
```

## Configuration Files

### Server Configuration (.env)
```bash
# Model settings
MODEL_CHECKPOINT=checkpoints/phase4_easy_final.zip
VECNORM_STATS_ID=vecnorm_baseline_001
OBS_VERSION=obs_v1.0
TRANSFORM_VERSION=tfm_v1.0

# Server settings
HOST=0.0.0.0
PORT=5000
CORS_ORIGINS=["http://localhost:3000"]

# Logging
LOG_DIR=inference_episodes
LOG_LEVEL=INFO

# Safety limits
RATE_MAX_RADPS=10.0
```

### Unity Configuration (ScriptableObject)
```csharp
[CreateAssetMenu(fileName = "RLAPIConfig", menuName = "Hlynr/API Config")]
public class RLAPIConfig : ScriptableObject
{
    public string serverUrl = "http://localhost:5000";
    public string obsVersion = "obs_v1.0";
    public string vecnormStatsId = "vecnorm_baseline_001";
    public float requestTimeout = 0.1f;  // 100ms timeout
    public bool enableDebugLogging = false;
    public bool visualizeCommands = true;
    public bool recordEpisodes = true;
}
```

## Error Handling

### Common Error Responses
```json
{
  "success": false,
  "error": "Model not loaded",
  "timestamp": 1692889200.789
}
```

### Unity Error Handler
```csharp
private void HandleAPIError(string error)
{
    Debug.LogError($"RL API Error: {error}");
    
    // Fallback to safety controller
    EnableSafetyController();
    
    // Notify UI
    ShowErrorNotification(error);
    
    // Log for analysis
    LogErrorToFile(error);
}
```

## Testing and Validation

### 1. Connection Test
```bash
# Test basic connectivity
curl -X GET http://localhost:5000/healthz
```

### 2. Inference Test
```bash
# Send test request
curl -X POST http://localhost:5000/v1/inference \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

### 3. Unity Integration Test
1. Start Python server: `python -m hlynr_bridge.server`
2. Open Unity scene with test interceptor
3. Enable API client component
4. Verify control commands are received and applied
5. Check visualization updates correctly

## Troubleshooting

### Issue: High Latency
- Check network connection
- Verify model is loaded in GPU memory
- Monitor CPU/GPU usage
- Reduce observation size if needed

### Issue: Coordinate Mismatch
- Verify `unity_lh: true` is set
- Check quaternion normalization
- Validate transform version matches

### Issue: Commands Not Applied
- Check safety clamp logs
- Verify command ranges
- Monitor error responses
- Check Unity script execution order

## References

- [Unity RL API Integration Feasibility](Unity_RL_API_Integration_Feasibility.md)
- [Unity Data Reference](UNITY_DATA_REFERENCE.md)
- [API Schemas](src/hlynr_bridge/schemas.py)
- [Coordinate Transforms](src/hlynr_bridge/transforms.py)