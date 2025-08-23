# Quick Test Guide for Hlynr Inference Server

## Prerequisites

Ensure you have the required dependencies installed:
```bash
pip install fastapi uvicorn stable-baselines3 aiohttp numpy
```

## Testing Procedure

### Step 1: Start the Test Client (FIRST)

Open a terminal and run the test client. It will wait for the server to start:

```bash
# From the repository root
python test_inference_client.py
```

The client will display:
```
Waiting for server to start...
  Waiting for server... (1/30)
  Waiting for server... (2/30)
```

Leave this running - it will automatically detect when the server starts.

### Step 2: Start the Inference Server (SECOND)

In a **new terminal**, start the inference server using the startup script:

```bash
# From the repository root - use the provided startup script
./start_inference_server.sh
```

**OR** manually with environment variables:

```bash
# From the repository root
cd src
export MODEL_CHECKPOINT="phase4_rl/checkpoints/phase4_easy_final.zip"
export VECNORM_STATS_ID="vecnorm_baseline_001"
export OBS_VERSION="obs_v1.0"
export TRANSFORM_VERSION="tfm_v1.0"
uvicorn hlynr_bridge.server:app --host 0.0.0.0 --port 5000 --reload
```

The server will:
1. Load the model checkpoint from `src/phase4_rl/checkpoints/`
2. Initialize the environment
3. Start listening on port 5000

### Step 3: Watch the Test Results

Once the server starts, the test client will automatically:
1. ✅ Detect the server is ready
2. ✅ Test the health endpoint
3. ✅ Send a test inference request
4. ✅ Display the response including:
   - Control commands (pitch/yaw/roll rates, thrust)
   - Diagnostics (latency, clipping)
   - **Simulation state (positions of both interceptor and threat)**
5. ✅ Run a 10-step simulation loop
6. ✅ Check metrics

## Expected Output

You should see something like:

```
✓ Server is ready!

============================================================
  Testing Health Endpoint
============================================================
Status Code: 200
Health Response:
  • Server OK: True
  • Policy Loaded: True
  • Policy ID: policy_123
  • VecNorm Stats ID: vecnorm_baseline_001

============================================================
  Testing Inference Endpoint  
============================================================
✓ Inference Response Received!

📍 Control Commands:
  • Pitch Rate: 0.234 rad/s
  • Yaw Rate: -0.156 rad/s
  • Roll Rate: 0.089 rad/s
  • Thrust: 0.750 [0-1]

🎮 Simulation State (Python → Unity):

  Interceptor (Blue):
    • Position: [100.0, 200.0, 50.0] m
    • Velocity: [150.0, 10.0, -5.0] m/s
    • Fuel: 75.00%

  Threat (Red):
    • Position: [500.0, 600.0, 100.0] m
    • Velocity: [-50.0, -40.0, -10.0] m/s

  📏 Distance: 538.5 m
```

## What This Tests

1. **Server Health**: Confirms model is loaded and ready
2. **Inference Pipeline**: Tests the full request → policy → response flow
3. **Simulation State**: Verifies both entity positions are returned
4. **Performance**: Measures latency (should be < 35ms)
5. **Safety Systems**: Checks if commands are being clamped
6. **Metrics**: Verifies performance tracking

## Troubleshooting

### Server Won't Start
- Check if port 5000 is already in use: `lsof -i :5000`
- Verify checkpoint exists: `ls src/phase4_rl/checkpoints/*.zip`
- Check Python dependencies: `pip list | grep fastapi`

### Model Not Loading
- The server needs a trained model in `src/phase4_rl/checkpoints/`
- Look for files like `phase4_easy_final.zip`
- Check server logs for specific error messages

### No Simulation State in Response
- This means the environment doesn't expose internal state
- Check server logs for "Environment doesn't expose internal state for visualization"
- The feature gracefully degrades - you'll still get control commands

### High Latency
- First request is always slower (model initialization)
- Subsequent requests should be < 35ms
- Check CPU usage and available RAM

## Using with Unity

Once you've confirmed the server works with this test client, Unity can:
1. Send the same JSON request format
2. Receive control commands AND entity positions
3. Visualize both interceptor and threat using `simulation_state`
4. No need to run physics in Unity - just display what Python tells you

## Stop the Test

- Press `Ctrl+C` in both terminals to stop
- The test client will exit gracefully
- The server will shut down cleanly