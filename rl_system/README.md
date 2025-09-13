# Hlynr Intercept - Blank Canvas System

A realistic missile interception RL system with **radar-only observations**, 6DOF physics, and production-ready deployment capabilities. The interceptor has **no direct knowledge** of missile positions and must rely entirely on simulated radar sensors, just like real-world missile defense systems.

## Features

- **🎯 Radar-Only Observations**: 17D sensor-based observation space with realistic limitations
- **📡 Authentic Radar Physics**: Range limits, beam width, noise, detection failures
- **🚀 6DOF Missile Dynamics**: Physics based on PAC-3/THAAD interceptor specifications  
- **🧠 PPO Training**: Stable training with adaptive features (entropy scheduling, LR decay, clip adaptation)
- **⚡ FastAPI Inference**: Real-time inference server with safety constraints
- **📊 Unified Logging**: Centralized timestamped logging for training, inference, and episodes
- **📈 TensorBoard Integration**: Built-in visualization support
- **🔄 Coordinate Transforms**: ENU ↔ Unity coordinate system conversion
- **🛡️ Safety Constraints**: Post-policy action clamping for safe operation

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train with default configuration
python train.py

# Train with custom config
python train.py --config my_config.yaml
```

Training will create timestamped logs in `logs/training_YYYYMMDD_HHMMSS/` containing:
- TensorBoard logs
- Model checkpoints
- Training metrics
- Episode data

### Inference

**Server Mode (Real-time API):**
```bash
# Start inference server
python inference.py --model checkpoints/best --config config.yaml --mode server

# Custom host/port
python inference.py --model checkpoints/best --host 0.0.0.0 --port 8080 --mode server
```

**Offline Mode (Batch evaluation with JSON export):**
```bash
# Run 100 episodes and save results
python inference.py --model checkpoints/best --mode offline --episodes 100

# Test on specific scenario
python inference.py --model checkpoints/best --mode offline --scenario hard --episodes 50
```

API endpoints (server mode):
- `GET /health` - Health check
- `GET /metrics` - Server metrics  
- `POST /infer` - Get action from radar observation
- `POST /episode/end` - End episode logging
- `POST /reset` - Reset server state

### API Usage Example

```python
import requests
import numpy as np

# Prepare observation (interceptor state + radar parameters)
# NOTE: You only provide missile state for radar simulation - 
# the interceptor has no direct knowledge of missile position!
observation = {
    "interceptor": {
        "position": [500.0, 500.0, 100.0],      # Perfect self-knowledge
        "velocity": [10.0, 10.0, 5.0],          # Internal sensors
        "orientation": [1.0, 0.0, 0.0, 0.0],    # IMU/GPS
        "fuel": 100.0                           # Fuel gauge
    },
    "missile": {
        "position": [100.0, 100.0, 400.0],      # True position (for radar simulation)
        "velocity": [100.0, 100.0, -40.0]       # True velocity (for radar simulation)
    },
    "radar_quality": 0.9,    # Environmental conditions
    "radar_noise": 0.05      # Measurement uncertainty
}

# Make inference request
response = requests.post(
    "http://localhost:8000/infer",
    json={
        "observation": observation,
        "coordinate_system": "ENU",
        "episode_id": "test_001"
    }
)

# Get action (with safety info)
result = response.json()
action = result["action"]
safety = result["safety"]

print(f"Thrust: {action['thrust']}")
print(f"Angular: {action['angular']}")
print(f"Safety clamped: {safety['clamped']}")
```

## Configuration

Main configuration in `config.yaml`:

### Environment
- Physics parameters (gravity, drag, wind)
- Spawn ranges for missiles and interceptors
- **Radar system** (range, beam width, noise, detection limits)

### Training
- PPO hyperparameters
- Network architecture
- Adaptive features (entropy/LR scheduling)
- Checkpoint frequency

### Inference
- API settings
- Coordinate system defaults
- Performance limits

### Logging
- Log directory structure
- Metrics intervals
- TensorBoard settings

## Scenarios

Pre-configured difficulty levels in `scenarios/`:
- `easy.yaml` - Simple interception, wide radar beam, low noise
- `medium.yaml` - Moderate difficulty, standard radar, environmental effects  
- `hard.yaml` - Challenging with narrow beam, high noise, and missile evasion

Load scenarios by modifying environment config or passing to training.

## Architecture

```
blank_canvas/
├── core.py           # 17D observations, transforms, safety
├── environment.py    # Gymnasium environment with physics
├── train.py         # PPO training with callbacks
├── inference.py     # FastAPI server for deployment
├── logger.py        # Unified logging system
├── config.yaml      # Main configuration
├── scenarios/       # Difficulty presets
└── requirements.txt # Dependencies
```

## Logging Structure

All logs use timestamped directories:

```
logs/
└── run_YYYYMMDD_HHMMSS/
    ├── system.log          # Python logging output
    ├── metrics.jsonl       # Performance metrics
    ├── training.jsonl      # Training progress
    ├── inference.jsonl     # Inference requests
    ├── episodes/           # Individual episode logs
    ├── tensorboard/        # TensorBoard events
    └── manifest.json       # Run metadata
```

## Key Components

### 17D Radar Observation Vector

**Target-Related (Radar-Dependent - Zero when not detected):**
1. **[0-2]** Relative position to target (3D, range-normalized)
2. **[3-5]** Relative velocity (3D, radar doppler)
3. **[13]** Time to intercept estimate (computed from radar data)
4. **[14]** Radar lock quality (0=no lock, 1=perfect)
5. **[15]** Closing rate (from radar measurements)
6. **[16]** Off-axis angle (target bearing vs. interceptor heading)

**Self-State (Perfect Internal Knowledge):**
7. **[6-8]** Interceptor velocity (3D, internal sensors)
8. **[9-11]** Interceptor orientation (3D euler, IMU)
9. **[12]** Fuel fraction (internal gauge)

**🎯 Key Insight**: When radar loses lock, target observations become zero - the agent must learn to handle detection failures!

### Radar System Limitations
- **Maximum range**: 5000m (configurable per scenario)  
- **Beam width**: 60° detection cone (must point at target)
- **Range-dependent noise**: Accuracy degrades with distance
- **Detection failures**: Signal loss in poor conditions or at extreme range
- **No omniscience**: Realistic sensor physics only - no cheating!

### Safety Constraints
- Maximum acceleration: 50 m/s² (5G)
- Maximum angular rate: 5 rad/s (285°/s)
- Gimbal limits: ±45° thrust vector control
- Fuel-based thrust limiting

### Coordinate Systems
- **ENU**: East-North-Up (aerospace standard)
- **Unity**: Game engine convention (left-handed)
- Automatic conversion in inference API

## Performance

- Training: ~100k steps/hour on 8 CPU cores
- Inference: <10ms latency per request
- Logging: Minimal overhead with buffered writes

## Troubleshooting

### Model not loading
- Check checkpoint path exists
- Verify `model.zip` or `best_model.zip` present
- Ensure `vec_normalize.pkl` is in same directory

### Poor training performance
- **Start with easy scenario** - wide radar beam for initial learning
- Adjust learning rate and entropy coefficient
- Increase number of parallel environments  
- **Check radar detection rate** - agent needs successful acquisitions to learn
- Verify reward shaping encourages target pursuit

### High inference latency
- Reduce batch size in config
- Use CPU inference for consistency
- Check logging buffer sizes

## License

Academic research use only. See main repository for license details.