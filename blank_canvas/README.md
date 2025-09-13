# Hlynr Intercept - Blank Canvas System

A clean, minimal implementation of the missile interception RL system with 17D radar observations, realistic physics, and comprehensive logging.

## Features

- **17D Radar Observations**: Consistent normalized observation space for RL
- **6DOF Physics**: Realistic missile and interceptor dynamics with wind effects
- **PPO Training**: Stable training with adaptive features (entropy scheduling, LR decay, clip adaptation)
- **FastAPI Inference**: Real-time inference server with safety constraints
- **Unified Logging**: Centralized timestamped logging for training, inference, and episodes
- **TensorBoard Integration**: Built-in visualization support
- **Coordinate Transforms**: ENU ↔ Unity coordinate system conversion
- **Safety Constraints**: Post-policy action clamping for safe operation

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

```bash
# Start inference server
python inference.py --model checkpoints/best --config config.yaml

# Custom host/port
python inference.py --model checkpoints/best --host 0.0.0.0 --port 8080
```

API endpoints:
- `GET /health` - Health check
- `GET /metrics` - Server metrics
- `POST /infer` - Get action from observation
- `POST /episode/end` - End episode logging
- `POST /reset` - Reset server state

### API Usage Example

```python
import requests
import numpy as np

# Prepare observation
observation = {
    "interceptor": {
        "position": [500.0, 500.0, 100.0],
        "velocity": [10.0, 10.0, 5.0],
        "orientation": [1.0, 0.0, 0.0, 0.0],  # Quaternion
        "fuel": 100.0
    },
    "missile": {
        "position": [100.0, 100.0, 400.0],
        "velocity": [100.0, 100.0, -40.0]
    },
    "radar_quality": 1.0,
    "radar_noise": 0.05
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

# Get action
action = response.json()["action"]
print(f"Thrust: {action['thrust']}")
print(f"Angular: {action['angular']}")
```

## Configuration

Main configuration in `config.yaml`:

### Environment
- Physics parameters (gravity, drag, wind)
- Spawn ranges for missiles and interceptors
- Radar settings

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
- `easy.yaml` - Simple interception, ideal conditions
- `medium.yaml` - Moderate difficulty, environmental effects
- `hard.yaml` - Challenging with evasion and poor radar

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

### 17D Observation Vector
1. Relative position to target (3D)
2. Relative velocity (3D)
3. Interceptor velocity (3D)
4. Interceptor orientation (3D euler)
5. Fuel fraction (1D)
6. Time to intercept (1D)
7. Radar lock quality (1D)
8. Closing rate (1D)
9. Off-axis angle (1D)

### Safety Constraints
- Maximum acceleration: 50 m/s²
- Maximum angular rate: 5 rad/s
- Gimbal limits: ±45°
- Fuel-based thrust limiting

### Coordinate Systems
- **ENU**: East-North-Up (right-handed)
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
- Adjust learning rate and entropy coefficient
- Increase number of parallel environments
- Check reward shaping in environment

### High inference latency
- Reduce batch size in config
- Use CPU inference for consistency
- Check logging buffer sizes

## License

Academic research use only. See main repository for license details.