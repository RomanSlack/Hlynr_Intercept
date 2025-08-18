# Hlynr Unity↔RL Inference API Bridge Server v1.0

A production-ready Unity↔RL Inference API Bridge implementing the PRP specification for serving low-latency RL policy control to Unity environments.

## Features

- **v1.0 API**: Pydantic-validated request/response schemas with versioning
- **Deterministic Transforms**: ENU↔Unity coordinate system transforms with versioning (tfm_v1.0)
- **VecNormalize Integration**: Versioned observation normalization statistics tracking
- **Safety Clamps**: Post-policy safety limits for angular rates and thrust commands
- **Episode Logging**: JSONL episode logs with manifest generation for Unity replay
- **Performance Monitoring**: Latency tracking (p50/p95/p99) and comprehensive metrics
- **Environment Configuration**: Full environment variable support with .env file loading

## Quick Start

### 1. Environment Setup

Copy the example environment file and configure it:

```bash
cp .env.example .env
# Edit .env with your model path and configuration
```

Required environment variables:
- `MODEL_CHECKPOINT`: Path to your trained PPO model (required)

### 2. Run the Server

Using environment variables (recommended):
```bash
python -m src.phase4_rl.hlynr_bridge_server
```

Using command-line arguments:
```bash
python -m src.phase4_rl.hlynr_bridge_server \
  --checkpoint /path/to/model.zip \
  --host 0.0.0.0 \
  --port 5000 \
  --vecnorm-stats-id your_vecnorm_id
```

### 3. Test the API

Health check:
```bash
curl http://localhost:5000/healthz
```

Metrics:
```bash
curl http://localhost:5000/metrics
```

## Configuration

### Environment Variables

The server supports comprehensive configuration via environment variables. See `.env.example` for all available options:

```bash
# Required
MODEL_CHECKPOINT=/path/to/model.zip

# Optional with defaults
HOST=0.0.0.0
PORT=5000
VECNORM_STATS_ID=your_stats_id
OBS_VERSION=obs_v1.0
TRANSFORM_VERSION=tfm_v1.0
CORS_ORIGINS=*
LOG_DIR=inference_episodes
RATE_MAX_RADPS=10.0
```

### Command Line Arguments

All environment variables can be overridden via command-line arguments:

```bash
python -m src.phase4_rl.hlynr_bridge_server --help
```

Print current configuration:
```bash
python -m src.phase4_rl.hlynr_bridge_server --print-config
```

## API Endpoints

### POST /v1/inference

Main inference endpoint implementing the v1.0 API specification.

**Request Schema:**
```json
{
  "meta": {
    "t": 1.234,
    "episode_id": "ep_001"
  },
  "frames": {
    "unity_lh": true
  },
  "blue": {
    "pos_m": [100, 200, 5000],
    "vel_mps": [0, 0, -100],
    "quat_wxyz": [1, 0, 0, 0],
    "ang_vel_radps": [0, 0, 0],
    "fuel_frac": 0.8
  },
  "red": {
    "pos_m": [0, 0, 0],
    "vel_mps": [0, 0, 0],
    "quat_wxyz": [1, 0, 0, 0]
  },
  "guidance": {
    "los_unit": [0, 0, -1],
    "los_rate_radps": [0, 0],
    "range_m": 5000,
    "closing_speed_mps": 100,
    "fov_ok": true,
    "g_limit_ok": true
  },
  "env": {
    "episode_step": 100,
    "max_steps": 1000,
    "wind_mps": [0, 0, 0]
  },
  "normalization": {
    "obs_version": "obs_v1.0",
    "vecnorm_stats_id": "your_stats_id",
    "transform_version": "tfm_v1.0"
  }
}
```

**Response Schema:**
```json
{
  "action": {
    "rate_cmd_radps": {
      "pitch": 0.1,
      "yaw": 0.05,
      "roll": 0.0
    },
    "thrust_cmd": 0.7,
    "aux": []
  },
  "diagnostics": {
    "policy_latency_ms": 12.5,
    "obs_clip_fractions": {
      "low": 0.0,
      "high": 0.02
    },
    "value_estimate": null
  },
  "safety": {
    "clamped": false,
    "clamp_reason": null
  }
}
```

### GET /healthz

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "loaded_models": {
    "ppo_policy": "/path/to/model.zip"
  },
  "vecnorm_stats_loaded": "your_stats_id",
  "transform_version": "tfm_v1.0"
}
```

### GET /metrics

Performance metrics endpoint.

**Response:**
```json
{
  "requests_served": 1000,
  "requests_failed": 5,
  "latency_p50_ms": 15.2,
  "latency_p95_ms": 28.7,
  "latency_p99_ms": 45.1,
  "latency_mean_ms": 18.3,
  "requests_per_second": 12.5,
  "safety_clamps_total": 23,
  "safety_clamp_rate": 0.023,
  "model_loaded_at": 1692123456.789,
  "vecnorm_stats_id": "your_stats_id"
}
```

## Performance

The server is designed to meet the following latency SLOs:
- **p50 < 20ms**: 50th percentile end-to-end latency
- **p95 < 35ms**: 95th percentile end-to-end latency

Configure latency targets via environment variables:
```bash
LATENCY_SLO_P50_MS=20.0
LATENCY_SLO_P95_MS=35.0
```

## Safety Features

### Angular Rate Limits
- Configurable maximum angular rates via `RATE_MAX_RADPS` (default: 10.0 rad/s)
- Hard clamps applied to pitch, yaw, and roll commands
- Safety violations logged with detailed statistics

### Thrust Limits
- Thrust commands clamped to [0, 1] range
- Safety clamp events tracked and reported in metrics

## Logging

### Episode Logging
- JSONL format for each timestep
- Manifest file for episode replay in Unity
- Comprehensive diagnostics and safety information
- Configurable via `ENABLE_LOGGING` environment variable

### Server Logging
- Structured logging with configurable levels
- Request/response logging for debugging
- Performance metrics and error tracking

## Testing

Run the test suite:
```bash
# Unit tests
python -m pytest src/phase4_rl/tests/ -v

# Performance validation
python -m src.phase4_rl.validate_performance
```

## Development

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up development environment
cp .env.example .env
# Edit .env with your configuration
```

### Configuration Management
The server uses a hierarchical configuration system:
1. Default values in code
2. Environment variables (from system or .env file)
3. Command-line arguments (highest priority)

### Module Structure
```
src/phase4_rl/
├── hlynr_bridge_server.py    # Main server implementation
├── env_config.py            # Environment configuration management
├── schemas.py               # Pydantic request/response schemas
├── transforms.py            # ENU↔Unity coordinate transforms
├── normalize.py             # VecNormalize statistics management
├── clamps.py               # Safety clamping system
├── inference_logger.py     # Episode logging system
└── tests/                  # Comprehensive test suite
```

## Unity Integration

### Coordinate Systems
- **Unity**: Left-handed coordinate system
- **RL Training**: ENU (East-North-Up) right-handed coordinate system
- **Transforms**: Deterministic, versioned transforms between systems

### Request Flow
1. Unity sends state in left-handed coordinates
2. Server transforms to ENU for RL policy
3. Policy computes action in ENU frame
4. Server transforms action back to Unity frame
5. Safety clamps applied before returning to Unity

### Versioning
- `obs_version`: Observation format version
- `vecnorm_stats_id`: Specific normalization statistics
- `transform_version`: Coordinate transform version
- Ensures deterministic, reproducible inference

## Troubleshooting

### Common Issues

**Model not loading:**
```bash
# Check model path
python -m src.phase4_rl.hlynr_bridge_server --print-config

# Verify model file exists
ls -la $MODEL_CHECKPOINT
```

**High latency:**
```bash
# Check metrics
curl http://localhost:5000/metrics

# Enable debug logging
DEBUG_MODE=true python -m src.phase4_rl.hlynr_bridge_server
```

**VecNormalize issues:**
```bash
# List available stats
python -c "from src.phase4_rl.normalize import get_vecnorm_manager; print(get_vecnorm_manager().list_stats())"

# Check specific stats ID
VECNORM_STATS_ID=your_id python -m src.phase4_rl.hlynr_bridge_server --print-config
```

### Logging Configuration
```bash
# Enable debug logging
LOG_LEVEL=DEBUG

# Disable episode logging
ENABLE_LOGGING=false

# Custom log directory
LOG_DIR=/custom/log/path
```

## License

MIT License - see LICENSE file for details.