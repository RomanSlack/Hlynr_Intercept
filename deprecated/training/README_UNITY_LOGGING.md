# Unity Episode Logging System

## Overview

The Phase 4 RL system includes an **optional** episode logging capability that records **full 6DOF 3D simulation data** in a format suitable for Unity replay visualization. This allows offline reconstruction and analysis of RL agent behavior with true 3D physics.

**Features:**
- Full 6DOF physics (3D position, quaternion orientation, 3D velocity, angular velocity)
- Real-time physics simulation with gravity, air resistance, and torque dynamics
- Body-frame control inputs (3D thrust + 3D torque)
- Fuel consumption and realistic flight dynamics

**Note: Episode logging is disabled by default. The `runs/` directory will only be created when you explicitly enable logging.**

## Architecture

The logging system consists of:
- `episode_logger.py` - Core logging module that handles file I/O and data formatting
- `fast_sim_env.py` - Environment wrapper with integrated logging hooks
- Configuration in `config.yaml` for logging parameters

## File Format

### Directory Structure
```
runs/
└── run_YYYY-MM-DD-HHMMSS/
    ├── manifest.json       # Episode index and metadata
    ├── ep_000001.jsonl    # Episode 1 data (JSON Lines format)
    ├── ep_000002.jsonl    # Episode 2 data
    └── ...
```

### Manifest Schema
The `manifest.json` file contains:
- `schema_version`: Format version (currently "1.0")
- `coord_frame`: Coordinate system ("ENU_RH" - East-North-Up Right-Handed)
- `units`: Physical units (meters, seconds, radians)
- `dt_nominal`: Sampling timestep (0.01s = 100Hz)
- `episodes`: List of episode metadata with file references

### Episode File Format
Each episode file uses JSON Lines format (one JSON object per line):
1. **Header** (t=0): Episode metadata and entity configurations
2. **Timesteps**: State snapshots with positions, velocities, actions
3. **Footer**: Episode summary with outcome and statistics

## Usage

### Training with Logging
```bash
# Enable logging during evaluation episodes only (recommended)
python train_radar_ppo.py --scenario easy --timesteps 1000000 --enable-episode-logging

# Specify custom log directory
python train_radar_ppo.py --scenario easy --enable-episode-logging --episode-log-dir my_runs
```

### Configuration Options
Edit `config.yaml` to configure logging behavior:
```yaml
episode_logging:
  enabled: false              # Master enable/disable
  output_dir: "runs"         # Output directory
  log_during_training: false # Log training episodes (impacts performance)
  log_during_eval: true      # Log evaluation episodes
  log_interval: 100          # Log every N episodes if training logging enabled
  coordinate_frame: "ENU_RH" # Coordinate system
  sampling_rate_hz: 100      # Data sampling rate
```

### Inference with Logging
```bash
python run_inference.py --checkpoint checkpoints/best_model.zip --enable-episode-logging
```

## Monitoring Training

### TensorBoard
```bash
# View training metrics
tensorboard --logdir logs

# Access at http://localhost:6006
```

### Training Logs
- `logs/progress.csv` - Training metrics in CSV format
- `logs/evaluations.npz` - Evaluation episode results
- `logs/*.tfevents.*` - TensorBoard event files

## Coordinate System

The system uses ENU_RH (East-North-Up Right-Handed) coordinates:
- X: East (positive right)
- Y: North (positive forward)  
- Z: Up (positive upward)

Unity integration requires coordinate transformation:
- Unity.X = Python.X (East)
- Unity.Y = Python.Z (Up)
- Unity.Z = Python.Y (North)

## Performance Considerations

- Episode logging adds minimal overhead (~1-2% during evaluation)
- Each episode generates approximately 0.5-2MB of data
- Logging during training is disabled by default to maintain performance
- Data is written incrementally to handle long episodes efficiently

## Quick Start

To see episode logging in action:
```bash
# Test the logging system
python test_episode_logging.py

# This creates test_runs/ directory with sample episodes
```

## Example Data Access

```python
import json

# Read manifest
with open('runs/run_2025-08-04-123456/manifest.json') as f:
    manifest = json.load(f)

# Read episode data
with open('runs/run_2025-08-04-123456/ep_000001.jsonl') as f:
    for line in f:
        record = json.loads(line)
        if 'agents' in record:
            # Process timestep data
            print(f"Time: {record['t']}, Interceptor pos: {record['agents']['interceptor_0']['p']}")
```

## Integration Notes

The logging system is designed to be non-intrusive:
- Disabled by default
- No impact on existing training workflows
- Compatible with parallel environments
- Handles missing entity data gracefully