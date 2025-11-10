# Hlynr Intercept - Blank Canvas System

A realistic missile interception RL system with **radar-only observations**, 6DOF physics, and production-ready deployment capabilities. The interceptor has **no direct knowledge** of missile positions and must rely entirely on simulated radar sensors, just like real-world missile defense systems.

## What's New - Hierarchical RL Support ✨

The system now includes **optional** Hierarchical RL training for modular, interpretable policies. Existing flat PPO workflows remain unchanged and fully backward compatible.

**New Capabilities**:
- 🧩 Modular architecture with pre-trainable specialists (Search, Track, Terminal)
- 🎯 Interpretable option transitions with forced physics-based switching
- 📊 Enhanced sample efficiency through curriculum learning
- 🔧 Per-phase reward tuning for fine-grained control

See [HRL Documentation](#hierarchical-rl-optional) below for details.

## Features

- **🎯 Radar-Only Observations**: 17D sensor-based observation space with realistic limitations
- **📡 Authentic Radar Physics**: Range limits, beam width, noise, detection failures
- **🚀 6DOF Missile Dynamics**: Physics based on PAC-3/THAAD interceptor specifications
- **🧠 PPO Training**: Stable training with adaptive features (entropy scheduling, LR decay, clip adaptation)
- **🧩 Hierarchical RL (NEW)**: Optional modular training with Search/Track/Terminal specialists
- **⚡ FastAPI Inference**: Real-time inference server with safety constraints
- **📊 Unified Logging**: Centralized timestamped logging for training, inference, and episodes
- **📈 TensorBoard Integration**: Built-in visualization support
- **🔄 Coordinate Transforms**: ENU ↔ Unity coordinate system conversion
- **🛡️ Safety Constraints**: Post-policy action clamping for safe operation
- **🌡️ Advanced Physics v2.0**: Realistic atmospheric models, sensor delays, Mach effects, and domain randomization

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training: Choose Your Approach

#### Option 1: Flat PPO (Recommended for Quick Start)

**Fast, proven, single monolithic policy**

```bash
# Train with optimized hyperparameters (5M steps, ~25-30 minutes)
python train.py --config config.yaml

# Monitor training progress with TensorBoard
tensorboard --logdir logs

# Access at: http://localhost:6006
```

**Expected Results**: 75-85% intercept success in 25-30 minutes

#### Option 2: Hierarchical RL (For Modular/Interpretable Policies)

**Modular, interpretable, option-based control**

```bash
# Full HRL pipeline (~2 hours total)
python scripts/train_hrl_full.py --config configs/hrl/hrl_curriculum.yaml

# Or train stages individually:
python scripts/train_hrl_pretrain.py --specialist all  # 45 min: Pre-train specialists
python scripts/train_hrl_selector.py                   # 20 min: Train selector

# Evaluate HRL
python scripts/evaluate_hrl.py --model checkpoints/hrl/selector/best/ --episodes 100
```

**Expected Results**: 70-85% intercept success, interpretable option transitions

**Comparison**:

| Feature | Flat PPO | Hierarchical RL |
|---------|----------|-----------------|
| Training time | 25-30 min | ~2 hours |
| Intercept rate | 75-85% | 70-85% |
| Interpretability | Low | High (option logs) |
| Modularity | Monolithic | Composable (reusable specialists) |
| Use case | Quick baselines, simple scenarios | Research, multi-phase behavior, customization |

See [HRL Documentation](#hierarchical-rl-optional) for detailed guide.

**Curriculum Learning (Recommended for Best Results):**
```bash
# Stage 1: Easy scenario - wide radar beam, close targets (1-2M steps)
python train.py --config scenarios/easy.yaml

# Stage 2: Standard difficulty - continue training
python train.py --config config.yaml

# Stage 3: Hard scenario - evaluate robustness
python inference.py --model checkpoints/best --mode offline --scenario hard --episodes 100
```

**Training Output:**

Training creates timestamped logs in `logs/training_YYYYMMDD_HHMMSS/`:
- **TensorBoard logs**: Real-time training metrics, reward curves, loss plots
- **Model checkpoints**: Saved every 10k steps in `checkpoints/`
- **Best model**: Auto-saved to `checkpoints/best/` based on eval performance
- **Training metrics**: JSON logs of key performance indicators
- **Episode data**: Detailed trajectory information

**TensorBoard Metrics:**
- `rollout/ep_rew_mean` - Average episode reward (track this for performance)
- `train/policy_gradient_loss` - Policy optimization progress
- `train/value_loss` - Value function accuracy
- `train/entropy_loss` - Exploration vs exploitation
- `train/approx_kl` - Policy update magnitude
- `train/clip_fraction` - PPO clipping activity
- `eval/mean_reward` - Evaluation performance (best model selection)

**Expected Performance:**
- **1M steps (~5 min)**: 30-40% interception success rate
- **3M steps (~15 min)**: 60-70% interception success rate
- **5M steps (~25 min)**: 75-85% interception success rate

**Key Improvements in Current System:**
- ✅ **Dense reward shaping** - Strong gradients for closing distance and radar tracking
- ✅ **Larger network** - [512, 512, 256] architecture for complex behavior
- ✅ **Better spawn geometry** - Pursuit configuration instead of head-on collision
- ✅ **Extended episodes** - 2000 steps (20 seconds) for full interception sequence
- ✅ **Radar tracking rewards** - Incentivizes maintaining lock on target

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

## Advanced Physics v2.0

The system includes state-of-the-art physics modeling for realistic missile dynamics and improved sim-to-real transfer.

### 🌡️ Atmospheric Modeling

**International Standard Atmosphere (ISA) Implementation:**
- **Troposphere (0-11km)**: Temperature lapse rate of 6.5K/km
- **Stratosphere (11-20km)**: Isothermal layer at 216.65K
- **High altitude (>20km)**: Exponential decay model

**Altitude-dependent properties:**
- Air density: 1.225 kg/m³ at sea level → 0.41 kg/m³ at 10km
- Temperature: 288.15K at sea level → 216.65K at 11km
- Pressure: Barometric formula with proper lapse rate
- Speed of sound: Temperature-dependent calculation

```yaml
# Enable/disable atmospheric model
physics_enhancements:
  atmospheric_model:
    enabled: true
    sea_level_temperature: 288.15  # K
    troposphere_lapse_rate: 0.0065  # K/m
```

### 🚀 Mach-Dependent Drag Effects

**Transonic Drag Rise Modeling:**
- **Subsonic (M < 0.8)**: Constant base drag coefficient
- **Transonic (0.8 < M < 1.2)**: Linear rise to 3x base drag
- **Supersonic (M > 1.2)**: Constant 2.5x base drag

**Realistic drag curve based on missile aerodynamics:**
```python
# Example: At Mach 1.0, drag coefficient increases ~2.5x
base_cd = 0.3      # Subsonic
transonic_cd = 0.75  # At Mach 1.0
supersonic_cd = 0.75 # Above Mach 1.2
```

### 📡 Sensor Delays and Measurement Lag

**Realistic radar processing delays:**
- **Default delay**: 30ms (configurable 10-50ms range)
- **Circular buffer**: Proper FIFO delay implementation
- **Initialization period**: No detections during buffer fill
- **Training impact**: Agents must learn predictive control

```yaml
# Configure sensor delays
physics_enhancements:
  sensor_delays:
    enabled: true
    radar_delay_ms: 30.0  # Realistic processing delay
```

### ⚡ Thrust Dynamics and Engine Response

**First-order lag model for solid rocket motors:**
- **Response time**: 100ms time constant (configurable)
- **Physical model**: `thrust_actual += (thrust_cmd - thrust_actual) * dt/tau`
- **Fuel consumption**: Based on actual (not commanded) thrust
- **Training benefit**: More realistic control authority

### 💨 Enhanced Wind and Turbulence

**Altitude-dependent wind profiles:**
- **Boundary layer**: Power-law wind profile below 1000m
- **Free atmosphere**: Constant wind above boundary layer
- **Turbulence**: Altitude-dependent intensity
- **Gusts**: Stochastic wind gusts with configurable probability

### 🎲 Domain Randomization Framework

**Physics parameter variation per episode:**
- **Drag coefficients**: ±20% variation
- **Air density**: ±10% variation
- **Sensor delays**: ±50% variation
- **Thrust response**: ±30% variation
- **Wind conditions**: ±30% variation

```yaml
# Enable domain randomization (use carefully - impacts training stability)
physics_enhancements:
  domain_randomization:
    enabled: false  # Disabled by default
    drag_coefficient_variation: 0.2
    randomize_per_episode: true
    log_randomization: true
```

### 🔧 Configuration and Backward Compatibility

**Master control switches:**
```yaml
physics_enhancements:
  enabled: true  # Master switch

  # Individual feature flags
  atmospheric_model: { enabled: true }
  mach_effects: { enabled: true }
  sensor_delays: { enabled: true }
  thrust_dynamics: { enabled: true }
  enhanced_wind: { enabled: true }
  domain_randomization: { enabled: false }

  # Backward compatibility
  fallback_to_simple_physics: true
```

**Performance monitoring:**
```yaml
physics_enhancements:
  performance:
    log_physics_timing: false
    max_physics_time_ms: 10.0  # Performance threshold
    enable_physics_validation: true  # Check for NaN/inf
```

### 📊 Training with Advanced Physics

**Recommended training progression:**

1. **Start with basic physics** (all enhancements disabled)
2. **Enable atmospheric + Mach effects** for altitude realism
3. **Add sensor delays** for control system realism
4. **Include thrust dynamics** for propulsion realism
5. **Domain randomization** only for final robustness training

**Expected performance impact:**
- **Physics computation**: <5ms additional per step
- **Training time**: ~10-15% increase with all features
- **Convergence**: May require 20-30% more training steps
- **Realism**: Dramatically improved sim-to-real transfer

### 🧪 Testing and Validation

**Comprehensive test suite:**
```bash
# Run physics validation tests
python tests/test_physics_enhancements.py

# Quick atmospheric model validation
python -c "
from physics_models import AtmosphericModel
atm = AtmosphericModel()
print('10km density:', atm.get_density(10000), 'kg/m³')  # Should be ~0.41
print('Mach 1 speed:', atm.get_speed_of_sound(10000), 'm/s')  # Should be ~299
"
```

**Physics validation against literature:**
- ✅ Atmospheric density matches US Standard Atmosphere 1976
- ✅ Mach drag curves match missile aerodynamic data
- ✅ Sensor delays match tactical radar specifications
- ✅ Performance benchmarks meet <10ms requirement

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

## Hierarchical RL (Optional)

### Overview

Hierarchical RL provides a modular alternative to flat PPO with three specialists:
- **Search Specialist**: Wide-area scanning for radar lock acquisition
- **Track Specialist**: Maintain lock and close distance
- **Terminal Specialist**: Final precision guidance

A high-level **Selector** policy chooses which specialist to use based on mission phase.

**Key Benefits**:
- 🔍 **Interpretability**: See exactly which phase the agent is in
- 🧩 **Modularity**: Train/replace specialists independently
- 🎯 **Sample Efficiency**: ~20-30% better with curriculum learning
- 🔧 **Customization**: Tune rewards per specialist

**Tradeoff**: Longer training time (~2 hrs vs 25 min) for enhanced modularity.

### Quick Start

```bash
# Full pipeline
python scripts/train_hrl_full.py --config configs/hrl/hrl_curriculum.yaml

# Compare with flat PPO
python scripts/compare_policies.py \
    --flat checkpoints/flat_ppo/best/ \
    --hrl checkpoints/hrl/selector/best/ \
    --episodes 100
```

### Architecture

```
Selector (1Hz)
  |
  ├─ SEARCH  → Acquire radar lock
  ├─ TRACK   → Maintain lock, close distance
  └─ TERMINAL → Final intercept

Each specialist: 104D obs → 6D action (LSTM-enabled)
Selector: 7D abstract state → {0,1,2} discrete option
```

### Forced Transitions

Physics-based option switching ensures realistic behavior:
- Lock quality > 0.7 → SEARCH → TRACK
- Lock quality < 0.3 → TRACK → SEARCH
- Distance < 100m → → TERMINAL

### Documentation

- **Architecture**: [docs/hrl/architecture.md](docs/hrl/architecture.md) - System design overview
- **Training Guide**: [docs/hrl/training_guide.md](docs/hrl/training_guide.md) - Step-by-step workflow
- **API Reference**: [docs/hrl/api_reference.md](docs/hrl/api_reference.md) - Function signatures
- **Migration Guide**: [docs/hrl/migration_guide.md](docs/hrl/migration_guide.md) - Upgrade instructions

### When to Use HRL

**Use HRL if you need**:
- Interpretable decision-making (see which phase is active)
- Modular components (reuse specialists across scenarios)
- Fine-grained control (tune rewards per phase)
- Research insights (analyze option transitions)

**Stick with Flat PPO if**:
- You need quick baselines
- Training time is critical
- Simple end-to-end learning suffices
- You prefer proven monolithic approaches

**Backward Compatibility**: All existing flat PPO workflows work unchanged. HRL is an optional enhancement.

---

## License

Academic research use only. See main repository for license details.