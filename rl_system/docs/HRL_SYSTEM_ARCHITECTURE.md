# HRL System Architecture Documentation

## Overview

This document describes the Hierarchical Reinforcement Learning (HRL) system for missile interception. The system uses a two-level hierarchy: a **selector policy** that chooses which specialist to activate, and three **specialist policies** that execute phase-specific behaviors.

---

## 1. System Architecture

### 1.1 Hierarchy Structure

```
                    ┌─────────────────┐
                    │    SELECTOR     │
                    │   (PPO Policy)  │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
    ┌───────▼───────┐ ┌──────▼──────┐ ┌───────▼───────┐
    │    SEARCH     │ │    TRACK    │ │   TERMINAL    │
    │  Specialist   │ │  Specialist │ │   Specialist  │
    │  (PPO/MLP)    │ │ (LSTM/PPO)  │ │   (PPO/MLP)   │
    └───────────────┘ └─────────────┘ └───────────────┘
```

### 1.2 Option Definitions

| Option | Purpose | Distance Range | Key Behaviors |
|--------|---------|----------------|---------------|
| SEARCH | Find and acquire target | >1000m | Radar scanning, initial approach |
| TRACK | Pursue and close distance | 300-1000m | Stable tracking, energy management |
| TERMINAL | Final approach and intercept | <300m | Precision guidance, intercept execution |

---

## 2. Observation Space

### 2.1 Base Observations (26D)

```python
observation = {
    # Interceptor State (10D)
    'position': [x, y, z],           # 3D position (meters)
    'velocity': [vx, vy, vz],        # 3D velocity (m/s)
    'orientation': [qw, qx, qy, qz], # Quaternion orientation

    # Target Relative (8D)
    'relative_position': [rx, ry, rz],  # Vector to target
    'relative_velocity': [rvx, rvy, rvz], # Closing velocity
    'distance': scalar,                  # Distance to target (m)
    'closing_rate': scalar,              # Rate of closure (m/s)

    # Sensor State (4D)
    'radar_lock': bool,               # Has radar lock
    'radar_signal_strength': float,   # Signal quality [0,1]
    'time_since_detection': float,    # Seconds since last detection
    'target_aspect_angle': float,     # Angle to target

    # Resources (4D)
    'fuel': float,                    # Remaining fuel [0,1]
    'time_remaining': float,          # Episode time budget
    'altitude': float,                # Current altitude
    'speed': float,                   # Current speed
}
```

### 2.2 Frame Stacking

Observations are frame-stacked (default: 4 frames) to provide temporal context:
- **Input dimension**: 26D × 4 = **104D**
- Frame stacking helps with velocity estimation and trajectory prediction

### 2.3 Observation Normalization (VecNormalize)

- Running mean/std normalization applied during training
- Critical for stable learning with varying scales
- **Must be saved and loaded** with model checkpoints
- Training: `VecNormalize(training=True)` - updates statistics
- Inference: `VecNormalize(training=False)` - fixed statistics

---

## 3. Action Space

### 3.1 Continuous Actions (5D)

```python
action_space = Box(low=-1, high=1, shape=(5,))

actions = {
    'thrust': [-1, 1],      # Forward/backward thrust
    'pitch': [-1, 1],       # Pitch rate
    'yaw': [-1, 1],         # Yaw rate
    'roll': [-1, 1],        # Roll rate
    'throttle': [-1, 1],    # Engine throttle
}
```

---

## 4. Reward Structure

### 4.1 Terminal Rewards (Episode End)

```python
def compute_reward(intercepted, terminated, distance, ...):
    if intercepted:
        # Base intercept reward
        reward = 5000.0

        # Time bonus (faster = better)
        time_bonus = (max_steps - steps) * 0.5
        reward += time_bonus

        # PRECISION BONUS (NEW): Closer hits get more reward
        # At 100m threshold: 10m hit = +900, 50m hit = +500, 99m hit = +10
        intercept_radius = get_current_intercept_radius()
        precision_ratio = max(0, 1.0 - distance / intercept_radius)
        precision_bonus = precision_ratio * 1000.0
        reward += precision_bonus

        return reward

    if terminated:  # Failure
        # Distance penalty (closer misses penalized less)
        distance_penalty = -distance * 0.5  # -0.5 per meter
        reward = max(distance_penalty, -2000.0)  # Cap at -2000

        # Additional failure mode penalties
        if missile_hit_target: reward -= 1000
        elif crashed: reward -= 500
        elif out_of_fuel: reward -= 300

        return reward
```

### 4.2 Per-Step Rewards (Shaping)

```python
# Distance reduction (gradient signal)
if distance < 500m:
    reward += distance_delta * 1.0  # Close range
else:
    reward += distance_delta * 0.5  # Far range

# Time penalty
reward -= 0.5  # Discourage long episodes
```

### 4.3 Intercept Radius (Success Threshold)

The **intercept_radius** determines what distance counts as a successful intercept:
- Default: **200m** (in config.yaml)
- Training curriculum can progressively tighten this
- Curriculum example: 100m → 50m over 2M steps

---

## 5. Curriculum Learning

### 5.1 Configuration

```yaml
curriculum:
  enabled: true
  initial_radius: 100.0    # Starting intercept radius (meters)
  final_radius: 50.0       # Target intercept radius
  curriculum_steps: 2000000 # Steps to reach final radius
```

### 5.2 Radius Progression

Linear interpolation based on training step:

```python
progress = current_step / curriculum_steps
current_radius = initial_radius - (initial_radius - final_radius) * progress
```

### 5.3 Step Counter Update

**CRITICAL**: The training step count must be passed to environments:

```python
# In training callback
self.training_env.env_method('set_training_step_count', self.num_timesteps)
```

---

## 6. Training Pipeline

### 6.1 Phase 1: Specialist Pre-Training

Each specialist is trained independently on phase-appropriate scenarios:

```bash
# Train SEARCH specialist (500k steps)
python scripts/train_hrl_pretrain.py --agent search --config configs/hrl/search_specialist.yaml

# Train TRACK specialist with LSTM (500k steps)
python scripts/train_hrl_pretrain.py --agent track --config configs/hrl/track_specialist.yaml

# Train TERMINAL specialist (2M steps with curriculum)
python scripts/train_hrl_pretrain.py --agent terminal --config configs/hrl/terminal_precision_v2.yaml
```

**Key Settings per Specialist:**

| Specialist | Steps | LSTM | Key Focus |
|------------|-------|------|-----------|
| SEARCH | 500k | No | Target acquisition, scanning |
| TRACK | 500k | Yes | Stable pursuit, memory needed |
| TERMINAL | 2M | No | Precision, curriculum learning |

### 6.2 Phase 2: Selector Training

Train the selector to choose between specialists:

```bash
python scripts/train_hrl_selector.py \
    --search checkpoints/hrl/specialists/search/.../model.zip \
    --track checkpoints/hrl/specialists/track/.../model.zip \
    --terminal checkpoints/hrl/specialists/terminal/.../model.zip \
    --config configs/hrl/hrl_curriculum.yaml
```

**Selector Training Details:**
- Uses frozen specialists (no gradient through specialists)
- Learns when to switch between options
- Trained on full mission scenarios

### 6.3 Fine-Tuning (Resume Training)

To continue training an existing model:

```bash
python scripts/train_hrl_pretrain.py --agent terminal \
    --config configs/hrl/terminal_finetune.yaml \
    --resume checkpoints/hrl/specialists/terminal/.../model.zip \
    --vecnorm checkpoints/hrl/specialists/terminal/.../vec_normalize.pkl
```

---

## 7. Model Architecture

### 7.1 Network Configuration

```yaml
training:
  net_arch: [512, 512, 256]    # Hidden layer sizes
  use_lstm: false              # MLP or LSTM
  use_layer_norm: true         # Layer normalization
  use_orthogonal_init: true    # Orthogonal weight init
  frame_stack: 4               # Temporal frames
```

### 7.2 PPO Hyperparameters

```yaml
training:
  learning_rate: 0.0003
  n_steps: 2048              # Rollout length
  batch_size: 512
  n_epochs: 10               # PPO epochs per update
  gamma: 0.99                # Discount factor
  gae_lambda: 0.95           # GAE lambda
  clip_range: 0.2            # PPO clip range
  ent_coef: 0.01             # Entropy coefficient
  vf_coef: 0.5               # Value function coefficient
  max_grad_norm: 0.5         # Gradient clipping
```

---

## 8. Inference / Evaluation

### 8.1 Running Evaluation

```bash
python scripts/evaluate_hrl.py \
    --selector checkpoints/hrl/selector/.../model.zip \
    --search checkpoints/hrl/specialists/search/.../model.zip \
    --track checkpoints/hrl/specialists/track/.../model.zip \
    --terminal checkpoints/hrl/specialists/terminal/.../model.zip \
    --episodes 50 --seed 44
```

### 8.2 Evaluation Metrics

```
PRECISION METRICS:
  Mean Min Distance:      XXXm    # Average closest approach
  Sub-10m Precision:      N/50    # Direct hits
  Sub-50m Precision:      N/50    # High precision
  Sub-100m Precision:     N/50    # Standard precision
  Sub-150m (baseline):    N/50    # Basic capability

PERFORMANCE METRICS:
  Success Rate:           XX%     # Intercept success
  Mean Reward:            XXXX    # Average episode reward

OPTION USAGE:
  SEARCH:                 XX%     # Time in search mode
  TRACK:                  XX%     # Time in track mode
  TERMINAL:               XX%     # Time in terminal mode
```

### 8.3 VecNormalize in Evaluation

**Important**: Evaluation currently runs **without** VecNormalize stats. This can cause observation distribution mismatch. Future improvement: load VecNormalize stats during evaluation.

---

## 9. File Structure

```
rl_system/
├── configs/
│   └── hrl/
│       ├── search_specialist.yaml
│       ├── track_specialist.yaml
│       ├── terminal_precision_v2.yaml
│       ├── terminal_finetune.yaml
│       └── hrl_curriculum.yaml
├── scripts/
│   ├── train_hrl_pretrain.py     # Specialist training
│   ├── train_hrl_selector.py     # Selector training
│   └── evaluate_hrl.py           # Evaluation
├── hrl/
│   ├── option_definitions.py     # Option enum and rules
│   ├── hierarchical_env.py       # HRL environment wrapper
│   └── wrappers.py               # Reward wrappers
├── checkpoints/
│   └── hrl/
│       ├── specialists/
│       │   ├── search/
│       │   ├── track/
│       │   └── terminal/
│       └── selector/
└── environment.py                # Base interception environment
```

---

## 10. Known Issues & Limitations

### 10.1 Current Issues

1. **Precision Clustering at Threshold**: All intercepts cluster at ~threshold distance (e.g., 99m when threshold is 100m). The new precision bonus should help.

2. **VecNormalize Mismatch**: When swapping specialists trained independently, observation normalization statistics may not match. Solution: Use --vecnorm flag when fine-tuning.

3. **CUDA/SubprocVecEnv Conflict**: Using SubprocVecEnv causes CUDA initialization errors. Use DummyVecEnv with GPU instead.

### 10.2 Best Practices

1. **Always save VecNormalize stats** with model checkpoints
2. **Use curriculum learning** for precision training
3. **Match frame_stack** between training and inference (default: 4)
4. **Monitor curriculum radius** in training logs

---

## 11. Performance Benchmarks

### Current Results (as of 2025-11-27)

| Configuration | Success Rate | Mean Min Distance | Best Approach |
|--------------|--------------|-------------------|---------------|
| HRL (200m radius) | 62% | ~198m | 190m |
| HRL (100m radius) | 35% | ~192m | 98m |
| Fine-tuned (100m) | 24% | ~180m | 98m |
| PPO Baseline | 28% | ~300m | ~200m |

### Target Performance

- Success Rate: >50% at 50m radius
- Mean Min Distance: <50m
- Sub-50m Precision: >30%

---

## 12. Future Improvements

1. **End-to-end Fine-tuning**: Train entire hierarchy jointly after specialist pre-training
2. **VecNormalize in Evaluation**: Properly load normalization stats during inference
3. **Adaptive Curriculum**: Adjust curriculum based on success rate
4. **Multi-target Scenarios**: Extend to intercept multiple missiles
5. **Sensor Noise**: Add realistic sensor modeling
