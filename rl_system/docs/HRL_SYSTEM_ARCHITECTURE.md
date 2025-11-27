# HRL System Architecture Documentation

**Version 2.0** - Updated with ZEM-based precision guidance

## Overview

This document describes the Hierarchical Reinforcement Learning (HRL) system for missile interception. The system uses a two-level hierarchy: a **selector policy** that chooses which specialist to activate, and three **specialist policies** that execute phase-specific behaviors.

### Version 2.0 Changes
- **Observation space expanded from 26D to 30D** with ZEM, time-to-go, and LOS rates
- **Reward function restructured** for precision-first with exponential scaling
- **LSTM enabled for terminal specialist** for trajectory prediction

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

### 2.1 Base Observations (30D) - v2.0

```python
# ONBOARD RADAR Components [0-16]
[0-2]:   Relative position (Kalman-filtered, normalized by max_range)
[3-5]:   Relative velocity (Kalman-filtered, normalized by max_velocity)
[6-8]:   Interceptor velocity (internal sensors, perfect)
[9-11]:  Interceptor orientation (Euler angles / pi)
[12]:    Fuel fraction [0,1]
[13]:    Time to intercept estimate (normalized)
[14]:    Track quality (Kalman filter uncertainty + radar quality)
[15]:    Closing rate (normalized by max_velocity)
[16]:    Off-axis angle (dot product with forward vector)

# GROUND RADAR Components [17-25]
[17-19]: Ground radar relative position
[20-22]: Ground radar relative velocity
[23]:    Ground radar quality
[24]:    Data link quality (ground-to-interceptor)
[25]:    Multi-radar fusion confidence

# PRECISION GUIDANCE Components [26-29] - NEW in v2.0
[26]:    ZEM (Zero-Effort-Miss) - predicted miss if no corrections
         Normalized: 0 = perfect collision course, 1 = 1000m+ miss
[27]:    Time-to-go - estimated seconds to closest approach
         Normalized: 0 = imminent, 1 = 30+ seconds
[28]:    LOS rate azimuth - line-of-sight rate horizontal (rad/s)
         Normalized: ±1 = ±0.1 rad/s (~5.7 deg/s)
[29]:    LOS rate elevation - line-of-sight rate vertical (rad/s)
         Normalized: ±1 = ±0.1 rad/s
```

**Why ZEM and LOS rates matter:**
- ZEM is the core concept in proportional navigation - nullifying ZEM = perfect intercept
- LOS rate of zero means collision course - the target appears stationary
- Time-to-go enables commit/abort decisions

### 2.2 Frame Stacking

Observations are frame-stacked (default: 4 frames) to provide temporal context:
- **Input dimension**: 30D × 4 = **120D**
- Frame stacking helps with velocity estimation and trajectory prediction
- Even with LSTM, frame stacking provides immediate acceleration information

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

## 4. Reward Structure (v2.0 - ZEM-based)

### 4.1 Terminal Rewards - Exponential Precision (NEW)

The reward function now uses **exponential scaling** to make precision the primary objective:

```python
def compute_reward(intercepted, terminated, distance, ...):
    if intercepted:
        # EXPONENTIAL precision reward (not linear!)
        # Decay constant = 30m means closer hits are MUCH better
        precision_reward = 5000.0 * exp(-distance / 30.0)

        # Reward breakdown:
        #   0m hit:   5000 * exp(0)     = 5000 points
        #   10m hit:  5000 * exp(-0.33) = 3594 points
        #   30m hit:  5000 * exp(-1)    = 1839 points
        #   50m hit:  5000 * exp(-1.67) = 946 points
        #   100m hit: 5000 * exp(-3.33) = 178 points

        # Small time bonus (10% of total, secondary to precision)
        time_bonus = (max_steps - steps) * 0.1

        return precision_reward + time_bonus

    if terminated:  # Failure
        # Exponential penalty (close misses much better than far)
        miss_penalty = -5000.0 * (1.0 - exp(-distance / 200.0))
        # Additional failure mode penalties...
        return miss_penalty
```

### 4.2 Per-Step Rewards - ZEM Shaping (NEW)

The key insight from guidance literature: **minimize ZEM, not just distance**.

```python
# ZEM reduction is the PRIMARY shaping signal
zem_delta = prev_zem - current_zem
reward += zem_delta * 0.1  # 10m ZEM reduction = 1 point

# Quadratic ZEM penalty (gradient toward collision course)
reward -= (current_zem / 1000.0) ** 2 * 0.5

# Distance reduction (secondary to ZEM)
if distance < 500m:
    reward += distance_delta * 0.5
else:
    reward += distance_delta * 0.2

# Small time penalty (don't discourage precision adjustments)
reward -= 0.1
```

### 4.3 Why ZEM-based Rewards Work

1. **ZEM = predicted miss if no corrections** - agents learn to nullify it
2. **LOS rate of zero = collision course** - ZEM implicitly captures this
3. **Exponential precision scaling** prevents "good enough" optimization
4. **Time penalty is small** so agents can make precision adjustments

### 4.4 Intercept Radius (Success Threshold)

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
