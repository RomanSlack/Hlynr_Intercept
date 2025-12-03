# Hlynr Intercept - Recent Changes Summary

## Overview
This document summarizes the recent development work on the missile interception RL system, focusing on achieving 360° direction-invariant interception capability.

---

## The Problem
The original system only worked when missiles came from one direction (quadrant 1: +X, +Y). Training with 360° spherical spawns caused catastrophic performance degradation.

**Root Cause**: The observation and action spaces were tied to **world coordinates**, so the same physical scenario produced different observations depending on approach direction.

---

## Key Concepts

### Line-of-Sight (LOS) Frame
Instead of world XYZ coordinates, use a frame relative to the interceptor-missile geometry:
- **LOS unit vector**: Points from interceptor toward missile
- **LOS horizontal**: Perpendicular to LOS in the horizontal plane
- **LOS vertical**: Perpendicular to LOS in the vertical plane

This makes observations direction-invariant: the same relative geometry produces the same observation regardless of where in the world it occurs.

### Proportional Navigation (PN)
Standard missile guidance approach:
- **Goal**: Achieve collision course where LOS direction stays constant
- **Method**: Apply thrust perpendicular to LOS to cancel LOS rotation rate
- **Equation**: `acceleration = N * closing_velocity * LOS_rate` (N = navigation gain, typically 3-5)

### Proximity Fuze
Real missiles don't hit targets directly - they detonate when closest approach is reached. The system now uses `proximity_kill_radius` (default 20-50m) as the success criterion.

---

## Changes Made

### 1. LOS Observation Mode (`core.py`)
**Commit**: `9f80a28` - "360 observation space rotational invariant"

Changed observation space to be direction-invariant:
```
obs[0]: Range to target (normalized)
obs[1]: Range rate (positive = closing)
obs[2]: LOS azimuth rate (horizontal bearing change)
obs[3]: LOS elevation rate (vertical bearing change)
obs[4]: Off-axis angle cosine (1.0 = on collision course)
obs[5]: Lead angle cosine
obs[6-8]: Interceptor velocity in LOS frame
```

### 2. LOS Action Transformation (`environment.py`)
Actions are now in LOS-relative coordinates:
```python
action[0]: Thrust along LOS (toward target)
action[1]: Thrust perpendicular (horizontal correction)
action[2]: Thrust perpendicular (vertical correction)
```

Transformed to world frame for physics:
```python
world_thrust = action[0] * los_unit + action[1] * los_horizontal + action[2] * los_vertical
```

### 3. Orthonormal Basis Fix
**Commit**: `30c2b18` - "Fixing still LOS stuff"

Fixed bug where LOS frame vectors weren't truly orthogonal:
```python
# CORRECT orthonormal basis
los_horizontal = normalize(cross(los_unit, world_up))
los_vertical = cross(los_unit, los_horizontal)
```

Previous implementation caused action effectiveness to vary with elevation angle.

### 4. Interceptor Velocity Mode
**Commit**: `10ac0f9` - "V6 fix: velocity_mode toward_missile"

Bug: Interceptor always launched toward +X, +Y regardless of where missile spawned.

Fix: Added `velocity_mode: "toward_missile"` so interceptor launches toward the threat:
```yaml
interceptor_spawn:
  velocity_mode: "toward_missile"
  speed_min: 50.0
  speed_max: 100.0
```

### 5. Reward Function Improvements
**Commits**: `f9da9a8`, `d115030`

Added aggressive pursuit rewards using radar-derived quantities:

```python
# Closing velocity reward (from radar range rate)
closing_velocity = distance_delta / dt
reward += clip(closing_velocity / 100.0, -0.5, 2.0) * 0.5

# Pursuit alignment (from LOS direction vs velocity)
pursuit_alignment = dot(velocity_unit, los_unit)
reward += pursuit_alignment * 0.3

# Forward thrust reward (direct action shaping)
forward_thrust = action[0]  # LOS thrust command
reward += forward_thrust * 0.4
```

Problem solved: Model was learning 50% thrust policies because there was no incentive for aggressive pursuit.

### 6. MLP Instead of LSTM
**Commit**: `8176d72` - "For track specialist: MLP instead of LSTM"

Switched from LSTM to MLP policy networks. The LOS-frame observations provide sufficient temporal context through range rates and LOS rates.

---

## Configuration

### Spherical Spawn (360°)
```yaml
missile_spawn:
  position_mode: "spherical"
  radius_min: 800.0
  radius_max: 1500.0
  azimuth_range: [0, 360]      # Full 360°
  elevation_range: [15, 60]    # Above horizon
  velocity_mode: "toward_target"
```

### LOS Mode
```yaml
environment:
  observation_mode: "los_frame"   # Direction-invariant observations
  proximity_fuze_enabled: true
  proximity_kill_radius: 50.0
```

---

## Validation

Hardcoded PN policy test proved environment correctness:
```
Pure Pursuit (N=0): 100% success, 48.9m mean
PN (N=3): 96% success, ~50m mean
```

This confirms:
1. LOS frame transformation is correct
2. Action-observation alignment works
3. Physics supports 360° interception
4. Trained model learning is the remaining challenge

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    InterceptEnvironment                  │
├─────────────────────────────────────────────────────────┤
│  Radar26DObservation (LOS mode)                         │
│  ├── Kalman filter for trajectory smoothing             │
│  ├── Ground radar fusion                                │
│  └── 26D observation vector (direction-invariant)       │
├─────────────────────────────────────────────────────────┤
│  LOS Action Transformation                              │
│  ├── _update_los_frame() - compute basis vectors        │
│  └── _transform_los_action_to_world() - transform       │
├─────────────────────────────────────────────────────────┤
│  6DOF Physics                                           │
│  ├── Atmospheric model (altitude-dependent)             │
│  ├── Mach-dependent drag                                │
│  ├── Thrust dynamics (first-order lag)                  │
│  └── Enhanced wind model                                │
└─────────────────────────────────────────────────────────┘
```

---

## Training Pipeline

```bash
# 1. Train specialists with LOS configs
python scripts/train_hrl_pretrain.py --agent terminal --config configs/hrl/terminal_los.yaml

# 2. Evaluate
python scripts/evaluate_hrl.py --config configs/eval_360_los.yaml --episodes 50
```

---

## Key Files

| File | Purpose |
|------|---------|
| `environment.py` | Main env with LOS transform, rewards, physics |
| `core.py` | 26D observation generator, radar simulation |
| `configs/eval_360_los.yaml` | 360° evaluation config |
| `configs/hrl/terminal_los.yaml` | Terminal specialist training |
| `HRL_LOS_FRAME_PLAN.md` | Detailed implementation notes |