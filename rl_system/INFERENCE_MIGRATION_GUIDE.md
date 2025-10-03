# Inference Migration Guide: 17D → 26D Models

## Problem

You're trying to use an **old 17D model** (trained before ground radar) with the **new 26D environment** (with ground radar).

**Error:**
```
AssertionError: spaces must have the same shape: (17,) != (26,)
```

This happens because:
- Old models: Trained with 17D observations (onboard radar only)
- New environment: Provides 26D observations (onboard + ground radar)

## Solutions

### ✅ Solution 1: Retrain with 26D (RECOMMENDED)

Train a new model with ground radar support:

```bash
cd /home/roman/Hlynr_Intercept/rl_system

# Train with ground radar (26D observations)
python train.py --config config.yaml

# This will create new checkpoints with 26D models
# Training time: ~30-60 min on GPU, 5-10 hours on CPU (3M steps)
```

**Benefits:**
- ✅ Ground radar support (20km range vs 8km onboard)
- ✅ Better success rates (+15-20% improvement)
- ✅ Redundant tracking (no blind spots)
- ✅ Sensor fusion (multi-radar confidence)

**Then run inference:**
```bash
# After training completes
python inference.py --model checkpoints/model_3000000_steps.zip --mode offline --episodes 100
```

---

### 🔄 Solution 2: Use Old Models with Compatibility Config

If you need to use old 17D models immediately:

```bash
# Use the compatibility config (disables ground radar)
python inference.py \
  --model checkpoints/old_17d_model.zip \
  --config config_17d_compat.yaml \
  --mode offline \
  --episodes 100
```

**Or manually disable ground radar:**
```bash
# Edit config.yaml
nano config.yaml

# Find ground_radar section and change:
ground_radar:
  enabled: false  # ← Set to false
```

**Then run inference:**
```bash
python inference.py --model checkpoints/best --mode offline --episodes 100
```

---

## How to Check Model Dimension

```bash
python3 << 'EOF'
from stable_baselines3 import PPO
model = PPO.load("checkpoints/best/best_model.zip")
print(f"Model observation space: {model.observation_space.shape}")
# Output: (17,) = old model, (26,) = new model
