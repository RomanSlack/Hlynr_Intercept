# Option A: Proximity Fuze Success Criteria

## Discovery Summary

### What We Found

Running evaluation with `precision_mode: true` revealed the model achieves **much better precision than metrics showed**:

| Episode | Min Distance | Final Distance | Notes |
|---------|--------------|----------------|-------|
| ep_0021 | **16.20m** | 282.29m | Sub-20m achieved! |
| ep_0005 | 30.81m | 490.80m | Sub-35m |
| ep_0017 | 31.13m | 306.56m | Sub-35m |
| ep_0033 | 35.62m | 130.57m | Sub-40m |
| ep_0029 | 41.97m | 203.05m | Sub-50m |

**The model gets close but flies past the target.** In real missile defense, this is actually a **successful intercept** - the proximity fuze would detonate when within kill radius.

### Real-World Context

- **Hit-to-kill (kinetic)**: 0-2m needed
- **Proximity fuze fragmentation**: 5-30m effective
- **Blast fragmentation**: 10-50m effective

A 16m minimum distance is well within **proximity fuze kill range**.

---

## Issue #1: Success Criteria

### Current Behavior
- Success = final distance < threshold at episode termination
- Model flies through target, final distance is large
- Shows 0% success rate despite achieving 16m proximity

### Fix Required
- Success = **minimum distance** < threshold at any point during episode
- This matches real proximity fuze behavior
- Episode should terminate when min distance threshold crossed

---

## Issue #2: Threat Approach Direction

### Current Spawn Configuration (from `config.yaml`)

```yaml
missile_spawn:
  position: [[800, 800, 800], [1500, 1500, 1500]]
  velocity: [[-60, -60, -30], [-100, -100, -50]]
```

**Problem:** Missiles always spawn in the **positive X, Y, Z octant** and fly toward origin. This is only ~12.5% of possible approach directions (1 of 8 octants).

### Required Fix
Missiles should spawn from **all directions** (full 360° azimuth, varied elevation) to ensure model generalizes.

---

## Implementation Plan

### Step 1: Fix Success Criteria in Environment

**File:** `rl_system/environment.py`

Add proximity fuze termination - when minimum distance drops below kill radius, terminate with success:

```python
# In step() method, after tracking min distance:
if self._episode_min_distance < self.proximity_kill_radius:
    intercepted = True
    terminated = True
```

Add config option:
```python
self.proximity_kill_radius = self.config.get('proximity_kill_radius', 20.0)
```

### Step 2: Fix Spawn Directions

**File:** `rl_system/config.yaml` (and training configs)

Change missile spawns to cover all directions:

```yaml
missile_spawn:
  # Spawn in sphere shell around origin, 800-1500m radius
  # Full 360° coverage
  position_mode: "spherical"  # New mode
  radius_min: 800.0
  radius_max: 1500.0
  azimuth_range: [0, 360]      # Full horizontal coverage
  elevation_range: [10, 60]    # Above horizon, realistic ballistic

  # Velocity toward target (origin)
  velocity_mode: "toward_target"
  speed_min: 80.0
  speed_max: 150.0
```

Or simpler approach - use symmetric spawn box:
```yaml
missile_spawn:
  position: [[-1500, -1500, 500], [1500, 1500, 1500]]  # All quadrants
  velocity: [[-100, -100, -50], [100, 100, -20]]       # Various directions
```

### Step 3: Update Evaluation Script

**File:** `rl_system/scripts/evaluate_hrl.py`

Change success metric to use minimum distance:

```python
# In _compute_metrics():
success = episode['min_distance'] < self.success_threshold
```

### Step 4: Retrain with Fixed Spawns

After fixing spawn directions, retrain specialists to handle all approach angles.

---

## Commands Used to Discover This

### Working HRL Evaluation (100m threshold, standard mode)
```bash
python scripts/evaluate_hrl.py \
  --selector checkpoints/hrl/selector/20251126_113539_500000steps/final/model.zip \
  --search checkpoints/hrl/specialists/search/20251125_174448_500000steps/final/model.zip \
  --track checkpoints/hrl/specialists/track/20251125_175356_500000steps/final/model.zip \
  --terminal checkpoints/hrl/specialists/terminal/20251129_130451_1500000steps/final/model.zip \
  --episodes 50 --seed 44
```
**Result:** 34% success, 168m mean, clustering at 99m threshold

### Precision Mode Evaluation (revealed true min distances)
```bash
python scripts/evaluate_hrl.py \
  --config configs/eval_precision.yaml \
  --selector checkpoints/hrl/selector/20251126_113539_500000steps/final/model.zip \
  --search checkpoints/hrl/specialists/search/20251125_174448_500000steps/final/model.zip \
  --track checkpoints/hrl/specialists/track/20251125_175356_500000steps/final/model.zip \
  --terminal checkpoints/hrl/specialists/terminal/20251129_130451_1500000steps/final/model.zip \
  --episodes 50 --seed 44
```
**Result:** 0% success (wrong metric), but **16.2m best min distance achieved!**

---

## Working Checkpoints Reference

| Component | Checkpoint Path |
|-----------|-----------------|
| Selector | `checkpoints/hrl/selector/20251126_113539_500000steps/final/model.zip` |
| Search | `checkpoints/hrl/specialists/search/20251125_174448_500000steps/final/model.zip` |
| Track | `checkpoints/hrl/specialists/track/20251125_175356_500000steps/final/model.zip` |
| Terminal | `checkpoints/hrl/specialists/terminal/20251129_130451_1500000steps/final/model.zip` |

---

## Priority Order

1. **High:** Add proximity fuze termination (quick win, shows true success rate)
2. **High:** Fix spawn directions for full coverage
3. **Medium:** Retrain with fixed spawns
4. **Low:** Fine-tune kill radius based on real PAC-3/THAAD specs

---

## Expected Outcome

After implementing proximity fuze termination with 20m kill radius:
- Current model should show ~2-4% sub-20m success rate
- ~16% sub-50m success rate
- Proper training with all-direction spawns should improve further
