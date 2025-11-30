# HRL Line-of-Sight Frame Implementation Plan

## Status Update (November 30, 2025)

### What We Tried

| Approach | Best Min Distance | Sub-50m Rate | Notes |
|----------|------------------|--------------|-------|
| **Single-octant (baseline)** | **19.87m** ✅ | 14% (7/50) | Works but only from one direction |
| 360° world-frame | 310m | 0% | Catastrophic failure |
| 360° body-frame rotation-invariant | 63m | 0% | Better but not learning |
| 360° all specialists retrained | 49.67m | 2% (1/50) | Slight improvement, still poor |

### Root Cause

Current observations use **absolute XYZ coordinates** (world-frame or body-frame). A missile from +X looks completely different from one from -X, even with body-frame transform. The model memorizes patterns for one direction and can't generalize.

### Literature Solution

Research papers use **Line-of-Sight (LOS) frame** observations:
- Only relative geometry matters (range, bearing rate, closing velocity)
- Naturally invariant to approach direction
- Same technique used in proportional navigation since 1950s

---

## Implementation Plan

### Phase 1: LOS Observation Space

Replace current 26D observations with LOS-based observations:

**Current (direction-dependent):**
```
[0-2]: Relative position XYZ (world or body frame)
[3-5]: Relative velocity XYZ
[6-8]: Interceptor velocity XYZ
[9-11]: Interceptor orientation (euler angles)
...
```

**New (direction-invariant):**
```
[0]: Range (distance to target)
[1]: Range rate (closing velocity, negative = closing)
[2]: LOS azimuth rate (bearing change rate, horizontal)
[3]: LOS elevation rate (bearing change rate, vertical)
[4]: Off-axis angle (angle between interceptor velocity and LOS)
[5]: Lead angle (angle between LOS and target velocity)
[6]: Interceptor speed (magnitude)
[7]: Target speed estimate (magnitude)
[8]: Time-to-intercept estimate
[9]: Fuel fraction
[10-11]: Interceptor pitch/yaw rates (for control feedback)
[12-15]: Ground radar equivalents (range, range rate, LOS rates)
[16-17]: Quality metrics (radar lock, fusion confidence)
```

**Key properties:**
- All scalars or angles relative to LOS
- Same observation for same geometry, regardless of world orientation
- ~18D instead of 26D (simpler)

### Phase 2: Modify `core.py`

Add new observation mode `los_frame`:

```python
def compute_los_observations(self, interceptor, missile, ...):
    # Geometry
    rel_pos = missile_pos - interceptor_pos
    range_to_target = np.linalg.norm(rel_pos)
    los_unit = rel_pos / (range_to_target + 1e-6)

    # Range rate (closing velocity)
    rel_vel = missile_vel - interceptor_vel
    range_rate = np.dot(rel_vel, los_unit)

    # LOS angular rates
    los_cross_vel = rel_vel - range_rate * los_unit
    los_rate = np.linalg.norm(los_cross_vel) / (range_to_target + 1e-6)

    # Decompose into azimuth/elevation rates
    # ... (project onto horizontal and vertical planes)

    # Off-axis angle
    int_vel_unit = interceptor_vel / (np.linalg.norm(interceptor_vel) + 1e-6)
    off_axis = np.arccos(np.clip(np.dot(int_vel_unit, los_unit), -1, 1))

    # Time to intercept
    tti = range_to_target / (-range_rate + 1e-6) if range_rate < 0 else 999

    return np.array([range_to_target, range_rate, los_az_rate, los_el_rate, ...])
```

### Phase 3: Config Flag

```yaml
environment:
  observation_mode: "los_frame"  # Options: "world_frame", "body_frame", "los_frame"
```

### Phase 4: Retrain All Specialists

With LOS observations, train fresh:
1. Search specialist (3M steps)
2. Track specialist (3M steps)
3. Terminal specialist (3M steps)
4. Selector (500k steps)

### Phase 5: Evaluate on 360° Spawns

Should now generalize to any approach direction.

---

## Expected Outcome

- **Direction invariance**: Same observation for same relative geometry
- **Simpler learning**: Fewer dimensions, more meaningful features
- **Proven approach**: Used in real missile guidance systems
- **Full 360° coverage** with single trained model

---

## Files to Modify

| File | Change |
|------|--------|
| `rl_system/core.py` | Add `compute_los_observations()` method |
| `rl_system/environment.py` | Add `observation_mode` config option |
| `rl_system/configs/*.yaml` | Add `observation_mode: los_frame` |

---

## References

- [MDPI: Study on RL-Based Missile Guidance Law](https://www.mdpi.com/2076-3417/10/18/6567) - Uses LOS rate + velocity as inputs
- [MIT: Reinforcement Metalearning for Interception](https://dspace.mit.edu/bitstream/handle/1721.1/145416/2004.09978.pdf)
- [Tandfonline: Deep Recurrent RL for Intercept Guidance](https://www.tandfonline.com/doi/full/10.1080/08839514.2024.2355023)

---

## Timeline Estimate

- Phase 1-3 (Implementation): 2-3 hours
- Phase 4 (Training): ~2 hours (3 specialists × 35 min + selector)
- Phase 5 (Evaluation): 10 minutes

---

## Working Checkpoint Reference (Current Best)

Until LOS is implemented, this single-octant config achieves 19.87m:

```bash
python scripts/evaluate_hrl.py \
  --config configs/eval_proximity_fuze.yaml \
  --selector checkpoints/hrl/selector/20251126_113539_500000steps/final/model.zip \
  --search checkpoints/hrl/specialists/search/20251125_174448_500000steps/final/model.zip \
  --track checkpoints/hrl/specialists/track/20251125_175356_500000steps/final/model.zip \
  --terminal checkpoints/hrl/specialists/terminal/20251129_130451_1500000steps/final/model.zip \
  --episodes 50 --seed 44
```
