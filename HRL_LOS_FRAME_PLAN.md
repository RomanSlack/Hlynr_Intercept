# HRL Line-of-Sight Frame Implementation Plan

## Status Update (November 30, 2025) - V2: Full LOS Frame

### Implementation Complete

**CRITICAL UPDATE**: LOS frame now includes BOTH observations AND actions.
- Observations: Direction-invariant geometry (range, LOS rates, off-axis angle)
- Actions: LOS-relative thrust (along LOS, perpendicular horizontal, perpendicular vertical)

This makes the entire control loop direction-invariant - the same policy works from any approach direction.

### Training History

| Approach | Best Min Distance | Sub-50m Rate | Notes |
|----------|------------------|--------------|-------|
| **Single-octant (baseline)** | **19.87m** ✅ | 14% (7/50) | Works but only from one direction |
| 360° world-frame | 310m | 0% | Catastrophic failure |
| 360° body-frame rotation-invariant | 63m | 0% | Better but not learning |
| 360° body-frame all specialists retrained | 49.67m | 2% (1/50) | Slight improvement, still poor |
| 360° LOS obs only (no LOS actions) | 425m mean, 82m best | 0% | Observations alone not enough |
| **360° Full LOS (obs + actions)** | TBD | TBD | **Current approach - needs training** |

### Root Cause Analysis

The previous LOS implementation only changed **observations** to be direction-invariant. But the **action space** was still world-frame thrust vectors `[thrust_x, thrust_y, thrust_z]`.

**The Problem**: With LOS observations but world-frame actions, the model must learn an implicit mapping:
- "I see range closing at X rate, LOS rotating at Y rate" → "I should thrust in world direction [a,b,c]"
- This mapping changes depending on WHERE in the world the engagement happens
- This defeats the purpose of direction-invariant observations!

**The Solution**: Transform actions to LOS-relative frame:
- `action[0]`: Thrust along LOS (toward target)
- `action[1]`: Thrust perpendicular to LOS (horizontal correction)
- `action[2]`: Thrust perpendicular to LOS (vertical correction)

Now the same action "thrust toward target with lateral correction" produces the same result regardless of world orientation.

---

## Implementation Details

### LOS Observation Space (26D, same as before)
```
[0]: Range to target (normalized, 0-1)
[1]: Range rate (positive = closing, normalized)
[2]: LOS azimuth rate (horizontal bearing change rate)
[3]: LOS elevation rate (vertical bearing change rate)
[4]: Off-axis angle cos (1.0 = on collision course)
[5]: Lead angle cos (LOS vs target velocity direction)
[6]: Interceptor speed (magnitude)
[7]: Vertical climb rate
[8]: Horizontal speed
[9-11]: Reserved (zeros in LOS mode)
[12]: Fuel fraction
[13-16]: TTI, quality, closing rate, off-axis
[17-19]: Ground radar (range, closing, LOS rate)
[20-25]: Quality metrics
```

### LOS Action Space (6D)
```
[0]: Thrust along LOS (positive = toward target)
[1]: Thrust perpendicular to LOS in horizontal plane (lateral correction)
[2]: Thrust perpendicular to LOS in vertical plane (altitude correction)
[3-5]: Angular rates (body-relative, unchanged)
```

### Transformation Logic

The environment maintains three orthogonal unit vectors defining the LOS frame:
- `los_unit`: Points from interceptor toward missile (along LOS)
- `los_horizontal`: Perpendicular to LOS in the horizontal plane
- `los_vertical`: Perpendicular to LOS in the vertical plane

Action transformation:
```python
world_thrust = (
    action[0] * los_unit +
    action[1] * los_horizontal +
    action[2] * los_vertical
)
```

### Proportional Navigation Intuition

The policy can learn proportional navigation (PN) directly:
- **To close distance**: `action[0] > 0` (thrust along LOS)
- **To achieve collision course**: `action[1,2] = -N * obs[2,3]` (thrust perpendicular to cancel LOS rotation)
- **N (navigation gain)**: The policy learns optimal N for different engagement phases

This is exactly how real missile guidance works!

---

## Files Modified

| File | Change |
|------|--------|
| `rl_system/core.py` | LOS observation computation (already done) |
| `rl_system/environment.py` | Added `_update_los_frame()` and `_transform_los_action_to_world()` |
| `rl_system/configs/hrl/search_los.yaml` | Updated docs for LOS actions |
| `rl_system/configs/hrl/track_los.yaml` | Updated docs for LOS actions |
| `rl_system/configs/hrl/terminal_los.yaml` | Updated docs for LOS actions |
| `rl_system/configs/hrl/selector_los.yaml` | Uses LOS mode |
| `rl_system/configs/eval_360_los.yaml` | Evaluation config for LOS models |

---

## Training Commands

```bash
# Train Search specialist with full LOS frame
python scripts/train_hrl_pretrain.py --agent search --config configs/hrl/search_los.yaml

# Train Track specialist
python scripts/train_hrl_pretrain.py --agent track --config configs/hrl/track_los.yaml

# Train Terminal specialist
python scripts/train_hrl_pretrain.py --agent terminal --config configs/hrl/terminal_los.yaml

# Train Selector (after specialists complete)
python scripts/train_hrl_selector.py \
  --config configs/hrl/selector_los.yaml \
  --specialist-dir checkpoints/hrl/specialists/

# Evaluate
python scripts/evaluate_hrl.py \
  --config configs/eval_360_los.yaml \
  --selector checkpoints/hrl/selector/<timestamp>/final/model.zip \
  --search checkpoints/hrl/specialists/search/<timestamp>/final/model.zip \
  --track checkpoints/hrl/specialists/track/<timestamp>/final/model.zip \
  --terminal checkpoints/hrl/specialists/terminal/<timestamp>/final/model.zip \
  --episodes 50 --seed 44
```

---

## Expected Outcome

With full LOS frame (observations + actions):
- **Perfect direction invariance**: Same observation AND action produces same result from any direction
- **Natural PN learning**: Action space directly supports proportional navigation
- **Simpler credit assignment**: "Thrust toward target" is action[0], not a complex world-frame vector
- **Full 360° coverage**: Single trained model should work from any approach direction

---

## References

- [MDPI: Study on RL-Based Missile Guidance Law](https://www.mdpi.com/2076-3417/10/18/6567)
- [MIT: Reinforcement Metalearning for Interception](https://dspace.mit.edu/bitstream/handle/1721.1/145416/2004.09978.pdf)
- [Tandfonline: Deep Recurrent RL for Intercept Guidance](https://www.tandfonline.com/doi/full/10.1080/08839514.2024.2355023)
- Zarchan, P. "Tactical and Strategic Missile Guidance" - Proportional Navigation theory

---

## Working Checkpoint Reference (Single-Octant Best)

For comparison, the single-octant config achieves 19.87m:

```bash
python scripts/evaluate_hrl.py \
  --config configs/eval_proximity_fuze.yaml \
  --selector checkpoints/hrl/selector/20251126_113539_500000steps/final/model.zip \
  --search checkpoints/hrl/specialists/search/20251125_174448_500000steps/final/model.zip \
  --track checkpoints/hrl/specialists/track/20251125_175356_500000steps/final/model.zip \
  --terminal checkpoints/hrl/specialists/terminal/20251129_130451_1500000steps/final/model.zip \
  --episodes 50 --seed 44
```
