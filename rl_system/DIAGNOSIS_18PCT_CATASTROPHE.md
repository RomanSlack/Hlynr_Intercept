# DIAGNOSIS: 18% Catastrophic Performance Degradation

**Date**: 2025-10-21
**Training**: `training_20251021_150516_15000000steps` (15M steps, vf_coef=1.0)
**Inference**: `offline_run_20251021_164906` (18% success)
**Previous**: 33% success (7M & 15M runs with vf_coef=0.5)

## Executive Summary

Performance **DEGRADED** from 33% → 18% despite fixes. However, the real problem is **NOT the vf_coef change** - the model is **fundamentally not learning** during training.

### Critical Findings

**Training Success Rate**: 0.7% (62 successes out of 8,649 episodes)
**Inference Success Rate**: 18% (tested on easy 200m radius)
**Conclusion**: Model never learned to intercept during training, even at easy difficulty

## Root Cause Analysis

### Issue 1: ❌ Training Intercept Rate is CATASTROPHICALLY LOW

**Evidence from TensorBoard**:
```
SUCCESS RATE DURING TRAINING:
  Step        0: 0.00%
  Step   3.75M: 0.00%
  Step   7.50M: 0.00%
  Step  11.25M: 0.00%
  Step  15.00M: 0.00%

INDIVIDUAL EPISODES:
  Total episodes: 8,649
  Successes: 62 (0.7%)
  Failures: 8,587 (99.3%)
```

**This is NOT normal** - even random policy should occasionally get lucky. Something is fundamentally broken in the training loop.

### Issue 2: ⚠️ Curriculum Shrinks Too Aggressively

**Intercept Radius During Training**:
| Step | Radius | Success Needed |
|------|--------|----------------|
| 0 | 200m | Get within 200m |
| 3.75M | 155m | Get within 155m |
| 7.5M | 110m | Get within 110m |
| 11.25M | 65m | Get within 65m |
| 15M | **20m** | Get within 20m! |

**Problem**: By 15M steps, the policy must intercept within **20m** (proximity fuse accuracy). But it never learned to intercept at 200m in the first place!

**Inference Testing**: Uses "easy" scenario with **200m radius**
- Model trained at 20m radius
- Tested at 200m radius
- Got 18% success with 10x easier threshold!

**Implication**: At training's final 20m radius, true success rate is probably <1%

### Issue 3: ❌ vf_coef=1.0 May Have Hurt Policy Learning

**Value Loss Progression**:
```
vf_coef=0.5 (33% model):  9,030 (never converged)
vf_coef=1.0 (18% model):  9,728 (WORSE!)
```

**Training Reward**:
```
vf_coef=0.5: 33,031 final reward
vf_coef=1.0: 33,587 final reward (similar)
```

**Analysis**: Doubling vf_coef did NOT improve value convergence. In fact, value loss is slightly WORSE. This suggests the value function is fundamentally unable to converge with current setup, and increasing vf_coef just **slowed down policy learning** without helping value.

### Issue 4: ✅ Success Rate Tracking Works (But Reveals Bad News)

The good news: Our callback fix WORKED! We can now see training metrics in real-time.

The bad news: The metrics reveal the model is not learning at all.

## Why Did 33% Model Work Better?

The previous 33% model likely:
1. **Trained longer at easier difficulties** (curriculum may have progressed slower in early steps)
2. **Had better policy/value balance** (vf_coef=0.5 let policy learn faster)
3. **Got lucky** with better initial random policy

## Comparison: 33% vs 18% Model

| Metric | 33% Model (vf=0.5) | 18% Model (vf=1.0) | Change |
|--------|--------------------|--------------------|--------|
| Training Success | Unknown (no tracking) | 0.7% | N/A |
| Value Loss (final) | 9,030 | 9,728 | +7.7% WORSE |
| Reward (final) | 33,031 | 33,587 | +1.7% better |
| Inference Success | 33% @ 200m | 18% @ 200m | -45% WORSE |
| Failure Distance | 2,081m avg | 437m avg | Much closer! |
| Steps (success) | 908 | 1,247 | +37% slower |

**Key Insight**: The 18% model is getting **much closer** (437m vs 2,081m failures) but succeeding less often. This suggests it's learning SOMETHING about navigation but not finalinterception.

## Failure Analysis

### Inference Results (18% model, 200m radius):
- **Successes**: 18/100 (18%)
  - Avg final distance: 199.3m (right at threshold)
  - Avg steps: 1,247 (12.5 seconds)

- **Failures**: 82/100 (82%)
  - Avg final distance: 437m (vs 2,081m for 33% model!)
  - Avg steps: 1,597 (16 seconds)
  - Early terminations: 0 (smart termination working correctly)

**Key Finding**: Failures are 5x CLOSER than 33% model (437m vs 2,081m). The model IS learning approach trajectory but failing at final intercept.

## Hypotheses for Why Training Failed

### Hypothesis 1: Curriculum Too Aggressive (MOST LIKELY)
**Problem**: Radius shrinks from 200m → 20m linearly over 15M steps
- Policy never gets enough training at ANY single difficulty
- By the time it sees a difficulty, curriculum has already moved on
- Final difficulty (20m) is impossibly hard for untrained policy

**Evidence**:
- 0.7% success rate during training
- Even at START (200m radius), policy wasn't succeeding
- Inference at 200m radius (easiest) still only 18%

**Solution**: Freeze curriculum at easy (200m) until policy learns basics

### Hypothesis 2: Observation Space Issue
**Problem**: 26D radar-only observations may not provide enough information
- No direct position data (only radar detections)
- Ground radar helps but may have gaps
- Policy can't build accurate mental model

**Evidence**:
- Consistent failure across all difficulty levels
- Value function won't converge (loss ~9,000-10,000)
- Failures are closer (437m) but still miss

**Solution**: Add more informative observations (but contradicts "radar-only" requirement)

### Hypothesis 3: Reward Function Insufficient
**Current**: 2-phase reward
- Phase 1 (>500m): distance_delta * 10
- Phase 2 (<500m): distance_delta * 20 + exponential proximity

**Problem**: May not provide enough gradient signal early in episode
- Episodes are 2000 steps (20 seconds)
- Most learning happens in first 10 seconds
- Reward only meaningful when <500m

**Evidence**:
- Policy learns approach (gets to ~437m)
- But fails at final intercept (<200m)

**Solution**: Add intermediate rewards for radar lock, velocity matching, etc.

### Hypothesis 4: vf_coef=1.0 Hurt Policy Learning (LIKELY)
**Problem**: Doubling value function updates slowed policy learning
- Value function STILL didn't converge (9,728 vs 9,030)
- Policy got fewer gradient updates relative to value
- Result: Worse policy, same bad value function

**Evidence**:
- Value loss WORSE with vf=1.0 (9,728 vs 9,030)
- Training success rate 0.7% (catastrophic)
- Previous model with vf=0.5 got 33%

**Solution**: Revert to vf_coef=0.5 OR try separate network architectures

## Recommended Actions

### IMMEDIATE: Freeze Curriculum at Easy
**Priority: CRITICAL**

**Change config.yaml**:
```yaml
curriculum:
  enabled: true
  initial_radius: 200.0
  final_radius: 200.0  # ← FREEZE at easy
  curriculum_steps: 15000000
```

**Rationale**: Policy must master basics (200m intercept) before facing harder challenges

**Expected**: Success rate during training should reach 50%+ if fundamentals work

### Action 2: Revert vf_coef
**Priority: HIGH**

```yaml
vf_coef: 0.5  # Revert from 1.0
```

**Rationale**: vf_coef=1.0 made things worse, not better

**Expected**: Policy should learn faster, like 33% model did

### Action 3: Simplify to Verify Basics Work
**Priority: HIGH**

**Temporarily disable physics complexity**:
```yaml
physics_enhancements:
  sensor_delays:
    enabled: false
  mach_effects:
    enabled: false
  thrust_dynamics:
    enabled: false
  enhanced_wind:
    enabled: false
```

**Rationale**: Rule out physics complexity as blocker

**Expected**: If policy still can't learn, problem is in core observation/reward

### Action 4: Add Intermediate Rewards
**Priority: MEDIUM**

**Enhance reward function** with:
- Radar lock bonus: +10 per step when target detected
- Velocity alignment: reward for matching target velocity vector
- Closing rate: reward for positive closing velocity

**Rationale**: Provide more gradient signal throughout episode

**Expected**: Policy learns incrementally rather than needing perfect intercept

### Action 5: Increase Network Size
**Priority: LOW (try others first)**

```yaml
net_arch: [256, 256]  # ← Up from [128, 128]
```

**Rationale**: 26D observation space may need more capacity

**Expected**: Value function might converge better

## Testing Protocol

### Test 1: Frozen Curriculum + Reverted vf_coef (5M steps)
```yaml
curriculum:
  initial_radius: 200.0
  final_radius: 200.0  # Frozen
vf_coef: 0.5  # Reverted
total_timesteps: 5000000  # Shorter test
```

**Success Criteria**:
- Training success rate > 30% by 2M steps
- Training success rate > 60% by 5M steps
- Value loss < 5,000 by 5M steps

**If SUCCESS**: Gradually enable curriculum
**If FAILURE**: Problem is more fundamental (observation/reward)

### Test 2: Add Intermediate Rewards (if Test 1 fails)
**Only if frozen curriculum doesn't work**

Add radar lock bonus and velocity matching rewards

**Success Criteria**:
- Training success rate > 20% by 2M steps
- Shows improving trend

### Test 3: Verify Environment Basics
**Sanity check if both above fail**

```python
# Test random policy for 100 episodes
for ep in range(100):
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
    print(f"Episode {ep}: {info['intercepted']}")
```

**Expected**: ~0% success (confirms task is hard)
**If >5% success**: Random policy works, RL should too

## Conclusion

The 18% result is actually a **training failure**, not just a performance degradation. The model never learned to intercept during training (0.7% success rate).

**Root cause**: Combination of:
1. **Curriculum too aggressive** (primary issue)
2. **vf_coef=1.0 hurt policy learning** (secondary issue)
3. **Possibly insufficient reward gradient**

**Next steps**:
1. Freeze curriculum at easy (200m radius)
2. Revert vf_coef to 0.5
3. Train for 5M steps and monitor success rate
4. If still fails, add intermediate rewards

**Do NOT need**: More steps or CPUs. 15M steps is plenty IF the setup works.
