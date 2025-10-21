# DIAGNOSIS: Fundamental Learning Failure

**Date**: 2025-10-21
**Result**: 20% inference success (5M steps, frozen curriculum @ 200m)
**Training Success**: 1.4% (41/2976 episodes) - **CATASTROPHICALLY LOW**

## Executive Summary

Freezing the curriculum did NOT fix the problem. The policy is fundamentally unable to learn this task with the current setup.

**Training success rate over 5M steps**: 0% → 4% → 1% → 0% → 0%
**Inference success rate**: 20% (better than training, but still terrible)

## The Core Problem: Policy CAN'T Learn from Experience

### Training Success Rate Progression
```
Step        0: 0.00%
Step   1.12M: 4.00%  ← Brief spike
Step   2.39M: 1.00%  ← Collapsed
Step   3.66M: 0.00%  ← Complete failure
Step   4.89M: 0.00%  ← No learning

Individual: 41 successes / 2,976 episodes = 1.4%
```

**This means**: Out of ~3,000 training episodes, only 41 succeeded. The policy is essentially **never** experiencing successful intercepts to learn from.

### Value Loss Remains High
```
Start: 38,421
2.5M:   6,578  (improved!)
End:    7,127  (still 70x target)
```

Value function improved but still far from converged (<100 target).

### Comparison: All Models Fail Training

| Model | Training Steps | Curriculum | vf_coef | Training Success | Inference |
|-------|----------------|------------|---------|------------------|-----------|
| 33% | 15M | 200m→20m | 0.5 | Unknown | **33%** |
| 18% | 15M | 200m→20m | 1.0 | **0.7%** | 18% |
| 20% | 5M | **FROZEN 200m** | 0.5 | **1.4%** | 20% |

**Key Insight**: Inference consistently performs 10-20x BETTER than training. This is backwards - training should be easier!

## Why Is Inference Better Than Training?

### Hypothesis: Training vs Inference Mismatch

**During Training**:
- Stochastic policy (exploration)
- Multiple parallel environments with varying scenarios
- Random initial conditions every episode
- Policy constantly updating (non-stationary)

**During Inference**:
- Deterministic policy (exploitation only)
- Sequential episodes
- Best model selected (eval callback chose best checkpoint)
- Stable policy

**Implication**: The policy IS learning something useful, but:
1. Exploration noise prevents it from succeeding during training
2. Best model checkpoint captures lucky good policy
3. Deterministic inference filters out bad actions

## Root Cause Analysis

### Issue 1: Task is Too Hard for Radar-Only Observations ❌

**Evidence**:
- Consistent failure across ALL configurations
- Success rate never exceeds 5% during training
- Value function won't converge (loss ~7,000)
- Even frozen at easiest difficulty (200m), can't learn

**26D Radar Observations**:
```
[0:3]   Relative position (if detected, else -2.0)
[3:6]   Relative velocity (if detected, else -2.0)
[6:9]   Interceptor velocity (body frame)
[9:12]  Interceptor angular velocity
[12:15] Interceptor orientation
[15]    Fuel remaining
[16]    Radar quality
[17]    Radar detected flag
[18]    Radar range (if detected)
[19]    Ground radar detected flag
[20:23] Ground relative position (if detected)
[23:26] Ground relative velocity (if detected)
```

**Problems**:
1. **Partial observability**: Radar dropouts create observation gaps
2. **No position history**: Can't build trajectory estimate
3. **Body frame velocity**: Harder to reason about than world frame
4. **Sentinel values (-2.0)**: Network must learn to handle missing data

**Comparison**: Omniscient systems (with full state) typically succeed 80-95%

### Issue 2: Reward Function Provides Insufficient Gradient ⚠️

**Current 2-Phase Reward**:
```python
if distance > 500:
    reward = distance_delta * 10.0
else:
    reward = distance_delta * 20.0 + 50 * exp(-distance/100)

reward -= 0.1  # Time penalty
```

**Problems**:
1. **Only rewards distance reduction**: No credit for good behaviors
2. **No intermediate rewards**: Radar acquisition, velocity matching, approach angle all ignored
3. **Sparse terminal reward**: +500 for intercept, -500 for miss
4. **Failure modes indistinguishable**: Missing by 50m same as missing by 2000m (both -500)

**Evidence**: Policy learns approach (failures at 585m vs 2,081m initially) but not final intercept

### Issue 3: Observation Space May Be Fundamentally Insufficient 🔴

**The radar-only constraint may be impossible for PPO to solve.**

Why?
- Partial observability requires memory (LSTM/Transformer)
- PPO with MLP has NO memory
- Can't integrate radar detections over time
- Can't estimate target trajectory from sparse observations
- Can't handle missing data (-2.0 sentinels) effectively

**Classic RL result**: POMDPs (Partially Observable MDPs) need recurrent policies or extensive domain knowledge in observations.

## Failed Experiments Summary

| Attempt | Change | Result | Conclusion |
|---------|--------|--------|------------|
| 1 | Smart early termination | 33% | Baseline |
| 2 | vf_coef=1.0 | 18% | **WORSE** - slowed policy learning |
| 3 | Frozen curriculum (200m) | 20% | **STILL BAD** - task too hard |

**Pattern**: No configuration improves training success rate above 5%

## The Uncomfortable Truth

**This task may be too difficult for**:
1. Radar-only observations (partial observability)
2. MLP policy (no memory)
3. 2-phase reward (insufficient gradient)
4. PPO algorithm (sample inefficient for sparse rewards)

## Recommended Solutions

### Option 1: Add Memory to Policy 🎯 **RECOMMENDED**
**Use LSTM or GRU recurrent policy**

```python
# In train.py
policy_kwargs = dict(
    net_arch=[128, 128],
    activation_fn=torch.nn.ReLU,
    lstm_hidden_size=128,  # ← Add LSTM
    n_lstm_layers=1
)

model = PPO(
    "MlpLstmPolicy",  # ← Change policy type
    ...
)
```

**Why this helps**:
- LSTM maintains hidden state across timesteps
- Can integrate radar detections over time
- Builds trajectory estimate from partial observations
- Handles missing data (-2.0 sentinels) naturally

**Expected**: Training success 20% → 50%+

**Cost**: 2-3x slower training, more complex

### Option 2: Enrich Observations 🎯 **HIGH IMPACT**
**Add derived features to help policy**:

```python
# Add to observation space:
- Time-averaged position (last 10 detections)
- Estimated target trajectory (Kalman filter)
- Predicted intercept point
- Closing velocity
- Time to closest approach
- Approach angle quality metric
```

**Why this helps**:
- Provides "domain knowledge" policy can't learn
- Reduces partial observability
- Gives intermediate goals to optimize

**Expected**: Training success 1.4% → 30%+

**Cost**: More engineering, deviates from pure radar

### Option 3: Dense Reward Shaping 🎯 **HIGH IMPACT**
**Add intermediate rewards for good behaviors**:

```python
# Reward components:
+10/step: Radar has lock on target
+20/step: Closing velocity > 50 m/s
+50/step: Within 1000m and approaching
+100/step: Within 500m and good angle
-10/step: No radar lock
-20/step: Increasing distance

# Terminal rewards stay the same:
+500: Intercept
-500: Miss
```

**Why this helps**:
- Provides learning signal every step
- Rewards incremental progress
- Distinguishes "getting closer" from "random motion"

**Expected**: Training success 1.4% → 20%+

**Cost**: Careful tuning needed (can create local optima)

### Option 4: Curriculum of Scenarios 🎯 **MEDIUM IMPACT**
**Instead of shrinking intercept radius, vary scenario difficulty**:

```yaml
# Stage 1 (0-2M steps): Easy scenarios
- Missile slow (50 m/s)
- Straight trajectory
- Close initial distance (1000m)
- Perfect radar

# Stage 2 (2M-4M): Medium
- Missile faster (100 m/s)
- Some evasion
- Medium distance (2000m)
- 90% radar reliability

# Stage 3 (4M+): Hard
- Missile very fast (150 m/s)
- Active evasion
- Far distance (3000m)
- 75% radar reliability
```

**Why this helps**:
- Policy learns basics on easy scenarios
- Gradually adds complexity
- Each stage buildson previous

**Expected**: Training success 1.4% → 10-15%

**Cost**: Requires scenario generation system

### Option 5: Switch to SAC or DDPG 🎯 **LOW PRIORITY**
**Use off-policy algorithm instead of PPO**:

- SAC (Soft Actor-Critic)
- DDPG (Deep Deterministic Policy Gradient)

**Why this might help**:
- More sample efficient
- Better for sparse rewards
- Maintains replay buffer of successes

**Expected**: Uncertain (may not fix fundamental issues)

**Cost**: Different hyperparameters, different training dynamics

## My Recommendation: Implement Options 1 + 2 + 3

**Phase 1: Quick Wins (1-2 hours)**
1. Add dense reward shaping (Option 3)
2. Add derived observation features (Option 2)
3. Train for 5M steps, monitor training success rate

**Expected**: Training success 1.4% → 20-30%

**Phase 2: If Phase 1 Insufficient (2-3 hours)**
4. Implement LSTM policy (Option 1)
5. Train for 10M steps with memory

**Expected**: Training success 30% → 60%+

**Phase 3: Polish (if needed)**
6. Implement scenario curriculum (Option 4)
7. Fine-tune reward weights

**Target**: Training success 60%+ → Inference 80-90%

## Why Current Setup Fails

The combination of:
1. Partial observability (radar dropouts)
2. No memory (MLP policy)
3. Sparse reward (distance only)
4. Sample-inefficient algorithm (PPO)

Creates an **impossible learning problem** for the current setup.

**The policy needs EITHER**:
- Memory (LSTM) to integrate observations over time
- OR richer observations with derived features
- OR much denser reward signal
- OR all three

## Conclusion

**It's not about MORE steps or MORE CPUs.** The fundamental setup is broken.

**Good news**: You now have visibility (success rate tracking works!)

**Bad news**: Success rate reveals the policy can't learn this task

**Next steps**: Implement combination of:
1. Dense reward shaping (fastest to implement)
2. Observation enrichment (medium effort)
3. LSTM policy (highest impact, most work)

Start with reward shaping + observation enrichment. If that doesn't get training success >30%, switch to LSTM policy.
