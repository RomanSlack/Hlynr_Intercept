# Training Diagnosis: 33% Success Rate Plateau

**Date**: 2025-10-21
**Training Run**: `training_20251021_121632_15000000steps`
**Inference Results**: `/home/roman/Hlynr_Intercept/rl_system/inference_results/offline_run_20251021_140253`

## Executive Summary

After 15M training steps with all optimizations applied (smart early termination, simplified 2-phase reward, [128,128] network, slow curriculum), the model achieved **33% success rate** - identical to the previous 7M model. This indicates a **performance plateau** rather than a regression.

## Key Findings

### 1. ✓ Smart Early Termination IS Working
- Code is correctly implemented in `environment.py:493-511`
- Tracks distance worsening over 500 steps
- Only terminates if distance > 2500m AND consistently worsening
- **Result**: Only 17/67 failures (25%) hit the ~2500m threshold
- Most failures (50/67 = 75%) are legitimate failures at 1400-2400m range

### 2. ✗ SUCCESS RATE NOT TRACKED DURING TRAINING
**CRITICAL ISSUE**: The `EpisodeLoggingCallback` is NOT successfully logging success rates to TensorBoard.

Evidence:
- TensorBoard has NO `episode/success_rate_pct` or `episode/intercepted` metrics
- Callback code exists (train.py:190-248) but is not firing correctly
- The callback checks `self.locals.get('dones', [False])[0]` which may not work correctly with VecEnv

**Impact**: We have been training BLIND without knowing actual success rate progression!

### 3. Training Metrics Analysis

From TensorBoard (15M steps):

| Metric | Start | Mid (7.5M) | End (15M) | Target | Status |
|--------|-------|------------|----------|--------|--------|
| Episode Reward | 19,367 | 32,039 | 33,031 | - | ✓ Converged |
| Value Loss | 36,932 | 5,685 | 9,030 | < 100 | ✗ **TOO HIGH** |
| Explained Variance | 0.000 | 0.962 | 0.957 | > 0.95 | ✓ Good |
| Entropy | -8.6 | -19.8 | -22.6 | (negative) | ✓ Decaying |

**CRITICAL**: Value loss of 9,030 is **90x higher than target** of < 100.

### 4. Performance Pattern Analysis

**Successes** (33/100 episodes):
- Final distance: ~199m (all hitting intercept radius threshold perfectly)
- Steps to intercept: 729-1163 (avg 908 steps = 9 seconds)
- Fuel efficient: 21-33 kg used
- **Pattern**: When it works, it works PERFECTLY

**Failures** (67/100 episodes):
- Final distance: 1400-2500m (avg 2081m)
- Steps: 1350-1824 (avg 1585 steps = 16 seconds)
- Fuel wasteful: 38-57 kg used
- **Pattern**: When it fails, it COMPLETELY fails - no "close misses"

### 5. Bimodal Behavior = Value Function Problem

The model exhibits stark bimodal behavior:
- Either intercepts at ~199m (perfect)
- Or completely fails at >1400m (catastrophic)
- **Zero episodes** in the 200-1400m range (no "almost made it")

This is a classic symptom of **poor value function convergence**:
- Policy doesn't know which trajectories lead to success
- Can't distinguish "on track" from "doomed" until too late
- Value loss of 9,030 confirms value function is poorly calibrated

## Root Cause Analysis

### Primary Issue: Value Function Not Converged
**Evidence**:
1. Value loss stuck at 9,030 (target < 100)
2. Bimodal outcomes (perfect or catastrophic, nothing in between)
3. No success rate tracking to guide curriculum transitions

**Why this matters**:
- PPO relies on value function to estimate returns
- Poor value estimates → poor advantage estimates → bad policy gradients
- Policy can't learn to "course correct" mid-flight

### Secondary Issue: Blind Training
**Evidence**:
1. No success rate metrics in TensorBoard
2. `EpisodeLoggingCallback` not working with VecEnv
3. Trained for 15M steps without knowing if curriculum was helping

**Why this matters**:
- Curriculum learning requires monitoring success rate
- Can't tell if difficulty transitions happened too early/late
- Can't diagnose training issues until after inference

### Tertiary Issue: Curriculum May Be Too Aggressive
**Current schedule** (from config.yaml):
- Intercept radius: 200m → 20m over 15M steps (linear)
- Beam width: 120° → 60° (8M-12M)
- Detection reliability: 100% → 75% (12M-14M)
- Measurement noise: 0% → 5% (14M-15M)

**At 15M steps**: All difficulty factors are at MAXIMUM
- 20m intercept radius (very tight)
- 60° beam width (narrow)
- 75% detection reliability (frequent dropouts)
- 5% measurement noise (significant)

**Problem**: Policy may have mastered easy (200m radius, 120° beam, 100% detection) but **never had enough time to adapt to hard difficulties**. The final 25% of training (11.25M-15M steps) was fighting all maximum difficulty factors simultaneously.

## Recommendations

### Priority 1: Fix Success Rate Tracking
**Action**: Repair `EpisodeLoggingCallback` to work correctly with VecEnv
- Problem: `self.locals.get('dones', [False])[0]` may not access correct info
- Solution: Check SB3 documentation for proper VecEnv callback usage
- Alternative: Use custom Monitor wrapper that logs success rate

**Impact**: Essential for diagnosing all future training runs

### Priority 2: Reduce Value Loss
**Options**:

A. **Increase value function learning** (recommended):
   - Increase `vf_coef` from 0.5 → 1.0 or 2.0
   - This forces more gradient updates on value function
   - May slow policy learning but improves convergence

B. **Extend training with frozen curriculum**:
   - Train another 5M steps at EASY difficulty (200m radius, perfect radar)
   - Let value function fully converge
   - Then resume curriculum

C. **Reduce batch complexity**:
   - Reduce `n_steps` from 2048 → 1024 (shorter rollouts)
   - Increase `batch_size` 512 → 1024 (more gradient updates per rollout)
   - More frequent value updates may help convergence

### Priority 3: Slow Down Curriculum Even More
**Current**: Final difficulty reached at 15M (100% of training)
**Proposed**: Final difficulty at 25M (extending training significantly)

**Alternative "Staged Mastery" approach**:
1. Train at EASY until value_loss < 100 AND success_rate > 80%
2. Then transition to MEDIUM difficulty
3. Train at MEDIUM until value_loss < 100 AND success_rate > 70%
4. Then transition to HARD
5. Final convergence

This requires **dynamic curriculum** based on performance, not fixed steps.

### Priority 4: Increase Network Capacity (If Value Loss Persists)
**Current**: [128, 128] with ~33k parameters
**Proposed**: [256, 256] with ~134k parameters

**Reasoning**:
- Smaller network was meant for faster convergence with simple reward
- But complex observation space (26D radar + ground data) may need more capacity
- Especially value function may need more capacity than policy

**Alternative**: Separate network architectures for policy and value
```python
policy_kwargs = dict(
    net_arch=dict(
        pi=[128, 128],  # Policy: smaller, faster
        vf=[256, 256]   # Value: larger, more accurate
    )
)
```

## Inference System Analysis

### Is Inference Broken?
**NO** - Inference is working correctly.

Evidence:
1. Success distances are consistently ~199m (exactly at intercept_radius threshold)
2. Failure patterns are consistent with training difficulties
3. Inference uses same environment code as training
4. Model loads correctly and produces reasonable actions

### Are Success Metrics Accurate?
**YES** - Metrics are accurately reporting model performance.

Evidence from `metrics.jsonl`:
- Successes: 33/100 episodes end with `"outcome": "intercepted"` and distance ~199m
- Failures: 67/100 episodes end with `"outcome": "failed"` and distance >1400m
- Total rewards align with expected values (successes: 23k-34k, failures: -434 to 12k)

## Conclusion

The model is **not broken** - it has simply **plateaued at 33% success rate** due to:
1. **Insufficient value function convergence** (value_loss 90x too high)
2. **Blind training** without success rate metrics
3. **Curriculum transitions possibly too aggressive** in final 25% of training

The **smart early termination** IS working correctly and is NOT the problem.

Next steps should focus on:
1. **Immediately**: Fix success rate tracking
2. **Next training run**: Increase vf_coef and monitor value loss carefully
3. **If still plateau**: Implement staged mastery curriculum with performance gates
