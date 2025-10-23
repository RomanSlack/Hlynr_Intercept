# MLP Baseline Experiment: Testing LSTM Instability Hypothesis

**Date**: 2025-10-23
**Experiment ID**: MLP-NoLSTM-ExtendedHorizon-FrameStack
**Status**: ✅ READY TO TRAIN

---

## Experiment Objective

Test the hypothesis from `COMPREHENSIVE_RL_DIAGNOSIS.md` that **LSTM causes training instability** and is responsible for the "inverse learning curve" pattern (performance peaks at 200k steps, then degrades to 0% by 10M steps).

**Key Question**: Can a standard MLP policy with frame-stacking and extended horizon achieve stable learning without LSTM divergence?

---

## Configuration Changes Applied

### 1. ✅ Removed LSTM Entirely

**Before** (LSTM policy):
```yaml
use_lstm: true
lstm_hidden_size: 256
n_lstm_layers: 1
enable_critic_lstm: true
```

**After** (MLP policy):
```yaml
use_lstm: false  # NO RECURRENCE
frame_stack: 4   # Stack last 4 obs for temporal context
```

**Impact**:
- Policy is now memoryless (no hidden state)
- Eliminates LSTM gradient instability
- Removes spurious correlation risk
- Observation space: 26D → 104D (4 frames × 26D)

---

### 2. ✅ Extended Time Horizon

**Before**:
```yaml
gamma: 0.99       # Effective horizon ~100 steps
gae_lambda: 0.95  # Credit assignment ~20 steps
n_steps: 1024     # Rollout length
```

**After**:
```yaml
gamma: 0.997      # Effective horizon ~333 steps (3.3× longer!)
gae_lambda: 0.97  # Credit assignment ~33 steps (1.65× longer)
n_steps: 2048     # Rollout length (2× longer)
```

**Impact**:
- GAE effective horizon: $(1 - \gamma \lambda)^{-1} = (1 - 0.997 \times 0.97)^{-1} \approx 333$ steps
- Credit assignment now reaches back ~333 steps (vs ~100 previously)
- Can attribute terminal reward to actions 300+ steps earlier
- Better gradient signal for long episodes (~1800 steps)

**Why This Matters**: Previous setup couldn't assign credit beyond ~100 steps due to exponential decay. Intercept episodes are 1800 steps, so actions at t=500 (acquisition) had essentially zero gradient from terminal reward at t=1800.

---

### 3. ✅ Frame-Stacking (4 Frames)

**Implementation**: `VecFrameStack(envs, n_stack=4)`

**Observation Structure**:
```
Frame t-3: [26D radar observations]
Frame t-2: [26D radar observations]
Frame t-1: [26D radar observations]
Frame t-0: [26D radar observations]
-------------------------------------------
Total:     [104D stacked observations]
```

**What Policy Sees**:
- Last 4 timesteps of observations (0.04 seconds @ 100Hz)
- Can compute velocities from position deltas between frames
- Can detect acceleration from velocity changes
- Provides short-term dynamics without recurrence

**Precedent**: Frame-stacking is standard in Atari DQN (Nature 2015) for vision-based RL. Proven to work with MLP policies for temporal tasks.

---

### 4. ✅ Observation Normalization (VecNormalize)

**Implementation**:
```python
VecNormalize(
    envs,
    norm_obs=True,         # Per-feature running mean/std normalization
    norm_reward=False,     # Don't normalize (already scaled)
    clip_obs=10.0,         # Clip to [-10σ, +10σ]
    training=True/False    # Update stats during training only
)
```

**Why This Matters**:
- Observations have different scales:
  - Positions: 0-3000m
  - Velocities: 0-200 m/s
  - Fuel: 0-1
  - Quality: 0-1
- Neural networks learn better with normalized inputs
- Running statistics adapt to changing observation distributions during curriculum

**Critical Detail**: Normalization statistics are **saved with model** and **frozen during evaluation**. This ensures consistent preprocessing between training and inference.

---

### 5. ✅ Larger MLP Architecture

**Before** (with LSTM):
```yaml
net_arch: [256, 256]  # 2 layers, 134k params total (78% in LSTM)
```

**After** (MLP only):
```yaml
net_arch: [512, 512, 256]  # 3 layers, larger capacity
```

**Network Structure**:
```
Input: 104D (frame-stacked observations)
↓
Layer 1: Linear(104 → 512) + LayerNorm + ReLU
↓
Layer 2: Linear(512 → 512) + LayerNorm + ReLU
↓
Layer 3: Linear(512 → 256) + LayerNorm + ReLU
↓
Policy Head: Linear(256 → 6) [actions]
Value Head: Linear(256 → 1) [state value]
```

**Total Parameters**: ~390k (vs ~340k with LSTM)

**Initialization**: Orthogonal weights with gain=√2 (recommended for RL)
**Normalization**: LayerNorm on each hidden layer (stabilizes training)

---

### 6. ✅ Increased Batch Size

**Before**:
```yaml
batch_size: 128  # Small for LSTM memory constraints
```

**After**:
```yaml
batch_size: 256  # Larger for MLP stability
```

**Impact**:
- More gradient updates per rollout
- Lower variance in policy updates
- Better value function convergence
- Standard for larger MLPs

---

### 7. ✅ Standard PPO Hyperparameters

**Reset to Defaults**:
```yaml
learning_rate: 0.0003    # 3e-4 (was 1e-4 for LSTM stability)
clip_range: 0.2          # Standard PPO clipping (was 0.15)
max_grad_norm: 0.5       # Standard gradient clipping (was 0.3)
ent_coef: 0.01          # Standard entropy (unchanged)
```

**Rationale**: LSTM required conservative hyperparameters. MLP can use standard PPO settings proven in literature.

---

## Observations Remain Radar-Only ✅

**Verified**: Policy receives ONLY sensor-realistic data
- ✅ Onboard radar measurements (5km range, 120° beam, 2% noise)
- ✅ Ground radar measurements (20km range, 5% packet loss)
- ✅ Kalman-filtered trajectory estimates (processed from radar)
- ✅ Interceptor self-state (IMU, fuel sensors)
- ✅ Derived features (closing rate, TTI, computed from radar)

**NOT Observable**:
- ❌ True missile position (only radar-measured position)
- ❌ True missile velocity (only Kalman-estimated velocity)
- ❌ Ground truth state
- ❌ Omniscient data

---

## Expected Outcomes

### Success Criteria (If MLP Solves Instability)

**Training Metrics**:
- ✅ Success rate: Stable or increasing through 10M steps (NOT inverse curve!)
- ✅ Value loss: Drops below 5000 by 5M steps (vs stuck at 9000)
- ✅ No catastrophic collapse after 2M steps

**Inference Performance**:
- ✅ Success rate >30% (match LSTM peak)
- ✅ Success rate >40% (exceed LSTM best)
- ✅ Stable across evaluation episodes (not bimodal)

**Timeline**: ~6-7 hours training on current hardware (10M steps, 16 envs)

### Failure Indicators (If MLP Also Fails)

**If success rate <15% after 5M steps**:
- Partial observability may be too hard for MLP
- LSTM was necessary (but needs different training approach)
- Consider: SAC with frame-stacking, Transformer policy, privileged learning

**If success rate peaks then collapses**:
- Problem is not LSTM-specific
- Likely reward structure or curriculum issue
- Investigate: reward function, episode length, exploration

---

## Decision Tree (Post-Training)

```
MLP Result After 10M Steps:
│
├─ Success >40% AND stable? ✅ SUCCESS
│  └─ Action: Drop LSTM permanently
│      - MLP is faster to train (no LSTM overhead)
│      - MLP is more stable (no recurrence issues)
│      - Use MLP for all future experiments
│      - Focus on improving MLP (larger network, more envs)
│
├─ Success 30-40% AND stable? ⚠️ ACCEPTABLE
│  └─ Action: MLP is viable alternative to LSTM
│      - Slightly lower performance but more stable
│      - Consider: Increase network size [1024, 1024, 512]
│      - Consider: Increase frame-stack to 8 frames
│      - Consider: Add attention mechanism for longer context
│
├─ Success 15-30% but STABLE? ⚠️ PARTIAL SUCCESS
│  └─ Action: MLP confirms LSTM was unstable
│      - Lower performance but no divergence
│      - Task may need more advanced architecture:
│        * Transformer for longer temporal context
│        * SAC for better exploration
│        * Privileged learning (train with full state, distill)
│
└─ Success <15% or collapses? ❌ FAILURE
   └─ Action: Problem is not LSTM-specific
       - Investigate reward function (still exploitable?)
       - Check curriculum (too aggressive?)
       - Test with omniscient observations (upper bound)
       - Consider task is too hard for current RL approach
```

---

## Training Command

```bash
cd /home/roman/Hlynr_Intercept/rl_system
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Aegis_11

# Start training with new MLP configuration
python train.py --config config.yaml

# Monitor progress (separate terminal)
python monitor_training.py

# TensorBoard (separate terminal)
tensorboard --logdir logs/
# Navigate to http://localhost:6006
```

---

## What to Monitor During Training

### TensorBoard Metrics (Critical)

**episode/success_rate_pct** (MOST IMPORTANT):
```
Expected Pattern (Success):
0-2M:    0% → 15%    (gradual learning)
2-5M:   15% → 30%    (steady improvement)
5-8M:   30% → 40%    (approaching asymptote)
8-10M:  40% stable   (converged)

RED FLAG Pattern (Failure - same as LSTM):
0-500k:  0% → 20%    (early lucky peak)
500k-3M: 20% → 10%   (gradual degradation)
3M-10M:  10% → 0%    (catastrophic collapse)
```

**train/value_loss**:
```
Expected: 40,000 → 5,000 → <2,000 (converging)
Red Flag: Stuck at >7,000 (not learning)
```

**rollout/ep_rew_mean**:
```
Expected: -500 → +1000 → +3000 → +5000 (increasing)
Red Flag: Oscillating wildly or decreasing
```

### Console Output (Every 50k Steps)

```
Episode 1000/10000  |  Steps: 1.2M/10M
├─ Success Rate: 18.4%  [✓ Good if trending up]
├─ Mean Reward: +1850   [✓ Good if positive]
├─ Value Loss: 8420     [⚠️ Still high, needs to drop]
└─ Explained Var: 0.89  [✓ Good if >0.8]
```

---

## Diagnostic Checkpoints

**@ 2M Steps** (~1.2 hours):
- **Expected**: Success rate 10-20%, value loss <10k
- **Abort if**: Success rate <5%, same inverse curve pattern

**@ 5M Steps** (~3 hours):
- **Expected**: Success rate 25-35%, value loss <5k
- **Abort if**: Success rate declining, value loss not improving

**@ 10M Steps** (~6 hours):
- **Final evaluation**: Run inference on best checkpoint
- **Compare**: 33% LSTM baseline vs MLP result

---

## Key Differences from LSTM Runs

| Aspect | LSTM Setup | MLP Setup | Why This Matters |
|--------|------------|-----------|------------------|
| **Recurrence** | Yes (256 hidden) | No (memoryless) | Removes gradient instability |
| **Temporal Context** | LSTM memory | Frame-stack (4×) | Simpler, proven approach |
| **Effective Horizon** | ~100 steps | ~333 steps | Better credit assignment |
| **Batch Size** | 128 | 256 | More stable updates |
| **Learning Rate** | 1e-4 (conservative) | 3e-4 (standard) | Faster learning |
| **Network Size** | [256, 256] + LSTM | [512, 512, 256] | More capacity |
| **Initialization** | Default | Orthogonal | Better gradient flow |
| **Normalization** | None (commented out) | VecNormalize + LayerNorm | Stable training |

---

## Architecture Justification

### Why Frame-Stacking Works

**Theoretical Basis**:
- Observation at time $t$: $o_t$
- Stacked observation: $\tilde{o}_t = [o_{t-3}, o_{t-2}, o_{t-1}, o_t]$
- Policy: $\pi(a_t | \tilde{o}_t)$ (memoryless but sees history)

**What MLP Can Compute from Stack**:
1. **Velocity**: $v \approx (o_t[0:3] - o_{t-1}[0:3]) / \Delta t$
2. **Acceleration**: $a \approx (v_t - v_{t-1}) / \Delta t$
3. **Trajectory trend**: Is target moving closer or farther?
4. **Radar dropout handling**: If $o_t$ is sentinel, use $o_{t-1}$

**Limitation**: Cannot reason beyond 4 timesteps (0.04 seconds). For longer context, would need:
- More frames (8-16 stack)
- Attention mechanism
- Transformer architecture

### Why Extended Horizon Works

**Credit Assignment Path**:
```
Action at t=500 (radar acquisition)
↓ [1300 steps later]
Terminal reward at t=1800 (intercept success)

With γλ = 0.99 × 0.95 = 0.9405:
  Credit at t=500: (0.9405)^1300 ≈ 0 (vanished!)

With γλ = 0.997 × 0.97 = 0.967:
  Credit at t=500: (0.967)^1300 ≈ 10^-18 (still vanished!)

Solution: Value function V(s) must predict long-term return
  - Action at t=500 changes V(s_500) → V(s_501)
  - Temporal difference: δ_500 = r_500 + γ V(s_501) - V(s_500)
  - This δ is used to update policy
  - Higher γ means V(s) estimates longer future
```

**Key Insight**: Extended horizon doesn't magically propagate gradients through 1800 steps. Instead, it makes the value function estimate further into the future, which provides better advantage estimates for policy updates.

---

## Files Modified

**Configuration**:
- `/home/roman/Hlynr_Intercept/rl_system/config.yaml`
  - Disabled LSTM
  - Added frame-stacking
  - Extended gamma/lambda
  - Increased batch size
  - Updated network architecture

**Training Script**:
- `/home/roman/Hlynr_Intercept/rl_system/train.py`
  - Added `CustomMLP` class with orthogonal init + layer norm
  - Added `VecFrameStack` wrapper
  - Added `VecNormalize` wrapper
  - Removed LSTM/RecurrentPPO code paths
  - Updated eval environment preprocessing

**Observations** (unchanged):
- `/home/roman/Hlynr_Intercept/rl_system/core.py`
  - Still radar-only (verified)
  - Kalman filter preprocessing
  - No omniscient data leakage

---

## Troubleshooting

### If Training Crashes at Start

**Error**: `AttributeError: 'NoneType' object has no attribute 'shape'`
- **Cause**: Frame-stacking with non-array observations
- **Fix**: Ensure environment returns numpy arrays

**Error**: `ValueError: setting an array element with a sequence`
- **Cause**: Observation space mismatch after frame-stacking
- **Fix**: Check observation space is (104,) not (26,)

### If Training is Very Slow

**Observation**: <5000 steps/second (expected: 8000-12000)
- **Cause 1**: Frame-stacking overhead
  - Normal, frame-stack adds ~20% overhead
- **Cause 2**: Large batch size on CPU
  - Consider reducing batch_size to 128
- **Cause 3**: GPU not being used
  - Check `nvidia-smi`, ensure PyTorch sees GPU

### If Success Rate is Flat at 0%

**After 1M Steps, Still 0%**:
- Check radar detection: `python diagnose_radar.py`
- Check reward: Should see positive rewards sometimes
- Check value loss: Should be decreasing
- May need longer training (patience until 2M steps)

---

## Post-Training Analysis

After training completes, run:

```bash
# Run inference on best model
python inference.py \
  --model checkpoints/training_TIMESTAMP/best/best_model.zip \
  --mode offline \
  --num_episodes 100

# Compare to LSTM baseline
cat inference_results/offline_run_TIMESTAMP/summary.txt

# Expected output:
# Success Rate: XX/100 (XX%)
# Mean Reward: +XXXX
# Mean Final Distance: XXXm
```

**Questions to Answer**:
1. Did success rate remain stable or increase through training?
2. Did it exceed 30% (LSTM baseline)?
3. Did value loss converge below 5000?
4. Are failures closer than LSTM (437m vs 2081m)?

---

## Conclusion

This experiment directly tests the core hypothesis from the comprehensive diagnosis: **LSTM training instability causes the inverse learning curve**.

**If MLP succeeds**: We have a stable training setup and can iterate to improve performance.

**If MLP fails similarly**: The problem is deeper (reward structure, task difficulty, sample efficiency) and we need more advanced solutions (SAC, privileged learning, HER).

**Status**: ✅ READY TO TRAIN

**Next Action**: Run training command and monitor TensorBoard for success_rate_pct metric.

---

**Good luck! 🚀**

*Expected training time: 6-7 hours for 10M steps*
