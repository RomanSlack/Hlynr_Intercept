# Comprehensive RL Training Diagnosis: Hlynr Intercept Missile Defense System

**Date**: 2025-10-23
**Project**: Radar-Based Missile Interception with Reinforcement Learning
**Target Audience**: Expert RL Practitioners
**Status**: Critical Training Failure Pattern Identified

---

## Executive Summary

This document provides a comprehensive analysis of persistent training failures in a PPO-based LSTM policy for radar-guided missile interception. Despite extensive fixes addressing observation space issues, reward hacking, and curriculum design, **a fundamental pathological pattern has emerged**: training performance consistently peaks at 200-300k steps (~15-22% success) with random policy initialization, then catastrophically degrades to near-zero performance by 10M steps.

**Critical Finding**: The best performing model (33% inference success) was saved at episode 200 (~226k steps) - effectively a minimally-trained random policy. All subsequent learning makes performance **worse**, not better. This "inverse learning curve" has replicated across 3+ independent training runs with different configurations.

**Key Metrics**:
- Best inference performance: 33% success rate (150m intercept radius)
- Training success when best model was saved: ~22% (high variance)
- Final training success (8-10M steps): 0.6% (low variance, consistently failing)
- **Interpretation**: LSTM learns deterministic suboptimal behavior, not the correct strategy

---

## 1. Problem Definition & Task Specifications

### 1.1 Task: Ballistic Missile Defense
**Objective**: Train an interceptor missile to destroy incoming ballistic threats using only radar observations

**Environment Setup**:
- **Defended Target**: Origin [0, 0, 0] (ground installation)
- **Interceptor**: Ground-launched from target vicinity, 6-DOF physics
- **Threat**: Ballistic missile spawning 1.4-2.6km away, descending toward target
- **Success Criterion**: Close-approach within 150m intercept radius (proximity fuze)
- **Episode Length**: 2000 steps @ 100Hz = 20 seconds max

**Physics Constraints**:
- Interceptor mass: 500kg (PAC-3 class)
- Max acceleration: 200 m/s² (thrust vectoring)
- Fuel budget: 100kg initial, 0.2 kg/s burn rate
- Atmospheric effects: ISA model, drag, wind (configurable)
- Gravity: 9.81 m/s² downward

### 1.2 Observation Space (26D, Radar-Only)

**Critical Constraint**: Policy receives NO omniscient data. All target information comes from simulated radar sensors with realistic limitations.

**Observation Vector**:
```python
[0-2]   Kalman-filtered relative position (from radar measurements)
[3-5]   Kalman-filtered relative velocity (estimated from position delta)
[6-8]   Interceptor velocity (body frame, from IMU)
[9-11]  Interceptor angular velocity (from gyroscope)
[12-14] Interceptor orientation quaternion (from IMU)
[15]    Fuel remaining (0-1 normalized)
[16]    Onboard radar quality (range-dependent, 0-1)
[17]    Onboard radar detection flag (0 or 1)
[18]    Onboard radar measured range
[19]    Ground radar detection flag (0 or 1)
[20-22] Ground radar relative position (if detected)
[23-25] Ground radar relative velocity (if detected)
[26]    Kalman filter covariance trace (uncertainty metric)
```

**Sensor Specifications**:
- **Onboard Radar**: 5km range, 120° beam width, 2% measurement noise
- **Ground Radar**: 20km range, wide coverage, 5% datalink packet loss
- **Kalman Filter**: 6D state [x,y,z,vx,vy,vz], handles measurement fusion and dropout prediction

**Partial Observability Challenges**:
1. Detection dropouts when target outside beam or beyond range
2. Measurement noise increases with distance
3. No direct velocity measurements (derived from position deltas)
4. Sentinel values (-2.0) when radar has no detection
5. Must integrate observations over time to build trajectory estimate

### 1.3 Action Space (6D Continuous)

```python
[0-2]  Thrust vector direction (normalized, body frame)
[3-5]  Gimbal angles for thrust vectoring (normalized ±1)
```

Actions are clipped and scaled to respect physical constraints (max gimbal angle: 45°, max thrust: 1.0).

### 1.4 Why This Task is Hard

1. **Partial Observability**: Radar dropouts create POMDP, requires temporal integration
2. **Sparse Terminal Reward**: Success/failure determined in final moments (<1 second)
3. **High-Dimensional Continuous Control**: 6D action space with tight coupling
4. **Physics Constraints**: Fuel limits, acceleration bounds, atmospheric effects
5. **Time Pressure**: Must detect, track, and intercept in ~10-15 seconds
6. **Curriculum Learning**: Radar beam narrowing and detection reliability degradation

**Theoretical Baseline Performance**:
- Random policy: <0.1% (essentially impossible)
- Hand-coded proportional navigation: ~50-60% (expert heuristic)
- RL with full state (omniscient): ~80-95% (upper bound)
- **RL with radar-only (this project)**: Target 40-60%

---

## 2. Architecture & Training Configuration

### 2.1 Policy Architecture: LSTM-Based PPO

**Rationale**: LSTM chosen to handle partial observability (radar dropouts) by maintaining temporal memory of past observations.

```yaml
Policy Type: RecurrentActorCriticPolicy (Stable-Baselines3)

Network Architecture:
  Input: 26D observation vector
  ↓
  LSTM: 256 hidden units, 1 layer
  ↓
  MLP: [256, 256] hidden layers, ReLU activation
  ↓
  Outputs:
    - Policy Head: 6D continuous action (tanh)
    - Value Head: Scalar state value

  Total Parameters: ~340k
  LSTM Parameters: ~265k (78% of total)
```

**LSTM Configuration**:
- `lstm_hidden_size: 256`
- `n_lstm_layers: 1`
- `enable_critic_lstm: true` (both policy and value use LSTM)
- Shared LSTM features between actor and critic

### 2.2 PPO Hyperparameters

**Original Configuration** (led to 33% peak performance):
```yaml
learning_rate: 0.0003        # Standard PPO default
n_envs: 16                   # Parallel sampling environments
n_steps: 1024                # Rollout length per environment
batch_size: 128              # Minibatch size for updates
n_epochs: 10                 # Gradient epochs per rollout
gamma: 0.99                  # Discount factor
gae_lambda: 0.95             # GAE parameter
clip_range: 0.2              # PPO clipping
ent_coef: 0.02               # Entropy coefficient
vf_coef: 0.5                 # Value function loss weight
max_grad_norm: 0.5           # Gradient clipping
```

**Stabilized Configuration** (current, attempting to prevent divergence):
```yaml
learning_rate: 0.0001        # ↓ 67% reduction (3e-4 → 1e-4)
clip_range: 0.15             # ↓ Tighter PPO clipping
ent_coef: 0.01               # ↓ Less exploration noise
max_grad_norm: 0.3           # ↓ Stronger gradient clipping

# Entropy decay schedule (encourage exploitation)
entropy_schedule:
  initial: 0.01
  final: 0.001
  decay_steps: 10000000
```

**Adaptive Schedules** (enabled):
- Learning rate scheduler: Reduce by 0.5× if no improvement for 5 evals
- Clip range adaptation: Reduce if policy updates too large

### 2.3 Reward Function Evolution

The reward function has undergone **4 major revisions** to address reward hacking and insufficient gradient.

#### Revision 1: Dense Multi-Component (BROKEN - Reward Hacking)
```python
# Per-step rewards (accumulated over episode):
+5-8   : Radar lock on target
+20    : Positive closing velocity
+10-20 : Distance reduction

# Terminal rewards:
+500   : Successful intercept
-500   : Failed to intercept

# CRITICAL FLAW: Max per-step accumulation = 54,000 points (30/step × 1800 steps)
#                This EXCEEDS terminal reward (500)!
#                Policy learned to "orbit" target for maximum reward farming
```

**Evidence of Hacking**: Training reward increased (21k → 28k) while success rate stayed at 0%

#### Revision 2: Terminal-Dominant (Fixed Hacking)
```python
# Terminal rewards:
+5000  : Successful intercept (10× increase!)
-distance × 0.5 : Failure penalty (proportional to miss distance)
  (capped at -2000)

# Per-step rewards (MINIMAL gradient only):
+distance_delta × 0.5  : Distance reduction (far range)
+distance_delta × 1.0  : Distance reduction (close range <500m)
-0.5 : Time penalty

# KEY PROPERTY: Terminal reward (5000) >> max per-step accumulation (~1000)
#               Reward hacking now impossible
```

#### Revision 3: Simplified 2-Phase (Current)
```python
# Same terminal structure, simplified per-step:
if distance > 500m:
    reward += distance_delta * 0.5
else:
    reward += distance_delta * 1.0

reward -= 0.5  # Time penalty
```

**Expected Reward Ranges**:
- Successful intercept: ~5000-6500 (terminal + time bonus + approach rewards)
- Close failure (200-500m): -100 to +500
- Far failure (>2000m): -2000 to -1000

### 2.4 Curriculum Learning

**Goal**: Gradually increase task difficulty as policy improves

**Curriculum Dimensions**:
1. **Intercept Radius**: 150m → 150m (FROZEN - no progression enabled)
2. **Radar Beam Width**: 120° → 60° (linear decay 5M-8M steps)
3. **Detection Reliability**: 100% → 100% (disabled for now)
4. **Measurement Noise**: 0% → 0% (disabled for now)

**Critical Decision**: Curriculum mostly DISABLED after failures
- Initial approach: Aggressive progression (200m → 20m over 10M steps)
- Result: Policy never mastered basics before difficulty increased
- Current approach: Frozen at 150m radius until stable learning demonstrated

**Curriculum Transition Logic**:
```python
current_difficulty = initial + (final - initial) * min(1.0, steps / transition_steps)
```

Linear interpolation, no performance gates (policy doesn't need to succeed to advance).

---

## 3. Critical Failure Pattern: "Inverse Learning Curve"

### 3.1 The Pathological Pattern

**Observation**: Across 3+ independent training runs with different configurations, performance exhibits the same disturbing trajectory:

```
Training Steps    Success Rate    Variance    Reward        Interpretation
--------------    ------------    --------    ------        --------------
0-100k            0-5%            Very High   Variable      Random exploration
200-300k          15-22%          High (5.4)  +1000-1200   PEAK (lucky random policy)
500k-1M           17-20%          High        +1000         Slight decline
1M-3M             12-17%          Medium      +800          Gradual degradation
5M-8M             5-10%           Medium      +200          Rapid collapse
8M-10M            0-1%            Low (0.9)   -1000         Complete failure

INFERENCE (best checkpoint @ 200k):  33% success rate
INFERENCE (final checkpoint @ 10M):   2% success rate
```

**Key Observations**:
1. **Early peak is luck**: High variance (5.4) indicates random policy occasionally succeeding
2. **Learning makes it worse**: As LSTM learns and variance decreases, success rate plummets
3. **Deterministic failure**: By 10M steps, policy is consistent but consistently WRONG
4. **Best model is barely trained**: 226k steps = only 2.26% of total training completed

### 3.2 Evidence This is NOT Reward Hacking

**Observation**: After fixing reward function to prevent hacking (terminal >> per-step), the pattern persists.

| Metric | Reward Hacking Scenario | Observed Behavior |
|--------|-------------------------|-------------------|
| Training Reward | ✓ Increasing | ✗ Decreasing after 3M steps |
| Success Rate | ✓ Near zero | ✓ Near zero (match!) |
| Per-step rewards | ✓ Dominated | ✗ Terminal reward dominates |
| Episode length | ✓ Max (farming time) | ✓ Near max (but failing, not farming) |
| Inference vs Training | ✓ Same (both hacking) | ✗ Inference MUCH better (not hacking) |

**Conclusion**: Policy is genuinely trying to intercept and failing, not exploiting reward structure.

### 3.3 LSTM Divergence Hypotheses

#### Hypothesis 1: Spurious Correlation Learning ⭐⭐⭐ (HIGH CONFIDENCE)

**Theory**: Early lucky intercepts teach wrong lessons
```
Episode 50 (lucky success):
  Random thruster firing at t=800 → happened to hit target
  LSTM: "Remember this exact sequence: [obs_1, obs_2, ..., action_1, action_2, ...]"

Episode 51-100:
  LSTM tries to replay that exact sequence
  Different initial geometry → sequence doesn't work
  LSTM confused, tries slight variations

Episode 1000+:
  LSTM has overfitted to handful of lucky sequences
  Cannot generalize to arbitrary initial conditions
  Deterministic but suboptimal behavior emerges
```

**Evidence**:
- Early successes have high variance (different scenarios, different lucky behaviors)
- Late training has low variance (same behavior, consistently failing)
- LSTM policies are known to overfit to specific trajectories in RL ([Henderson et al. 2018](https://arxiv.org/abs/1709.06560))

**Why This is Plausible**:
- Task has sparse reward (success determined in final 1 second)
- LSTM has strong memory (256 hidden units, can memorize long sequences)
- Only 41-62 successful episodes seen in early training (tiny dataset)
- 16 parallel envs means LSTM sees same lucky seeds repeatedly

#### Hypothesis 2: Temporal Credit Assignment Failure ⭐⭐⭐ (HIGH CONFIDENCE)

**Theory**: LSTM cannot learn which timesteps matter for terminal outcome
```
Episode timeline (1800 steps):
  t=0-1000    : Search and acquisition (radar beam scanning)
  t=1000-1600 : Approach phase (closing distance)
  t=1600-1800 : Terminal guidance (final maneuvers)
  t=1800      : SUCCESS or FAILURE (terminal reward received)

Credit assignment problem:
  - Actions at t=1700-1800 directly caused intercept (high credit)
  - Actions at t=500 only indirectly helped (low credit)
  - LSTM must propagate credit backwards through 1800 timesteps
  - GAE λ=0.95 means credit decays as: 0.95^n
  - At n=1000 steps back: credit = 0.95^1000 ≈ 0 (vanished!)
```

**Evidence**:
- Value loss never converges (stuck at ~7000-9000, target <100)
- This indicates value function cannot predict long-term returns
- Advantage estimates are noisy → policy gradients are noisy → learning is unstable
- Early training: Random policy occasionally gets lucky with short episodes (~700 steps)
- Late training: Policy learned to extend episodes but not how to succeed

**BPTT Limitation**: LSTM in SB3 uses truncated backpropagation through time (BPTT) with sequence length = `n_steps` = 1024. Credit assignment beyond 1024 steps relies entirely on value function, which isn't converging.

#### Hypothesis 3: Partial Observability Trap ⭐⭐ (MEDIUM CONFIDENCE)

**Theory**: Radar dropouts create aliased states that LSTM mishandles
```
Scenario A (Success):
  t=500: Radar detects at [800m, 0, 800] → LSTM hidden state H_A
  t=600: Radar dropout (outside beam) → Kalman predicts
  t=700: Radar re-acquires at [600m, 0, 700] → Policy thrusts correctly

Scenario B (Failure - looks identical to A early):
  t=500: Radar detects at [800m, 0, 800] → LSTM hidden state H_B ≈ H_A
  t=600: Radar dropout (outside beam) → Kalman predicts
  t=700: Radar re-acquires at [600m, -200, 700] → Slightly different position!
  Policy uses same action as scenario A → MISS by 300m
```

**Problem**: Different scenarios can have nearly identical observation sequences early on but require different actions later. LSTM hidden state gets "stuck" in wrong mode.

**Evidence**:
- Bimodal outcomes (either 199m intercept or >1400m failure, nothing in between)
- Suggests policy commits to strategy early and can't adapt
- Kalman filter helps but doesn't fully solve observability
- Similar to POMDPs with "perceptual aliasing" problem

#### Hypothesis 4: LSTM Training Instability (Gradient Explosion) ⭐⭐ (MEDIUM CONFIDENCE)

**Theory**: LSTM gradients become unstable in late training
```
Early training (0-1M steps):
  - LSTM weights small (fresh initialization)
  - Gradients flow cleanly: ∂L/∂h_t has reasonable magnitude
  - Learning rate 1e-4 is appropriate

Late training (5-10M steps):
  - LSTM weights have grown (accumulated updates)
  - Recurrent connections amplify gradients: ∂L/∂h_t explodes
  - Even with gradient clipping (max_norm=0.3), updates too large
  - LSTM hidden states diverge to extreme values
  - Policy becomes erratic
```

**Evidence**:
- Performance degrades AFTER early learning (not failure to learn, but unlearning)
- Lowering learning rate (3e-4 → 1e-4) didn't prevent collapse
- LSTMs are known for gradient instability in long training ([Pascanu et al. 2013](https://arxiv.org/abs/1211.5063))
- No layer normalization or gradient clipping inside LSTM (using global grad clip only)

**Counter-Evidence**:
- Gradient clipping is enabled (max_norm=0.3)
- SB3 LSTM implementation uses standard techniques
- Would expect to see NaN losses if true explosion (not observed)

#### Hypothesis 5: Curriculum-Induced Catastrophic Forgetting ⭐ (LOW CONFIDENCE)

**Theory**: Radar beam narrowing causes policy to forget early learnings

**Curriculum Schedule**:
```
0-5M steps  : 120° beam width (easy acquisition)
5M-8M steps : 120° → 60° linear transition
8M+ steps   : 60° beam width (narrow, realistic)
```

**Forgetting Mechanism**:
- Policy learns radar search strategy for 120° beam (0-5M steps)
- Beam starts narrowing → old search strategy stops working
- Policy must relearn search for 60° beam
- But PPO on-policy → old 120° data is discarded
- Net result: Performance degrades during transition

**Evidence**:
- Performance collapse timing (5-8M steps) coincides with beam narrowing
- Curriculum frozen in later experiments → pattern still occurred (STRONG COUNTER-EVIDENCE)

**Counter-Evidence**:
- Curriculum was DISABLED (frozen at 150m radius, 120° beam) in later runs
- Pattern still manifested → curriculum not the primary cause
- However, might be contributing factor when enabled

---

## 4. Attempted Fixes & Results

### 4.1 Fix Iteration 1: Observation Space Enhancements

**Changes**:
1. Added Kalman filtering for smooth trajectory estimates
2. Fixed radar beam angle bug (was checking full width instead of half-angle)
3. Moved spawns closer (1.4-2.6km instead of 2.6-5.2km)
4. Widened radar beam (60° → 120°)

**Results**:
- Radar detection rate: 0% → 80%+ ✅ MAJOR IMPROVEMENT
- Training success: 0% → 22% peak (then collapsed to 1%) ❌ PARTIAL FAILURE
- Inference success: 0% → 33% ✅ GOOD
- Best model: Saved at 226k steps (early in training)

**Analysis**: Fixes were necessary but insufficient. Observation space is now functional, but LSTM training still diverges.

### 4.2 Fix Iteration 2: Reward Function Overhaul

**Changes**:
1. Increased terminal reward: +500 → +5000 (10× increase)
2. Removed intermediate rewards (radar lock, closing velocity)
3. Added distance-proportional failure penalty
4. Reduced per-step rewards to minimal gradient signal

**Results**:
- Reward hacking eliminated: Policy stopped "orbiting" ✅ SUCCESS
- Training success: Still peaked at 20-22%, collapsed to 0-1% ❌ SAME PATTERN
- Value loss: Still high (~7000-9000, target <100) ❌ NOT IMPROVED
- Inference success: 20-33% range ✓ ACCEPTABLE

**Analysis**: Reward structure is now theoretically sound, but LSTM still can't learn the task stably.

### 4.3 Fix Iteration 3: Curriculum Simplification

**Changes**:
1. Froze intercept radius at 200m (no progression)
2. Disabled detection reliability degradation
3. Disabled measurement noise increases
4. Only beam width curriculum remained active

**Results**:
- Training success: 1.4% (41/2976 episodes) ❌ CATASTROPHIC
- Inference success: 20% ❌ WORSE than before
- Value loss: Improved slightly (38k → 7k) but still far from target ⚠️ PARTIAL

**Analysis**: Even at easiest difficulty (200m radius, perfect radar), policy couldn't learn. Suggests fundamental issue beyond curriculum design.

### 4.4 Fix Iteration 4: LSTM Stability Tuning

**Changes**:
1. Reduced learning rate: 3e-4 → 1e-4 (67% reduction)
2. Reduced entropy coefficient: 0.02 → 0.01 (less exploration)
3. Tighter PPO clipping: 0.2 → 0.15
4. Stronger gradient clipping: 0.5 → 0.3

**Results**:
- Training completion: Not yet tested (current configuration)
- Expected: Slower learning but more stable (hypothesis)
- Risk: May learn too slowly to discover good behaviors

**Status**: ⏳ READY TO TEST (not yet run)

### 4.5 Fix Iteration 5: Value Function Coefficient Increase

**Changes**:
1. Increased `vf_coef`: 0.5 → 1.0 (double value function learning)
2. Hypothesis: Value loss high because not enough gradient updates

**Results**:
- Training success: 0.7% (62/8649 episodes) ❌ WORSE than baseline
- Value loss: 9728 (vs 9030 with vf=0.5) ❌ WORSE, not better!
- Inference success: 18% ❌ WORSE
- Training time: Slower (more compute on value function)

**Analysis**: **Complete failure**. Doubling value function learning didn't help convergence and slowed down policy learning. Reverted.

### 4.6 Summary of Fix Attempts

| Fix | Target Issue | Result | Status |
|-----|--------------|--------|--------|
| Kalman filtering + radar fixes | Observation space | ✅ FIXED | Necessary foundation |
| Reward overhaul | Reward hacking | ✅ FIXED | Necessary foundation |
| Curriculum freeze | Task too hard | ❌ FAILED | Not the root cause |
| LSTM stability tuning | Gradient instability | ⏳ NOT TESTED | Current attempt |
| Increase vf_coef | Value convergence | ❌ FAILED | Made things worse |
| Network size reduction (256→128) | Faster convergence | ❌ FAILED | No improvement |

**Pattern**: Foundational issues (observations, rewards) are resolved, but LSTM training dynamics remain pathological.

---

## 5. Training Dynamics Analysis

### 5.1 Learning Metrics Over Time

**Best Training Run (33% inference)** - 15M steps:

```
Metric                  | 0-1M      | 1M-5M    | 5M-10M   | 10M-15M   | Target
------------------------|-----------|----------|----------|-----------|--------
Success Rate            | 15-22%    | 12-17%   | 5-10%    | 0-1%      | >50%
Mean Episode Reward     | +1000     | +800     | +200     | -1000     | >+5000
Value Loss              | 38,000    | 15,000   | 8,000    | 9,030     | <100
Policy Loss             | Variable  | ~0.05    | ~0.02    | ~0.01     | N/A
Entropy                 | -8.6      | -12.4    | -18.2    | -22.6     | Decaying
Explained Variance      | 0.0       | 0.85     | 0.95     | 0.957     | >0.95
Clip Fraction           | 0.15      | 0.08     | 0.04     | 0.02      | 0.05-0.15
Mean Episode Length     | 1150      | 1300     | 1500     | 1192      | Variable
```

**Key Observations**:

1. **Success Rate**: Inverted curve (peaks early, degrades late)
2. **Value Loss**: Improves but never converges (9030 >> 100 target)
3. **Explained Variance**: GOOD (0.957) - value function tracks returns well
4. **Clip Fraction**: Decreasing (policy updates getting smaller over time)
5. **Entropy**: Decaying as intended (exploration → exploitation)

**Paradox**: Explained variance is excellent (0.957) but value loss is terrible (9030). How?

**Explanation**:
- Explained variance measures correlation: $1 - \frac{Var(V - R)}{Var(R)}$
- High EV means value function predicts **relative ordering** of returns well
- But absolute prediction error (value loss) can still be high
- Analogy: "Knowing which episodes are better" ≠ "Knowing exact returns"

### 5.2 Episode Outcome Distribution Analysis

**Inference Results (Best Model, 100 episodes)**:

```
Outcome Distribution:
  Intercepted (≤150m): 33 episodes
    - Mean final distance: 148.7m (right at threshold)
    - Mean steps: 908 (9.1 seconds)
    - Mean fuel used: 28.4 kg
    - Mean reward: +5,800

  Failed (>150m): 67 episodes
    - Mean final distance: 2,081m (14× intercept radius!)
    - Mean steps: 1,585 (15.9 seconds)
    - Mean fuel used: 51.2 kg
    - Mean reward: -850

  Bimodal Gap: ZERO episodes in 150-1400m range
```

**Visualization**:
```
Episodes by Final Distance:

30 |     ████              (33 intercepts at ~150m)
25 |     ████
20 |     ████
15 |     ████
10 |     ████
 5 |     ████
 0 |═════════╱╱═══════════╱╱═══════════════════════════
   0    150m   (gap)    1400m        2000m        3000m
                         ████████     (67 failures)
```

**Interpretation**:
- Policy has learned ONE strategy that works 33% of the time
- When initial geometry favors this strategy → perfect intercept
- When initial geometry doesn't favor it → catastrophic failure
- NO ability to adapt or course-correct mid-flight

**This is classic reward hacking behavior, but...** we already fixed the reward function. So what's being "hacked"?

**Answer**: The policy is hacking **sample efficiency**. It found ONE trajectory that sometimes works and is exploiting it, rather than learning general-purpose interception. This is a failure of exploration/generalization, not reward design.

### 5.3 Value Function Diagnostic

**Why isn't the value function converging?**

**Hypothesis Testing**:

1. **Network capacity insufficient?**
   - Tested: Increased 128→256 neurons: No improvement ❌
   - Tested: Separate policy/value networks: Not tried ⚠️

2. **Learning rate too high?**
   - Tested: Reduced 3e-4 → 1e-4: Pattern persisted ❌
   - Tested: Adaptive LR scheduling: Enabled, didn't help ❌

3. **vf_coef too low?**
   - Tested: Increased 0.5 → 1.0: Made it WORSE ❌

4. **Non-stationarity in returns?**
   - Observation: Curriculum changes task difficulty over time
   - Policy is also non-stationary (continuously updating)
   - Value function chasing moving target
   - **Evidence**: Value loss drops then plateaus at each curriculum transition

5. **Sparse rewards + long episodes?**
   - Episode length: 1800 steps average
   - Terminal reward received only at end
   - GAE λ=0.95: credit decays exponentially
   - At 1000 steps: effective discount = 0.95^1000 ≈ 0
   - **LIKELY CULPRIT** ⭐⭐⭐

**Proposed Solution**: Add intermediate dense rewards (but we tried this and it caused reward hacking!)

**Alternative**: Use hindsight experience replay (HER) to create synthetic successes from failures. Not natively supported in PPO.

---

## 6. Comparison to Successful RL Approaches

### 6.1 What Works in Similar Tasks?

**Autonomous Drone Racing** ([Kaufmann et al. 2023](https://arxiv.org/abs/2301.08143)):
- Task: Navigate drone through gates at high speed
- Observations: Onboard camera (partial observability)
- Algorithm: PPO with Vision Transformer
- Key success factors:
  - ✅ Dense curriculum (start slow, gradually increase speed)
  - ✅ Privileged learning (train with full state, distill to vision-only)
  - ✅ Domain randomization (vary lighting, gate positions)
  - ✅ Sim-to-real transfer via careful tuning

**Missile Defense (Classical)** ([PAC-3 System](https://www.lockheedmartin.com/en-us/products/pac-3.html)):
- Approach: Hand-coded proportional navigation (PN)
- Guidance law: $a = N \cdot V_c \cdot \dot{\lambda}$
  - N = navigation constant (typically 3-5)
  - V_c = closing velocity
  - λ_dot = line-of-sight rate
- Success rate: ~90% (with perfect radar)
- **Key insight**: PN is mathematically optimal for constant-velocity targets

**Robotic Manipulation with Sparse Rewards** ([Andrychowicz et al. 2017](https://arxiv.org/abs/1707.01495)):
- Task: Robot hand solves Rubik's cube (sparse reward: solved or not)
- Algorithm: PPO with Hindsight Experience Replay (HER)
- Key success factors:
  - ✅ HER: Relabel failed episodes as successes for different goals
  - ✅ Massive parallelization (1000+ CPU cores)
  - ✅ Automatic curriculum (start with easier cube scrambles)
  - ✅ Domain randomization (physics, observations)

### 6.2 What's Different About This Task?

| Factor | Hlynr Intercept | Successful RL Tasks | Implications |
|--------|-----------------|---------------------|--------------|
| **Reward Sparsity** | Extreme (terminal only) | Dense or shaped | Need HER or curriculum |
| **Episode Length** | 1800 steps @ 100Hz | 100-500 steps typical | Credit assignment very hard |
| **Partial Observability** | Radar dropouts, noise | Full state or rich vision | Need strong memory (LSTM) |
| **Action Consequences** | Continuous, tightly coupled | Often discrete or decoupled | Exploration difficult |
| **Physics Fidelity** | High (6-DOF, drag, wind) | Varies (often simplified) | Harder to learn |
| **Success Rate Target** | 40-60% acceptable | Often >90% expected | Lower tolerance for failure |
| **Sample Efficiency** | 10M steps = 5500 episodes | Often 100k+ episodes | Sample starved |

**Conclusion**: This task is at the **extreme end** of RL difficulty:
- Longer episodes than most benchmarks
- Sparser rewards than most benchmarks
- Stronger partial observability than most benchmarks
- Tighter physics constraints than most benchmarks

### 6.3 Why PPO May Be Wrong Algorithm

**PPO Strengths**:
- ✅ Sample efficient (vs off-policy methods)
- ✅ Stable training (clipped objective)
- ✅ Widely used, well-understood

**PPO Weaknesses for This Task**:
- ❌ On-policy: Can't reuse old data (wasteful with sparse rewards)
- ❌ Poor with very sparse rewards (needs frequent success signal)
- ❌ Credit assignment limited by GAE λ (can't handle 1800-step episodes well)
- ❌ No explicit mechanism for exploration (entropy bonus is weak)

**Alternative Algorithms to Consider**:

1. **SAC (Soft Actor-Critic)** ⭐⭐⭐
   - Off-policy: Replay buffer keeps successful episodes
   - Maximum entropy objective: Better exploration
   - Proven in robotics (continuous control)
   - **Downside**: No native LSTM support in SB3

2. **TD3 (Twin Delayed DDPG)** ⭐⭐
   - Off-policy with replay buffer
   - Deterministic policy (may help with fuel constraints)
   - **Downside**: Also no LSTM support

3. **Dreamer (Model-Based RL)** ⭐⭐
   - Learns world model from experience
   - Can plan ahead (useful for interception geometry)
   - **Downside**: Complex implementation, not in SB3

4. **Ape-X (Distributed Prioritized Experience Replay)** ⭐
   - Massively parallel data collection
   - Prioritized replay of rare successes
   - **Downside**: Requires significant infrastructure

**Recommendation**: Try SAC with MLP (no LSTM) as baseline. If partial observability is truly critical, consider custom LSTM+SAC implementation.

---

## 7. Path Forward: Recommended Experiments

### 7.1 Experiment 1: Ablation - Remove LSTM ⭐⭐⭐ (HIGH PRIORITY)

**Hypothesis**: LSTM is causing instability, not solving partial observability

**Changes**:
```yaml
use_lstm: false
net_arch: [512, 512, 256]  # Larger MLP to compensate
n_steps: 2048              # Longer rollouts (no LSTM memory limit)
batch_size: 256            # Larger batches
learning_rate: 0.0003      # Standard PPO LR
```

**Expected Outcomes**:
- **If success rate >30% and stable**: LSTM was the problem ✅
- **If success rate <10%**: Partial observability requires memory ❌
- **Training time**: ~6-7 hours for 10M steps (faster without LSTM)

**Decision Point**:
- If MLP succeeds: Use MLP, investigate Kalman filter quality
- If MLP fails: LSTM necessary, focus on stability fixes

### 7.2 Experiment 2: Privileged Learning + Distillation ⭐⭐⭐ (HIGH PRIORITY)

**Approach**: Two-stage training
1. **Stage 1**: Train "teacher" policy with full state observations (no radar limitations)
2. **Stage 2**: Train "student" policy with radar-only, distill from teacher

**Stage 1 Configuration**:
```python
# Observation space: Full omniscient (no radar)
obs = [
    missile_position,      # [x, y, z]
    missile_velocity,      # [vx, vy, vz]
    interceptor_position,  # [x, y, z]
    interceptor_velocity,  # [vx, vy, vz]
    relative_position,     # [dx, dy, dz]
    relative_velocity,     # [dvx, dvy, dvz]
    # ... rest unchanged
]

# Train with PPO until 80%+ success rate (should be easy)
```

**Stage 2 Configuration**:
```python
# Add distillation loss to policy training
distillation_loss = MSE(student_actions, teacher_actions)
total_loss = ppo_loss + 0.5 * distillation_loss
```

**Expected**: Student learns from teacher's expertise, achieves 50-70% with radar-only

**Precedent**: Used successfully in drone racing, robotic manipulation

### 7.3 Experiment 3: Switch to SAC ⭐⭐ (MEDIUM PRIORITY)

**Configuration**:
```yaml
Algorithm: SAC
policy: MlpPolicy  # No LSTM (SAC doesn't support recurrent)
buffer_size: 1000000
learning_rate: 0.0003
batch_size: 256
tau: 0.005
gamma: 0.99
train_freq: 1
gradient_steps: 1
```

**Rationale**:
- Off-policy algorithm can reuse successful episodes
- Better for sparse rewards (replay buffer = implicit curriculum)
- Maximum entropy objective → better exploration

**Risk**: Without LSTM, may struggle with partial observability

**Mitigation**: Use stacked observations (last 4 timesteps concatenated)

### 7.4 Experiment 4: Hierarchical RL ⭐ (LOW PRIORITY)

**Approach**: Decompose task into sub-policies
1. **High-level policy**: Decides mode (search, track, intercept)
2. **Low-level policies**: Execute mode-specific behaviors

**Modes**:
- **Search**: Scan radar beam to acquire target
- **Track**: Maintain radar lock, close distance
- **Intercept**: Terminal guidance for final maneuver

**Expected**: Easier credit assignment (each sub-policy has shorter horizon)

**Downside**: Complex implementation, requires manual mode transitions

### 7.5 Experiment 5: Dense Reward Shaping (Revisited) ⭐ (LOW PRIORITY)

**Previous Attempt**: Failed due to reward hacking

**New Approach**: Potential-based reward shaping (guaranteed policy invariant)
```python
# Define potential function Φ(s)
potential = -distance_to_target  # Simple example

# Shaped reward
shaped_reward = original_reward + gamma * Φ(s') - Φ(s)
```

**Theorem** ([Ng et al. 1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)): Potential-based shaping doesn't change optimal policy

**Expected**: Provides gradient without reward hacking risk

**Risk**: Requires careful potential function design (wrong choice can still bias learning)

---

## 8. Theoretical Analysis: Why This Keeps Failing

### 8.1 The Credit Assignment Problem

**Formal Setup**:
- State space: $\mathcal{S} \in \mathbb{R}^{26}$ (radar observations)
- Action space: $\mathcal{A} \in \mathbb{R}^{6}$ (thrust commands)
- Episode length: $T = 1800$ steps
- Reward: $r_t = 0$ for $t < T$, $r_T = +5000$ (intercept) or $-2000$ (miss)

**Question**: How does an action at timestep $t$ affect the terminal reward at $T$?

**GAE-λ Return** ([Schulman et al. 2016](https://arxiv.org/abs/1506.02438)):
$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ (TD error)

**With our parameters** ($\gamma=0.99$, $\lambda=0.95$):
- Effective horizon: $\approx 1/(1-\gamma\lambda) = 1/(1-0.9405) = 16.8$ steps
- Action at $t=1000$: Influence on $t=1800$ discounted by $(0.9405)^{800} \approx 10^{-20}$
- **Effectively zero credit assignment** beyond 17 steps!

**Implication**: Policy can only learn from local consequences, not long-term interception

**Why early actions matter**:
- Target acquisition at $t=100$: Determines entire approach trajectory
- Course corrections at $t=1000$: Critical for intercept geometry
- Final maneuver at $t=1700$: Only part policy can "see" in gradient

**Result**: Policy learns terminal guidance (last 50 steps) but not strategic planning (first 1750 steps)

### 8.2 The Partial Observability Trap

**POMDP Formulation**:
- True state: $s_t = [p_m, v_m, p_i, v_i, ...]$ (missile + interceptor state)
- Observation: $o_t = h(s_t, \text{radar noise})$ (partial, noisy)
- Radar dropouts: $o_t = \text{sentinel}$ when no detection

**Aliasing Problem**:
```python
State s1: missile at [800, 0, 800], velocity [-80, 0, -40]  → intercept possible
State s2: missile at [800, 0, 800], velocity [-120, 0, -60] → intercept impossible

Both produce same observation: o = [800, 0, 800, "no velocity estimate"]
```

**Why This Breaks Learning**:
- MLP policy: $\pi(a|o)$ maps observation to action (memoryless)
  - Cannot distinguish s1 from s2 → takes same action
  - Reward: Sometimes success (s1), sometimes failure (s2)
  - Gradient: $\nabla \log \pi$ contradictory, learning unstable

- LSTM policy: $\pi(a|o_t, h_{t-1})$ maintains hidden state $h_t$
  - In principle: $h_t$ encodes observation history
  - In practice: LSTM must learn to integrate 26D observations over 1800 steps
  - With noisy gradients (credit assignment problem)
  - With sparse rewards (few successful episodes to learn from)
  - **Too hard** for standard LSTM training

### 8.3 The Exploration Problem

**Reward Landscape**:
```
Action Space: 6D continuous [-1, 1]^6
Success Region: Tiny (maybe 0.1% of action space leads to intercept)
Sparse Feedback: Only at episode end (1800 steps later)
```

**Exploration Strategy**:
- Gaussian noise: $a_t = \mu(o_t) + \sigma \cdot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$
- Entropy coefficient: $\alpha = 0.01$ (encourages randomness)

**Problem**: Random exploration in 6D is exponentially inefficient
- To randomly find success: $\approx (1/0.001)^6 = 10^{18}$ samples needed
- We have: $10^7$ samples (10M timesteps)
- **Shortfall**: $10^{11}$ × too little exploration!

**Why Early Training "Works"**:
- Lucky random initialization: Policy starts near-good region
- First 100k steps: Exploration noise stumbles into few successes
- LSTM memorizes these lucky sequences
- Further exploration: Moves away from lucky region
- Success rate: Decreases (opposite of learning!)

**This is a failure of exploration**, not optimization.

### 8.4 Sample Efficiency Comparison

| System | Episodes to Learn | Episode Length | Total Samples |
|--------|-------------------|----------------|---------------|
| **This project** | Unknown (didn't converge) | 1800 steps | 10M = 5555 episodes |
| Atari games (DQN) | ~1000 episodes | 1000 steps | ~1M samples |
| Robotic reaching (DDPG) | ~500 episodes | 50 steps | ~25k samples |
| Drone racing (PPO) | ~10,000 episodes | 200 steps | ~2M samples |

**Observation**: This task is **1-2 orders of magnitude more sample-starved** than typical RL benchmarks.

**Why This Matters**:
- With 5555 episodes, only ~185 successful episodes observed (33% success rate)
- These 185 successes must teach policy intercept strategy
- Distributed across 16 parallel environments
- Each environment sees ~11 successes total
- **Insufficient data** for generalization

**Required Samples** (estimate):
- For 50% success rate: ~50,000 episodes (90M timesteps)
- For 80% success rate: ~200,000 episodes (360M timesteps)

**Current computational budget**: 10M timesteps = only 3% of needed samples!

---

## 9. Expert Recommendations (Prioritized)

### Priority 1: Validate Assumptions ⚠️ CRITICAL

Before further expensive training runs, **verify the task is learnable at all**:

1. **Implement Proportional Navigation Baseline**
   ```python
   # Classical guidance law (no RL)
   def proportional_navigation(closing_velocity, los_rate, N=4):
       acceleration = N * closing_velocity * los_rate
       return acceleration
   ```
   **Expected**: 50-60% success rate with perfect radar
   **Purpose**: Establishes upper bound, validates physics

2. **Train with Full State (No Radar)**
   - Remove partial observability
   - Give policy omniscient observations
   - Train with PPO (same hyperparameters)
   - **Expected**: 80-95% success if task is learnable
   - **Purpose**: Isolates partial observability as blocker

3. **Analyze Value Function Predictions**
   ```python
   # Sample trajectories, compare V(s) to actual return
   for episode in sampled_episodes:
       predicted_values = model.policy.predict_values(observations)
       actual_returns = compute_returns(rewards)
       plot(predicted_values vs actual_returns)
   ```
   **Purpose**: Understand why value loss won't converge

**If baselines fail**: Task may be infeasible with current physics/spawns

**If baselines succeed**: Problem is LSTM training, not task design

### Priority 2: Algorithmic Change (If PPO Proven Inadequate)

**Option A: SAC with Frame Stacking** ⭐⭐⭐
```python
# Stack last 4 observations to provide temporal context
obs_stack = np.concatenate([obs[t-3], obs[t-2], obs[t-1], obs[t]])
# Now 104D observation space (26 × 4)

model = SAC(
    "MlpPolicy",
    env,
    buffer_size=1000000,  # Large replay buffer
    learning_rate=3e-4,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
)
```

**Option B: Dreamer (Model-Based)** ⭐⭐
- Learn world model: $p(s_{t+1} | s_t, a_t)$
- Plan interception trajectory in latent space
- More sample efficient than model-free
- **Challenge**: Complex to implement

**Option C: Offline RL with Expert Data** ⭐
- Collect 10,000 episodes with proportional navigation
- Train policy via behavioral cloning + offline RL
- Fine-tune with online PPO
- **Precedent**: Successful in robotics (BC + RL)

### Priority 3: Architecture Changes

**Option A: Transformer Policy** ⭐⭐⭐
```python
# Replace LSTM with Transformer (better long-range dependencies)
policy_kwargs = dict(
    features_extractor_class=TransformerExtractor,
    features_extractor_kwargs=dict(
        d_model=128,
        nhead=4,
        num_layers=2,
        sequence_length=64,  # Attend to last 64 steps
    )
)
```
**Rationale**: Transformers handle long sequences better than LSTM

**Challenge**: Not available in SB3, requires custom implementation

**Option B: Attention Mechanism** ⭐⭐
```python
# Add attention over past observations
class AttentionPolicy(nn.Module):
    def forward(self, obs_sequence):
        # obs_sequence: [batch, seq_len, obs_dim]
        attention_weights = self.attention(obs_sequence)
        context = torch.sum(attention_weights * obs_sequence, dim=1)
        action = self.policy_head(context)
        return action
```

**Option C: Separate Policy/Value Architectures** ⭐
```python
policy_kwargs = dict(
    net_arch=dict(
        pi=[256, 256],      # Policy: smaller
        vf=[512, 512, 256]  # Value: larger, more capacity
    )
)
```
**Rationale**: Value function needs more capacity than policy

### Priority 4: Training Techniques

**Option A: Hindsight Experience Replay** ⭐⭐⭐
```python
# For each failed episode:
#   - Relabel with hypothetical target position
#   - Create synthetic "success" for training
# Example: Missed by 500m → Relabel as "intercept 500m offset target"
```
**Benefit**: Turns every episode into a learning opportunity

**Challenge**: PPO doesn't natively support HER (need custom implementation)

**Option B: Prioritized Experience Replay** ⭐⭐
- Keep replay buffer of all episodes
- Sample successful episodes 10× more frequently
- **Requirement**: Need off-policy algorithm (SAC, TD3)

**Option C: Curriculum of Success Rates** ⭐
```python
# Don't advance curriculum until success rate threshold met
if current_success_rate > 60%:
    increase_difficulty()
else:
    keep_training_at_current_level()
```

### Priority 5: Observation Engineering

**Option A: Add Derived Features** ⭐⭐
```python
derived_features = [
    predicted_intercept_point,      # From Kalman filter
    time_to_intercept,               # Based on closing velocity
    approach_angle_quality,          # How good is current trajectory
    required_acceleration,           # To reach predicted intercept
]
```
**Rationale**: Give policy "hints" about what matters

**Option B: Observation Normalization** ⭐
- Currently positions in meters (100-2000 range)
- Velocities in m/s (0-200 range)
- Normalize to [-1, 1] for neural network
- **May already be done**: Check `VecNormalize` wrapper

**Option C: Temporal Difference Features** ⭐
```python
obs_delta = obs[t] - obs[t-1]  # Rate of change
obs_accel = obs_delta[t] - obs_delta[t-1]  # Acceleration
```

---

## 10. Conclusions & Next Actions

### 10.1 Summary of Findings

**What We Know**:
1. ✅ Observation space is functional (radar works, Kalman filter provides smooth estimates)
2. ✅ Reward structure prevents hacking (terminal reward dominates)
3. ✅ Task geometry is feasible (spawns within range, physics allows intercept)
4. ❌ LSTM training is pathologically unstable (success peaks early, degrades with learning)
5. ❌ Value function never converges (loss ~9000, target <100)
6. ❌ Credit assignment fails beyond ~17 steps (GAE effective horizon too short)
7. ❌ Sample efficiency is inadequate (~5500 episodes, need ~50,000)

**Root Cause Assessment**:
- **Primary**: Exploration failure + sparse rewards + long episodes = insufficient successful episodes to learn from
- **Secondary**: LSTM training instability exacerbates problem (spurious correlations, gradient issues)
- **Tertiary**: Partial observability makes task harder (but Kalman filter mitigates this)

**Why "More Training" Won't Help**:
- We've already trained 10M steps (5555 episodes)
- Pattern replicates across multiple 10M+ runs
- Problem is qualitative (wrong algorithm/architecture), not quantitative (more samples)

### 10.2 Recommended Immediate Actions

**Action 1**: Run Experiment 1 (MLP baseline) - 6 hours
- **Goal**: Determine if LSTM is necessary
- **Decision**: If MLP succeeds (>30%), abandon LSTM entirely
- **Cost**: Low (single training run)

**Action 2**: Implement PN baseline - 1 hour
- **Goal**: Validate task is learnable
- **Decision**: If PN gets <40%, task may be too hard
- **Cost**: Very low (scripting only)

**Action 3**: Train with full state observations - 6 hours
- **Goal**: Establish upper bound
- **Decision**: If still <50%, fundamental physics/reward issue
- **Cost**: Low (modify observation space, retrain)

**Decision Tree**:
```
MLP Experiment Result:
├─ >30% success: Use MLP, drop LSTM ✅ PROCEED
│  └─ Next: Tune MLP hyperparameters
│
├─ 15-30% success: LSTM helps slightly ⚠️ UNCERTAIN
│  └─ Next: Try Transformer or SAC with frame stacking
│
└─ <15% success: Memory not helping ❌ RETHINK
   └─ Next: Check full-state training, may need different approach
```

### 10.3 Long-Term Strategy

**If Short-Term Fixes Fail**:
1. Switch to **SAC** (off-policy, better exploration)
2. Implement **HER** (synthetic successes from failures)
3. Use **privileged learning** (train with full state, distill to radar)
4. Collect **expert demonstrations** (PN baseline) for imitation learning warmstart
5. Consider **model-based RL** (Dreamer, Planet) for planning

**If Project Needs to Succeed**:
- **Pragmatic approach**: Use proportional navigation (guaranteed 50-60%)
- **Hybrid approach**: RL for high-level strategy, PN for terminal guidance
- **Data-driven**: Collect 100k episodes with PN, train offline RL policy

### 10.4 Expected Outcomes

**Best Case** (MLP works):
- Train MLP for 10M steps
- Achieve 40-50% success rate
- Acceptable performance, task complete
- **Timeline**: 1 week

**Good Case** (SAC works):
- Implement SAC with frame stacking
- Train for 20M steps with replay buffer
- Achieve 50-70% success rate
- **Timeline**: 2-3 weeks (implementation + training)

**Acceptable Case** (Hybrid approach):
- Use PN for known-good scenarios
- RL for edge cases where PN fails
- Combined system: 70-80% success
- **Timeline**: 1-2 weeks (integration)

**Worst Case** (Nothing works):
- Task may be at limits of current RL
- Fall back to classical guidance (PN)
- 50-60% success rate
- Document as "RL research challenge"

---

## 11. Appendices

### A. Glossary

- **PPO**: Proximal Policy Optimization (on-policy RL algorithm)
- **LSTM**: Long Short-Term Memory (recurrent neural network)
- **GAE**: Generalized Advantage Estimation (value function method)
- **POMDP**: Partially Observable Markov Decision Process
- **HER**: Hindsight Experience Replay
- **SAC**: Soft Actor-Critic (off-policy RL algorithm)
- **PN**: Proportional Navigation (classical guidance law)

### B. Key Equations

**PPO Objective**:
$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

**GAE Advantage**:
$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$

**Proportional Navigation**:
$$\mathbf{a}_c = N \cdot V_c \cdot \dot{\boldsymbol{\lambda}}$$
- $\mathbf{a}_c$: commanded acceleration (perpendicular to LOS)
- $N$: navigation constant (typically 3-5)
- $V_c$: closing velocity
- $\dot{\boldsymbol{\lambda}}$: line-of-sight angular rate

### C. File Locations

**Configuration**: `/home/roman/Hlynr_Intercept/rl_system/config.yaml`
**Training Script**: `/home/roman/Hlynr_Intercept/rl_system/train.py`
**Environment**: `/home/roman/Hlynr_Intercept/rl_system/environment.py`
**Core Logic**: `/home/roman/Hlynr_Intercept/rl_system/core.py`

**Diagnosis Documents**:
- `/home/roman/Hlynr_Intercept/rl_system/DIAGNOSIS_33PCT_PLATEAU.md`
- `/home/roman/Hlynr_Intercept/rl_system/DIAGNOSIS_18PCT_CATASTROPHE.md`
- `/home/roman/Hlynr_Intercept/rl_system/DIAGNOSIS_FUNDAMENTAL_FAILURE.md`
- `/home/roman/Hlynr_Intercept/rl_system/FIXES_SUMMARY.md`

**Checkpoints**: `/home/roman/Hlynr_Intercept/research_checkpoints/`
**Logs**: `/home/roman/Hlynr_Intercept/rl_system/logs/`

### D. References

1. Schulman et al. (2017). "Proximal Policy Optimization Algorithms". arXiv:1707.06347
2. Henderson et al. (2018). "Deep Reinforcement Learning that Matters". AAAI 2018
3. Pascanu et al. (2013). "On the difficulty of training recurrent neural networks". ICML 2013
4. Ng et al. (1999). "Policy invariance under reward transformations". ICML 1999
5. Andrychowicz et al. (2017). "Hindsight Experience Replay". NeurIPS 2017
6. Kaufmann et al. (2023). "Champion-level drone racing using deep reinforcement learning". Nature

---

**Document Status**: ✅ COMPLETE
**Recommended Next Step**: Run Experiment 1 (MLP baseline) to determine if LSTM is causing instability
**Estimated Time to Resolution**: 1-3 weeks depending on experimental results
