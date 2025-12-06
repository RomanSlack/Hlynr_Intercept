# HRL Fix Plan: From 200m to Actually Good Performance

## STATUS: FIXES APPLIED ✅

The following fixes have been implemented as of November 2025:

### Fixed Issues:

1. **Distance Calculation** ✅ - Now uses world positions from `info['interceptor_pos']` and `info['missile_pos']`
2. **Frame-Stacking** ✅ - Evaluation now applies `FrameStackObservation` with configurable `--frame-stack` flag
3. **Sanity Checks** ✅ - Evaluation prints warnings for implausible results (< 10cm distances)
4. **Specialist Testing** ✅ - New `test_specialist.py` script validates specialists individually
5. **Training Validation** ✅ - Training callback now logs real distances and warns on suspicious values

---

## What Went Wrong (Historical)

### 1. **Measurement Bug** ❌ → ✅ FIXED
- **Problem**: Evaluated distance using `np.linalg.norm(obs[0:3])` where `obs[0:3]` is **normalized** relative position (divided by max_range and clipped to [-1, 1])
- **Result**: Thought we had 0.05m precision, actually had ~200m precision
- **Impact**: Complete false positive - celebrated a "breakthrough" that didn't exist
- **Fix**: Now uses `info['interceptor_pos']` and `info['missile_pos']` for true world distances

### 2. **Frame-Stacking Mismatch** ❌ → ✅ FIXED
- **Problem**: Specialists trained with frame-stacking (104D = 26D × 4 frames), but evaluation runs without it (26D)
- **Result**: Can't even load the specialist models during evaluation
- **Impact**: Been running evaluation with **stub policies** (random actions) instead of trained specialists
- **Fix**: Evaluation now applies `FrameStackObservation` wrapper, configurable via `--frame-stack` flag

### 3. **No Validation** ❌ → ✅ FIXED
- **Problem**: Never checked actual world-space positions in logs
- **Result**: Accepted implausible results without verification
- **Impact**: Wasted time celebrating fake results instead of debugging
- **Fix**: Added sanity checks that warn when distances are physically implausible

## Root Cause Analysis

Looking at the actual logged positions:
```
Interceptor: [460.20, 414.13, 62.76]
Missile:     [537.02, 591.67, 112.22]
Distance:    199.67m (NOT 0.01m!)
```

The HRL policy is:
1. ✅ Switching between options (SEARCH → TRACK → TERMINAL)
2. ❌ NOT achieving close intercepts (~200m miss distance)
3. ❌ Specialists might not have learned effective policies
4. ❌ Or specialists aren't being used properly during training/evaluation

## The Fix Plan

### Phase 1: Fix Evaluation (Immediate)

**Goal**: Get accurate measurements of current performance

1. **Fix distance calculation** in `evaluate_hrl.py`:
   ```python
   # WRONG (current):
   rel_pos = obs[0:3]  # Normalized values!
   distance = np.linalg.norm(rel_pos)

   # RIGHT:
   # Use actual world positions from info dict
   int_pos = info['interceptor_pos']
   mis_pos = info['missile_pos']
   distance = np.linalg.norm(int_pos - mis_pos)
   ```

2. **Add frame-stacking** to evaluation environment to match training:
   ```python
   from stable_baselines3.common.vec_env import VecFrameStack
   env = VecFrameStack(env, n_stack=4)  # Match training config
   ```

3. **Verify specialists are loaded** and being used (not stubs)

### Phase 2: Diagnose Specialist Training

**Goal**: Understand if specialists learned anything useful

1. **Check specialist training curves**:
   - Did they converge?
   - What rewards did they achieve?
   - Were they trained long enough?

2. **Test specialists individually**:
   - Run each specialist policy standalone
   - Measure their performance on their specific subtask
   - Verify they're better than random

3. **Check reward shaping**:
   - Are specialist rewards properly decomposed?
   - Does each specialist get reward signal for its subtask?

### Phase 3: Fix Training Configuration

**Goal**: Retrain with correct setup

#### Option A: Quick Fix - Retrain Specialists Without Frame-Stacking
```bash
# Train each specialist with 26D observations (no frame-stacking)
python scripts/train_hrl_pretrain.py --option search --steps 100000 --no-frame-stack
python scripts/train_hrl_pretrain.py --option track --steps 100000 --no-frame-stack
python scripts/train_hrl_pretrain.py --option terminal --steps 100000 --no-frame-stack

# Train selector
python scripts/train_hrl_selector.py --steps 10000 --no-frame-stack

# Evaluate properly
python scripts/evaluate_hrl.py \
  --selector checkpoints/hrl/selector/latest/model.zip \
  --search checkpoints/hrl/specialists/search/latest/model.zip \
  --track checkpoints/hrl/specialists/track/latest/model.zip \
  --terminal checkpoints/hrl/specialists/terminal/latest/model.zip \
  --episodes 100
```

#### Option B: Proper Fix - Use Frame-Stacking Consistently
```bash
# Train with frame-stacking (slower but potentially better)
python scripts/train_hrl_pretrain.py --option search --steps 200000 --frame-stack 4
python scripts/train_hrl_pretrain.py --option track --steps 200000 --frame-stack 4
python scripts/train_hrl_pretrain.py --option terminal --steps 200000 --frame-stack 4

# Train selector
python scripts/train_hrl_selector.py --steps 20000 --frame-stack 4

# Evaluate with frame-stacking
python scripts/evaluate_hrl.py \
  --selector ... --search ... --track ... --terminal ... \
  --episodes 100 --frame-stack 4
```

### Phase 4: Validation

**Goal**: Never lie to ourselves again

1. **Sanity Checks**:
   - ✅ Verify actual world-space positions match claimed precision
   - ✅ Check that distances make physical sense (can't get closer than missile radius)
   - ✅ Compare against baseline (flat PPO at ~150m)

2. **Progressive Testing**:
   ```python
   # Test 1: Individual specialists
   - Search: Can it point toward missile? (measure pointing error)
   - Track: Can it follow trajectory? (measure tracking error)
   - Terminal: Can it close distance? (measure final approach distance)

   # Test 2: Full HRL
   - Does selector switch at appropriate times?
   - Do specialists hand off smoothly?
   - Is final performance better than flat PPO?
   ```

3. **Visualization**:
   - Plot actual 3D trajectories
   - Visualize option switches on trajectory
   - Show distance over time

## Expected Outcomes

### Realistic Goals

1. **Short term**: Get HRL working at all (~100-150m precision, matching flat PPO)
2. **Medium term**: Beat flat PPO by 2-3x (50-75m precision)
3. **Long term**: Achieve actual sub-meter precision with proper reward shaping and longer training

### Why HRL Should Help (Eventually)

- **Search**: Specialist can focus on wide-angle scanning and approach
- **Track**: Specialist can learn to maintain stable tracking
- **Terminal**: Specialist can learn aggressive close-in maneuvering

But this only works if:
1. Specialists are actually trained on their subtasks
2. Selector learns when to switch
3. Handoffs between specialists are smooth
4. Each specialist gets proper reward signal

## Implementation Checklist

- [x] Fix `evaluate_hrl.py` distance calculation to use world positions
- [x] Add frame-stacking option to evaluation (`--frame-stack` flag)
- [ ] Test current specialists individually to see if they learned anything
- [ ] Check training logs/curves for specialists
- [x] Update training scripts with validation hooks (real distance logging)
- [x] Add sanity checks to prevent future false positives
- [ ] Create visualization tools for trajectory analysis
- [ ] Document actual performance honestly
- [ ] Retrain specialists from scratch with proper validation

---

## CORRECTED TRAINING WORKFLOW

### Step 1: Train Specialists (WITH VALIDATION)

```bash
cd /home/roman/Hlynr_Intercept/rl_system

# Train each specialist (15 min each)
python scripts/train_hrl_pretrain.py --agent search --config configs/hrl/search_specialist.yaml
python scripts/train_hrl_pretrain.py --agent track --config configs/hrl/track_specialist.yaml
python scripts/train_hrl_pretrain.py --agent terminal --config configs/hrl/terminal_specialist.yaml
```

**Monitor TensorBoard during training:**
```bash
tensorboard --logdir logs --port 6006
```

Watch for:
- `specialists/*/real_distance` - Should be in 50-500m range, NOT sub-meter!
- `specialists/*/success_rate` - Should improve over training
- If you see sub-meter distances, STOP - there's still a measurement bug

### Step 2: TEST SPECIALISTS INDIVIDUALLY (CRITICAL!)

```bash
# Test each specialist BEFORE integrating into HRL
python scripts/test_specialist.py --specialist search \
    --model checkpoints/hrl/specialists/search/best/model.zip \
    --episodes 50 --frame-stack 4

python scripts/test_specialist.py --specialist track \
    --model checkpoints/hrl/specialists/track/best/model.zip \
    --episodes 50 --frame-stack 4

python scripts/test_specialist.py --specialist terminal \
    --model checkpoints/hrl/specialists/terminal/best/model.zip \
    --episodes 50 --frame-stack 4
```

**Expected Results (REALISTIC):**
- Search: >80% lock acquisition rate
- Track: >70% lock maintenance rate
- Terminal: Mean min distance <100m (NOT sub-meter!)

**If any specialist fails these checks, DO NOT proceed to selector training!**

### Step 3: Train Selector

Only after specialists pass validation:

```bash
python scripts/train_hrl_selector.py --config configs/hrl/hrl_curriculum.yaml
```

### Step 4: Evaluate Full HRL System

```bash
python scripts/evaluate_hrl.py \
    --selector checkpoints/hrl/selector/best/best_model.zip \
    --search checkpoints/hrl/specialists/search/best/model.zip \
    --track checkpoints/hrl/specialists/track/best/model.zip \
    --terminal checkpoints/hrl/specialists/terminal/best/model.zip \
    --episodes 100 \
    --frame-stack 4 \
    --config config.yaml
```

**Sanity check the output:**
- If you see "SANITY CHECK WARNINGS" - STOP and investigate
- Distances should be in the 50-200m range initially
- Sub-10m distances are good, sub-1m is suspicious, sub-10cm is DEFINITELY WRONG

---

## Timeline

- **Day 1**: Fix evaluation, measure actual current performance
- **Day 2**: Diagnose specialist training, test individually
- **Day 3**: Retrain with correct configuration
- **Day 4**: Validate and compare to baseline
- **Day 5**: Iterate on reward shaping if needed

## Success Criteria

We'll know HRL is working when:
1. ✅ Actual world-space distance < 100m consistently
2. ✅ Better than flat PPO baseline (not worse!)
3. ✅ Specialists demonstrably better than random on subtasks
4. ✅ Selector switches at sensible times
5. ✅ Results are reproducible and verifiable

---

**Bottom line**: We have a lot of work to do. The current HRL is fundamentally broken and we need to rebuild from the ground up with proper validation at each step.
