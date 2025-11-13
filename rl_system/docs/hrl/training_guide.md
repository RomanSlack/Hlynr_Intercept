# HRL Training Guide

## Quick Start

### Three-Stage Training (Recommended)

```bash
cd /Users/quinnhasse/Hlynr_Intercept/rl_system

# Stage 1: Pre-train specialists (~45 min)
python scripts/train_hrl_pretrain.py --config configs/hrl/hrl_curriculum.yaml --specialist all
# OR train individually:
  python scripts/train_hrl_pretrain.py --specialist search
  python scripts/train_hrl_pretrain.py --specialist track
  python scripts/train_hrl_pretrain.py --specialist terminal

# Stage 2: Train selector (~20 min)
python scripts/train_hrl_selector.py --config configs/hrl/hrl_curriculum.yaml

# Stage 3: Evaluate
python scripts/evaluate_hrl.py --selector checkpoints/hrl/selector/20251113_110656_12000steps/best/best_model.zip --episodes 100 
```

**Expected Runtime**: ~65 minutes total
**Expected Performance**: 70-85% intercept success rate

**Note on Checkpoints**: The HRL selector training now saves models to `best/` during training via automatic evaluation. The best model is selected based on evaluation performance at intervals specified in the config (default: every 5000 timesteps).

### Evaluation Commands

  # Basic evaluation
  python scripts/evaluate_hrl.py --selector checkpoints/hrl/selector/<timestamp>/final --episodes 25

  # With specific scenario
  python scripts/evaluate_hrl.py --model checkpoints/hrl/selector/best/ --episodes 100 --scenario medium

  # Compare with flat PPO
  python scripts/compare_policies.py --flat checkpoints/flat_ppo/best/ --hrl
  checkpoints/hrl/selector/best/ --episodes 100

```

**Expected Runtime**: ~65 minutes total
**Expected Performance**: 70-85% intercept success rate

### Monitor Training

```bash
tensorboard --logdir logs --port 6006
# Access: http://localhost:6006
```

---

## Training Workflow

### Stage 1: Pre-train Specialists (45 minutes)

Each specialist learns its specific objective independently:

#### Search Specialist (15 min)

**Objective**: Acquire radar lock quickly
**Training Focus**: Wide-area scanning, angular diversity
**Curriculum**: Easy (wide radar) → Medium scenarios

```bash
python scripts/train_hrl_pretrain.py \
    --config configs/hrl/search_specialist.yaml \
    --specialist search
```

**Success Metrics**:
- Lock quality > 0.7 within 200 steps
- Average lock acquisition time decreasing
- Reward curve stabilizing

**Key Config** (`configs/hrl/search_specialist.yaml`):
```yaml
rewards:
  lock_acquisition_bonus: 50.0    # Primary objective
  lock_improvement: 10.0          # Reward incremental progress
  angular_diversity_bonus: 0.5    # Encourage scanning
  fuel_waste_penalty: -5.0        # Efficiency constraint
```

#### Track Specialist (15 min)

**Objective**: Maintain lock while closing distance
**Training Focus**: Smooth tracking, approach optimization
**Initial Conditions**: Starts with lock_quality ≥ 0.7

```bash
python scripts/train_hrl_pretrain.py \
    --config configs/hrl/track_specialist.yaml \
    --specialist track
```

**Success Metrics**:
- Lock maintained (quality > 0.5) for >80% of episode
- Distance consistently decreasing
- Positive closing rate

**Key Config**:
```yaml
rewards:
  lock_maintenance_bonus: 2.0     # Steady bonus per step
  distance_reduction: 1.0         # Per meter closed
  closing_rate_bonus: 0.5         # Approach speed
  jerky_movement_penalty: -0.2    # Smooth control
```

#### Terminal Specialist (15 min)

**Objective**: Minimize miss distance with high precision
**Training Focus**: Maximum thrust, fine corrections
**Initial Conditions**: Starts at distance < 200m

```bash
python scripts/train_hrl_pretrain.py \
    --config configs/hrl/terminal_specialist.yaml \
    --specialist terminal
```

**Success Metrics**:
- Miss distance < 10m
- High closing rate at intercept
- Consistent interceptions

**Key Config**:
```yaml
rewards:
  proximity_bonus_scale: 10.0     # Exponential for close approach
  closing_rate_bonus: 1.0         # High impact speed
  max_thrust_bonus: 0.5           # Encourage full thrust
```

---

### Stage 2: Train Selector (20 minutes)

**Objective**: Learn when to switch between specialists
**Input**: 7D abstract state
**Output**: Discrete option {SEARCH=0, TRACK=1, TERMINAL=2}

```bash
python scripts/train_hrl_selector.py --config configs/hrl/hrl_curriculum.yaml
```

**What Selector Learns**:
- When to switch from SEARCH → TRACK (after lock)
- When to abort TRACK → SEARCH (if lock lost)
- When to enter TERMINAL (close range)
- Balancing fuel efficiency vs speed

**Training Flow**:
1. Load frozen specialist checkpoints
2. Wrap environment with HRL manager (training_selector=True mode)
3. Train discrete PPO on abstract observations
4. Automatic evaluation saves best model to `checkpoints/hrl/selector/<timestamp>/best/`
5. Final model saved to `checkpoints/hrl/selector/<timestamp>/final/`

**Success Metrics**:
- All 3 options used regularly (check `option_distribution` in TensorBoard)
- Intercept success rate > 70%
- Average option switches per episode: 2-3

---

### Stage 3: Evaluation and Comparison

#### Evaluate HRL System

```bash
# Run 100 episodes with best model
SEL="checkpoints/hrl/selector/$(ls -t checkpoints/hrl/selector | head -1)/best"
python scripts/evaluate_hrl.py \
    --selector "$SEL" \
    --episodes 100

# Or use a specific checkpoint
python scripts/evaluate_hrl.py \
    --selector checkpoints/hrl/selector/20251113_110656_12000steps/best \
    --episodes 100

# Results saved to: results/hrl_eval_*.json
```

**Metrics Collected**:
- Intercept success rate
- Average miss distance
- Fuel efficiency
- Option usage statistics
- Forced transitions per episode

#### Compare with Flat PPO

```bash
python scripts/compare_policies.py \
    --flat checkpoints/flat_ppo/best/ \
    --hrl checkpoints/hrl/selector/best/ \
    --episodes 100 \
    --scenario medium
```

**Comparison Report Includes**:
- Success rate difference (statistical significance)
- Fuel efficiency comparison
- Training time vs performance tradeoff
- Recommendation: which approach for your use case

---

## Key Hyperparameters

### Selector Policy

**Location**: `configs/hrl/hrl_base.yaml`

```yaml
hrl:
  hierarchy:
    decision_interval: 100        # High-level frequency (1Hz)

  selector:
    learning_rate: 0.0003         # Conservative for discrete
    n_steps: 2048                 # Rollout length
    batch_size: 64                # Mini-batch size
    ent_coef: 0.01                # Low entropy (exploit specialists)
    net_arch: [256, 256]          # Network size
```

**Tuning Guidance**:
- **decision_interval**: Lower = more frequent switching, higher overhead
  - Too low (< 50): Unstable, frequent switches
  - Too high (> 200): Slow to adapt to state changes
- **learning_rate**: Start 0.0003, reduce if unstable
- **ent_coef**: Keep low (0.01) - specialists already explore

### Specialist Policies

**Location**: Individual specialist configs

```yaml
specialist:
  learning_rate: 0.0003           # Standard PPO rate
  n_steps: 2048                   # Match environment episode length
  use_lstm: true                  # Enable for temporal reasoning
  lstm_hidden_dim: 256            # Hidden state size
  net_arch: [512, 512, 256]       # Larger than flat PPO
  ent_coef: 0.02                  # Search: higher; Track/Terminal: lower
```

**Tuning Guidance**:
- **use_lstm**: Recommended for Track and Terminal (temporal dependencies)
- **ent_coef**:
  - Search: 0.02 (explore scan patterns)
  - Track: 0.01 (balanced)
  - Terminal: 0.005 (exploit precision)
- **net_arch**: Can reduce to [256, 256] for faster training

### Forced Transitions

**Location**: `configs/hrl/hrl_base.yaml`

```yaml
hrl:
  thresholds:
    radar_lock_quality_min: 0.3       # Below: TRACK → SEARCH
    radar_lock_quality_search: 0.7    # Above: SEARCH → TRACK
    close_range_threshold: 100.0      # Below: → TERMINAL (meters)
    terminal_fuel_min: 0.1            # Min fuel for TERMINAL
```

**Tuning Guidance**:
- **lock_quality_min**: Too low → stays in TRACK without lock
- **lock_quality_search**: Too high → stays in SEARCH too long
- **close_range_threshold**: Larger → earlier terminal phase
  - 150m: Conservative (more time for precision)
  - 100m: Balanced
  - 50m: Aggressive (less time to correct)

---

## TensorBoard Metrics

### Specialist Training

```
specialists/search/
  - reward: Should increase to ~50-100
  - lock_acquisition_rate: Should approach 80-90%
  - avg_lock_time: Should decrease

specialists/track/
  - reward: Should stabilize at ~200-400
  - distance_reduction: Should be positive and increasing
  - lock_maintenance_rate: Should be >80%

specialists/terminal/
  - reward: Should increase as miss_distance decreases
  - miss_distance: Should drop to <10m
  - intercept_success_rate: Should approach 90%+
```

### Selector Training

```
selector/
  - reward: Should increase gradually
  - option_distribution: Check all 3 options used
  - option_switches_per_episode: Expect 2-3
  - forced_transitions_per_episode: Expect 1-2

overall/
  - intercept_success_rate: Target 70-85%
  - avg_fuel_remaining: Higher = more efficient
```

---

## Common Issues and Solutions

### Issue: Selector Always Chooses Same Option

**Symptoms**:
- `option_distribution` shows 90%+ for one option
- Other specialists rarely used

**Causes**:
- One specialist much better than others
- Abstract observation not informative enough
- Reward imbalance between options

**Solutions**:
1. Check specialist pre-training convergence:
   ```bash
   # Re-train underperforming specialist
   python scripts/train_hrl_pretrain.py --specialist track --config configs/hrl/track_specialist.yaml
   ```

2. Increase decision_interval to give selector more data:
   ```yaml
   decision_interval: 200  # Up from 100
   ```

3. Verify abstract observation includes enough info:
   ```python
   # Check observation_abstraction.py
   # Ensure lock_quality, distance, fuel all normalized correctly
   ```

### Issue: Training Very Slow

**Symptoms**:
- Specialist training > 30 min each
- Selector training > 1 hour

**Causes**:
- Network too large
- LSTM overhead
- Too many parallel environments

**Solutions**:
1. Reduce network size:
   ```yaml
   net_arch: [256, 256]  # Down from [512, 512, 256]
   ```

2. Disable LSTM for Search specialist:
   ```yaml
   use_lstm: false  # Search is mostly reactive
   ```

3. Profile training:
   ```bash
   python -m cProfile -o profile.stats scripts/train_hrl_pretrain.py ...
   ```

### Issue: Frequent Option Switching (Every Step)

**Symptoms**:
- `option_switches_per_episode` > 20
- Unstable behavior

**Causes**:
- decision_interval too short
- Selector still learning
- Forced transitions too sensitive

**Solutions**:
1. Increase decision_interval:
   ```yaml
   decision_interval: 200  # Up from 100
   ```

2. Tune forced transition thresholds:
   ```yaml
   radar_lock_quality_min: 0.2    # More tolerance before switching
   radar_lock_quality_search: 0.8  # Higher bar to exit search
   ```

3. Let selector train longer:
   ```yaml
   total_timesteps: 5000000  # Up from 3M
   ```

### Issue: Poor Intercept Success (<60%)

**Symptoms**:
- HRL performs worse than flat PPO
- Miss distances > 20m

**Causes**:
- Terminal specialist underfitted
- Selector switching too late
- Reward weights suboptimal

**Solutions**:
1. Increase Terminal specialist training:
   ```yaml
   terminal:
     pretrain:
       episodes: 300  # Up from 200
   ```

2. Lower close_range_threshold:
   ```yaml
   close_range_threshold: 150.0  # Earlier terminal entry
   ```

3. Increase terminal proximity reward:
   ```yaml
   proximity_bonus_scale: 20.0  # Up from 10.0
   ```

---

## Advanced Topics

### Curriculum Learning

For difficult scenarios, use progressive difficulty:

```yaml
# In hrl_curriculum.yaml
curriculum:
  stages:
    - scenario: easy
      episodes: 100
      specialist: search

    - scenario: medium
      episodes: 100
      specialist: search

    - scenario: hard
      episodes: 50
      specialist: search
```

### Joint Fine-Tuning (Optional)

After Stage 2, optionally fine-tune all policies together:

```bash
python scripts/train_hrl_joint.py --config configs/hrl/hrl_curriculum.yaml
```

**Benefits**: +5-10% performance improvement
**Risks**: May destabilize pre-trained policies
**Recommendation**: Use 0.1× learning rate, monitor carefully

### Parallel Specialist Training

Train all specialists simultaneously:

```bash
# Terminal 1
python scripts/train_hrl_pretrain.py --specialist search &

# Terminal 2
python scripts/train_hrl_pretrain.py --specialist track &

# Terminal 3
python scripts/train_hrl_pretrain.py --specialist terminal &

wait  # Wait for all to complete (~15 min instead of 45)
```

---

## Expected Runtimes and Resources

| Stage | Time | CPU | Memory | GPU |
|-------|------|-----|--------|-----|
| Search specialist | 15 min | 4-8 cores | 4GB | Optional |
| Track specialist | 15 min | 4-8 cores | 4GB | Optional |
| Terminal specialist | 15 min | 4-8 cores | 4GB | Optional |
| Selector training | 20 min | 4-8 cores | 4GB | Optional |
| **Total** | **65 min** | **4-8 cores** | **4GB** | **Optional** |

**Note**: GPU can speed up by 2-3×, but CPU is sufficient for this task size.

---

## Next Steps

1. **Training Complete?** → See [api_reference.md](api_reference.md) for deployment
2. **Upgrading from Flat PPO?** → See [migration_guide.md](migration_guide.md)
3. **Architecture Questions?** → See [architecture.md](architecture.md)
4. **Issues?** → Check [../../hrl/HRL_REFACTORING_DESIGN.md](../../hrl/HRL_REFACTORING_DESIGN.md) Troubleshooting section
