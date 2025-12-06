# HRL Precision Training Progress

## Current Best Results (November 29, 2025)

**34% success rate at 100m threshold, 168m mean min distance**

| Metric | Before | After |
|--------|--------|-------|
| Success Rate | 24% | 34% |
| Mean Min Distance | 188m | 168m |
| Mean Reward | 1603 | 2212 |
| Sub-100m Precision | 12/50 | 17/50 |

---

## Working HRL Checkpoint Configuration

```bash
python scripts/evaluate_hrl.py \
  --selector checkpoints/hrl/selector/20251126_113539_500000steps/final/model.zip \
  --search checkpoints/hrl/specialists/search/20251125_174448_500000steps/final/model.zip \
  --track checkpoints/hrl/specialists/track/20251125_175356_500000steps/final/model.zip \
  --terminal checkpoints/hrl/specialists/terminal/20251129_130451_1500000steps/final/model.zip \
  --episodes 50 --seed 44
```

### Checkpoint Details

| Specialist | Checkpoint | Training Reward |
|------------|------------|-----------------|
| Selector | `20251126_113539_500000steps` | Rule-based + learned |
| Search | `20251125_174448_500000steps` | +852 |
| Track | `20251125_175356_500000steps` | +2384 |
| Terminal | `20251129_130451_1500000steps` | Fine-tuned with precision_mode |

---

## Key Changes Made

### 1. Implemented `precision_mode` in Environment

**File:** `rl_system/environment.py`

Without `precision_mode`: Episode terminates when distance < threshold. Model learns "99m = 5m = success".

With `precision_mode`: Episode continues after threshold crossing. Rewards based on minimum distance achieved with exponential scaling.

```python
# In curriculum config:
precision_mode: true

# Effect: Episode continues, reward based on min distance with bonuses:
# - exp(-dist/25) * 500 for sub-50m
# - exp(-dist/10) * 1000 for sub-20m
# - exp(-dist/3) * 500 for sub-5m
```

### 2. Updated Terminal Entry Thresholds

**File:** `rl_system/hrl/option_definitions.py`

```python
# Changed from 90-100m to 200m to match training
'close_range_threshold': 200.0      # Was 100.0
'terminal_enter_distance': 200.0    # Was 90.0
'terminal_exit_distance': 250.0     # Was 130.0
'miss_imminent_distance': 400.0     # Was 200.0
```

### 3. Created Precision Training Config

**File:** `rl_system/configs/hrl/terminal_precision_v6.yaml`

Key settings:
- **Resumes from working checkpoint** (`20251127_123710_2000000steps`)
- **Uses base config spawns** (800-1500m) - critical for compatibility
- **Lower learning rate** (0.0001) for fine-tuning
- **Curriculum 100m → 30m** with `precision_mode: true`

---

## Training Command

```bash
python scripts/train_hrl_pretrain.py \
  --agent terminal \
  --config configs/hrl/terminal_precision_v6.yaml \
  --resume checkpoints/hrl/specialists/terminal/20251127_123710_2000000steps/final/model.zip
```

---

## What Didn't Work

1. **Wrong spawn config** (`terminal_longrange_v1.yaml` with 200-400m spawns)
   - Trained for different scenario than evaluation uses
   - Result: 853m mean distance (worse than baseline)

2. **Training too long** (>2M steps)
   - Catastrophic forgetting observed
   - Sweet spot: 1-1.5M steps

3. **Fixed 30m threshold from start**
   - Too hard without curriculum
   - Need progressive 100m → 30m

---

## Next Steps for Sub-30m

1. Continue fine-tuning with tighter curriculum (50m → 20m)
2. Longer training with early stopping based on eval
3. Consider LSTM for trajectory prediction in terminal phase

---

## File References

| Purpose | File |
|---------|------|
| Environment with precision_mode | `rl_system/environment.py` |
| Transition thresholds | `rl_system/hrl/option_definitions.py` |
| Training config | `rl_system/configs/hrl/terminal_precision_v6.yaml` |
| Training script | `rl_system/scripts/train_hrl_pretrain.py` |
| Evaluation script | `rl_system/scripts/evaluate_hrl.py` |
| Standalone test | `rl_system/scripts/test_terminal_standalone.py` |