# Paper Benchmark Plan

## Overview

The benchmark suite (`rl_system/scripts/paper_benchmark_suite.py`) runs comprehensive experiments for paper-quality results.

## Experiments

### 1. Baseline Comparison (`--experiments baselines`)

Compares your system against classical and learned baselines:

| Method | Description | What it tests |
|--------|-------------|---------------|
| **Pure Pursuit** | Always fly toward target | Naive baseline (worst case) |
| **PN N=3 (true)** | Proportional Nav with ground truth LOS | Upper bound (unrealistic) |
| **PN N=3 (radar)** | PN using radar-derived LOS rates | Realistic classical baseline |
| **PN N=4 (radar)** | Higher gain PN | Tuned classical baseline |
| **PN N=5 (radar)** | Aggressive PN | Over-tuned classical |
| **Augmented PN** | PN + target acceleration estimate | Best classical baseline |
| **Flat PPO** | Your monolithic learned policy | RL baseline without hierarchy |
| **HRL (Full)** | Search + Track + Terminal specialists | Your main contribution |

**Outputs:**
- `baselines.json` - Raw results
- `baselines_table.tex` - LaTeX table for paper
- `baselines_figure.png` - Bar chart comparison

### 2. Ablation Studies (`--experiments ablations`)

Tests which components of HRL are essential:

| Ablation | What's changed |
|----------|----------------|
| HRL_NoForcedTransitions | Selector controls all transitions (no physics-based forcing) |
| HRL_SearchOnly | Only Search specialist (others stubbed) |
| HRL_TerminalOnly | Only Terminal specialist |
| HRL_NoSelector | Rule-based selector with trained specialists |

**Key question:** Does hierarchy help, or is it just the specialists?

### 3. Radar Degradation (`--experiments radar_degradation`)

Tests robustness to sensing conditions:

| Parameter | Values tested |
|-----------|---------------|
| `radar_noise` | 0.05, 0.10, 0.15, 0.20, 0.30 |
| `radar_beam_width` | 120°, 90°, 60°, 45°, 30° |

**Key claim:** HRL degrades gracefully; PN fails catastrophically at high noise.

### 4. Approach Angles (`--experiments approach_angles`)

Verifies LOS-frame direction invariance:

Tests at: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°

**Key claim:** <10% variance across angles proves direction invariance.

### 5. Lock Loss Recovery (`--experiments lock_loss_recovery`)

Tests Search phase effectiveness:

Scenarios with forced lock losses at varying rates.

**Key metric:** Lock recovery rate (recoveries / losses)

---

## Usage

```bash
cd rl_system

# Quick sanity check (10 episodes, 1 seed) - ~5 min
python scripts/paper_benchmark_suite.py --quick --config config.yaml

# Full benchmark (100 episodes, 5 seeds) - ~2-3 hours
python scripts/paper_benchmark_suite.py --full \
    --ppo-model checkpoints/best_model.zip \
    --hrl-selector checkpoints/hrl/selector/best/best_model.zip \
    --hrl-search checkpoints/hrl/specialists/search/final/model.zip \
    --hrl-track checkpoints/hrl/specialists/track/final/model.zip \
    --hrl-terminal checkpoints/hrl/specialists/terminal/final/model.zip

# Specific experiments only
python scripts/paper_benchmark_suite.py --experiments baselines radar_degradation \
    --episodes 50 --seeds 3

# Generate figures from existing results
python scripts/paper_benchmark_suite.py --analyze-only benchmark_results/benchmark_20241214/
```

---

## Expected Results (Based on Codebase Analysis)

From your training configs and previous runs:

| Method | Expected Success Rate | Notes |
|--------|----------------------|-------|
| Pure Pursuit | 20-30% | Misses maneuvering targets |
| PN N=3 (true) | 70-80% | Upper bound with cheating |
| PN N=4 (radar) | 50-60% | Realistic classical |
| Augmented PN | 55-65% | Best classical |
| Flat PPO | 60-75% | Depends on training |
| **HRL** | **75-90%** | Your contribution |

---

## Paper Figures to Generate

1. **Main comparison bar chart** - Success rate: PN vs PPO vs HRL
2. **Radar degradation curves** - Success vs noise level (HRL vs PN)
3. **360° polar plot** - Success rate by approach angle
4. **Lock loss/recovery** - Recovery rate under varying lock loss
5. **Phase transition diagram** - Example trajectory showing Search→Track→Terminal

---

## Statistical Rigor

- **5 random seeds** per configuration
- **95% confidence intervals** on all metrics
- **100 episodes** per seed (500 total per method)
- Metrics: Mean ± SEM reported

---

## Timeline

| Task | Time |
|------|------|
| Quick sanity check | 5 min |
| Baselines only | 30 min |
| Full benchmark | 2-3 hours |
| Analysis/figures | 10 min |
