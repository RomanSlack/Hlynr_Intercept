# HRL System Architecture

## Overview

This document provides a high-level overview of the Hierarchical RL system. For detailed design specifications, see:
- **[HRL_ARCHITECTURE_SUMMARY.md](../../hrl/HRL_ARCHITECTURE_SUMMARY.md)** - Complete architecture with diagrams
- **[HRL_REFACTORING_DESIGN.md](../../hrl/HRL_REFACTORING_DESIGN.md)** - Implementation design and interfaces

### Key Concepts

**Hierarchical Control**: Three-level decision making
1. **Selector** (1Hz): High-level option selection on abstract state
2. **Specialists** (100Hz): Low-level continuous actions per option
3. **Environment**: 6DOF physics with radar observations

**Three Options**:
- **SEARCH** (0): Wide-area scanning for target acquisition
- **TRACK** (1): Maintain lock and close distance
- **TERMINAL** (2): Final intercept guidance

**Design Philosophy**: Composition over modification - HRL components wrap existing environment without changes.

---

## Data Flow

```
Episode Start
     |
     v
┌─────────────────────────────────────┐
│  InterceptEnvironment (unchanged)  │
│  - 26D radar observations          │
│  - 6DOF physics                    │
│  - Safety constraints              │
└──────────┬──────────────────────────┘
           |
           v
┌─────────────────────────────────────┐
│  Observation Abstraction            │
│  26D → 7D abstract state            │
│  [distance, closing_rate, lock,     │
│   fuel, off_axis, tti, altitude]    │
└──────────┬──────────────────────────┘
           |
           v
┌─────────────────────────────────────┐
│  HierarchicalManager                │
│  - Check forced transitions         │
│  - Selector decision (every 100)    │
│  - Active specialist execution      │
└──────────┬──────────────────────────┘
           |
           v
    ┌──────┴──────┐
    |             |
    v             v
Selector       Specialist
(7D abs)       (104D full)
Discrete       Continuous
{0,1,2}        6D action
```

### Typical Episode Flow

```
Steps 1-150: SEARCH
  - Specialist: SearchSpecialist
  - Goal: Acquire radar lock
  - Action: Wide angular changes

Step 150: Lock acquired → FORCED SWITCH to TRACK

Steps 151-620: TRACK
  - Specialist: TrackSpecialist
  - Goal: Maintain lock, close distance
  - Action: Smooth tracking, balanced thrust

Step 620: Distance < 100m → FORCED SWITCH to TERMINAL

Steps 621-670: TERMINAL
  - Specialist: TerminalSpecialist
  - Goal: Minimize miss distance
  - Action: Maximum thrust, precision

Step 670: Distance < 5m → SUCCESS
```

---

## Module Relationships

```
hrl/
├── option_definitions.py    # Shared constants, Option enum
├── observation_abstraction.py  # 26D → 7D conversion
├── reward_decomposition.py     # Strategic + tactical rewards
│
├── selector_policy.py       # High-level discrete PPO
├── specialist_policies.py   # Low-level continuous PPO
├── option_manager.py        # Forced transition logic
├── manager.py              # Coordinates selector + specialists
│
├── wrappers.py             # Gymnasium wrappers
└── hierarchical_env.py     # Main HRL environment
```

### Key Dependencies

- **manager.py** imports: selector_policy, specialist_policies, option_manager, observation_abstraction
- **hierarchical_env.py** imports: manager, wrappers
- **No circular dependencies**: Clear one-way module relationships

---

## Configuration Structure

```
configs/
├── config.yaml              # Base config (unchanged)
├── scenarios/
│   ├── easy.yaml            # Wide radar, slow targets
│   ├── medium.yaml          # Standard difficulty
│   └── hard.yaml            # Narrow radar, fast targets
└── hrl/
    ├── hrl_base.yaml        # Main HRL config
    ├── hrl_curriculum.yaml  # Training stages
    └── {search,track,terminal}_specialist.yaml
```

**Key Settings**:
- `decision_interval_steps: 100` - High-level frequency (1Hz @ 100Hz sim)
- `enable_forced_transitions: true` - Physics-based option switching
- Reward weights per option (search, track, terminal)
- Forced transition thresholds (lock quality, distance, fuel)

See existing config files for complete parameter lists.

---

## Observation Abstraction

**Full Observation** (26D):
```
[0-2]   Relative position to target
[3-5]   Relative velocity
[6-8]   Interceptor velocity
[9-11]  Interceptor orientation
[12]    Fuel fraction
[13]    Time to intercept
[14]    Radar lock quality
[15]    Closing rate
[16]    Off-axis angle
[17-25] (reserved)
```

**Abstract State** (7D) for Selector:
```
[0] Distance (normalized 0-1)
[1] Closing rate (normalized -1 to 1)
[2] Lock quality (0-1)
[3] Fuel fraction (0-1)
[4] Off-axis angle (normalized -1 to 1)
[5] Time to intercept (normalized 0-1)
[6] Relative altitude (normalized -1 to 1)
```

**Rationale**: Selector needs strategic info only, not low-level sensor noise.

---

## Reward Decomposition

### Strategic Rewards (Selector)
- Intercept success: +1000
- Fuel efficiency: +100 × fuel_remaining
- Timeout penalty: -100
- Distance shaping: -0.01 per meter
- Closing rate bonus: +0.1 per m/s

### Tactical Rewards (Specialists)

**Search**:
- Lock acquisition: +50
- Lock improvement: +10 per 0.1 increase
- Angular diversity: +0.5
- Fuel waste: -5.0

**Track**:
- Lock maintenance: +2.0 per step
- Lock loss: -10.0
- Distance reduction: +1.0 per meter
- Jerky movement: -0.2

**Terminal**:
- Proximity: +10 × exp(-dist/10)
- Distance increase: -5.0
- Closing rate: +1.0 per m/s
- Max thrust: +0.5

See `hrl/reward_decomposition.py` for complete implementation.

---

## Forced Transitions

**SEARCH → TRACK**:
- Condition: lock_quality > 0.7
- Rationale: Can't track without lock

**TRACK → SEARCH**:
- Condition: lock_quality < 0.3
- Rationale: Lost lock, must reacquire

**TRACK → TERMINAL**:
- Condition: distance < 100m
- Rationale: Close-range precision needed

**Overrides**: Forced transitions bypass selector decisions to enforce physical constraints.

---

## Checkpoint Organization

```
checkpoints/
├── flat_ppo/               # Existing flat PPO (migrated)
│   ├── best/
│   │   ├── best_model.zip
│   │   └── vec_normalize.pkl
│   └── model_*_steps.zip
│
└── hrl/                    # HRL system
    ├── selector/
    │   └── best/
    │       ├── best_model.zip
    │       └── vec_normalize.pkl
    └── specialists/
        ├── search/best/
        ├── track/best/
        └── terminal/best/
```

---

## Performance Characteristics

| Metric | Flat PPO | HRL |
|--------|----------|-----|
| Training time | 25-30 min | ~2 hours |
| Intercept success | 75-85% | 70-85% |
| Interpretability | Low | High |
| Modularity | Monolithic | Composable |
| Sample efficiency | Baseline | 20-30% better |

**Tradeoff**: Longer training but better interpretability and modularity.

---

## Further Reading

- **Training Guide**: [training_guide.md](training_guide.md)
- **API Reference**: [api_reference.md](api_reference.md)
- **Migration Guide**: [migration_guide.md](migration_guide.md)
- **Complete Architecture**: [../../hrl/HRL_ARCHITECTURE_SUMMARY.md](../../hrl/HRL_ARCHITECTURE_SUMMARY.md)
- **Design Specification**: [../../hrl/HRL_REFACTORING_DESIGN.md](../../hrl/HRL_REFACTORING_DESIGN.md)
