# HRL Architecture: Visual Summary

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HIERARCHICAL MANAGER                            │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────┐    │
│  │              SELECTOR POLICY (High-Level)                     │    │
│  │                                                               │    │
│  │  Input:  Abstract State (7D)                                 │    │
│  │          [distance, closing_rate, lock, fuel, ...]           │    │
│  │                                                               │    │
│  │  Output: Option {SEARCH, TRACK, TERMINAL}                    │    │
│  │                                                               │    │
│  │  Decision Frequency: Every 100 steps (1Hz)                   │    │
│  └───────────────────────────────────────────────────────────────┘    │
│                                   │                                    │
│                                   ▼                                    │
│  ┌──────────────────┬──────────────────┬──────────────────┐          │
│  │  SEARCH          │  TRACK           │  TERMINAL        │          │
│  │  SPECIALIST      │  SPECIALIST      │  SPECIALIST      │          │
│  │                  │                  │                  │          │
│  │  Objective:      │  Objective:      │  Objective:      │          │
│  │  Acquire lock    │  Maintain lock   │  Minimize miss   │          │
│  │  Scan coverage   │  Close distance  │  Final guidance  │          │
│  │                  │                  │                  │          │
│  │  Input: 104D     │  Input: 104D     │  Input: 104D     │          │
│  │  (26D × 4 stack) │  (26D × 4 stack) │  (26D × 4 stack) │          │
│  │                  │                  │                  │          │
│  │  Output: 6D      │  Output: 6D      │  Output: 6D      │          │
│  │  (3D thrust +    │  (3D thrust +    │  (3D thrust +    │          │
│  │   3D angular)    │   3D angular)    │   3D angular)    │          │
│  │                  │                  │                  │          │
│  │  Uses: LSTM      │  Uses: LSTM      │  Uses: LSTM      │          │
│  │  [512,512,256]   │  [512,512,256]   │  [512,512,256]   │          │
│  └──────────────────┴──────────────────┴──────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  INTERCEPT ENVIRONMENT       │
                    │  (Unchanged)                 │
                    │                              │
                    │  - 6DOF Physics              │
                    │  - Radar Simulation          │
                    │  - Safety Constraints        │
                    │  - 26D Observations          │
                    └──────────────────────────────┘
```

## Data Flow: Training vs Inference

### Training Flow (3-Stage Curriculum)

```
STAGE 1: PRE-TRAIN SPECIALISTS (Parallel)
==========================================

┌─────────────────────────────────────────────────────────────────┐
│  SEARCH SPECIALIST TRAINING                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Environment: Start without lock, wide search space       │  │
│  │ Reward: Lock acquisition, scan coverage                  │  │
│  │ Episodes: 200                                            │  │
│  │ Checkpoint: checkpoints/hrl/specialists/search/          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  TRACK SPECIALIST TRAINING                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Environment: Start with lock, medium distance            │  │
│  │ Reward: Lock maintenance, approach progress              │  │
│  │ Episodes: 200                                            │  │
│  │ Checkpoint: checkpoints/hrl/specialists/track/           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  TERMINAL SPECIALIST TRAINING                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Environment: Start close (<200m), good lock              │  │
│  │ Reward: Miss distance minimization                       │  │
│  │ Episodes: 200                                            │  │
│  │ Checkpoint: checkpoints/hrl/specialists/terminal/        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘


STAGE 2: TRAIN SELECTOR (Specialists Frozen)
=============================================

┌─────────────────────────────────────────────────────────────────┐
│  SELECTOR POLICY TRAINING                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Load: All 3 pre-trained specialists (weights frozen)     │  │
│  │ Input: Abstract state (7D)                               │  │
│  │ Action: Choose option {0, 1, 2}                          │  │
│  │ Reward: Strategic (intercept success, efficiency)        │  │
│  │ Episodes: 1000                                           │  │
│  │ Checkpoint: checkpoints/hrl/selector/                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘


STAGE 3: JOINT FINE-TUNING (Optional)
======================================

┌─────────────────────────────────────────────────────────────────┐
│  END-TO-END TRAINING                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Train: Both selector AND specialists                     │  │
│  │ Learning Rate: 10x lower for stability                   │  │
│  │ Episodes: 500                                            │  │
│  │ Risk: May destabilize pre-trained policies               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Inference Flow (Real-Time Decision Making)

```
EPISODE START
═════════════
│
├─► Reset HierarchicalManager
│   ├─► Current option = SEARCH (default)
│   ├─► Steps in option = 0
│   └─► Reset all specialist LSTM states
│
▼


SIMULATION LOOP (Every 0.01s = 100Hz)
═════════════════════════════════════

Step 1-100: SEARCH PHASE
┌─────────────────────────────────────────────────────────────┐
│ 1. Observe: Get 26D radar obs from environment             │
│    - No lock yet (lock_quality = 0.0)                      │
│                                                             │
│ 2. Check Forced Transitions:                               │
│    - Lock acquired? NO                                     │
│    - Continue SEARCH                                       │
│                                                             │
│ 3. Execute Specialist:                                     │
│    - SEARCH specialist predicts 6D action                  │
│    - Wide angular changes to scan area                     │
│                                                             │
│ 4. Apply Action:                                           │
│    - Environment steps with action                         │
│    - Get reward, next_obs                                  │
│                                                             │
│ 5. Update State:                                           │
│    - steps_in_option += 1                                  │
└─────────────────────────────────────────────────────────────┘

Step 100: HIGH-LEVEL DECISION
┌─────────────────────────────────────────────────────────────┐
│ 1. Time for selector decision (every 100 steps)            │
│                                                             │
│ 2. Abstract Observation:                                   │
│    - Convert 26D → 7D abstract state                       │
│    - [distance=800m, closing_rate=50, lock=0.0, ...]       │
│                                                             │
│ 3. Selector Predicts:                                      │
│    - Input: 7D abstract state                              │
│    - Output: Option = SEARCH (continue)                    │
│                                                             │
│ 4. No option switch, continue SEARCH                       │
└─────────────────────────────────────────────────────────────┘

Step 150: LOCK ACQUIRED! (Forced Transition)
┌─────────────────────────────────────────────────────────────┐
│ 1. Observe: lock_quality = 0.75 (above threshold)          │
│                                                             │
│ 2. Check Forced Transitions:                               │
│    ✓ Lock quality > 0.7 → FORCE SWITCH to TRACK           │
│                                                             │
│ 3. Option Switch:                                          │
│    - Reset SEARCH specialist LSTM                          │
│    - Switch to TRACK specialist                            │
│    - steps_in_option = 0                                   │
│                                                             │
│ 4. Execute TRACK specialist:                               │
│    - Smooth tracking, maintain lock                        │
└─────────────────────────────────────────────────────────────┘

Step 150-600: TRACK PHASE
┌─────────────────────────────────────────────────────────────┐
│ 1. TRACK specialist maintains lock                         │
│    - Balanced thrust + smooth angular changes              │
│    - Distance decreasing: 800m → 400m → 200m → 120m        │
│                                                             │
│ 2. High-level decisions every 100 steps:                   │
│    - Selector sees: distance decreasing, lock maintained   │
│    - Continues TRACK option                                │
└─────────────────────────────────────────────────────────────┘

Step 620: TERMINAL RANGE (Forced Transition)
┌─────────────────────────────────────────────────────────────┐
│ 1. Observe: distance = 95m (below 100m threshold)          │
│                                                             │
│ 2. Check Forced Transitions:                               │
│    ✓ Distance < 100m → FORCE SWITCH to TERMINAL           │
│                                                             │
│ 3. Option Switch:                                          │
│    - Reset TRACK specialist LSTM                           │
│    - Switch to TERMINAL specialist                         │
│    - steps_in_option = 0                                   │
│                                                             │
│ 4. Execute TERMINAL specialist:                            │
│    - Maximum thrust, high-precision guidance               │
└─────────────────────────────────────────────────────────────┘

Step 620-670: TERMINAL PHASE
┌─────────────────────────────────────────────────────────────┐
│ 1. TERMINAL specialist minimizes miss distance             │
│    - High thrust, precise corrections                       │
│    - Distance: 95m → 50m → 20m → 5m → 0.8m                │
│                                                             │
│ 2. Step 670: INTERCEPT!                                    │
│    - Distance < 5m → SUCCESS                               │
│    - Episode terminates                                    │
│    - Reward: +1000 (intercept bonus)                       │
└─────────────────────────────────────────────────────────────┘

EPISODE END
═══════════
Total steps: 670
Options used: SEARCH → TRACK → TERMINAL
Final outcome: SUCCESS
Miss distance: 0.8m
```

## Module Interaction Map

```
┌────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                             │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  scripts/train_hrl_full.py                                        │
│  ├─► scripts/train_hrl_pretrain.py                               │
│  │   └─► Train each specialist separately                         │
│  │                                                                 │
│  ├─► scripts/train_hrl_selector.py                               │
│  │   └─► Train selector with frozen specialists                   │
│  │                                                                 │
│  └─► scripts/train_hrl_joint.py (optional)                       │
│      └─► Joint fine-tuning                                        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                       HRL CORE LAYER                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  hrl/manager.py (HierarchicalManager)                             │
│  ├─► Coordinates selector + specialists                           │
│  ├─► Manages option switching                                     │
│  └─► Handles LSTM state persistence                               │
│                                                                    │
│  hrl/selector_policy.py (SelectorPolicy)                          │
│  ├─► High-level discrete PPO                                      │
│  ├─► Input: 7D abstract state                                     │
│  └─► Output: Option index {0, 1, 2}                               │
│                                                                    │
│  hrl/specialist_policies.py                                       │
│  ├─► SearchSpecialist, TrackSpecialist, TerminalSpecialist       │
│  ├─► Low-level continuous PPO with LSTM                           │
│  ├─► Input: 104D stacked observation                              │
│  └─► Output: 6D continuous action                                 │
│                                                                    │
│  hrl/option_manager.py (OptionManager)                            │
│  ├─► Forced transition logic                                      │
│  ├─► Check state-based triggers                                   │
│  └─► Return required option or None                               │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                    ABSTRACTION LAYER                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  hrl/observation_abstraction.py                                   │
│  ├─► abstract_observation(): 26D → 7D                            │
│  └─► extract_env_state_for_transitions()                         │
│                                                                    │
│  hrl/reward_decomposition.py                                      │
│  ├─► compute_strategic_reward() - for selector                    │
│  └─► compute_tactical_reward() - for specialists                  │
│      ├─► compute_search_reward()                                  │
│      ├─► compute_track_reward()                                   │
│      └─► compute_terminal_reward()                                │
│                                                                    │
│  hrl/wrappers.py                                                  │
│  ├─► HierarchicalActionWrapper                                    │
│  └─► AbstractObservationWrapper                                   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌────────────────────────────────────────────────────────────────────┐
│                   ENVIRONMENT LAYER                                │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  environment.py (InterceptEnvironment)                            │
│  ├─► 6DOF physics simulation                                      │
│  ├─► Radar sensor model                                           │
│  ├─► Safety constraints                                           │
│  └─► Returns 26D observations                                     │
│                                                                    │
│  core.py                                                           │
│  ├─► Radar26DObservation                                          │
│  ├─► SafetyClamp                                                  │
│  └─► Kalman filtering                                             │
│                                                                    │
│  physics_models.py                                                │
│  ├─► AtmosphericModel                                             │
│  ├─► MachDragModel                                                │
│  └─► EnhancedWindModel                                            │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Configuration Hierarchy

```
configs/
│
├── config.yaml                    # BASE CONFIG (unchanged)
│   ├─► Environment settings
│   ├─► Physics parameters
│   ├─► PPO hyperparameters
│   └─► Used by flat PPO (backward compat)
│
├── scenarios/                     # SCENARIO PRESETS
│   ├── easy.yaml                  # Wide radar, slow missile
│   ├── medium.yaml                # Default settings
│   └── hard.yaml                  # Narrow radar, fast missile
│
└── hrl/                           # HRL-SPECIFIC CONFIGS
    │
    ├── hrl_base.yaml              # MAIN HRL CONFIG
    │   ├─► Inherits from: ../config.yaml
    │   ├─► Hierarchy settings
    │   ├─► Selector configuration
    │   ├─► Specialist configurations
    │   ├─► Reward decomposition
    │   ├─► Forced transition thresholds
    │   └─► Training curriculum
    │
    ├── selector_config.yaml       # SELECTOR DETAILS
    │   ├─► Network architecture
    │   ├─► Learning rate schedule
    │   └─► Checkpoint settings
    │
    ├── search_specialist.yaml     # SEARCH SPECIALIST
    │   ├─► Pre-training scenarios
    │   ├─► Reward weights
    │   └─► LSTM configuration
    │
    ├── track_specialist.yaml      # TRACK SPECIALIST
    │   ├─► Initial conditions (with lock)
    │   ├─► Reward weights
    │   └─► LSTM configuration
    │
    ├── terminal_specialist.yaml   # TERMINAL SPECIALIST
    │   ├─► Initial conditions (close range)
    │   ├─► Reward weights
    │   └─► LSTM configuration
    │
    └── hrl_curriculum.yaml        # TRAINING CURRICULUM
        ├─► Stage 1: Pre-train (600 episodes)
        ├─► Stage 2: Selector (1000 episodes)
        └─► Stage 3: Joint fine-tuning (500 episodes)
```

## Checkpoint Organization

```
checkpoints/
│
├── flat_ppo/                      # EXISTING FLAT PPO (migrated)
│   ├── best/
│   │   ├── best_model.zip
│   │   └── vec_normalize.pkl
│   │
│   └── model_*_steps.zip          # Training checkpoints
│
└── hrl/                           # NEW HRL CHECKPOINTS
    │
    ├── selector/                  # High-level policy
    │   ├── model.zip
    │   ├── vec_normalize.pkl
    │   └── training_progress.json
    │
    └── specialists/               # Low-level policies
        │
        ├── search/
        │   ├── model.zip
        │   ├── vec_normalize.pkl
        │   └── training_metrics.json
        │
        ├── track/
        │   ├── model.zip
        │   ├── vec_normalize.pkl
        │   └── training_metrics.json
        │
        └── terminal/
            ├── model.zip
            ├── vec_normalize.pkl
            └── training_metrics.json
```

## Testing Strategy

```
tests/
│
├── Unit Tests (Individual Modules)
│   ├── test_option_definitions.py
│   │   └─► Enum values, metadata, thresholds
│   │
│   ├── test_observation_abstraction.py
│   │   ├─► 26D → 7D conversion
│   │   ├─► Normalization ranges
│   │   └─► Frame stacking handling
│   │
│   ├── test_specialist_policies.py
│   │   ├─► Model initialization
│   │   ├─► Predict with/without LSTM
│   │   ├─► Save/load checkpoints
│   │   └─► LSTM state reset
│   │
│   ├── test_selector_policy.py
│   │   ├─► Discrete action space
│   │   ├─► Abstract observation input
│   │   └─► Save/load checkpoints
│   │
│   ├── test_option_manager.py
│   │   ├─► Forced transition logic
│   │   ├─► Threshold checks
│   │   └─► Edge cases
│   │
│   └── test_hrl_manager.py
│       ├─► Option switching
│       ├─► LSTM state management
│       ├─► Action selection pipeline
│       └─► Statistics tracking
│
├── Integration Tests (Module Interactions)
│   ├── test_hierarchical_env.py
│   │   ├─► Full episode with HRL
│   │   ├─► Wrapper functionality
│   │   └─► Info dict propagation
│   │
│   ├── test_reward_decomposition.py
│   │   ├─► Strategic reward calculation
│   │   ├─► Tactical rewards per option
│   │   └─► Episode termination handling
│   │
│   └── test_wrappers.py
│       ├─► HierarchicalActionWrapper
│       ├─► AbstractObservationWrapper
│       └─► Wrapper stacking
│
├── End-to-End Tests (Full Pipeline)
│   ├── test_end_to_end.py
│   │   ├─► Complete episode execution
│   │   ├─► All options used
│   │   ├─► Forced transitions triggered
│   │   └─► Performance benchmarks
│   │
│   └── test_backward_compatibility.py
│       ├─► Flat PPO training unchanged
│       ├─► Config loading (old paths)
│       ├─► Checkpoint migration
│       └─► Existing scripts work
│
└── Fixtures & Utilities
    ├── conftest.py
    │   ├─► Pytest fixtures (dummy env, policies)
    │   └─► Mock objects
    │
    └── test_utils.py
        ├─► Helper functions
        └─► Test data generators
```

## Performance Comparison: Flat PPO vs HRL

```
METRIC                  │ FLAT PPO      │ HRL (Target)  │ Notes
────────────────────────┼───────────────┼───────────────┼──────────────────
Intercept Success Rate  │ 75-85%        │ 70-90%        │ Should be comparable
Training Time           │ 120-180 min   │ 480-720 min   │ Includes pre-training
                        │ (5M steps)    │ (total)       │
Fuel Efficiency         │ Baseline      │ 10-20% better │ HRL can optimize phases
Sample Efficiency       │ Baseline      │ 20-30% better │ Curriculum learning
Interpretability        │ Low           │ High          │ Clear option transitions
Adaptation              │ Moderate      │ High          │ Can swap specialists
Multi-Task Transfer     │ Poor          │ Good          │ Reuse specialists
```

## Quick Reference: Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **3 Options** | Matches natural phases: search → track → terminal |
| **100-step decision interval** | 1Hz high-level @ 100Hz sim = human-scale decisions |
| **Forced transitions** | Ensure physical constraints (can't track without lock) |
| **7D abstract state** | Strategic info only, filters noise for selector |
| **Curriculum learning** | Pre-train specialists → selector → optional joint |
| **LSTM for specialists** | Temporal reasoning for tracking, terminal guidance |
| **No LSTM for selector** | Abstract state is mostly Markovian |
| **Reward decomposition** | Strategic (long-term) vs tactical (option-specific) |
| **Wrapper pattern** | Zero changes to existing environment |
| **Symlink for scenarios/** | Backward compatibility without code changes |

---

## Summary

This architecture provides:

1. **Modularity:** HRL components are isolated and composable
2. **Backward Compatibility:** Existing flat PPO workflow unchanged
3. **Testability:** Comprehensive unit, integration, and E2E tests
4. **Extensibility:** Clear interfaces for future enhancements
5. **Maintainability:** Clean separation of concerns, minimal coupling

The design achieves hierarchical control while preserving the production-ready qualities of the existing system.
