# HRL System Dependency Diagram

## Visual Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CORE SYSTEM (UNCHANGED)                             │
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │ environment. │────▶│    core.py   │     │  physics_    │                │
│  │     py       │     │              │────▶│  models.py   │                │
│  │              │     │  26D Radar   │     │              │                │
│  │ Intercept    │     │  Observation │     │  6DOF        │                │
│  │ Environment  │     │              │     │  Dynamics    │                │
│  └──────────────┘     └──────────────┘     └──────────────┘                │
│         ▲                                                                    │
│         │                                                                    │
│         │ Composition (No Modification)                                     │
└─────────┼──────────────────────────────────────────────────────────────────┘
          │
          │
┌─────────┼──────────────────────────────────────────────────────────────────┐
│         │                  HRL SYSTEM (NEW MODULES)                         │
│         │                                                                    │
│  ┌──────▼────────────────────────────────────────────────────────────────┐ │
│  │                     PHASE 5: INTEGRATION LAYER                         │ │
│  │                                                                         │ │
│  │  ┌────────────────────────────────────────────────────────┐           │ │
│  │  │  hierarchical_env.py                                   │           │ │
│  │  │  ┌──────────────────────────────────────────────────┐ │           │ │
│  │  │  │  make_hrl_env(config)                            │ │           │ │
│  │  │  │  - Creates InterceptEnvironment                  │ │           │ │
│  │  │  │  - Wraps with HRLActionWrapper                   │ │           │ │
│  │  │  │  - Applies VecFrameStack, VecNormalize           │ │           │ │
│  │  │  └──────────────────────────────────────────────────┘ │           │ │
│  │  └────────────────────────────────────────────────────────┘           │ │
│  │         │                                                               │ │
│  │         ├──────────────────────────────────────────────────────────┐   │ │
│  │         │                                                           │   │ │
│  │  ┌──────▼─────────────┐  ┌────────────────────┐  ┌────────────────▼─┐ │ │
│  │  │ train_hrl_pretrain │  │ train_hrl_selector │  │ train_hrl_full   │ │ │
│  │  │      .py           │  │        .py         │  │      .py         │ │ │
│  │  │                    │  │                    │  │                  │ │ │
│  │  │ Pre-train          │  │ Train selector     │  │ End-to-end       │ │ │
│  │  │ specialists        │  │ with frozen        │  │ orchestrator     │ │ │
│  │  │                    │  │ specialists        │  │                  │ │ │
│  │  └────────────────────┘  └────────────────────┘  └──────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│           │                         │                       │                │
│           ▼                         ▼                       ▼                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     PHASE 4: WRAPPER LAYER                          │   │
│  │                                                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │  wrappers.py                                                 │  │   │
│  │  │                                                              │  │   │
│  │  │  ┌────────────────────────────────────────────────────┐    │  │   │
│  │  │  │  HRLActionWrapper(env, manager)                    │    │  │   │
│  │  │  │  - Wraps InterceptEnvironment                      │    │  │   │
│  │  │  │  - Intercepts step() calls                         │    │  │   │
│  │  │  │  - Delegates action selection to manager           │    │  │   │
│  │  │  │  - Returns enhanced info dict with HRL metadata   │    │  │   │
│  │  │  └────────────────────────────────────────────────────┘    │  │   │
│  │  │                                                              │  │   │
│  │  │  ┌────────────────────────────────────────────────────┐    │  │   │
│  │  │  │  AbstractObservationWrapper(env)                   │    │  │   │
│  │  │  │  - Converts 26D → 7D abstract state                │    │  │   │
│  │  │  │  - Used for selector training                      │    │  │   │
│  │  │  └────────────────────────────────────────────────────┘    │  │   │
│  │  │                                                              │  │   │
│  │  │  ┌────────────────────────────────────────────────────┐    │  │   │
│  │  │  │  RewardShapingWrapper(env, reward_mode)            │    │  │   │
│  │  │  │  - Replaces env reward with HRL reward             │    │  │   │
│  │  │  │  - Supports strategic/tactical modes               │    │  │   │
│  │  │  └────────────────────────────────────────────────────┘    │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   PHASE 3: REWARD SYSTEM                            │   │
│  │                                                                      │   │
│  │  ┌──────────────────────────────────────────────────────────────┐  │   │
│  │  │  reward_decomposition.py                                     │  │   │
│  │  │                                                              │  │   │
│  │  │  ┌────────────────────────────────────────────────────┐    │  │   │
│  │  │  │  compute_strategic_reward(env_state, ...)         │    │  │   │
│  │  │  │  - For selector policy                             │    │  │   │
│  │  │  │  - Intercept success: +1000                        │    │  │   │
│  │  │  │  - Distance shaping: -0.01/m                       │    │  │   │
│  │  │  └────────────────────────────────────────────────────┘    │  │   │
│  │  │                                                              │  │   │
│  │  │  ┌────────────────────────────────────────────────────┐    │  │   │
│  │  │  │  compute_tactical_reward(env_state, option, ...)  │    │  │   │
│  │  │  │  - Dispatches to option-specific functions:        │    │  │   │
│  │  │  │    • compute_search_reward()                       │    │  │   │
│  │  │  │    • compute_track_reward()                        │    │  │   │
│  │  │  │    • compute_terminal_reward()                     │    │  │   │
│  │  │  └────────────────────────────────────────────────────┘    │  │   │
│  │  └──────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│           │                                                                  │
│           ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              PHASE 2: CORE HRL COMPONENTS (COMPLETE ✅)             │   │
│  │                                                                      │   │
│  │  ┌────────────────────────────────────────────────────────────┐    │   │
│  │  │  manager.py                                                │    │   │
│  │  │  ┌──────────────────────────────────────────────────────┐ │    │   │
│  │  │  │  HierarchicalManager                                 │ │    │   │
│  │  │  │  - Coordinates selector and specialists             │ │    │   │
│  │  │  │  - Manages LSTM states                              │ │    │   │
│  │  │  │  - Enforces decision intervals                      │ │    │   │
│  │  │  │  - Returns enhanced info dict                       │ │    │   │
│  │  │  │                                                      │ │    │   │
│  │  │  │  select_action(full_obs, env_state, deterministic)  │ │    │   │
│  │  │  │    1. Check forced transitions                      │ │    │   │
│  │  │  │    2. Selector decision (if at interval)            │ │    │   │
│  │  │  │    3. Switch option if needed                       │ │    │   │
│  │  │  │    4. Execute active specialist                     │ │    │   │
│  │  │  │    5. Update state & return action                  │ │    │   │
│  │  │  └──────────────────────────────────────────────────────┘ │    │   │
│  │  └────────────────────────────────────────────────────────────┘    │   │
│  │           │                   │                                     │   │
│  │           ▼                   ▼                                     │   │
│  │  ┌──────────────────┐  ┌────────────────────────────────────┐     │   │
│  │  │ selector_policy  │  │  specialist_policies.py            │     │   │
│  │  │      .py         │  │                                     │     │   │
│  │  │                  │  │  ┌──────────────────────────────┐  │     │   │
│  │  │  SelectorPolicy  │  │  │  SpecialistPolicy (Base)     │  │     │   │
│  │  │  - Discrete(3)   │  │  │  - 6D continuous actions     │  │     │   │
│  │  │  - 7D abstract   │  │  │  - 104D observations (26*4)  │  │     │   │
│  │  │    state input   │  │  │                              │  │     │   │
│  │  │  - Mode: learned │  │  ├──────────────────────────────┤  │     │   │
│  │  │    or rules      │  │  │  SearchSpecialist            │  │     │   │
│  │  │                  │  │  │  - Wide scanning             │  │     │   │
│  │  │  predict()       │  │  │  - Lock acquisition          │  │     │   │
│  │  │  → Option index  │  │  ├──────────────────────────────┤  │     │   │
│  │  └──────────────────┘  │  │  TrackSpecialist             │  │     │   │
│  │                        │  │  - Maintain lock             │  │     │   │
│  │                        │  │  - Close distance            │  │     │   │
│  │                        │  ├──────────────────────────────┤  │     │   │
│  │                        │  │  TerminalSpecialist          │  │     │   │
│  │                        │  │  - Final intercept           │  │     │   │
│  │                        │  │  - High precision            │  │     │   │
│  │                        │  └──────────────────────────────┘  │     │   │
│  │                        └─────────────────────────────────────┘     │   │
│  │           │                                                         │   │
│  │           ▼                                                         │   │
│  │  ┌──────────────────────────────────────────────────────────┐     │   │
│  │  │  option_manager.py                                       │     │   │
│  │  │  ┌────────────────────────────────────────────────────┐ │     │   │
│  │  │  │  OptionManager                                     │ │     │   │
│  │  │  │  - Forced transition logic                         │ │     │   │
│  │  │  │  - Hysteresis bands                                │ │     │   │
│  │  │  │  - Minimum dwell enforcement                       │ │     │   │
│  │  │  │  - Transition statistics                           │ │     │   │
│  │  │  │                                                     │ │     │   │
│  │  │  │  get_forced_transition(current_option, env_state)  │ │     │   │
│  │  │  │    Rule 1: Lost lock → SEARCH                      │ │     │   │
│  │  │  │    Rule 2: Lock acquired → TRACK                   │ │     │   │
│  │  │  │    Rule 3: Close range → TERMINAL                  │ │     │   │
│  │  │  │    Rule 4: Miss imminent → TRACK                   │ │     │   │
│  │  │  └────────────────────────────────────────────────────┘ │     │   │
│  │  └──────────────────────────────────────────────────────────┘     │   │
│  │           │                                                         │   │
│  │           ▼                                                         │   │
│  │  ┌──────────────────────────────────────────────────────────┐     │   │
│  │  │  observation_abstraction.py                              │     │   │
│  │  │  ┌────────────────────────────────────────────────────┐ │     │   │
│  │  │  │  abstract_observation(full_obs)                    │ │     │   │
│  │  │  │  - 26D/104D → 7D conversion                        │ │     │   │
│  │  │  │  - Normalized features:                            │ │     │   │
│  │  │  │    [0] distance_to_target                          │ │     │   │
│  │  │  │    [1] closing_rate                                │ │     │   │
│  │  │  │    [2] radar_lock_quality                          │ │     │   │
│  │  │  │    [3] fuel_fraction                               │ │     │   │
│  │  │  │    [4] off_axis_angle                              │ │     │   │
│  │  │  │    [5] time_to_intercept_estimate                  │ │     │   │
│  │  │  │    [6] relative_altitude                           │ │     │   │
│  │  │  └────────────────────────────────────────────────────┘ │     │   │
│  │  │  ┌────────────────────────────────────────────────────┐ │     │   │
│  │  │  │  extract_env_state_for_transitions(full_obs)       │ │     │   │
│  │  │  │  - Extracts: lock_quality, distance, fuel,         │ │     │   │
│  │  │  │              closing_rate                           │ │     │   │
│  │  │  └────────────────────────────────────────────────────┘ │     │   │
│  │  └──────────────────────────────────────────────────────────┘     │   │
│  │           │                                                         │   │
│  │           ▼                                                         │   │
│  │  ┌──────────────────────────────────────────────────────────┐     │   │
│  │  │  option_definitions.py                                   │     │   │
│  │  │  ┌────────────────────────────────────────────────────┐ │     │   │
│  │  │  │  Option (IntEnum)                                  │ │     │   │
│  │  │  │    - SEARCH = 0                                    │ │     │   │
│  │  │  │    - TRACK = 1                                     │ │     │   │
│  │  │  │    - TERMINAL = 2                                  │ │     │   │
│  │  │  └────────────────────────────────────────────────────┘ │     │   │
│  │  │  ┌────────────────────────────────────────────────────┐ │     │   │
│  │  │  │  OPTION_METADATA                                   │ │     │   │
│  │  │  │    - name, description, expected_duration          │ │     │   │
│  │  │  │    - forced_exit_conditions                        │ │     │   │
│  │  │  │    - color (for visualization)                     │ │     │   │
│  │  │  └────────────────────────────────────────────────────┘ │     │   │
│  │  │  ┌────────────────────────────────────────────────────┐ │     │   │
│  │  │  │  FORCED_TRANSITION_THRESHOLDS                      │ │     │   │
│  │  │  │    - radar_lock_quality_min: 0.3                   │ │     │   │
│  │  │  │    - radar_lock_quality_search: 0.7                │ │     │   │
│  │  │  │    - close_range_threshold: 100.0 m                │ │     │   │
│  │  │  │    - terminal_fuel_min: 0.1                        │ │     │   │
│  │  │  └────────────────────────────────────────────────────┘ │     │   │
│  │  └──────────────────────────────────────────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

### Training Flow (Specialist Pre-Training)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. SPECIALIST PRE-TRAINING (train_hrl_pretrain.py)                      │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ make_hrl_env(config, mode="training")                                    │
│   1. Create InterceptEnvironment(config)                                 │
│   2. Apply RewardShapingWrapper(tactical, option=SEARCH)                 │
│   3. Apply VecFrameStack(n_stack=4)                                      │
│   4. Apply VecNormalize()                                                │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ PPO Training Loop                                                         │
│   - Observation: 104D (26D * 4 frames)                                   │
│   - Action: 6D continuous                                                │
│   - Reward: Tactical (option-specific)                                   │
│   - Policy: LSTM-enabled MlpPolicy                                       │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Save Checkpoint                                                           │
│   checkpoints/hrl/specialists/{search|track|terminal}/model.zip          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Training Flow (Selector)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 2. SELECTOR TRAINING (train_hrl_selector.py)                            │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Load Pre-Trained Specialists (FROZEN)                                    │
│   - SearchSpecialist.load('checkpoints/.../search/model')                │
│   - TrackSpecialist.load('checkpoints/.../track/model')                  │
│   - TerminalSpecialist.load('checkpoints/.../terminal/model')            │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ make_hrl_env(config, selector=None, specialists={...}, mode="training")  │
│   1. Create InterceptEnvironment(config)                                 │
│   2. Create HierarchicalManager(selector=None, specialists={...})        │
│   3. Apply HRLActionWrapper(env, manager)                                │
│   4. Apply AbstractObservationWrapper(env)  ← Converts to 7D             │
│   5. Apply RewardShapingWrapper(strategic)                               │
│   6. Apply VecNormalize()                                                │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ PPO Training Loop (Selector)                                             │
│   - Observation: 7D abstract state                                       │
│   - Action: Discrete(3) - option selection                               │
│   - Reward: Strategic (high-level objectives)                            │
│   - Policy: MlpPolicy (no LSTM)                                          │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Save Checkpoint                                                           │
│   checkpoints/hrl/selector/model.zip                                     │
└──────────────────────────────────────────────────────────────────────────┘
```

### Inference Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 3. INFERENCE (inference.py --hrl)                                        │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Load All Policies                                                         │
│   - selector = SelectorPolicy.load('checkpoints/hrl/selector/model')     │
│   - search_spec = SearchSpecialist.load('checkpoints/.../search/model')  │
│   - track_spec = TrackSpecialist.load('checkpoints/.../track/model')     │
│   - terminal_spec = TerminalSpecialist.load('checkpoints/.../terminal')  │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Create HierarchicalManager                                               │
│   manager = HierarchicalManager(                                         │
│       selector=selector,                                                 │
│       specialists={SEARCH: search_spec, TRACK: track_spec, ...},         │
│       decision_interval=100,                                             │
│   )                                                                      │
└──────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ Episode Loop                                                              │
│   obs, info = env.reset()                                                │
│   manager.reset()                                                        │
│                                                                          │
│   while not done:                                                        │
│       # Extract state for forced transitions                             │
│       env_state = extract_env_state_for_transitions(obs)                 │
│                                                                          │
│       # Get action from HRL manager                                      │
│       action, hrl_info = manager.select_action(                          │
│           full_obs=obs,                                                  │
│           env_state=env_state,                                           │
│           deterministic=True                                             │
│       )                                                                  │
│       #   ▲                                                              │
│       #   │                                                              │
│       #   └─ manager internally:                                         │
│       #      1. Checks forced transitions (OptionManager)                │
│       #      2. Calls selector if at decision_interval                   │
│       #      3. Executes active specialist                               │
│       #      4. Returns 6D action + metadata                             │
│                                                                          │
│       # Step environment (unchanged API)                                 │
│       obs, reward, terminated, truncated, info = env.step(action)        │
│       done = terminated or truncated                                     │
│                                                                          │
│       # Log option transitions                                           │
│       if hrl_info['hrl/option_switched']:                                │
│           print(f"Switched to {hrl_info['hrl/option']}")                 │
└──────────────────────────────────────────────────────────────────────────┘
```

## Comparison: Flat PPO vs HRL

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FLAT PPO (EXISTING)                            │
└─────────────────────────────────────────────────────────────────────────┘

    InterceptEnvironment
            │
            ▼
    [26D Observation] × 4 frames = 104D
            │
            ▼
    ┌─────────────────┐
    │   PPO Policy    │  Single monolithic policy
    │   (LSTM + MLP)  │  Must learn all phases simultaneously
    └─────────────────┘
            │
            ▼
    [6D Continuous Action]
            │
            ▼
    [Single Reward Signal]

    Pros:
    ✓ Simple architecture
    ✓ Proven to work (75%+ intercept rate)

    Cons:
    ✗ No explicit phase structure
    ✗ Policy must learn when to search/track/intercept
    ✗ Difficult to debug failures
    ✗ No transfer learning across phases

┌─────────────────────────────────────────────────────────────────────────┐
│                     HRL (NEW IMPLEMENTATION)                            │
└─────────────────────────────────────────────────────────────────────────┘

    InterceptEnvironment
            │
            ▼
    [26D Observation] × 4 frames = 104D
            │
            ├────────────────────────────────────────────┐
            │                                            │
            ▼                                            ▼
    [7D Abstract State]                        [104D Full Observation]
            │                                            │
            ▼                                            ▼
    ┌─────────────────┐                     ┌──────────────────────┐
    │ Selector Policy │  High-level         │  Specialist Policies │
    │   (Discrete)    │  decision-making    │    (Continuous)      │
    │                 │                     │                      │
    │ Decides every   │                     │  ┌────────────────┐ │
    │ 100 steps:      │                     │  │ SearchSpecialist│ │
    │ SEARCH/TRACK/   │                     │  │ TrackSpecialist │ │
    │ TERMINAL        │                     │  │TerminalSpecialist│ │
    └─────────────────┘                     │  └────────────────┘ │
            │                                └──────────────────────┘
            └────────────────┬────────────────┘
                            │
                            ▼
                  [6D Continuous Action]
                            │
                            ├─────────────────────────────────┐
                            │                                 │
                            ▼                                 ▼
                  [Strategic Reward]              [Tactical Reward]
                  (Intercept success,             (Option-specific:
                   efficiency)                     lock acquisition,
                                                   tracking, precision)

    Pros:
    ✓ Explicit phase structure (interpretable)
    ✓ Specialists can be trained independently
    ✓ Forced transitions ensure physical correctness
    ✓ Transfer learning across scenarios
    ✓ Easier debugging (know which option failed)
    ✓ Reward shaping per phase

    Cons:
    ✗ More complex architecture
    ✗ Longer training time (pre-training + selector)
    ✗ Requires careful coordination between levels

```

## Module Size Estimates

```
Phase 2 (Complete ✅):
  option_definitions.py       ~130 lines
  observation_abstraction.py  ~180 lines
  manager.py                  ~230 lines
  selector_policy.py          ~180 lines
  specialist_policies.py      ~200 lines
  option_manager.py           ~240 lines
  Total: ~1160 lines

Phase 3 (To Implement):
  reward_decomposition.py     ~350 lines (est.)
    - compute_strategic_reward()     ~60 lines
    - compute_tactical_reward()      ~30 lines
    - compute_search_reward()        ~60 lines
    - compute_track_reward()         ~80 lines
    - compute_terminal_reward()      ~70 lines
    - Helper functions               ~50 lines

Phase 4 (To Implement):
  wrappers.py (enhanced)      ~250 lines (est.)
    - HRLActionWrapper           ~120 lines
    - AbstractObservationWrapper ~40 lines
    - RewardShapingWrapper       ~90 lines

Phase 5 (To Implement):
  hierarchical_env.py         ~150 lines (est.)
  train_hrl_pretrain.py       ~300 lines (est.)
  train_hrl_selector.py       ~250 lines (est.)
  train_hrl_full.py           ~200 lines (est.)

Total Estimated Lines of Code: ~2660 lines
  (Excluding tests, docs, configs)

Test Coverage Target: >80%
  Estimated test code: ~2000 lines
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-07
**Purpose:** Visual architecture reference for HRL implementation
