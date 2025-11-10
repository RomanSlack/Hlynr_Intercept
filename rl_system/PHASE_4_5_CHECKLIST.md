# Phase 4 & Phase 5 Implementation Checklist

**Document Version:** 1.0
**Generated From:** HRL_REFACTORING_DESIGN.md
**Scope:** Training infrastructure (Phase 4) + Testing, Migration, Documentation (Phase 5)
**Prerequisites:** Phases 1-3 completed (core modules, reward decomposition, environment wrappers)

---

## PHASE 4: TRAINING INFRASTRUCTURE (Days 7-9)

### Overview
Establish scalable training scripts, configuration management, and curriculum learning capabilities. Maintain full backward compatibility with existing flat PPO workflows.

---

## 4.1 FLAT PPO REFACTORING (Day 7)

### Goal
Extract training logic from `train.py` into reusable module without breaking existing workflows.

#### 4.1.1 Create Flat PPO Training Module
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/scripts/train_flat_ppo.py`

**Specific Functionality Required:**
- [ ] Extract all training logic from current `train.py`
- [ ] Create `main()` function that accepts config path from command line
- [ ] Implement logging integration with `logger.py`
- [ ] Support all existing command-line arguments: `--config`, `--seed`, `--verbose`
- [ ] Preserve exact training behavior (PPO hyperparameters, frame stacking, LSTM settings)
- [ ] Return exit code 0 on success, non-zero on failure

**Code Structure Expected:**
```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Load config, create env, train PPO
    # Save checkpoints, log metrics
    return exit_code

if __name__ == "__main__":
    exit(main())
```

**Dependencies:**
- Must NOT break: `environment.py`, `core.py`, `physics_*.py`
- Must use: `logger.py`, `config.yaml` structure
- Optional: `tensorboard` integration

**Acceptance Criteria:**
- [ ] `pytest tests/test_backward_compatibility.py` passes
- [ ] Training produces identical results to existing `train.py` for first 100k steps
- [ ] All checkpoint formats remain unchanged
- [ ] TensorBoard event files are compatible with existing logs

---

#### 4.1.2 Modify train.py as Backward-Compatible Wrapper
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/train.py`

**Specific Functionality Required:**
- [ ] Keep existing public interface (CLI args, config handling)
- [ ] Import and delegate to `scripts.train_flat_ppo.main()`
- [ ] Detect HRL mode flag (e.g., `--hrl`) for future extensibility
- [ ] Emit deprecation warning if `--hrl` flag is used
- [ ] Support all legacy config paths (both `scenarios/` and `configs/scenarios/`)
- [ ] Maintain identical exit codes and output format

**Code Structure Expected:**
```python
def main():
    # Detect HRL mode
    if "--hrl" in sys.argv:
        warnings.warn("Use scripts/train_hrl_full.py instead", DeprecationWarning)
        # Call HRL training (Phase 4 later)
    else:
        # Call flat PPO training
        from scripts.train_flat_ppo import main as flat_main
        return flat_main()

if __name__ == "__main__":
    exit(main())
```

**Dependencies:**
- Must support all existing argument parsers
- Must be compatible with existing shell scripts/documentation
- Config resolution fallback needed

**Acceptance Criteria:**
- [ ] `python train.py --config config.yaml` produces identical output
- [ ] `python train.py --config scenarios/easy.yaml` still works (backward compat)
- [ ] Help text shows all options: `python train.py --help`

---

### Validation Checkpoint
Run before proceeding to Day 8:
```bash
cd /Users/quinnhasse/Hlynr_Intercept/rl_system
python train.py --config config.yaml --seed 42 &
sleep 60
pkill -f "train.py"
# Verify logs/run_*/checkpoints created
```

**Expected Result:** Training starts and produces logs in expected format.

---

## 4.2 HRL PRE-TRAINING SCRIPT (Day 8)

### Goal
Create script to independently pre-train each specialist (search, track, terminal) for curriculum learning.

#### 4.2.1 Create HRL Pre-training Script
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/scripts/train_hrl_pretrain.py`

**Specific Functionality Required:**
- [ ] Accept config path: `--config configs/hrl/hrl_curriculum.yaml`
- [ ] Accept specialist selection: `--specialist search|track|terminal|all`
- [ ] Create base InterceptEnvironment with standard physics
- [ ] Loop through curriculum stages (easy → medium → hard scenarios)
- [ ] Train specialist with scenario-specific reward shaping
- [ ] Save checkpoints: `checkpoints/hrl/specialists/{name}/model_*.zip`
- [ ] Log metrics to TensorBoard with namespace: `specialists/{name}/`
- [ ] Support resuming from checkpoint: `--resume-from path/to/model.zip`
- [ ] Implement early stopping: stop if no improvement in 50k steps

**Training Configuration Flow:**
1. Load `configs/hrl/hrl_curriculum.yaml` (defines stages, episode counts)
2. For each specialist:
   - Stage 1 (easy): 100 episodes on `configs/scenarios/easy.yaml`
   - Stage 2 (medium): 100 episodes on `configs/scenarios/medium.yaml`
3. Save final model to `checkpoints/hrl/specialists/{name}/best/best_model.zip`

**Code Structure Expected:**
```python
def train_specialist(config, specialist_name):
    """Train single specialist through curriculum."""
    specialist = create_specialist_policy(specialist_name, obs_dim=104)

    for stage in config['hrl']['specialists'][specialist_name]['curriculum']:
        env = InterceptEnvironment(stage_config)
        ppo = PPO(policy='MlpLstmPolicy', env=env, ...)
        ppo.learn(total_timesteps=stage['episodes'] * max_steps)
        specialist.model = ppo

    specialist.save(f"checkpoints/hrl/specialists/{specialist_name}/best/")

if __name__ == "__main__":
    args = parse_args()
    if args.specialist == 'all':
        for spec in ['search', 'track', 'terminal']:
            train_specialist(config, spec)
    else:
        train_specialist(config, args.specialist)
```

**Dependencies:**
- `hrl/specialist_policies.py`: Must be implemented
- `hrl/option_definitions.py`: Option enum, specialist mappings
- `environment.py`: Base environment with curriculum stages
- `configs/hrl/hrl_curriculum.yaml`: Stage definitions

**Acceptance Criteria:**
- [ ] Train search specialist 100 episodes: completes in <15 minutes
- [ ] Train track specialist 100 episodes: completes in <15 minutes
- [ ] Train terminal specialist 100 episodes: completes in <15 minutes
- [ ] Checkpoints saved to correct paths with correct filenames
- [ ] TensorBoard logs show learning curves for each specialist
- [ ] Early stopping works: stops if loss plateaus

---

#### 4.2.2 Create HRL Configuration Files
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/configs/hrl/search_specialist.yaml`
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/configs/hrl/track_specialist.yaml`
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/configs/hrl/terminal_specialist.yaml`

**Specific Functionality Required:**

For `search_specialist.yaml`:
- [ ] Base environment settings (dt, max_steps, physics)
- [ ] Specialist-specific reward weights:
  - lock_acquisition_bonus: 50.0
  - lock_improvement: 10.0
  - angular_diversity_bonus: 0.5
  - fuel_waste_penalty: -5.0
- [ ] Network architecture: [512, 512, 256]
- [ ] LSTM settings: hidden_dim: 256, enabled: true
- [ ] Learning rate: 0.0003
- [ ] Entropy coefficient: 0.02 (higher for exploration)
- [ ] Curriculum stages: easy (100 eps), medium (100 eps)

For `track_specialist.yaml`:
- [ ] Base environment settings
- [ ] Track-specific reward weights:
  - lock_maintenance_bonus: 2.0
  - lock_loss_penalty: -10.0
  - distance_reduction: 1.0
  - closing_rate_bonus: 0.5
  - jerky_movement_penalty: -0.2
- [ ] Initial lock quality: 0.7 (starts near existing target)
- [ ] Network: [512, 512, 256], LSTM: 256, LR: 0.0003

For `terminal_specialist.yaml`:
- [ ] Base environment settings
- [ ] Terminal-specific reward weights:
  - proximity_bonus_scale: 10.0
  - distance_increase_penalty: -5.0
  - closing_rate_bonus: 1.0
  - max_thrust_bonus: 0.5
- [ ] Initial distance max: 200m (starts close to target)
- [ ] Network: [512, 512, 256], LSTM: 256, LR: 0.0003

**Format:**
Each config should follow YAML structure with hierarchical keys matching `hrl_base.yaml`.

**Acceptance Criteria:**
- [ ] All three configs are valid YAML (can be loaded with `yaml.safe_load()`)
- [ ] Each config has required keys: environment, hrl, physics_enhancements
- [ ] Reward weights are reasonable (within 1-100 range)
- [ ] Specialist can be instantiated from config

---

### Validation Checkpoint
Run before proceeding to Day 9:
```bash
cd /Users/quinnhasse/Hlynr_Intercept/rl_system
# Pre-train search specialist (100 episodes ≈ 10k steps)
python scripts/train_hrl_pretrain.py --config configs/hrl/hrl_curriculum.yaml --specialist search
# Expected: Takes 2-3 minutes, saves to checkpoints/hrl/specialists/search/
```

---

## 4.3 HRL SELECTOR TRAINING (Day 9, Part 1)

### Goal
Create script to train high-level option selector with frozen specialist checkpoints.

#### 4.3.1 Create HRL Selector Training Script
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/scripts/train_hrl_selector.py`

**Specific Functionality Required:**
- [ ] Accept config path: `--config configs/hrl/hrl_curriculum.yaml`
- [ ] Load specialist checkpoints from: `checkpoints/hrl/specialists/{search,track,terminal}/best/`
- [ ] Create specialist policy instances and freeze them (disable backprop)
- [ ] Create selector policy with observation dimension: 7D (abstract state)
- [ ] Instantiate HierarchicalManager with frozen specialists
- [ ] Wrap base environment with HierarchicalEnv and AbstractObservationWrapper
- [ ] Train selector PPO on abstract observations (7D vector)
- [ ] Selector action space: Discrete(3) for {SEARCH, TRACK, TERMINAL}
- [ ] Save selector checkpoints: `checkpoints/hrl/selector/model_*.zip`
- [ ] Log metrics namespace: `selector/`
- [ ] Support training duration: `--total-steps 3000000` (default)

**Training Flow:**
1. Load frozen specialists from checkpoints
2. Create HierarchicalManager with decision_interval=100
3. Wrap environment with HRL wrapper
4. Train PPO selector policy (discrete action space, 7D observations)
5. Periodically evaluate on test episodes (no option switching)

**Code Structure Expected:**
```python
def train_selector(config):
    # Load frozen specialists
    search_specialist = SearchSpecialist(
        model_path="checkpoints/hrl/specialists/search/best/best_model"
    )
    # ... track, terminal similarly

    # Create manager
    manager = HierarchicalManager(
        selector_policy=SelectorPolicy(obs_dim=7),
        specialists={
            Option.SEARCH: search_specialist,
            Option.TRACK: track_specialist,
            Option.TERMINAL: terminal_specialist,
        },
        decision_interval=100,
    )

    # Wrap environment
    env = InterceptEnvironment(config)
    env = HierarchicalEnvWrapper(env, manager)
    env = AbstractObservationWrapper(env)

    # Train selector PPO
    ppo = PPO(policy='MlpPolicy', env=env, ...)
    ppo.learn(total_timesteps=3e6)

    manager.selector.save("checkpoints/hrl/selector/best/")

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train_selector(config)
```

**Dependencies:**
- `hrl/manager.py`: HierarchicalManager class
- `hrl/specialist_policies.py`: Load frozen specialists
- `hrl/selector_policy.py`: SelectorPolicy with discrete action space
- `hrl/wrappers.py`: HierarchicalEnvWrapper, AbstractObservationWrapper
- `hrl/observation_abstraction.py`: Abstract state computation
- Checkpoints: `checkpoints/hrl/specialists/{search,track,terminal}/best/`

**Acceptance Criteria:**
- [ ] Script loads 3 frozen specialist checkpoints without error
- [ ] Specialist parameters are not updated during selector training
- [ ] Environment wrapping produces 7D observations to selector
- [ ] Selector action space is Discrete(3)
- [ ] Training runs for specified total_steps
- [ ] Checkpoints saved with correct naming convention

---

#### 4.3.2 Create HRL Full Training Script
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/scripts/train_hrl_full.py`

**Specific Functionality Required:**
- [ ] Orchestrate complete HRL training pipeline
- [ ] Accept config: `--config configs/hrl/hrl_curriculum.yaml`
- [ ] Stage 1: Run `train_hrl_pretrain.py` (all 3 specialists)
- [ ] Stage 2: Run `train_hrl_selector.py` with frozen specialists
- [ ] Stage 3 (optional): Run joint fine-tuning with reduced learning rates
- [ ] Save full pipeline artifacts:
  - Specialist models: `checkpoints/hrl/specialists/{name}/`
  - Selector model: `checkpoints/hrl/selector/`
  - Pipeline metadata: `checkpoints/hrl/pipeline.json`
- [ ] Implement progress tracking with intermediate validation
- [ ] Support resume from specific stage: `--resume-stage selector`
- [ ] Log overall metrics to TensorBoard: `hrl_pipeline/`

**Training Sequence:**
1. **Pre-training Phase:** Train specialists independently
   - Command: `train_hrl_pretrain.py --specialist all`
   - Duration: ~45 minutes (15 min each specialist)
   - Output: `checkpoints/hrl/specialists/{search,track,terminal}/best/`

2. **Selector Training Phase:** Train selector with frozen specialists
   - Command: `train_hrl_selector.py`
   - Duration: ~20 minutes (3M steps @ ~150k steps/min)
   - Output: `checkpoints/hrl/selector/best/`

3. **Joint Fine-tuning Phase (Optional):** Fine-tune all together
   - Duration: ~15 minutes (1.5M steps @ reduced learning rate)
   - Output: `checkpoints/hrl/selector_finetuned/`, `checkpoints/hrl/specialists_finetuned/`

**Code Structure Expected:**
```python
def main():
    config = load_config(args.config)

    # Stage 1: Pre-train specialists
    if should_run_stage('pretrain', args.resume_stage):
        log_stage_start("Pre-training specialists")
        train_hrl_pretrain(config)
        log_stage_complete()

    # Stage 2: Train selector
    if should_run_stage('selector', args.resume_stage):
        log_stage_start("Training selector")
        train_hrl_selector(config)
        log_stage_complete()

    # Stage 3: Joint fine-tuning (optional)
    if config.get('curriculum', {}).get('joint_finetuning', {}).get('enabled'):
        if should_run_stage('finetune', args.resume_stage):
            log_stage_start("Joint fine-tuning")
            train_hrl_joint(config)
            log_stage_complete()

    save_pipeline_metadata(config)

if __name__ == "__main__":
    exit(main())
```

**Dependencies:**
- `scripts/train_hrl_pretrain.py`: Must be callable/importable
- `scripts/train_hrl_selector.py`: Must be callable/importable
- `hrl/callbacks.py`: Optional progress callbacks (Phase 3 artifact)
- Config: `configs/hrl/hrl_curriculum.yaml`

**Acceptance Criteria:**
- [ ] All three stages complete successfully in sequence
- [ ] Intermediate checkpoints saved at each stage
- [ ] Resume functionality works: `--resume-stage selector`
- [ ] TensorBoard logs show all three stages
- [ ] Pipeline metadata saved with timestamps and hyperparameters
- [ ] Total pipeline time: <2 hours

---

### Validation Checkpoint
Run before Phase 5:
```bash
cd /Users/quinnhasse/Hlynr_Intercept/rl_system
# Full HRL pipeline (expect ~1.5-2 hours)
python scripts/train_hrl_full.py --config configs/hrl/hrl_curriculum.yaml --log-dir logs/hrl_test
# Expected: All checkpoints created, TensorBoard logs generated
```

---

## 4.4 TRAINING INFRASTRUCTURE SUMMARY

### Scripts Created (4.1-4.3)
| Script | Location | Purpose | Status |
|--------|----------|---------|--------|
| train_flat_ppo.py | scripts/ | Refactored flat PPO training | [ ] |
| train_hrl_pretrain.py | scripts/ | Specialist pre-training | [ ] |
| train_hrl_selector.py | scripts/ | Selector training with frozen specialists | [ ] |
| train_hrl_full.py | scripts/ | End-to-end orchestrator | [ ] |

### Configuration Files Created (4.2)
| Config | Location | Purpose | Status |
|--------|----------|---------|--------|
| search_specialist.yaml | configs/hrl/ | Search specialist settings | [ ] |
| track_specialist.yaml | configs/hrl/ | Track specialist settings | [ ] |
| terminal_specialist.yaml | configs/hrl/ | Terminal specialist settings | [ ] |

### Backward Compatibility Maintained
- [ ] `python train.py --config config.yaml` unchanged
- [ ] Existing checkpoints still load
- [ ] TensorBoard format compatible
- [ ] All scenarios paths work (forward and backward compat)

---

---

# PHASE 5: TESTING, MIGRATION, DOCUMENTATION (Days 10-12)

## Overview
Validate complete HRL pipeline, migrate existing infrastructure, and document for users.

---

## 5.1 INTEGRATION TESTING (Day 10)

### Goal
Comprehensive testing of HRL system end-to-end and backward compatibility verification.

#### 5.1.1 Create End-to-End Integration Tests
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/tests/test_end_to_end.py`

**Specific Functionality Required:**
- [ ] Test full HRL episode from reset to termination:
  - [ ] Initialize HierarchicalManager
  - [ ] Create environment with HRL wrapper
  - [ ] Run 500 steps (enough for option switches)
  - [ ] Verify option switching occurs
  - [ ] Verify all options are used at least once
  - [ ] Check for memory leaks (peak memory stays constant)

- [ ] Test specialist-only inference (no selector):
  - [ ] Load frozen specialist models
  - [ ] Run 500 steps deterministic inference
  - [ ] Verify consistent action distribution
  - [ ] Check output shapes and ranges

- [ ] Test selector-only decision making:
  - [ ] Load selector model
  - [ ] Feed abstract observations
  - [ ] Verify option selection is valid (0-2)
  - [ ] Check decision frequency (every decision_interval steps)

- [ ] Test forced transitions:
  - [ ] Manually set env_state with low lock quality
  - [ ] Verify option switches from TRACK to SEARCH
  - [ ] Verify close range triggers TERMINAL
  - [ ] Verify transitions only occur when enabled

**Test Structure Expected:**
```python
@pytest.mark.integration
def test_hrl_full_episode():
    """Full episode with HRL system."""
    config = load_test_config()
    env = create_hrl_env(config)

    obs, info = env.reset()
    assert obs.shape == (26,)  # Full observation

    options_used = set()
    for step in range(500):
        obs, reward, terminated, truncated, info = env.step(None)
        options_used.add(info['option'])

        assert 'option' in info
        assert 'steps_in_option' in info

    assert len(options_used) > 1  # Multiple options used
    env.close()

@pytest.mark.integration
def test_forced_transition():
    """Forced transitions by environment state."""
    manager = create_manager_with_frozen_specialists()

    # High lock quality → can stay in TRACK
    env_state = {
        'lock_quality': 0.8,
        'distance': 1000.0,
        'fuel': 0.7,
        'closing_rate': 50.0,
    }

    manager.state.current_option = Option.TRACK
    obs = np.random.randn(26)
    action, info = manager.select_action(obs, env_state)

    assert info['forced_transition'] == False
    assert manager.state.current_option == Option.TRACK

    # Low lock quality → forced to SEARCH
    env_state['lock_quality'] = 0.2
    action, info = manager.select_action(obs, env_state)

    assert info['forced_transition'] == True
    assert manager.state.current_option == Option.SEARCH
```

**Dependencies:**
- All Phase 2 modules: manager, specialists, selector
- All Phase 3 modules: wrappers, hierarchical_env
- Test fixtures in `tests/conftest.py`

**Acceptance Criteria:**
- [ ] All tests pass: `pytest tests/test_end_to_end.py -v`
- [ ] >95% test execution time <30 seconds
- [ ] No memory leaks detected (use tracemalloc)
- [ ] All edge cases covered (empty obs, invalid actions, forced transitions)

---

#### 5.1.2 Create Backward Compatibility Tests
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/tests/test_backward_compatibility.py`

**Specific Functionality Required:**
- [ ] Test flat PPO training still works:
  - [ ] Load config.yaml
  - [ ] Create environment without HRL
  - [ ] Train 50k steps PPO
  - [ ] Verify reward increases over time
  - [ ] Save and load checkpoint
  - [ ] Checkpoint loads with identical weights

- [ ] Test config path resolution:
  - [ ] Load `scenarios/easy.yaml` (old path)
  - [ ] Load `configs/scenarios/easy.yaml` (new path)
  - [ ] Both load identical config content
  - [ ] Deprecation warning issued for old path

- [ ] Test existing checkpoint loading:
  - [ ] List existing checkpoints in repo
  - [ ] Load each checkpoint
  - [ ] Run inference on random observation
  - [ ] Verify action shape and range

- [ ] Test environment interface unchanged:
  - [ ] InterceptEnvironment methods unchanged
  - [ ] reset() returns (obs, info) tuple
  - [ ] step() returns (obs, reward, terminated, truncated, info)
  - [ ] Observation shape is 26D (or 104D with frame stack)
  - [ ] Action space is Box(6,) continuous

**Test Structure Expected:**
```python
def test_flat_ppo_training():
    """Flat PPO training still works."""
    config = load_config('config.yaml')
    env = InterceptEnvironment(config)

    ppo = PPO('MlpPolicy', env=env)
    ppo.learn(total_timesteps=50000)

    # Check learning occurred
    assert hasattr(ppo, 'policy')
    assert ppo.num_timesteps > 0

def test_config_backward_compat():
    """Old config paths still work."""
    # Old path should work with deprecation warning
    with pytest.warns(DeprecationWarning):
        config1 = load_config('scenarios/easy.yaml')

    # New path should work without warning
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        config2 = load_config('configs/scenarios/easy.yaml')

    # Content should be identical
    assert config1 == config2

def test_existing_checkpoint_loading():
    """Existing checkpoints still load."""
    checkpoint_paths = glob('checkpoints/flat_ppo/model_*.zip')

    for path in checkpoint_paths[:3]:  # Test first 3
        model = PPO.load(path)
        obs = np.random.randn(26)
        action, _ = model.predict(obs)
        assert action.shape == (6,)
```

**Dependencies:**
- `environment.py`: Base environment
- `core.py`: Observation/physics
- Existing checkpoints in `/checkpoints/`
- All config files in `configs/` and `scenarios/`

**Acceptance Criteria:**
- [ ] All backward compatibility tests pass
- [ ] No changes to InterceptEnvironment interface
- [ ] Existing checkpoints load and run inference
- [ ] Config paths resolve correctly (old and new)
- [ ] Deprecation warnings issued appropriately

---

#### 5.1.3 Create Evaluation Comparison Script
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/scripts/evaluate_hrl.py`

**Specific Functionality Required:**
- [ ] Accept arguments:
  - `--model-path` - Path to HRL selector checkpoint
  - `--episodes` - Number of eval episodes (default 100)
  - `--seed` - Random seed
  - `--scenario` - Difficulty (easy/medium/hard)

- [ ] Run evaluation:
  - [ ] Load HRL system (selector + specialists)
  - [ ] Run deterministic inference (no exploration)
  - [ ] Collect metrics per episode:
    - Intercept success (binary)
    - Final miss distance
    - Fuel efficiency (remaining/initial)
    - Option usage counts
    - Average steps in each option
    - Forced transitions per episode

- [ ] Save evaluation results:
  - [ ] JSON file with per-episode data
  - [ ] Summary statistics (mean, std)
  - [ ] Option usage distribution (pie chart)
  - [ ] Success rate by scenario

**Code Structure Expected:**
```python
def evaluate_hrl(model_path, num_episodes, scenario):
    """Evaluate HRL policy."""
    manager = load_hrl_manager(model_path)
    env = InterceptEnvironment(load_scenario_config(scenario))
    env = HierarchicalEnvWrapper(env, manager)

    results = []
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_data = {
            'success': False,
            'miss_distance': float('inf'),
            'fuel_remaining': 0.0,
            'option_usage': {},
            'forced_transitions': 0,
        }

        while not done:
            action, hrl_info = manager.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track metrics
            if hrl_info['option'] not in episode_data['option_usage']:
                episode_data['option_usage'][hrl_info['option']] = 0
            episode_data['option_usage'][hrl_info['option']] += 1

            if hrl_info.get('forced_transition'):
                episode_data['forced_transitions'] += 1

        results.append(episode_data)

    return aggregate_results(results)

if __name__ == "__main__":
    args = parse_args()
    results = evaluate_hrl(args.model_path, args.episodes, args.scenario)
    save_results(results, args.output_dir)
    print_summary(results)
```

**Dependencies:**
- `hrl/manager.py`: Load HRL manager
- `hrl/wrappers.py`: HierarchicalEnvWrapper
- `environment.py`: Base environment

**Acceptance Criteria:**
- [ ] Evaluation runs 100 episodes in <15 minutes
- [ ] Metrics collected: success rate, miss distance, fuel efficiency
- [ ] Results saved to JSON with per-episode and aggregate data
- [ ] Summary printed to stdout with key statistics

---

#### 5.1.4 Create Comparison Script (Flat vs HRL)
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/scripts/compare_policies.py`

**Specific Functionality Required:**
- [ ] Accept arguments:
  - `--flat` - Path to flat PPO checkpoint
  - `--hrl` - Path to HRL selector checkpoint
  - `--episodes` - Comparison episodes (default 100)
  - `--scenario` - Test difficulty

- [ ] Run side-by-side evaluation:
  - [ ] Flat PPO policy on environment
  - [ ] HRL policy on same environment
  - [ ] Use same random seeds for fair comparison

- [ ] Collect comparative metrics:
  - [ ] Success rate (flat vs HRL)
  - [ ] Average miss distance
  - [ ] Fuel efficiency
  - [ ] Average reward
  - [ ] Training efficiency (steps to convergence)

- [ ] Generate comparison report:
  - [ ] Statistical significance test (t-test)
  - [ ] Performance deltas (%)
  - [ ] Visualization: success rate, miss distance plots
  - [ ] Recommendation: which policy is better

**Acceptance Criteria:**
- [ ] Comparison completes in <20 minutes for 100 episodes
- [ ] Statistical tests applied correctly
- [ ] Report shows clear performance differences
- [ ] Visualization(s) generated and saved

---

### Validation Checkpoint (Day 10)
```bash
cd /Users/quinnhasse/Hlynr_Intercept/rl_system
# Run all integration tests
pytest tests/test_end_to_end.py -v --cov=hrl
# Expected: All pass, >80% coverage

# Run backward compatibility tests
pytest tests/test_backward_compatibility.py -v
# Expected: All pass, existing code unchanged

# Evaluate HRL
python scripts/evaluate_hrl.py --model-path checkpoints/hrl/selector/best/ --episodes 50
# Expected: Results JSON, summary stats printed

# Compare flat vs HRL
python scripts/compare_policies.py --flat checkpoints/flat_ppo/best/ --hrl checkpoints/hrl/selector/best/ --episodes 50
# Expected: Comparison report generated
```

---

## 5.2 CHECKPOINT MIGRATION (Day 11)

### Goal
Organize and migrate existing checkpoints to new directory structure.

#### 5.2.1 Create Checkpoint Migration Script
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/scripts/migrate_checkpoints.py`

**Specific Functionality Required:**
- [ ] Accept arguments:
  - `--source` - Source checkpoint directory (default: current)
  - `--dest` - Destination (default: checkpoints/flat_ppo/)
  - `--backup` - Create backup before migration

- [ ] Scan for existing checkpoints:
  - [ ] Find all `model_*.zip` files
  - [ ] Find all `vec_normalize.pkl` files
  - [ ] Find best model indicators

- [ ] Organize into flat_ppo structure:
  - [ ] `checkpoints/flat_ppo/model_000_steps.zip`
  - [ ] `checkpoints/flat_ppo/best/best_model.zip`
  - [ ] `checkpoints/flat_ppo/best/vec_normalize.pkl`

- [ ] Update references:
  - [ ] Scan `README.md` for old paths, note changes needed
  - [ ] Create migration summary report

- [ ] Verify migration:
  - [ ] Load each migrated checkpoint
  - [ ] Run inference test (10 steps)
  - [ ] Confirm weights unchanged

**Code Structure Expected:**
```python
def migrate_checkpoints(source_dir, dest_dir, backup=True):
    """Migrate checkpoints to new directory structure."""

    # Create backup
    if backup:
        backup_dir = f"{source_dir}_backup_{datetime.now().isoformat()}"
        shutil.copytree(source_dir, backup_dir)
        print(f"Backup created: {backup_dir}")

    # Find all checkpoints
    checkpoints = glob(f"{source_dir}/**/model_*.zip", recursive=True)
    vec_norms = glob(f"{source_dir}/**/vec_normalize.pkl", recursive=True)

    # Create destination structure
    os.makedirs(f"{dest_dir}/best/", exist_ok=True)

    # Copy and organize
    for checkpoint in checkpoints:
        filename = os.path.basename(checkpoint)
        shutil.copy(checkpoint, os.path.join(dest_dir, filename))

    # Identify and copy best model
    best_checkpoint = find_best_checkpoint(checkpoints)
    if best_checkpoint:
        shutil.copy(
            best_checkpoint,
            os.path.join(dest_dir, "best/best_model.zip")
        )

    # Verify migration
    for checkpoint in glob(f"{dest_dir}/model_*.zip"):
        try:
            model = PPO.load(checkpoint)
            print(f"✓ {checkpoint} loads successfully")
        except Exception as e:
            print(f"✗ {checkpoint} failed: {e}")

    return dest_dir

if __name__ == "__main__":
    args = parse_args()
    migrate_checkpoints(args.source, args.dest, args.backup)
```

**Dependencies:**
- `stable_baselines3`: Load checkpoints for verification
- Standard library: shutil, glob, os

**Acceptance Criteria:**
- [ ] All existing checkpoints migrated
- [ ] Directory structure matches specification
- [ ] All migrated checkpoints load successfully
- [ ] Backup created if requested
- [ ] Migration report generated

---

#### 5.2.2 Update Directory Structure Manifest
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/DIRECTORY_STRUCTURE.md`

**Specific Functionality Required:**
- [ ] Document final directory layout
- [ ] Explain purpose of each directory
- [ ] List expected files in each location
- [ ] Include symlink explanations (scenarios → configs/scenarios)
- [ ] Document checkpoint organization (flat_ppo vs hrl)
- [ ] Show file naming conventions

**Format Expected:**
```markdown
# Hlynr Intercept Directory Structure

## Root: rl_system/

### Core Files (No Changes)
- core.py - 26D radar observations, Kalman filtering
- environment.py - Gymnasium environment
- train.py - Training entry point (backward compatible)
- inference.py - Inference/evaluation server
- physics_models.py - 6DOF dynamics
- physics_randomizer.py - Domain randomization
- logger.py - Unified logging
- ... (other utilities)

### New Directories

#### hrl/
All HRL-specific implementation
- __init__.py - Public API
- manager.py - HierarchicalManager
- selector_policy.py - High-level option selection
- specialist_policies.py - Low-level specialists
- option_manager.py - Forced transitions
- observation_abstraction.py - 26D → 7D abstraction
- reward_decomposition.py - Strategic + tactical rewards
- wrappers.py - Gymnasium wrappers
- hierarchical_env.py - Main HRL environment
- callbacks.py - Training callbacks
- metrics.py - Option tracking
- ... (utilities)

#### scripts/
Training and evaluation scripts
- train_flat_ppo.py - Refactored flat PPO
- train_hrl_pretrain.py - Pre-train specialists
- train_hrl_selector.py - Train selector
- train_hrl_full.py - End-to-end orchestrator
- evaluate_hrl.py - HRL evaluation
- compare_policies.py - Flat vs HRL comparison
- migrate_checkpoints.py - Checkpoint migration

#### configs/
Configuration files
- config.yaml - Main config (backward compat)
- scenarios/ → symlink to configs/scenarios/
  - easy.yaml
  - medium.yaml
  - hard.yaml
- hrl/
  - hrl_base.yaml
  - search_specialist.yaml
  - track_specialist.yaml
  - terminal_specialist.yaml
  - hrl_curriculum.yaml
  - hrl_full_training.yaml

#### checkpoints/
Model checkpoints
- flat_ppo/
  - model_000_steps.zip
  - model_050_steps.zip
  - best/
    - best_model.zip
    - vec_normalize.pkl
- hrl/
  - selector/
    - model_*.zip
    - best/
      - best_model.zip
  - specialists/
    - search/
      - model_*.zip
      - best/
        - best_model.zip
    - track/
    - terminal/

#### tests/
Unit and integration tests
- test_observation_abstraction.py
- test_hrl_manager.py
- test_selector_policy.py
- test_specialist_policies.py
- test_end_to_end.py
- test_backward_compatibility.py
- conftest.py - Pytest fixtures

#### docs/
Documentation
- hrl/
  - architecture.md
  - training_guide.md
  - api_reference.md
  - migration_guide.md

### Symlinks (Backward Compatibility)
- scenarios/ → configs/scenarios/
```

**Acceptance Criteria:**
- [ ] Document is complete and accurate
- [ ] All directories listed with purposes
- [ ] File naming conventions documented
- [ ] Symlink relationships explained
- [ ] Checkpoint organization clear

---

### Validation Checkpoint (Day 11)
```bash
cd /Users/quinnhasse/Hlynr_Intercept/rl_system
# Run migration
python scripts/migrate_checkpoints.py --source checkpoints/ --dest checkpoints/flat_ppo/ --backup
# Expected: All checkpoints copied, verified, backup created

# Verify structure
tree -L 2 checkpoints/
# Expected: flat_ppo/ with models, hrl/ with selector and specialists subdirs
```

---

## 5.3 DOCUMENTATION (Day 12)

### Goal
Create comprehensive documentation for HRL system, architecture, and usage.

#### 5.3.1 Create Architecture Documentation
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/docs/hrl/architecture.md`

**Specific Functionality Required:**
- [ ] Overview section:
  - [ ] What is HRL in this context
  - [ ] Why hierarchical approach (missile defense motivation)
  - [ ] Key concepts: options, specialists, selector

- [ ] System architecture section:
  - [ ] Three-level hierarchy: selector → specialists → low-level actions
  - [ ] Data flow diagrams (text/ASCII or referenced)
  - [ ] 26D → 7D abstraction explanation
  - [ ] Reward decomposition: strategic vs tactical

- [ ] Module descriptions (one section per core module):
  - [ ] HierarchicalManager: Role, responsibilities, state management
  - [ ] SelectorPolicy: Input/output, abstract observation usage
  - [ ] SpecialistPolicies: Search, Track, Terminal behaviors
  - [ ] OptionManager: Forced transitions, thresholds

- [ ] Design patterns used:
  - [ ] Mediator pattern (Manager coordinates)
  - [ ] Wrapper pattern (Environment wrapping)
  - [ ] Strategy pattern (Specialist policies)

- [ ] State diagrams:
  - [ ] Option transition graph (SEARCH → TRACK → TERMINAL)
  - [ ] Forced transition conditions
  - [ ] Episode lifecycle

**Content Outline Expected:**
```markdown
# HRL Architecture

## Overview
Hierarchical Reinforcement Learning for Missile Defense...

## System Design

### Three-Level Hierarchy
```
┌─────────────────────────────────┐
│  Selector (High-Level)          │
│  - Input: 7D abstract state     │
│  - Output: Option {0,1,2}       │
│  - Frequency: Every 100 steps   │
└──────────────┬──────────────────┘
               │
       ┌───────┴──────┬─────────────┐
       │              │             │
┌──────▼────┐  ┌──────▼────┐  ┌───▼──────┐
│ Search    │  │  Track    │  │Terminal  │
│Specialist │  │Specialist │  │Specialist│
└──────┬────┘  └──────┬────┘  └───┬──────┘
       │              │            │
       └──────────────┼────────────┘
                      │
                  ┌───▼────────┐
                  │ 6D Action  │
                  │ to Motor   │
                  └────────────┘
```

### Observation Abstraction
- 26D Full → 7D Abstract
  - Distance, closing rate, lock quality, fuel, off-axis, TTI, altitude

### Reward Decomposition
- Strategic (Selector): Intercept success, fuel efficiency, timeout penalties
- Tactical (Specialists): Option-specific objectives (lock, distance, precision)

## Module Details

### HierarchicalManager
Coordinates selector and specialists...

### SelectorPolicy
High-level option selection on abstract state...

### SpecialistPolicies
Search: Wide-area scanning
Track: Maintain lock and approach
Terminal: Final intercept guidance

### OptionManager
Forced transitions when:
- Lock quality < 0.3: TRACK → SEARCH
- Lock quality > 0.7 in SEARCH: SEARCH → TRACK
- Distance < 100m: → TERMINAL

## Design Patterns

### Mediator Pattern
Manager coordinates between selector and specialists...

### Wrapper Pattern
Environment wrapped with HierarchicalActionWrapper...

## State Machine

### Option Transitions
```
SEARCH ─ (lock_quality > 0.7) ──→ TRACK
  ↑                                  │
  │                                  ↓
  └─ (lock_quality < 0.3) ─ (distance < 100m) → TERMINAL
```
```

**Dependencies:**
- None (document only)
- Reference to module code for examples

**Acceptance Criteria:**
- [ ] Document covers all core concepts
- [ ] Diagrams are clear (text or ASCII art acceptable)
- [ ] Module descriptions match actual code
- [ ] Design patterns explained with rationale
- [ ] Sufficient detail for developer onboarding

---

#### 5.3.2 Create Training Guide
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/docs/hrl/training_guide.md`

**Specific Functionality Required:**
- [ ] Quick start section:
  - [ ] Minimal commands to run HRL training
  - [ ] Expected output and runtime

- [ ] Detailed training workflow:
  - [ ] Stage 1: Specialist pre-training
    - [ ] What each specialist learns
    - [ ] How to train individually
    - [ ] Expected convergence metrics
  - [ ] Stage 2: Selector training
    - [ ] Why freeze specialists
    - [ ] How selector learns options
    - [ ] Expected learning curves
  - [ ] Stage 3: Joint fine-tuning (optional)
    - [ ] Why needed (higher performance)
    - [ ] Risks (forgetting pre-training)
    - [ ] Recommended learning rate reduction

- [ ] Configuration guide:
  - [ ] Key hyperparameters and their effects
  - [ ] Reward weight tuning
  - [ ] Network architecture choices
  - [ ] Curriculum difficulty progression

- [ ] Monitoring and debugging:
  - [ ] TensorBoard metrics to watch
  - [ ] What good training looks like (expected curves)
  - [ ] Common issues and solutions
  - [ ] How to interpret option usage statistics

- [ ] Evaluation and testing:
  - [ ] How to evaluate trained policies
  - [ ] Comparing flat PPO vs HRL
  - [ ] Performance metrics to track
  - [ ] Test scenarios (easy/medium/hard)

**Content Outline Expected:**
```markdown
# HRL Training Guide

## Quick Start

```bash
# Full pipeline
python scripts/train_hrl_full.py --config configs/hrl/hrl_curriculum.yaml

# Monitor training
tensorboard --logdir logs --port 6006
# Access: http://localhost:6006
```

Expected runtime: ~2 hours
Expected result: Trained selector + specialists in checkpoints/hrl/

## Detailed Workflow

### Stage 1: Specialist Pre-training (45 min)

#### Search Specialist (15 min)
- Learns: Wide-area scanning, lock acquisition
- Curriculum: easy (100 ep) → medium (100 ep)
- Success metric: Lock quality > 0.7 within 10 steps
- TensorBoard: Check `specialists/search/reward`

```bash
python scripts/train_hrl_pretrain.py --specialist search
```

#### Track Specialist (15 min)
- Learns: Maintain lock, approach target
- Curriculum: medium → hard
- Success metric: Distance decreases, lock maintained
- TensorBoard: Check `specialists/track/distance_reduction`

#### Terminal Specialist (15 min)
- Learns: Final guidance, precision
- Curriculum: Close-range scenarios
- Success metric: Miss distance < 10m
- TensorBoard: Check `specialists/terminal/miss_distance`

### Stage 2: Selector Training (20 min)
- High-level policy learns when to switch options
- Input: 7D abstract state
- Output: Discrete option {0, 1, 2}
- Expected metrics:
  - Reward increases over 3M steps
  - Option usage: all three used regularly
  - Intercept success rate: >70%

### Stage 3: Joint Fine-tuning (15 min, optional)
- All policies update together
- Recommended: Use 0.1x learning rate
- Better performance but risk of forgetting

## Configuration Guide

### Hyperparameters

Key tunable parameters in `configs/hrl/hrl_base.yaml`:

1. **decision_interval** (100 steps)
   - Time between high-level decisions
   - Lower = more frequent switching
   - Tradeoff: Higher = fewer decision overhead

2. **Selector learning_rate** (0.0003)
   - Start point: 0.0003
   - If unstable: Reduce to 0.0001
   - If too slow: Increase to 0.0005

3. **Specialist entropy_coef** (0.02, search only)
   - Higher = more exploration
   - Search needs more exploration than track
   - Too high = random behavior

### Reward Weights

Tune in `configs/hrl/hrl_base.yaml` under `rewards.tactical`:

- **Search specialist:**
  - `lock_acquisition_bonus`: 50 (encourage lock getting)
  - `lock_improvement`: 10 (per 0.1 quality increase)
  - `angular_diversity_bonus`: 0.5 (scanning diversity)

- **Track specialist:**
  - `lock_maintenance_bonus`: 2 (per step with good lock)
  - `distance_reduction`: 1 (per meter closed)
  - `closing_rate_bonus`: 0.5 (approach speed)

- **Terminal specialist:**
  - `proximity_bonus_scale`: 10 (exponential for close)
  - `closing_rate_bonus`: 1 (high speed focus)

## Monitoring Training

### TensorBoard Metrics to Watch

```
# Overall performance
rollout/ep_rew_mean - Should increase gradually

# Specialist performance (each)
specialists/search/reward - Should stabilize
specialists/track/distance_reduction - Should increase
specialists/terminal/miss_distance - Should decrease

# Selector performance
selector/reward - Should increase
selector/option_distribution - Show all 3 options used
selector/option_switches - Reasonable frequency
```

### Expected Learning Curves

1000 steps:
- All specialists learning (reward increasing)
- High reward variance (normal for early training)

10k steps:
- Search converges to >0.7 lock quality
- Track shows distance reduction
- Terminal shows decreasing miss distance

100k steps:
- Selector starts learning option timing
- Intercept success rate >50%
- Option usage becoming strategic

## Evaluation

### Individual Specialist Performance

```bash
# Run 50 episodes with frozen specialist
python scripts/evaluate_hrl.py --model-path checkpoints/hrl/specialists/search/best/ --episodes 50
```

### Full HRL Performance

```bash
# Run 100 episodes with selector + specialists
python scripts/evaluate_hrl.py --model-path checkpoints/hrl/selector/best/ --episodes 100
```

### Compare Flat vs HRL

```bash
# Side-by-side comparison
python scripts/compare_policies.py \
    --flat checkpoints/flat_ppo/best/ \
    --hrl checkpoints/hrl/selector/best/ \
    --episodes 100
```

## Common Issues

### Issue: Selector learns to always use same option
**Cause:** Reward for switching too low or specialist performance imbalanced
**Fix:**
- Increase `closing_rate_bonus` in track specialist
- Check that all specialists converge to >0 reward
- Verify abstract observation gives enough information

### Issue: Training very slow
**Cause:** decision_interval too long or network too large
**Fix:**
- Reduce decision_interval from 100 to 50
- Reduce network size: [256, 256] instead of [512, 512, 256]
- Increase learning_rate from 0.0003 to 0.001

### Issue: Frequent option switching (every step)
**Cause:** decision_interval too short or selector still learning
**Fix:**
- Increase decision_interval to 200
- Let training continue longer (selector learns stability)
```

**Acceptance Criteria:**
- [ ] Quick start section has 3-4 copy-paste commands
- [ ] All three training stages documented with expectations
- [ ] Key hyperparameters explained with tuning guidance
- [ ] TensorBoard metrics listed with interpretation
- [ ] Common issues have solutions
- [ ] Expected runtimes and memory usage documented

---

#### 5.3.3 Create API Reference
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/docs/hrl/api_reference.md`

**Specific Functionality Required:**
- [ ] Public API reference for each module:
  - [ ] HierarchicalManager class
  - [ ] SelectorPolicy class
  - [ ] SpecialistPolicy and subclasses
  - [ ] observation_abstraction functions
  - [ ] reward_decomposition functions
  - [ ] HierarchicalEnvWrapper class

- [ ] For each class/function:
  - [ ] Signature with type hints
  - [ ] Parameter descriptions
  - [ ] Return value descriptions
  - [ ] Usage example
  - [ ] Raises section (exceptions)

- [ ] Data structures:
  - [ ] HRLState dataclass
  - [ ] Option enum values and meanings
  - [ ] Configuration structure (YAML keys)

- [ ] Constants:
  - [ ] FORCED_TRANSITION_THRESHOLDS
  - [ ] OPTION_METADATA
  - [ ] Reward weight ranges

**Format Example Expected:**
```markdown
# HRL API Reference

## hrl.manager

### class HierarchicalManager
High-level controller coordinating selector and specialists.

#### Constructor
```python
def __init__(
    self,
    selector_policy: SelectorPolicy,
    specialists: Dict[Option, SpecialistPolicy],
    decision_interval: int = 100,
    enable_forced_transitions: bool = True,
    default_option: Option = Option.SEARCH,
)
```

**Parameters:**
- `selector_policy` (SelectorPolicy): High-level policy
- `specialists` (Dict[Option, SpecialistPolicy]): Three specialist policies
- `decision_interval` (int): Steps between selector decisions (default 100)
- `enable_forced_transitions` (bool): Allow environment-based forced switches
- `default_option` (Option): Initial option on reset (default SEARCH)

**Example:**
```python
from hrl.manager import HierarchicalManager
from hrl.selector_policy import SelectorPolicy
from hrl.specialist_policies import SearchSpecialist, TrackSpecialist, TerminalSpecialist
from hrl.option_definitions import Option

selector = SelectorPolicy(obs_dim=7)
specialists = {
    Option.SEARCH: SearchSpecialist(),
    Option.TRACK: TrackSpecialist(),
    Option.TERMINAL: TerminalSpecialist(),
}

manager = HierarchicalManager(selector, specialists, decision_interval=100)
```

#### Method: select_action
Select action using hierarchical policy.

```python
def select_action(
    self,
    full_obs: np.ndarray,
    env_state: Optional[Dict[str, Any]] = None,
    deterministic: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]
```

**Parameters:**
- `full_obs` (np.ndarray): 26D or 104D (frame-stacked) observation
- `env_state` (Dict[str, Any]): Environment state for forced transitions
- `deterministic` (bool): Use deterministic action selection

**Returns:**
- `action` (np.ndarray): 6D continuous action
- `info` (Dict[str, Any]): Debugging information with keys:
  - `option`: Current option name (str)
  - `option_index`: Option index 0-2 (int)
  - `steps_in_option`: Steps in current option (int)
  - `option_switched`: Whether switched this step (bool)
  - `switch_reason`: "forced", "selector", or "continue"
  - `forced_transition`: Was transition forced (bool)

**Example:**
```python
obs, _ = env.reset()
env_state = {
    'lock_quality': 0.5,
    'distance': 1000.0,
    'fuel': 0.8,
    'closing_rate': 50.0,
}

action, info = manager.select_action(obs, env_state, deterministic=True)
print(f"Selected option: {info['option']}")
```

**Raises:**
- ValueError: If obs shape is invalid
- KeyError: If env_state missing required keys

### class HRLState
Current state of hierarchical controller.

```python
@dataclass
class HRLState:
    current_option: Option
    steps_in_option: int
    lstm_states: Dict[Option, Any]
    last_decision_step: int
    total_steps: int
```

## hrl.selector_policy

### class SelectorPolicy
High-level policy for option selection.

#### Constructor
```python
def __init__(
    self,
    obs_dim: int,
    n_options: int = 3,
    model_path: Optional[str] = None,
    device: str = 'auto',
)
```

**Parameters:**
- `obs_dim` (int): Abstract state dimension (typically 7)
- `n_options` (int): Number of options (default 3)
- `model_path` (str): Path to pre-trained model (optional)
- `device` (str): 'cuda', 'cpu', or 'auto'

#### Method: predict
Select option based on abstract observation.

```python
def predict(
    self,
    abstract_obs: np.ndarray,
    deterministic: bool = True,
) -> int
```

**Parameters:**
- `abstract_obs` (np.ndarray): 7D abstract state vector
- `deterministic` (bool): Use deterministic policy

**Returns:**
- (int): Option index {0, 1, 2}

## hrl.observation_abstraction

### Function: abstract_observation
Convert 26D radar observation to 7D abstract state.

```python
def abstract_observation(full_obs: np.ndarray) -> np.ndarray
```

**Parameters:**
- `full_obs` (np.ndarray): 26D base or 104D frame-stacked observation

**Returns:**
- (np.ndarray): 7D abstract state:
  - [0] distance_to_target (normalized 0-1)
  - [1] closing_rate (normalized -1 to 1)
  - [2] radar_lock_quality (0-1)
  - [3] fuel_fraction (0-1)
  - [4] off_axis_angle (normalized -1 to 1)
  - [5] time_to_intercept (normalized 0-1)
  - [6] relative_altitude (normalized -1 to 1)

**Example:**
```python
from hrl.observation_abstraction import abstract_observation

full_obs = np.random.randn(26)
abstract_state = abstract_observation(full_obs)
assert abstract_state.shape == (7,)
```

## hrl.option_definitions

### Enum: Option
Options available to selector.

```python
class Option(IntEnum):
    SEARCH = 0      # Wide-area scanning
    TRACK = 1       # Maintain lock and approach
    TERMINAL = 2    # Final intercept guidance
```

### Constants
```python
FORCED_TRANSITION_THRESHOLDS = {
    'radar_lock_quality_min': 0.3,      # Below: lose track
    'radar_lock_quality_search': 0.7,   # Above: exit search
    'close_range_threshold': 100.0,     # meters
    'terminal_fuel_min': 0.1,           # Fraction
}

OPTION_METADATA = {
    Option.SEARCH: {
        'name': 'Search',
        'description': '...',
        'expected_duration': 200,
        'forced_exit_conditions': ['radar_lock_acquired'],
        'color': '#FF6B6B',
    },
    # ... TRACK and TERMINAL
}
```
```

**Dependencies:**
- None (reference material only)

**Acceptance Criteria:**
- [ ] All public classes and functions documented
- [ ] Each has signature, parameters, returns, example
- [ ] Exceptions/errors documented
- [ ] Data structures (enums, dataclasses) explained
- [ ] Constants listed with values and meanings

---

#### 5.3.4 Create Migration Guide
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/docs/hrl/migration_guide.md`

**Specific Functionality Required:**
- [ ] Overview: What's changing, what's not

- [ ] For existing users (flat PPO):
  - [ ] "You don't need to change anything"
  - [ ] Backward compatible examples
  - [ ] How to access HRL if interested (optional)

- [ ] For new users:
  - [ ] Quick start with HRL (recommended path)
  - [ ] Understanding the three training stages
  - [ ] How to monitor progress

- [ ] Directory structure changes:
  - [ ] Old `scenarios/` still works (symlink)
  - [ ] New `configs/` structure
  - [ ] Checkpoint organization (flat_ppo vs hrl)

- [ ] If upgrading from older version:
  - [ ] Migration steps
  - [ ] Checkpoint migration script
  - [ ] Updated config format

- [ ] Troubleshooting:
  - [ ] Common issues after update
  - [ ] How to revert if needed
  - [ ] Getting help

**Content Outline Expected:**
```markdown
# HRL Migration Guide

## Overview

The Hlynr Intercept system now includes optional Hierarchical RL (HRL) capabilities. The existing flat PPO training pipeline remains unchanged and fully backward compatible.

### What's New
- Hierarchical training pipeline (optional)
- Pre-trainable specialist policies
- High-level option selection
- Improved modular architecture

### What's Unchanged
- `train.py` behavior (same interface)
- All existing checkpoints
- `core.py`, `environment.py` (no changes)
- Inference and evaluation

## For Existing Users (Flat PPO)

### No Action Required
Your existing workflow continues to work exactly as before:

```bash
cd rl_system/
python train.py --config config.yaml
python inference.py --model checkpoints/best --episodes 100
```

### Optional: Try HRL
If you want to experiment with HRL:

```bash
# Full HRL training pipeline
python scripts/train_hrl_full.py --config configs/hrl/hrl_curriculum.yaml

# Evaluate HRL
python scripts/evaluate_hrl.py --model-path checkpoints/hrl/selector/best/ --episodes 100

# Compare with flat PPO
python scripts/compare_policies.py \
    --flat checkpoints/flat_ppo/best/ \
    --hrl checkpoints/hrl/selector/best/
```

## For New Users

### Recommended Starting Point

Start with flat PPO for simplicity, then move to HRL if desired:

**Stage 1: Flat PPO Training (Quickest)**
```bash
python train.py --config config.yaml  # 25-30 minutes
python inference.py --model checkpoints/best --episodes 100
```

**Stage 2: HRL Training (More Complex, Better Performance)**
```bash
# Full pipeline: ~2 hours
python scripts/train_hrl_full.py --config configs/hrl/hrl_curriculum.yaml

# Evaluate and compare
python scripts/compare_policies.py --flat checkpoints/flat_ppo/best/ --hrl checkpoints/hrl/selector/best/
```

## Directory Structure Changes

### Files You Interact With

Old paths (still work with backward compat):
```
scenarios/easy.yaml       → Now: configs/scenarios/easy.yaml
checkpoints/best/         → Now: checkpoints/flat_ppo/best/
```

New paths (recommended):
```
configs/scenarios/        # Scenario difficulty levels
configs/hrl/              # HRL-specific configs
checkpoints/flat_ppo/     # Flat PPO checkpoints
checkpoints/hrl/          # HRL checkpoints (selector + specialists)
scripts/                  # Training and evaluation scripts
```

### Symlinks for Backward Compatibility

```
scenarios/ → configs/scenarios/   (old paths still work)
```

If you reference `scenarios/easy.yaml` in scripts, both paths work:
- Old: `scenarios/easy.yaml` ✓
- New: `configs/scenarios/easy.yaml` ✓

## Checkpoint Migration

### If You Have Old Checkpoints

Run migration script:

```bash
cd rl_system/
python scripts/migrate_checkpoints.py \
    --source checkpoints/ \
    --dest checkpoints/flat_ppo/ \
    --backup
```

This will:
1. Create backup: `checkpoints_backup_2025-01-20T10:30:00/`
2. Move checkpoints to `checkpoints/flat_ppo/`
3. Verify all checkpoints load correctly

Your training continues to work:
```bash
python train.py --config config.yaml  # Still works!
```

## Breaking Changes

**None.** This release is 100% backward compatible.

If you encounter any issues:
1. Check `TROUBLESHOOTING.md`
2. Verify `python train.py --config config.yaml` works
3. If not, restore from backup and contact team

## Config File Format

No changes to core config format. New optional HRL configs:

```yaml
# Existing config.yaml (unchanged)
environment:
  dt: 0.01
  max_steps: 1000
# ... rest unchanged

# New: configs/hrl/hrl_base.yaml
hrl:
  enabled: true
  hierarchy:
    levels: 2
    decision_interval: 100
  # ... HRL-specific settings
```

If you created custom configs, they continue to work unchanged.

## Troubleshooting

### Issue: "scenarios/ directory not found"
**Solution:** Use `configs/scenarios/` or let symlink handle it

### Issue: Old checkpoints don't load
**Solution:** Run migration script and check `checkpoints/flat_ppo/`

### Issue: Import errors for new modules
**Solution:** Ensure `pip install -r requirements.txt` ran

### Issue: HRL training much slower than flat PPO
**Expected behavior:** HRL has overhead from decision-making
**Optimization:** Increase `decision_interval` from 100 to 200 steps

## Getting Help

- Architecture questions: See `docs/hrl/architecture.md`
- Training questions: See `docs/hrl/training_guide.md`
- API usage: See `docs/hrl/api_reference.md`
- Issues: Check `README.md` or contact development team
```

**Acceptance Criteria:**
- [ ] Clearly states: "No action required for existing users"
- [ ] Shows backward compatible examples
- [ ] Explains directory structure changes
- [ ] Migration script documented
- [ ] Troubleshooting section covers common issues
- [ ] Links to other documentation

---

#### 5.3.5 Update Main README
**File:** `/Users/quinnhasse/Hlynr_Intercept/rl_system/README.md`

**Specific Functionality Required:**
- [ ] Add HRL section to table of contents
- [ ] Add HRL quick start (copy-paste commands)
- [ ] Add link to HRL documentation (`docs/hrl/`)
- [ ] Add section: "What's New (HRL)"
- [ ] Keep existing flat PPO documentation unchanged
- [ ] Add comparison table: Flat PPO vs HRL
- [ ] Update feature list

**Content to Add/Modify:**

```markdown
# Hlynr Intercept: Hierarchical RL Missile Defense Simulator

## What's New - Hierarchical RL Support

The system now supports optional Hierarchical RL training for improved policy learning and faster convergence. Existing flat PPO workflows remain unchanged.

## Quick Start

### Option 1: Flat PPO (Existing - Unchanged)
```bash
cd rl_system/
python train.py --config config.yaml
python inference.py --model checkpoints/best --episodes 100
```
Expected time: 25-30 minutes, 75%+ intercept rate

### Option 2: Hierarchical RL (New - Recommended)
```bash
cd rl_system/
python scripts/train_hrl_full.py --config configs/hrl/hrl_curriculum.yaml
python scripts/evaluate_hrl.py --model-path checkpoints/hrl/selector/best/ --episodes 100
```
Expected time: 2 hours, similar or better performance

## Approach Comparison

| Feature | Flat PPO | Hierarchical RL |
|---------|----------|-----------------|
| Training time | 25-30 min | ~2 hours |
| Intercept rate | 75-85% | 70-85% |
| Interpretability | Low | High (option usage) |
| Composability | Monolithic | Modular (search/track/terminal) |
| Customization | Limited | High (per-option reward tuning) |
| Real-world transfer | Direct | Potential via option dynamics |

### When to Use Each

**Flat PPO:**
- Quick baseline establishment
- Simple scenarios
- When interpretability not needed
- Minimal computational resources

**Hierarchical RL:**
- Complex multi-phase behavior needed
- Interpretable decision-making required
- Fine-tuned control per phase
- Research and analysis

## Documentation

- **Getting Started:** See `CLAUDE.md` for setup
- **HRL Architecture:** See `docs/hrl/architecture.md`
- **HRL Training Guide:** See `docs/hrl/training_guide.md`
- **API Reference:** See `docs/hrl/api_reference.md`
- **Migration Guide:** See `docs/hrl/migration_guide.md`

## Backward Compatibility

100% backward compatible. All existing commands work unchanged:
```bash
python train.py --config config.yaml      # Still works!
python inference.py --model checkpoints/best
```

No breaking changes. Existing checkpoints load correctly.
```

**Acceptance Criteria:**
- [ ] HRL section added to README
- [ ] Quick start shows both flat PPO and HRL options
- [ ] Comparison table accurate and useful
- [ ] Documentation links correct
- [ ] Backward compatibility emphasized
- [ ] Total README still readable (not overwhelming)

---

### Validation Checkpoint (Day 12)
```bash
cd /Users/quinnhasse/Hlynr_Intercept/rl_system

# Verify all documentation exists
ls -la docs/hrl/
# Expected: architecture.md, training_guide.md, api_reference.md, migration_guide.md

# Test documentation examples
grep -r "python scripts/" docs/hrl/
# Expected: Multiple code examples

# Verify README updated
grep -i "hierarchical" README.md
# Expected: HRL section present
```

---

## 5.4 DOCUMENTATION & TESTING SUMMARY

### Documentation Created
| Document | Location | Purpose | Status |
|----------|----------|---------|--------|
| architecture.md | docs/hrl/ | System design and concepts | [ ] |
| training_guide.md | docs/hrl/ | Training workflow and tuning | [ ] |
| api_reference.md | docs/hrl/ | API documentation | [ ] |
| migration_guide.md | docs/hrl/ | Upgrade path and changes | [ ] |
| DIRECTORY_STRUCTURE.md | rl_system/ | Directory layout reference | [ ] |
| README.md | rl_system/ | Updated with HRL info | [ ] |

### Tests Created (Day 10)
| Test File | Location | Coverage | Status |
|-----------|----------|----------|--------|
| test_end_to_end.py | tests/ | Full HRL pipeline | [ ] |
| test_backward_compatibility.py | tests/ | Legacy code unchanged | [ ] |

### Evaluation Scripts Created (Day 10)
| Script | Location | Purpose | Status |
|--------|----------|---------|--------|
| evaluate_hrl.py | scripts/ | HRL policy evaluation | [ ] |
| compare_policies.py | scripts/ | Flat vs HRL comparison | [ ] |

### Migration Completed (Day 11)
| Task | Status |
|------|--------|
| Migrate existing checkpoints to `checkpoints/flat_ppo/` | [ ] |
| Create backups | [ ] |
| Verify all checkpoints load | [ ] |
| Generate migration report | [ ] |

---

## CRITICAL DEPENDENCIES & ORDER

### Must Complete Phases 1-3 First
- [ ] Phase 1: Directory structure, backward compat symlinks
- [ ] Phase 2: Core HRL modules (manager, specialists, selector)
- [ ] Phase 3: Reward decomposition, environment wrappers

### Phase 4 Must Complete Before Phase 5
- [ ] Day 7: Flat PPO refactoring
- [ ] Day 8: HRL pre-training script + configs
- [ ] Day 9: HRL selector and full training scripts

### Phase 5 Validation Order
- [ ] Day 10: Integration tests (requires all Phase 4 scripts)
- [ ] Day 11: Checkpoint migration (requires verified scripts)
- [ ] Day 12: Documentation (reference finalized code)

---

## AMBIGUITIES & CLARIFICATION QUESTIONS

### Before Implementation Starts

1. **Checkpoint Organization**
   - [ ] Should flat PPO checkpoints be migrated automatically or manually?
   - [ ] Should old `checkpoints/` be preserved as backup or removed?
   - [ ] Recommendation: Create backup, organize into flat_ppo/, keep backup for 1 month

2. **Configuration Inheritance**
   - [ ] Should HRL configs completely define all settings or inherit from config.yaml?
   - [ ] How to handle conflicts (HRL-specific vs base settings)?
   - [ ] Recommendation: HRL configs inherit from base, override HRL-specific sections only

3. **Training Duration**
   - [ ] Is 2-hour total pipeline acceptable or should it be faster?
   - [ ] Can specialists train in parallel to reduce time?
   - [ ] Recommendation: 2 hours is acceptable for initial v1; parallelization is future optimization

4. **Test Coverage**
   - [ ] Is 80% code coverage sufficient or target 90%+?
   - [ ] Should all edge cases be tested or just happy path?
   - [ ] Recommendation: 80% minimum for Phase 1; edge cases add polish later

5. **Performance Baselines**
   - [ ] What intercept rate is acceptable for HRL to be considered successful?
   - [ ] Should HRL match or exceed flat PPO performance?
   - [ ] Recommendation: Within 10% of flat PPO (±75-85%) sufficient for Phase 1

6. **Documentation Scope**
   - [ ] Should docs include theoretical background on HRL?
   - [ ] Or focus on practical usage only?
   - [ ] Recommendation: Practical usage primary; reference papers for theory

7. **Backward Compatibility**
   - [ ] If old config paths fail, should we auto-migrate or error with helpful message?
   - [ ] How long should symlinks be maintained?
   - [ ] Recommendation: Auto-migrate with deprecation warning; maintain symlinks for 2+ releases

### Decisions Needed from Team

Before starting Phase 4, confirm:
- [ ] Acceptable timeline: 3 days (Days 7-9) for Phase 4 scripts
- [ ] Acceptable testing scope: Unit + integration, 80% coverage
- [ ] Documentation depth: Practical guides + API reference
- [ ] Checkpoint handling: Migrate automatically with backup
- [ ] HRL performance target: Within 10% of flat PPO

---

## SUCCESS CRITERIA SUMMARY

### Functional (Required)
- [ ] All Phase 4 scripts runnable without error
- [ ] All Phase 5 tests pass (>80% coverage)
- [ ] HRL training runs end-to-end for 2 hours
- [ ] Checkpoint migration preserves all models
- [ ] All documentation complete and accurate

### Performance (Target)
- [ ] Specialist training: <30 min per specialist
- [ ] Selector training: <20 minutes
- [ ] Full pipeline: <2 hours total
- [ ] Inference overhead: <5ms per step

### Quality (Essential)
- [ ] Zero breaking changes to existing code
- [ ] All backward compatibility tests pass
- [ ] Code coverage >80% for new modules
- [ ] Documentation examples are tested and working

---

**Prepared by:** Claude Code (Requirements Analyst)
**Generated from:** HRL_REFACTORING_DESIGN.md
**Last Updated:** 2025-11-09
**Next Steps:** Review this checklist with team, create feature branch, begin Phase 4 implementation
