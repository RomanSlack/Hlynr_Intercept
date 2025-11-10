# Hierarchical Reinforcement Learning for Missile Intercept
## An Engineering Primer

---

## 1. Why HRL for Long-Horizon Intercept

### The Core Problem

Our current flat PPO agent faces a brutal challenge: it must learn a single monolithic policy that handles 5000+ timesteps (50+ seconds at 100Hz) with almost zero feedback until the very end. Success gives +5000 reward, failure gives 0. That's like teaching someone to land an aircraft by only saying "good job" or "you crashed" after each 50-second attempt, with no feedback during the flight.

**The credit assignment nightmare:**
- Timestep 1: Should I turn the radar left or right? (No feedback for 5000 steps)
- Timestep 2500: Was that radar sweep useful? (No feedback for 2500 steps)
- Timestep 4999: Was my approach angle correct? (Feedback in 1 step, but too late)

The agent must somehow figure out which of those 5000 decisions mattered. This is exponentially hard.

### How Temporal Abstraction Helps

Humans don't think at 100Hz. A pilot intercept mission breaks naturally into phases:

```
SEARCH (5-20 sec)     →  TRACK (20-35 sec)    →  TERMINAL (5-10 sec)
"Where is it?"           "Stay locked on it"      "Hit it precisely"
Wide radar scans         Maintain radar lock      Aggressive maneuvering
Low G-forces             Moderate G-forces        Max G-forces
```

Hierarchical RL exploits this structure by learning:
1. **Low-level specialists**: Each learns ONE phase really well (shorter horizon, faster learning)
2. **High-level selector**: Learns WHEN to switch between specialists (simple decision, easy to learn)

**Why this reduces exploration space:**

Flat PPO must explore: 6-dimensional action space^5000 timesteps = absurdly large

HRL explores:
- Search specialist: 6D action space^500 timesteps (10x easier)
- Track specialist: 6D action space^2000 timesteps (2.5x easier)
- Terminal specialist: 6D action space^500 timesteps (10x easier)
- Selector: 3-option choice^10 decisions (trivial)

Each subproblem is independently tractable. The selector only makes ~10 decisions per episode at 1Hz, not 5000.

### Natural Phase Decomposition

The interceptor domain has gift-wrapped phase boundaries:

| Phase | Natural Trigger | Reward Signal | Observation Focus |
|-------|----------------|---------------|-------------------|
| Search | Start → Radar lock acquired | Dense: +1 per radar sweep covering target area | Position uncertainty, fuel, time |
| Track | Lock acquired → Close range (<500m) | Dense: +1 per timestep maintaining lock | Lock quality, closing rate, relative velocity |
| Terminal | Close range → Impact/miss | Dense: +10 per meter closer | Off-axis angle, time-to-intercept, acceleration needed |

Each phase has:
- **Clear success criteria**: Lock acquired, lock maintained, impact achieved
- **Dense intermediate rewards**: No more waiting 5000 steps for feedback
- **Focused state relevance**: Search doesn't care about precise angles, terminal doesn't care about fuel conservation

### Sample Efficiency Gains

Conservative estimates based on HRL literature and our domain:

**Current flat PPO:**
- 5M steps to reach 75-85% success (25-30 minutes training)
- Struggles with cold-start exploration (first 1M steps often <30% success)
- Curriculum required: Easy → Medium → Hard scenarios

**Expected with HRL:**
- **Phase 1 - Pre-train specialists** (parallel training):
  - Search specialist: 200k steps on "detect target" task (~1 min)
  - Track specialist: 250k steps on "maintain lock" task (~1.5 min)
  - Terminal specialist: 100k steps on "close intercept" task (~30 sec)
  - Total: 550k steps, but parallelizable → ~3 minutes wall-clock

- **Phase 2 - Train selector**: 200k steps learning when to switch (~1 min)
  - Uses frozen specialists initially
  - Fine-tunes specialists later if needed

**Total: ~750k steps (~4 minutes) to reach equivalent performance**

**Sample efficiency improvement: 6-7x faster**

Additional benefits:
- **Debuggability**: Can isolate which phase is failing (current PPO is a black box)
- **Transferability**: Pre-trained search specialist works across all scenarios
- **Robustness**: Terminal specialist trained on close-range starts is more reliable

---

## 2. Core HRL Concepts for Engineers

### Options/Skills Framework

Think of options as "subroutines" or "macro-actions":

```python
class Option:
    def initiation_set(self, state):
        """Can this option run now? (e.g., Track needs radar lock)"""
        return state.radar_locked

    def policy(self, observation):
        """Specialist policy: 17D radar obs → 6D thrust+angular"""
        return self.neural_network(observation)

    def termination_condition(self, state):
        """Should we stop this option? (e.g., Track ends at close range)"""
        return state.range_to_target < 500  # meters
```

**Key insight**: Options have temporal extent. The high-level selector doesn't choose actions every timestep—it chooses which option runs, and that option executes for multiple timesteps (100-2000 steps).

This is fundamentally different from flat RL where the policy chooses actions every single timestep.

### Policy Switching vs Goal-Conditioned Approaches

**Policy Switching (our approach):**
```
Selector: "Use Track policy"  [decision at t=0]
  Track policy executes: a₀, a₁, a₂, ..., a₁₉₉₉  [2000 timesteps]
Selector: "Switch to Terminal policy"  [decision at t=2000]
  Terminal policy executes: a₂₀₀₀, a₂₀₀₁, ..., a₄₉₉₉  [3000 timesteps]
```

**Goal-Conditioned (alternative):**
```
High-level: "Achieve position (x=1000, y=500, z=300)" [abstract goal]
  Low-level: Takes actions a₀, a₁, a₂... trying to reach that position
High-level: "Achieve position (x=200, y=100, z=50)" [new goal]
  Low-level: Takes actions trying to reach new position
```

Policy switching is simpler: the selector just picks "which brain to use" rather than generating complex spatial goals.

### Hierarchical Observation Spaces

**Low-level specialists** see full sensory details:
- 17D radar observations (ranges, velocities, angles, lock quality)
- Updated at 100Hz
- Noisy, high-dimensional, partially observable

**High-level selector** sees abstract state summary:
- 8D strategic state: range_to_target, closing_rate, radar_lock_quality, fuel_remaining, time_elapsed, current_altitude, target_altitude_estimate, off_axis_angle
- Updated at 1Hz (downsampled)
- Cleaner, low-dimensional, aggregated over time

This is like a pilot (low-level) seeing the instrument panel in real-time, while the mission commander (high-level) sees the tactical situation map updated every few seconds.

### Reward Decomposition

**Strategic rewards (selector):**
- +10 for successful phase transition (e.g., acquiring radar lock)
- -5 for premature termination (e.g., losing lock in track phase)
- +5000 for mission success (same as flat PPO)
- -1000 for mission failure (crash, fuel depletion, timeout)

**Tactical rewards (specialists):**
- Search: +1 per radar sweep covering likely target region, -0.1 per G-force (fuel conservation)
- Track: +5 per timestep maintaining lock, -1 per degree of lock degradation
- Terminal: +10 per meter closer to target, +100 for impact, +50 for near-miss (<10m)

Notice how specialists get **dense feedback** every timestep. This is the magic: instead of waiting 5000 steps, they learn from immediate consequences.

### Pre-training vs Joint Training vs Curriculum

**Pre-training (our recommended approach):**
1. Train each specialist independently on isolated tasks
2. Freeze specialists and train selector
3. Optionally fine-tune end-to-end

Advantages: Fast, stable, debuggable
Disadvantages: Specialists might not coordinate perfectly

**Joint training:**
Train selector and specialists simultaneously from scratch

Advantages: Better coordination possible
Disadvantages: Unstable, slow, hard to debug

**Curriculum:**
Gradually increase task difficulty

Our approach: Pre-train on easy scenarios, then curriculum for selector

---

## 3. Why Policy Switching Fits Our Interceptor

### Discrete Phases Map Naturally to Options

Our missile intercept has hard physical boundaries:

```
Phase Transitions (Observable in State):

SEARCH → TRACK: Triggered by radar_lock_quality > 0.7
  Physical: Radar beam centered on target, doppler locked

TRACK → TERMINAL: Triggered by range_to_target < 500m
  Physical: Entering close-combat maneuvering envelope

TERMINAL → END: Triggered by miss_distance < 5m OR timeout
  Physical: Intercept or miss declared
```

These aren't soft, learned transitions—they're grounded in radar physics and engagement geometry. Policy switching exploits this structure directly.

### Radar-Only Partial Observability Benefits from LSTM Specialists

With radar-only observations, the interceptor has memory requirements:

**Search phase**: "I scanned left 3 seconds ago, now scan right"
- LSTM tracks: Previous scan directions, time since last detection

**Track phase**: "Target was here, moving this fast, 0.5 seconds ago"
- LSTM tracks: Target motion history for Kalman filtering

**Terminal phase**: "Target's evasive pattern over last 2 seconds"
- LSTM tracks: Acceleration patterns for prediction

Each phase has different memory needs. Specialist LSTMs can focus on phase-relevant history:
- Search LSTM: 10-second scan pattern history
- Track LSTM: 5-second motion estimation history
- Terminal LSTM: 1-second maneuver prediction history

A single flat LSTM must handle all three patterns simultaneously, which is harder to learn.

### Pre-trained Specialists = Proven Stability

We can validate each specialist independently:

```bash
# Test search specialist in isolation
python test_specialist.py --phase search --episodes 100
# Expected: 95%+ radar acquisition rate

# Test track specialist with perfect initial lock
python test_specialist.py --phase track --episodes 100
# Expected: 85%+ lock maintenance rate

# Test terminal specialist starting at 500m with lock
python test_specialist.py --phase terminal --episodes 100
# Expected: 70%+ intercept rate within 5m
```

If search fails 50% of the time, we **know** the search specialist needs more training—not the selector, not the other specialists. This debuggability is impossible with flat PPO.

### Comparison to Alternatives

**vs HIRO (Goal-Conditioned HRL):**

HIRO has high-level policy output abstract goals in state space (e.g., "reach position x,y,z in 10 seconds"), then low-level policy tries to achieve those goals.

Why this doesn't work for radar-only partial observability:
- **Problem 1**: With radar, we don't know our absolute position relative to target with precision—only noisy range/bearing
- **Problem 2**: Goals like "be at position (x, y, z)" assume full observability, but we only observe what radar sees
- **Problem 3**: Target position is estimated with Kalman filter uncertainty; giving precise spatial goals is meaningless
- **Our solution**: Policy switching doesn't require precise goals—just "use the tracking behavior" which handles uncertainty internally

**vs FeUdal Networks (Velocity-Space Goals):**

FeUdal uses directional goals in latent "velocity" space (e.g., "move in this direction in learned feature space").

Why this is complex for our domain:
- **Problem 1**: 6DOF dynamics (3D position + 3D orientation) make velocity-space goals 6-dimensional and hard to interpret
- **Problem 2**: Thrust and gimbal actions have coupled effects (thrust direction affects rotation, rotation affects thrust effectiveness)
- **Problem 3**: Debugging is nearly impossible—what does "move in direction [0.3, -0.7, 0.4, 0.1, -0.2, 0.9] in feature space" mean physically?
- **Our solution**: Policy switching uses interpretable phases—"you're in track phase" is clear to engineers and pilots

**vs Option Discovery (Learned Phases):**

Algorithms like DIAYN or VIC discover options automatically from data without hand-design.

Why hand-designed phases are better here:
- **Problem 1**: Learned options often don't align with human-interpretable phases (debuggability lost)
- **Problem 2**: Discovery requires huge amounts of exploration (expensive in long-horizon tasks)
- **Problem 3**: No guarantee learned options match safety requirements (e.g., "don't deplete fuel in search phase")
- **Our solution**: Hand-designed phases incorporate domain knowledge—radar physics, engagement geometry, safety constraints—directly

### Expected Benefits Summary

**Sample Efficiency:**
- 6-7x faster training (750k steps vs 5M steps)
- Parallelizable specialist training

**Interpretability:**
- Can visualize which phase is active at each timestep
- Can debug failures per phase: "Track phase loses lock at 1500m range—needs better LSTM memory"
- Can ablate specialists: "Does terminal specialist even need LSTM or is feedforward enough?"

**Robustness:**
- Search specialist trained on easy wide-beam radar transfers to hard narrow-beam scenarios
- Terminal specialist trained on 100m starts works for 500m starts
- Curriculum becomes easier: Train specialists on simple tasks, selector handles complexity

**Safety:**
- Per-phase fuel budgets: Search can't burn all fuel
- Per-phase G-limits: Terminal can use max G, search stays gentle
- Graceful degradation: If track fails, can fall back to search

---

## 4. Practical Implementation Sketch

### Architecture Overview

```
                    ┌─────────────────────────┐
                    │   High-Level Selector   │
                    │    (Updated at 1 Hz)    │
                    └───────────┬─────────────┘
                                │
                   Abstract State (8D):
                   [range, closing_rate, lock_quality,
                    fuel, time, alt, target_alt, off_axis]
                                │
                    ┌───────────┴─────────────┐
                    │   Option: "TRACK"       │
                    │   (Active for 2000 steps)│
                    └───────────┬─────────────┘
                                │
                   Radar Observations (17D):
                   [rel_pos, rel_vel, self_vel, orientation,
                    fuel, tti, lock, closing_rate, angle]
                                │
                    ┌───────────┴─────────────┐
                    │   Track Specialist      │
                    │   LSTM + MLP Policy     │
                    │   (Updated at 100 Hz)   │
                    └───────────┬─────────────┘
                                │
                    Actions (6D): [thrust_x, thrust_y, thrust_z,
                                   gimbal_pitch, gimbal_yaw, gimbal_roll]
```

### Pseudocode Implementation

```python
class HierarchicalInterceptor:
    def __init__(self):
        # High-level selector (simple feedforward MLP)
        self.selector = PPO(
            observation_dim=8,  # Abstract state
            action_dim=3,       # {search, track, terminal}
            policy_network=[64, 64],  # Small network
            update_frequency=1  # Hz
        )

        # Low-level specialists (LSTM + MLP)
        self.search_specialist = PPO(
            observation_dim=17,  # Full radar obs
            action_dim=6,        # Continuous control
            policy_network=[512, 512, 256],  # Large network
            lstm_hidden=256,
            update_frequency=100  # Hz
        )

        self.track_specialist = PPO(
            observation_dim=17,
            action_dim=6,
            policy_network=[512, 512, 256],
            lstm_hidden=256,
            update_frequency=100
        )

        self.terminal_specialist = PPO(
            observation_dim=17,
            action_dim=6,
            policy_network=[512, 512, 256],
            lstm_hidden=128,  # Shorter memory needed
            update_frequency=100
        )

        self.active_option = None
        self.option_start_time = 0
        self.lstm_state = None

    def get_abstract_state(self, full_observation):
        """Compress 17D radar obs to 8D strategic state"""
        return np.array([
            full_observation[0],   # range (rel_pos magnitude)
            full_observation[15],  # closing_rate
            full_observation[14],  # lock_quality
            full_observation[12],  # fuel_remaining
            self.time_elapsed,
            full_observation[9],   # altitude (from orientation)
            full_observation[2],   # target_altitude_estimate
            full_observation[16]   # off_axis_angle
        ])

    def step(self, observation, timestep):
        # High-level decision at 1 Hz
        if timestep % 100 == 0:  # Every second
            abstract_state = self.get_abstract_state(observation)
            new_option = self.selector.select_action(abstract_state)

            # Check if option should terminate
            if self.should_terminate_option(observation):
                self.active_option = new_option
                self.option_start_time = timestep
                self.lstm_state = None  # Reset specialist memory

        # Low-level action at 100 Hz
        if self.active_option == 0:  # Search
            action, self.lstm_state = self.search_specialist.act(
                observation, lstm_state=self.lstm_state
            )
        elif self.active_option == 1:  # Track
            action, self.lstm_state = self.track_specialist.act(
                observation, lstm_state=self.lstm_state
            )
        elif self.active_option == 2:  # Terminal
            action, self.lstm_state = self.terminal_specialist.act(
                observation, lstm_state=self.lstm_state
            )

        return action

    def should_terminate_option(self, observation):
        """Hard transitions based on physical state"""
        lock_quality = observation[14]
        range_to_target = np.linalg.norm(observation[0:3])

        if self.active_option == 0:  # Search
            return lock_quality > 0.7  # Lock acquired
        elif self.active_option == 1:  # Track
            return range_to_target < 500 or lock_quality < 0.3  # Terminal or lost lock
        elif self.active_option == 2:  # Terminal
            return False  # Runs until episode end

        return False
```

### Training Pipeline

```bash
# Step 1: Pre-train specialists (parallel)
python train_specialist.py --phase search --config scenarios/easy.yaml --steps 200000 &
python train_specialist.py --phase track --config scenarios/easy.yaml --steps 250000 &
python train_specialist.py --phase terminal --config scenarios/easy.yaml --steps 100000 &
wait

# Step 2: Train selector with frozen specialists
python train_hrl.py --freeze-specialists --config config.yaml --steps 200000

# Step 3: Fine-tune end-to-end (optional)
python train_hrl.py --finetune --config config.yaml --steps 200000

# Step 4: Evaluate full hierarchy
python inference.py --model checkpoints/hrl_best --mode offline --episodes 100
```

### Key Implementation Details

**Specialist Training:**
Each specialist is trained on a modified environment that:
- **Search**: Starts with no radar lock, ends when lock acquired (or timeout)
- **Track**: Starts with lock, random range 1000-3000m, ends at 500m or lock lost
- **Terminal**: Starts with lock at 500m, ends on impact/miss

**Selector Training:**
Uses specialists as fixed "actions." Reward shaping:
- +10 for each successful phase transition
- -5 for failed transition (e.g., losing lock)
- +5000 for mission success (intercept)

**Termination Conditions:**
Hard-coded based on observable state to ensure interpretability and safety.

---

## Conclusion

Hierarchical RL is not magic—it's structured problem decomposition. By exploiting the natural phases of missile intercept, we transform an intractable long-horizon problem into three tractable short-horizon problems plus one simple scheduling problem.

The key insight: **Interceptor phases are real physical regimes, not arbitrary abstractions.** Search, track, and terminal guidance are concepts from decades of missile engineering. HRL simply allows our AI to leverage this existing structure.

Expected outcome: 6-7x sample efficiency improvement, full interpretability, and robust performance across scenarios. The path from 5M steps (30 minutes) to 750k steps (4 minutes) while improving success rate from 75% to 85%+.
