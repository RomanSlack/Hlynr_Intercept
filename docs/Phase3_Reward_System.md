# Phase 3 Reward System Documentation

## Overview

The Phase 3 reward system implements a sophisticated sparse reward structure designed to guide reinforcement learning agents through complex 6DOF missile intercept scenarios. The system balances mission success rewards with progressive guidance signals while maintaining compatibility with the curriculum learning framework.

## Reward Architecture

### Dual-Mode Design

The reward system operates in two distinct modes to support curriculum progression:

1. **3DOF Compatibility Mode** (`_step_3dof_mode()` - lines 399-463)
2. **6DOF Advanced Mode** (`_step_6dof_mode()` - lines 465-540)

This dual approach enables seamless transition from simple 3DOF scenarios to complex 6DOF missions within the curriculum framework.

## 3DOF Mode Rewards

### Termination-Based Rewards

Located in `aegis_6dof_env.py`, lines 452-462:

#### Success Rewards
```python
if intercepted:
    reward = 15.0 + (self.fuel_remaining / self.max_fuel) * 5.0
    self.episode_stats['intercept_method'] = 'explosion' if exploded else 'proximity'
```

**Components:**
- **Base success reward**: +15.0 points
- **Fuel efficiency bonus**: Up to +5.0 points based on remaining fuel
- **Total possible**: +20.0 points maximum

#### Failure Penalties
```python
elif missile_hit:
    reward = -8.0
elif out_of_bounds:
    reward = -2.0
```

**Penalty Structure:**
- **Mission failure** (missile hits target): -8.0 points
- **Boundary violation**: -2.0 points

#### Progressive Guidance
```python
else:
    # Distance-based reward
    reward = max(0, 1.0 - intercept_dist / 200.0) - 0.02
```

**Progressive Signal:**
- **Distance reward**: 0 to +1.0 based on proximity to missile
- **Time penalty**: -0.02 per step (encourages efficiency)

## 6DOF Mode Rewards

### Advanced Reward Calculation

The 6DOF reward system (`_calculate_6dof_reward()` - lines 647-716) implements a multi-faceted approach:

### Success Rewards (Intercepted)

```python
if intercepted:
    base_reward = 20.0
    fuel_bonus = (self.fuel_remaining / self.max_fuel) * 8.0
    time_bonus = (self.max_steps - self.step_count) / self.max_steps * 5.0
    method_bonus = 3.0 if exploded else 0.0
```

**Success Components:**
- **Base reward**: +20.0 points
- **Fuel efficiency bonus**: Up to +8.0 points
- **Time efficiency bonus**: Up to +5.0 points  
- **Method bonus**: +3.0 for explosion intercepts
- **Precision bonus**: +5.0 for close intercepts (<10m), +2.0 for medium (<20m)

**Maximum possible success reward**: +41.0 points

### Failure Penalties

```python
elif missile_hit:
    return -15.0
elif structural_failure:
    return -8.0  
elif out_of_bounds:
    return -5.0
```

**Failure Structure:**
- **Mission failure**: -15.0 points (increased severity from 3DOF)
- **Structural failure**: -8.0 points (new constraint for 6DOF)
- **Boundary violation**: -5.0 points

### Progressive Guidance Signals

#### Distance-Based Rewards
```python
max_dist = 600.0
distance_reward = max(0, 2.0 * (1.0 - intercept_dist / max_dist))
```

**Distance Signal:**
- **Range**: 0 to +2.0 points
- **Function**: Linear decay with distance
- **Purpose**: Guides interceptor toward missile

#### Intercept Geometry Rewards
```python
geometry = intercept_geometry_6dof(
    self.interceptor_6dof.position, self.interceptor_6dof.velocity,
    self.missile_6dof.position, self.missile_6dof.velocity
)

if geometry['closing_velocity'] > 0:
    closing_reward = min(1.0, geometry['closing_velocity'] / 50.0)
else:
    closing_reward = -0.5

aspect_reward = 0.5 * (1.0 - geometry['aspect_angle'] / np.pi)
```

**Geometry Components:**
- **Closing velocity reward**: 0 to +1.0 for positive closing rates
- **Closing velocity penalty**: -0.5 for diverging trajectories
- **Aspect angle reward**: 0 to +0.5 favoring head-on intercepts

#### Efficiency Penalties
```python
fuel_penalty = -0.1 * (1.0 - self.fuel_remaining / self.max_fuel)
time_penalty = -0.03
```

**Continuous Penalties:**
- **Fuel consumption**: 0 to -0.1 based on fuel used
- **Time pressure**: -0.03 per step

## Sparse Reward Structure

### Design Philosophy

The reward system implements controlled sparsity to promote robust learning:

#### Terminal Rewards (Dense)
- **High-value** success/failure signals at episode termination
- **Clear differentiation** between success and failure modes
- **Bonus structures** rewarding efficiency and precision

#### Progressive Signals (Sparse)
- **Limited intermediate rewards** to avoid local optima
- **Geometry-based guidance** only when meaningful
- **Penalty signals** for constraint violations

### Sparsity Levels by Phase

#### Phase 1-2 (3DOF)
- **Primary signal**: Distance-based continuous reward
- **Sparsity level**: Medium (continuous distance feedback)
- **Purpose**: Basic guidance for initial learning

#### Phase 3-5 (6DOF)  
- **Primary signal**: Terminal success/failure rewards
- **Secondary signals**: Sparse geometry and efficiency feedback
- **Sparsity level**: High (minimal intermediate rewards)
- **Purpose**: Robust policy development

## Curriculum Integration

### Phase-Dependent Scaling

The curriculum manager influences reward calculation through environment parameters:

#### Threshold Scaling
```python
# From curriculum_manager.py lines 254-321
"intercept_threshold": 35.0,  # Phase 1 - easier
"intercept_threshold": 30.0,  # Phase 2 - moderate  
"intercept_threshold": 25.0,  # Phase 3 - harder
"intercept_threshold": 20.0,  # Phase 4 - challenging
"intercept_threshold": 15.0,  # Phase 5 - expert
```

**Impact on Rewards:**
- **Larger thresholds** make success easier to achieve
- **Smaller thresholds** require more precise intercepts
- **Progressive tightening** increases difficulty

#### Success Rate Thresholds
```python
"success_rate_threshold": 0.70,  # Phase 1
"success_rate_threshold": 0.75,  # Phase 2
"success_rate_threshold": 0.65,  # Phase 3 (6DOF adjustment)
"success_rate_threshold": 0.70,  # Phase 4
"success_rate_threshold": 0.80,  # Phase 5
```

**Advancement Logic:**
- **Performance-based progression** ensures readiness
- **Temporary reduction** in Phase 3 accommodates 6DOF complexity
- **Increasing requirements** in advanced phases

### Dynamic Difficulty Adjustment

The curriculum system modifies reward-relevant parameters based on performance:

#### Difficulty Increase (lines 526-545)
```python
def _increase_difficulty(self, config: PhaseConfig):
    config.intercept_threshold = max(10.0, config.intercept_threshold * 0.95)
    config.wind_strength = min(2.0, config.wind_strength * 1.1)
    config.max_steps = max(200, int(config.max_steps * 0.98))
```

#### Difficulty Decrease (lines 547-566)
```python
def _decrease_difficulty(self, config: PhaseConfig):
    config.intercept_threshold = min(50.0, config.intercept_threshold * 1.05)
    config.wind_strength = max(0.1, config.wind_strength * 0.9)
    config.max_steps = min(600, int(config.max_steps * 1.02))
```

## Reward Signal Analysis

### Signal Composition by Scenario

#### Early Training (Phase 1-2)
- **Dominant signal**: Distance-based continuous feedback (+1.0 max)
- **Secondary signal**: Terminal success/failure (+20.0/-8.0)
- **Purpose**: Rapid initial learning with guidance

#### Advanced Training (Phase 4-5)
- **Dominant signal**: Terminal success/failure (+41.0/-15.0)
- **Secondary signals**: Sparse geometry feedback (+1.5 max)
- **Purpose**: Robust policy without over-guidance

### Temporal Reward Distribution

#### Episode Timeline
1. **Early steps**: Small negative time penalty (-0.03/step)
2. **Mid-episode**: Distance and geometry rewards (0-3.0 total)
3. **Terminal**: Large success/failure signal (±15.0 to ±41.0)

#### Learning Phases
- **Exploration phase**: Relies on sparse terminal signals
- **Refinement phase**: Utilizes geometry and efficiency feedback
- **Mastery phase**: Optimizes for maximum terminal rewards

## Implementation Details

### Fuel System Integration

```python
if self.enable_fuel_system:
    fuel_consumed = np.linalg.norm(thrust_force) / 1000.0 * self.fuel_burn_rate * self.dt
    self.fuel_remaining = max(0.0, self.fuel_remaining - fuel_consumed)
```

**Fuel Constraints:**
- **Consumption rate**: Proportional to thrust magnitude
- **Depletion effect**: Zero thrust when fuel exhausted
- **Reward impact**: Efficiency bonuses favor fuel conservation

### Structural Limits

```python
g_force = np.linalg.norm(thrust_force) / (self.interceptor_6dof.aero_props.mass * 9.81)
structural_failure = g_force > 20.0  # 20G limit
```

**G-Force Constraints:**
- **Limit**: 20G maximum acceleration
- **Failure mode**: Episode termination with -8.0 penalty
- **Learning impact**: Encourages realistic control strategies

### Boundary Conditions

#### 3DOF Boundaries
```python
out_of_bounds = (
    self.interceptor_pos_3d[2] > self.world_size * 2 or
    self.interceptor_pos_3d[0] < 0 or self.interceptor_pos_3d[0] > self.world_size * 2 or
    self.interceptor_pos_3d[1] < 0 or self.interceptor_pos_3d[1] > self.world_size * 2
)
```

#### 6DOF Boundaries  
```python
out_of_bounds = (
    self.interceptor_6dof.position[2] > self.world_size * 3 or
    self.interceptor_6dof.position[0] < -self.world_size or 
    self.interceptor_6dof.position[0] > self.world_size * 3 or
    self.interceptor_6dof.position[1] < -self.world_size or 
    self.interceptor_6dof.position[1] > self.world_size * 3
)
```

**Boundary Design:**
- **Larger 6DOF boundaries** accommodate higher complexity
- **Penalty consistency** maintains learning stability
- **Progressive scaling** with curriculum phases

## Performance Metrics

### Reward-Based Advancement

The curriculum system uses reward-derived metrics for phase progression:

#### Success Rate Calculation
```python
@property
def success_rate(self) -> float:
    if self.episodes_completed == 0:
        return 0.0
    return self.episodes_successful / self.episodes_completed
```

#### Average Reward Tracking
```python
@property  
def average_reward(self) -> float:
    if self.episodes_completed == 0:
        return 0.0
    return self.total_reward / self.episodes_completed
```

### Efficiency Metrics

#### Fuel Efficiency
```python
@property
def average_fuel_efficiency(self) -> float:
    if self.episodes_successful == 0:
        return 0.0
    return self.total_fuel_used / self.episodes_successful
```

#### Time Efficiency
```python
@property
def average_intercept_time(self) -> float:
    if self.episodes_successful == 0:
        return 0.0
    return self.total_time_to_intercept / self.episodes_successful
```

## Reward Optimization Strategies

### Hyperparameter Tuning

#### Success Reward Scaling
- **Base reward magnitude**: Balances with continuous signals
- **Bonus structure**: Provides optimization targets beyond success
- **Penalty severity**: Discourages failure without over-penalization

#### Progressive Signal Tuning
- **Distance reward range**: Calibrated to environment scale
- **Geometry reward weights**: Balanced to avoid premature convergence
- **Time penalty rate**: Encourages efficiency without rushing

### Curriculum Adaptation

#### Phase-Specific Adjustments
- **Threshold progression**: Gradually increases precision requirements
- **Reward complexity**: Adds efficiency considerations in advanced phases
- **Sparsity control**: Reduces guidance as capabilities develop

## Common Patterns and Edge Cases

### Reward Exploitation Prevention

#### Distance Reward Clipping
```python
distance_reward = max(0, 2.0 * (1.0 - intercept_dist / max_dist))
```
**Prevents**: Negative rewards for large distances that could encourage avoidance

#### Geometry Reward Bounds
```python
closing_reward = min(1.0, geometry['closing_velocity'] / 50.0)
```
**Prevents**: Unbounded rewards for extreme closing velocities

### Learning Stability

#### Smooth Transitions
- **Consistent reward scales** across curriculum phases
- **Gradual threshold changes** in dynamic difficulty adjustment
- **Preserved reward structure** during phase transitions

#### Robustness Measures
- **Clipped action spaces** prevent extreme control inputs
- **Bounded reward ranges** maintain learning stability
- **Structured failure modes** provide clear learning signals

## Integration with Training Loop

### Episode-Level Updates

From `train_ppo_phase3_6dof.py` lines 382-385:
```python
if args.enable_curriculum:
    curriculum_manager.update_performance(
        episode_reward, episode_success, fuel_used, intercept_time
    )
```

**Curriculum Feedback:**
- **Reward accumulation** drives advancement decisions
- **Success detection** based on reward thresholds
- **Performance tracking** enables dynamic adjustment

### Tensorboard Logging

```python
writer.add_scalar("charts/episodic_return", final_return, global_step)
writer.add_scalar("charts/success_rate", rolling_success_rate, global_step)
```

**Monitoring Capabilities:**
- **Real-time reward tracking** for training analysis
- **Success rate visualization** for curriculum progression
- **Performance trend analysis** for hyperparameter optimization

## Conclusion

The Phase 3 reward system implements a sophisticated balance between sparse terminal rewards and progressive guidance signals. The dual-mode architecture supports seamless curriculum progression while the advanced 6DOF reward structure promotes robust learning in complex scenarios. The integration with the curriculum framework enables adaptive difficulty and performance-based advancement, resulting in improved learning efficiency and higher success rates compared to previous phases.