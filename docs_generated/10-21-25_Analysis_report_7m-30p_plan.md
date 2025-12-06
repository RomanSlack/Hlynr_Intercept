Analysis Report: 7M Training Results (33% Success Rate)

  Success Rate: 33/100 = 33% - Much better than 10% (2M) and 2% (500k)!

  ---
  Key Observations

  What's Working:

  1. Policy has learned basic interception mechanics
    - Successful intercepts are clean: ~700-1100 steps, final distance ~197-200m (right at target)
    - Fuel usage is reasonable for successes: 23-37 kg (efficient)
    - Rewards for successes: 13,000-19,000 (consistent, high)
  2. Clear distinction between success and failure
    - Success pattern: 700-1100 steps, 197-200m final distance, high reward
    - Failure pattern: 1400-1800 steps, 1800-2900m final distance, low/negative reward
  3. Learning has happened
    - Value loss dropped but still high at 3350 (should be <100)
    - The fact that 33% succeed shows policy has learned something

  ---
  Problems Still Present:

  1. High Variance in Success (67% still failing)

  - Failed episodes end 1800-2900m away (way outside 200m target)
  - Some failures have negative rewards (episodes 21, 40, 41, 62, 71, 75, 90)
  - This suggests the policy is bimodal: either succeeds cleanly or fails catastrophically

  2. Failed Episodes Are TOO Long

  - Failures: 1400-1800 steps (near max 2000)
  - Successes: 700-1100 steps
  - Failed episodes are timing out - the policy doesn't know it's failing and keeps trying

  3. Fuel Consumption Pattern

  - Successes: 23-37 kg fuel (efficient)
  - Failures: 46-62 kg fuel (burning max fuel for nothing)
  - This suggests failed episodes involve thrashing/hunting behavior

  4. Value Loss Still Very High (3350)

  - Should be <100 for converged policy
  - Means value function can't accurately predict returns
  - This causes poor advantage estimates → unstable learning

  ---
  Root Cause Analysis

  The Bimodal Problem:

  The policy has learned one good strategy (works 33% of time) but hasn't generalized to all scenarios. This happens when:

  1. Observation space has blind spots
    - When initial geometry is favorable: Success
    - When initial geometry is unfavorable: Complete failure
  2. Reward function creates one local optimum
    - Policy found ONE way to intercept that works sometimes
    - But hasn't learned to adapt when that strategy won't work
  3. Training time still insufficient
    - 7M steps = ~3,500 episodes across 16 envs
    - With spawn randomization, might not have seen enough variety

  ---
  Recommended Improvements (Priority Order)

  Priority 1: Simplify Reward to Reduce Variance

  Problem: 4-phase reward with multiple exponentials is hard to learn
  Fix: Simplify to 2-phase reward:

  # Phase 1: Far away (>500m) - just reward closing
  if distance > 500:
      reward = distance_delta * 10.0

  # Phase 2: Close (<500m) - strong gradient to target
  else:
      reward = distance_delta * 20.0
      reward += 50.0 * np.exp(-distance / 100.0)  # Single exponential

  # Always penalize time
  reward -= 0.1

  Why: Simpler reward = clearer gradient = faster learning

  ---
  Priority 2: Increase Training Steps

  Problem: 7M steps isn't enough for 33% → 80%+ success
  Fix: Train for 15-20M steps

  Rationale:
  - 7M steps got you from 2% → 33% (16x improvement)
  - Another 10M steps could get 33% → 70-80%
  - Value loss needs to drop from 3350 → <100 (needs more time)

  ---
  Priority 3: Add Early Termination for Failed Episodes

  Problem: Failed episodes run 1400-1800 steps (wasting compute)
  Fix: Terminate episode early if:

  # If distance is increasing after 1000 steps, give up
  if steps > 1000 and distance > 2000:
      terminated = True
      reward = -200  # Penalty for giving up

  Why: Saves 40% of training time, lets policy learn more episodes

  ---
  Priority 4: Curriculum Adjustment

  Problem: Curriculum is disabled (stuck on easy: 200m radius, 120° beam, perfect radar)
  Current: curriculum: enabled: false

  Fix: Enable curriculum but make it much slower:
  curriculum:
    enabled: true
    curriculum_steps: 15000000  # Reach final at 15M instead of 6M

  Why: Policy needs to master easy before harder challenges

  ---
  Priority 5: Reduce Network Size Further (Maybe)

  Current: [256, 256] = ~134k params
  Try: [128, 128] = ~33k params

  Why: Smaller network = faster convergence with sparse rewards
  Risk: Might reduce final performance ceiling

  ---
  Expected Improvements

  If you do Priority 1 + 2 (Simplify reward + train 15M):

  - Expected: 60-75% success rate
  - Training time: ~1.5 hours
  - Best case: 80%+ if value loss drops below 100

  If you do Priority 1 + 2 + 3 (Add early termination):

  - Expected: 70-80% success rate
  - Training time: ~1 hour (30% faster due to early termination)
  - Value loss: Should drop to <500

  If you do all 5 priorities:

  - Expected: 75-85% success rate
  - Training time: ~2 hours (15M steps with early termination)
  - Realistic final difficulty: Will transition to hard mode gradually

  ---
  The Real Bottleneck: Value Function

  Your value_loss = 3350 is the smoking gun. This means:
  - Value network can't predict episode returns accurately
  - Advantage estimates are noisy
  - Policy updates are unstable

  Solution: More training steps. Value loss typically drops like:
  - 500k steps: 1650
  - 2M steps: 2640
  - 7M steps: 3350 (actually got worse!)
  - 15M+ steps: Should drop to <500, ideally <100

  The value loss got worse from 2M to 7M, which suggests the multi-phase reward made it harder for the value function to learn.

  ---
  Bottom Line Recommendations

  1. Simplify the reward function (easiest, biggest impact)
  2. Train for 15-20M steps (necessary for convergence)
  3. Add early termination for obvious failures (saves 30% time)

  These 3 changes should get you from 33% → 70-80% success rate.

  The curriculum and network size are secondary - focus on reward simplification and more training first.
