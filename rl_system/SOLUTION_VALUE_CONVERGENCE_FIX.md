# Solution: Value Function Convergence & Success Rate Tracking

**Date**: 2025-10-21
**Issue**: Model plateaued at 33% success rate after 15M steps
**Root Cause**: Poor value function convergence (loss 9,030 vs target <100) + blind training without success metrics

## Changes Applied

### 1. Fixed Success Rate Tracking (train.py:203-259)

**Problem**: `EpisodeLoggingCallback` was not properly handling VecEnv structure
- Only checked `dones[0]` instead of iterating through all environments
- Did not verify 'episode' dict exists (added by Monitor wrapper only on episode end)

**Solution**:
```python
def _on_step(self) -> bool:
    dones = self.locals.get('dones', [])
    infos = self.locals.get('infos', [])

    # Iterate through ALL environments
    for i, (done, info) in enumerate(zip(dones, infos)):
        if done and 'episode' in info:  # Monitor wrapper check
            intercepted = info.get('intercepted', False)
            self.interceptions.append(1.0 if intercepted else 0.0)

            # Log immediately to TensorBoard
            self.tb_writer.add_scalar('episode/intercepted', ...)
            self.tb_writer.add_scalar('episode/reward', ...)

            # Log rolling success rate every N episodes
            if self.episode_count % self.log_interval == 0:
                success_rate = np.mean(self.interceptions[-100:]) * 100
                self.tb_writer.add_scalar('episode/success_rate_pct', success_rate, ...)
```

**Impact**:
- ✓ Now tracks EVERY episode completion across all 16 parallel environments
- ✓ Logs individual intercepts to TensorBoard in real-time
- ✓ Computes rolling 100-episode success rate
- ✓ Provides visibility into training progress

**TensorBoard Metrics Added**:
- `episode/intercepted`: Binary 0/1 for each episode
- `episode/reward`: Episode cumulative reward
- `episode/success_rate_pct`: Rolling 100-episode success rate

### 2. Increased Value Function Learning (config.yaml:131)

**Problem**: Value loss stuck at 9,030 (90x higher than target <100)
- Value function not converging properly
- Poor value estimates → poor advantage estimates → suboptimal policy gradients
- Bimodal behavior (perfect or catastrophic, no middle ground)

**Solution**:
```yaml
vf_coef: 1.0  # INCREASED from 0.5
```

**Impact**:
- ✓ Doubles gradient updates for value function
- ✓ Forces network to prioritize value accuracy
- ✓ Should reduce value loss toward target <100
- ✓ Better advantage estimates → better policy learning

**Tradeoff**:
- Policy learning may be slightly slower
- But value convergence is CRITICAL for PPO performance
- Better to train longer with good value function than fast with poor one

## Expected Results

### Immediate (within 1M steps):
1. **TensorBoard will show new metrics**:
   - `episode/intercepted` will show 0/1 pattern
   - `episode/success_rate_pct` will show rolling average
   - Can visually track learning progress

2. **Value loss should decrease faster**:
   - Target: Drop from 9,030 → <1,000 within 2M steps
   - Ultimate target: <100 by 10M steps

### Medium-term (5-10M steps):
1. **Success rate should break 33% plateau**:
   - With better value estimates, policy can learn to course-correct
   - Should see gradual improvement: 33% → 40% → 50%+

2. **Reduced bimodal behavior**:
   - More "close misses" in 500-1400m range
   - Shows policy learning to recover from errors
   - Fewer catastrophic failures at >2000m

### Long-term (15M+ steps):
1. **Target success rate: 75-85%**:
   - With properly converged value function
   - With full curriculum completion
   - With visible success metrics for validation

## Validation Steps

### Test 1: Short Training Run (100k steps)
**Purpose**: Verify callback is working before committing to full 15M run

```bash
cd rl_system

# Temporarily reduce total_timesteps in config.yaml
# total_timesteps: 100000  # Just for testing

python train.py --config config.yaml

# Check TensorBoard
tensorboard --logdir logs/
# Navigate to: http://localhost:6006
# Verify these metrics exist:
#   - episode/intercepted
#   - episode/reward
#   - episode/success_rate_pct
```

**Expected**: Should see metrics after ~50k steps (first few hundred episodes)

**If metrics missing**: Callback still has issues - investigate further

**If metrics present**: ✓ Proceed to full training

### Test 2: Full Training (15M steps)
```bash
cd rl_system
python train.py --config config.yaml
```

**Monitor during training**:
1. **Value loss trajectory** (should decrease steadily)
   - Target: <1,000 by 2M steps
   - Target: <500 by 5M steps
   - Target: <100 by 10M steps

2. **Success rate progression** (should increase gradually)
   - Baseline: ~33% initially (from previous training)
   - Target: >40% by 5M steps
   - Target: >60% by 10M steps
   - Target: >75% by 15M steps

3. **Reward progression** (should increase and stabilize)
   - Already at ~33k (good)
   - Should remain stable or increase slightly

**If value loss not decreasing**:
- Consider increasing vf_coef to 2.0
- Or implement separate networks: `net_arch=dict(pi=[128,128], vf=[256,256])`

**If success rate not improving**:
- Value function may need more time
- Extend training to 20M steps
- Or freeze curriculum at easy difficulty until value converges

## Code Changes Summary

### Modified Files
1. **train.py** (lines 203-259):
   - Fixed `EpisodeLoggingCallback._on_step()` to iterate all VecEnv environments
   - Added check for Monitor 'episode' dict
   - Added immediate TensorBoard logging for each episode
   - Added rolling success rate calculation

2. **config.yaml** (line 131):
   - Increased `vf_coef` from 0.5 to 1.0
   - Added comment explaining the change

### No Changes Required
- environment.py: Already correctly passes 'intercepted' in info dict
- core.py: Observation generation working correctly
- Reward function: Already simplified to 2-phase
- Early termination: Smart termination working correctly

## Diagnosis Summary

The model is **not broken** - training infrastructure and environment are working.

**Real issues**:
1. ✗ Value function not converged (loss 90x too high)
2. ✗ No visibility into success rate during training
3. ✓ Smart early termination working correctly
4. ✓ Inference system accurate
5. ✓ Reward function appropriate
6. ~ Curriculum possibly too aggressive (but can't verify without metrics)

**After fixes**:
1. ✓ Success rate visible in TensorBoard
2. ✓ Value function gets 2x more gradient updates
3. Next training run will have full visibility
4. Can make data-driven decisions about curriculum timing

## Next Steps

1. **Run short test** (100k steps) to verify callback fixes
2. **If test passes**: Start full 15M step training run
3. **Monitor TensorBoard** during training:
   - Check `episode/success_rate_pct` every ~1M steps
   - Ensure `train/value_loss` is decreasing
4. **If value loss stuck**: Increase vf_coef to 2.0 or use separate networks
5. **If success rate plateaus again**: Extend training or freeze curriculum

## Risk Assessment

**Low risk**:
- Callback fix is straightforward (proper VecEnv iteration)
- vf_coef increase is standard PPO tuning
- Both changes are reversible

**Medium risk**:
- Higher vf_coef may slow policy learning
- Mitigation: Monitor both value_loss and success_rate

**No risk**:
- Smart early termination already working
- Environment and reward function unchanged
- Previous checkpoints still available if needed
