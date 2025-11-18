# What Actually Happened: HRL "Breakthrough" Post-Mortem

## TL;DR

**We measured the wrong thing and celebrated fake results.**

- **Claimed**: 0.05m (5cm) precision, 100% success rate, "500x better than PPO"
- **Reality**: ~200m precision, 10% success rate, **WORSE than PPO** (which got ~150m)

## The Bug

### What We Did Wrong

```python
# In evaluate_hrl.py (WRONG):
rel_pos = obs[0:3]  # First 3 elements of observation
distance = np.linalg.norm(rel_pos)
```

### Why It's Wrong

The observation `obs[0:3]` contains **NORMALIZED** relative position:
```python
# From core.py line 708:
obs[0:3] = np.clip(radar_rel_pos / self.max_range, -1.0, 1.0)
#                                  ^^^^^^^^^^^^^^
#                                  Division by 10,000m!
```

So when `obs[0:3] = [0.05, 0.03, 0.02]`:
- We thought: "0.06m distance! Amazing!"
- Reality: `0.06 * 10,000m = 600m distance`

### The Smoking Gun

Checked actual logged positions:
```
Interceptor: [460.20, 414.13, 62.76]
Missile:     [537.02, 591.67, 112.22]
Actual Distance: 199.67 meters (NOT 0.01m!)
```

## Why Nobody Caught It

### 1. Confirmation Bias
- **Wanted** HRL to work after months of PPO struggles
- Results seemed "too good to be true" → accepted anyway
- Cherry-picked positive interpretations

### 2. No Sanity Checks
- Never plotted actual trajectories
- Never verified against world coordinates
- Never asked "can a missile physically get that close?"

### 3. Plausible Code
- The distance calculation *looked* reasonable
- Comment said "first 3 elements are relative position" (technically true!)
- Didn't specify they were NORMALIZED

### 4. Supporting "Evidence"
- 100% success rate seemed to confirm good performance
- But success was measured at 150m radius (easy to hit at 200m distance)
- Option switching was happening (SEARCH→TRACK→TERMINAL)
- Made it seem like the system was working

## Additional Problems Found

### Frame-Stacking Mismatch

**Training**: Specialists trained with 104D observations (26D × 4 frames)
**Evaluation**: Running with 26D observations (no frame-stacking)
**Result**: Can't even load the trained specialists!

Been running evaluation with **stub policies** (random actions) the whole time.

### Specialists May Not Have Learned

Even if we fix frame-stacking:
- No evidence specialists learned useful policies
- Training logs missing
- No individual specialist testing
- Reward decomposition might be wrong

## How To Actually Fix This

See `HRL_FIX_PLAN.md` for detailed steps, but summary:

### Step 1: Fix Measurement (DONE ✅)
```python
# Use actual world positions:
int_pos = info['interceptor_pos']
mis_pos = info['missile_pos']
distance = np.linalg.norm(int_pos - mis_pos)
```

### Step 2: Fix Frame-Stacking
Either:
- **A)** Retrain specialists without frame-stacking (faster)
- **B)** Fix evaluation to use frame-stacking (matches current models)

### Step 3: Validate Specialists
Test each specialist individually:
- Does SEARCH point toward target?
- Does TRACK maintain lock?
- Does TERMINAL close distance?

### Step 4: Retrain Properly
With validation at each step:
- Check training curves
- Test on held-out scenarios
- Compare to baseline (flat PPO)
- Plot actual trajectories

### Step 5: Never Lie Again
Add sanity checks:
```python
assert distance > 0.1, "Can't get closer than 10cm to a missile!"
assert distance < 50000, "Distance can't exceed initial separation!"
# ... etc
```

## Lessons Learned

### For Future Development

1. **Always validate against ground truth**
   - If results seem too good, they probably are
   - Check actual logged data, not just summary statistics
   - Plot trajectories, don't just trust numbers

2. **Add sanity checks everywhere**
   - Physical constraints (min/max distances)
   - Comparison to baseline
   - Cross-validation with multiple metrics

3. **Document units explicitly**
   - Is this meters? Normalized? Radians?
   - Add comments with expected ranges
   - Type hints for measurement units

4. **Test components individually**
   - Don't just test the whole system
   - Verify each piece works before integration
   - Specialists should be tested standalone

5. **Be skeptical of breakthroughs**
   - Real breakthroughs are rare
   - Usually results come from careful iteration
   - "500x better" is a red flag, not a celebration

## Current Status

### What Works
- ✅ HRL architecture is implemented
- ✅ Option switching happens
- ✅ Selector can choose between specialists
- ✅ Logging infrastructure captures data

### What's Broken
- ❌ Specialists may not have learned useful policies
- ❌ Frame-stacking mismatch prevents using trained models
- ❌ Performance worse than flat PPO
- ❌ No validation was done during development
- ❌ Evaluation was measuring wrong metric

### What's Fixed
- ✅ Distance calculation now uses actual world positions
- ✅ We know the real performance (~200m)
- ✅ We understand what went wrong
- ✅ We have a plan to fix it

## Next Steps

1. Run evaluation with fixed distance calc to confirm ~200m performance
2. Fix frame-stacking to actually load specialists
3. Test specialists individually
4. Decide: retrain from scratch or try to salvage?
5. Implement proper validation pipeline
6. Actually make HRL work (the hard part)

---

**Bottom Line**: We got excited about fake results and didn't do due diligence. Now we know better. Time to build it properly.
