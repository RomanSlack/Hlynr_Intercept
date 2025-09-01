# 🚀 Quick Start - Fixed RL Training & Inference

## Train a Model
```bash
python simple_train_fixed.py --timesteps 10000 --checkpoint-dir my_model
```
 .
## Run Inference
```bash
python simple_inference.py my_model/fixed_model_final.zip --episodes 5 --output-dir results
```

## What's Fixed
- ✅ 17D radar observations working
- ✅ Reward function fixed (positive rewards for good behavior)
- ✅ Action interpretation fixed (world-frame thrust)
- ✅ JSON output for Unity visualization

## Files Generated
- **Training:** `my_model/fixed_model_final.zip`
- **Inference:** `results/episode_*.jsonl` + `results/summary.json`

That's it! 🎯