#!/usr/bin/env python3
"""
Quick test of the 17D observation fix.
"""

import sys
sys.path.insert(0, 'src/phase4_rl')

from fast_sim_env import FastSimEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def test_17d_fix():
    print("🔧 Testing 17D observation fix...")
    
    try:
        # Test single environment
        print("\n1. Testing single FastSimEnv...")
        env = FastSimEnv(scenario_name="easy")
        obs, _ = env.reset(seed=42)
        print(f"   ✓ Observation shape: {obs.shape}")
        assert obs.shape == (17,), f"Expected (17,), got {obs.shape}"
        
        # Test step
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info = env.step(action)
        print(f"   ✓ Step observation shape: {obs2.shape}")
        assert obs2.shape == (17,), f"Expected (17,), got {obs2.shape}"
        
        # Test vectorized environment
        print("\n2. Testing VecEnv...")
        def make_env():
            return FastSimEnv(scenario_name="easy")
        
        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        
        obs = vec_env.reset()
        print(f"   ✓ VecEnv observation shape: {obs.shape}")
        assert obs.shape == (1, 17), f"Expected (1, 17), got {obs.shape}"
        
        # Test vectorized step
        action = vec_env.action_space.sample()
        obs2, reward, done, info = vec_env.step(action)
        print(f"   ✓ VecEnv step observation shape: {obs2.shape}")
        assert obs2.shape == (1, 17), f"Expected (1, 17), got {obs2.shape}"
        
        print("\n✅ All tests passed! 17D observation system is working.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_17d_fix()
    sys.exit(0 if success else 1)