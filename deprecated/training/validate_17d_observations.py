#!/usr/bin/env python3
"""
Validate that the 17-dimensional observation system is working correctly.
"""

import numpy as np
from fast_sim_env import FastSimEnv
from radar_env import RadarEnv


def validate_observations():
    """Validate that observations are consistently 17-dimensional."""
    print("=" * 60)
    print("17D Observation System Validation")
    print("=" * 60)
    
    # Test FastSimEnv
    print("\n1. Testing FastSimEnv...")
    try:
        env = FastSimEnv(scenario="easy")
        obs, _ = env.reset(seed=42)
        
        print(f"   ✓ Observation shape: {obs.shape}")
        assert obs.shape == (17,), f"Expected shape (17,), got {obs.shape}"
        print(f"   ✓ Observation dtype: {obs.dtype}")
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
        
        # Check observation bounds
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0), "Observations out of bounds [-1, 1]"
        print(f"   ✓ Observations within bounds: [{obs.min():.3f}, {obs.max():.3f}]")
        
        # Take a step
        action = env.action_space.sample()
        obs2, _, _, _, _ = env.step(action)
        assert obs2.shape == (17,), f"Step observation shape mismatch: {obs2.shape}"
        print(f"   ✓ Step observation shape: {obs2.shape}")
        
        print("   ✅ FastSimEnv validation PASSED")
    except Exception as e:
        print(f"   ❌ FastSimEnv validation FAILED: {e}")
        return False
    
    # Test RadarEnv
    print("\n2. Testing RadarEnv...")
    try:
        env = RadarEnv(num_missiles=1, num_interceptors=1)
        obs, _ = env.reset(seed=42)
        
        print(f"   ✓ Observation shape: {obs.shape}")
        assert obs.shape == (17,), f"Expected shape (17,), got {obs.shape}"
        print(f"   ✓ Observation dtype: {obs.dtype}")
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"
        
        # Check observation bounds
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0), "Observations out of bounds [-1, 1]"
        print(f"   ✓ Observations within bounds: [{obs.min():.3f}, {obs.max():.3f}]")
        
        print("   ✅ RadarEnv validation PASSED")
    except Exception as e:
        print(f"   ❌ RadarEnv validation FAILED: {e}")
        return False
    
    # Test observation consistency
    print("\n3. Testing observation consistency...")
    try:
        env1 = FastSimEnv(scenario="easy")
        env2 = RadarEnv(num_missiles=1, num_interceptors=1)
        
        # Reset both with same seed
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        
        print(f"   ✓ FastSimEnv obs shape: {obs1.shape}")
        print(f"   ✓ RadarEnv obs shape: {obs2.shape}")
        assert obs1.shape == obs2.shape == (17,), "Shape mismatch between environments"
        
        print("   ✅ Consistency validation PASSED")
    except Exception as e:
        print(f"   ❌ Consistency validation FAILED: {e}")
        return False
    
    # Test with VecNormalize
    print("\n4. Testing VecNormalize compatibility...")
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        
        def make_env():
            return FastSimEnv(scenario="easy")
        
        vec_env = DummyVecEnv([make_env])
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        
        obs = vec_env.reset()
        print(f"   ✓ VecNormalize observation shape: {obs.shape}")
        assert obs.shape == (1, 17), f"Expected shape (1, 17), got {obs.shape}"
        
        # Take a step
        action = vec_env.action_space.sample()
        obs, _, _, _ = vec_env.step(action)
        assert obs.shape == (1, 17), f"Step observation shape mismatch: {obs.shape}"
        
        print("   ✅ VecNormalize compatibility PASSED")
    except Exception as e:
        print(f"   ❌ VecNormalize compatibility FAILED: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL VALIDATIONS PASSED - 17D OBSERVATION SYSTEM READY")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys
    success = validate_observations()
    sys.exit(0 if success else 1)