#!/usr/bin/env python3
"""
Simple training script to test 17D system with minimal setup.
"""

import sys
import os
sys.path.insert(0, 'src/phase4_rl')

from fast_sim_env import FastSimEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def simple_training_test():
    """Test training with 17D observations for a few steps."""
    print("🚀 Testing simple 17D training...")
    
    # Create environment
    def make_env():
        return FastSimEnv(scenario_name="easy")
    
    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    print(f"✓ Environment created with observation space: {vec_env.observation_space.shape}")
    
    # Create PPO model
    model = PPO('MlpPolicy', vec_env, verbose=1, n_steps=64)
    print("✓ PPO model created")
    
    # Train for a few steps
    print("🔄 Training for 1000 timesteps...")
    model.learn(total_timesteps=1000)
    print("✅ Training completed successfully!")
    
    # Test inference
    obs = vec_env.reset()
    for i in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        print(f"Step {i+1}: reward={reward[0]:.3f}, obs_shape={obs.shape}")
        if done[0]:
            obs = vec_env.reset()
    
    print("✅ Inference test completed!")
    return True

if __name__ == "__main__":
    try:
        success = simple_training_test()
        print("\n🎉 17D system is ready for full training!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)