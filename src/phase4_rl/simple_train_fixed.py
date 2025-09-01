#!/usr/bin/env python3
"""
Simple training script for the fixed 17D radar environment.
This bypasses the complex VecNormalize issues and trains a model directly.
"""

import os
import numpy as np
import torch
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from config import get_config, reset_config
from scenarios import get_scenario_loader, reset_scenario_loader  
from radar_env import RadarEnv


def make_env():
    """Create a single radar environment."""
    reset_config()
    reset_scenario_loader()
    
    config_loader = get_config()
    scenario_loader = get_scenario_loader()
    scenario_config = scenario_loader.create_environment_config('easy', config_loader._config)
    
    return RadarEnv(
        config=scenario_config,
        scenario_name='easy', 
        num_missiles=1,
        num_interceptors=1,
        render_mode=None
    )


def train_simple_model(timesteps=10000, checkpoint_dir="checkpoints_simple_fixed"):
    """Train a simple model without VecNormalize complications."""
    
    print(f"🚀 Starting simple training with fixed reward and action systems")
    print(f"   Timesteps: {timesteps}")
    print(f"   Checkpoint dir: {checkpoint_dir}")
    
    # Create directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)
    
    # Create environment
    env = DummyVecEnv([make_env])
    
    # Test environment
    obs = env.reset()
    print(f"✅ Environment created successfully")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Action space: {env.action_space}")
    
    # Create model
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=1,
        device='cpu',  # Use CPU to avoid GPU warnings
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(checkpoint_path / "tensorboard")
    )
    
    print(f"✅ PPO model created successfully")
    
    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
        save_path=str(checkpoint_path),
        name_prefix="fixed_model"
    )
    
    print(f"🏋️  Starting training...")
    
    # Train the model
    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    print(f"✅ Training completed!")
    
    # Save final model
    final_path = checkpoint_path / "fixed_model_final.zip"
    model.save(final_path)
    print(f"   Final model saved: {final_path}")
    
    # Test the trained model
    print(f"\n🧪 Testing trained model...")
    
    obs = env.reset()
    total_reward = 0
    
    for step in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        
        if done[0]:
            print(f"   Episode finished at step {step}")
            break
    
    print(f"   Test episode reward: {total_reward:.2f}")
    
    return model, str(final_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple training with fixed systems')
    parser.add_argument('--timesteps', type=int, default=10000, help='Training timesteps')
    parser.add_argument('--checkpoint-dir', default='checkpoints_simple_fixed', help='Checkpoint directory')
    
    args = parser.parse_args()
    
    model, model_path = train_simple_model(
        timesteps=args.timesteps,
        checkpoint_dir=args.checkpoint_dir
    )
    
    print(f"\n🎉 Training successful!")
    print(f"   Model saved at: {model_path}")
    print(f"   Ready for inference testing!")