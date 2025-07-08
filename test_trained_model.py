#!/usr/bin/env python3
"""
Test script for evaluating trained AegisIntercept models.

This script loads a trained model and runs evaluation episodes to test performance.
"""

import argparse
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import gymnasium as gym

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from aegis_intercept.envs.aegis_6dof_env import Aegis6DOFEnv


def dynamic_reset(env):
    """Handle both old and new Gym API reset returns."""
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
        return obs, info
    else:
        return reset_result, {}


def dynamic_step(env, action):
    """Handle both old (4-value) and new (5-value) Gym API step returns."""
    step_result = env.step(action)
    
    # Handle vectorized env info (list of dicts) vs single env info (dict)
    def process_info(info):
        if isinstance(info, list):
            # Vectorized env returns list of info dicts, take the first one
            return info[0] if len(info) > 0 else {}
        return info
    
    if len(step_result) == 4:
        # Old API: obs, reward, done, info
        obs, reward, done, info = step_result
        info = process_info(info)
        return obs, reward, done, done, info  # terminated = truncated = done
    elif len(step_result) == 5:
        # New API: obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = step_result
        info = process_info(info)
        return obs, reward, terminated, truncated, info
    else:
        raise ValueError(f"Unexpected step return format: {len(step_result)} values")


def test_model(model_path: str, vec_normalize_path: str = None, num_episodes: int = 10):
    """Test a trained model."""
    
    print(f"Loading model from: {model_path}")
    
    # Create environment
    env = Aegis6DOFEnv()
    
    # Load VecNormalize if available
    if vec_normalize_path and Path(vec_normalize_path).exists():
        print(f"Loading VecNormalize from: {vec_normalize_path}")
        # Wrap single environment in DummyVecEnv for VecNormalize compatibility
        env = DummyVecEnv([lambda: env])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = True   # Keep training mode for statistics matching
        env.norm_reward = False  # Don't normalize rewards during evaluation
    
    # Load trained model
    model = PPO.load(model_path)
    
    # Test the model
    results = []
    print(f"\nTesting model for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        # Reset with different seed for variety
        if vec_normalize_path and Path(vec_normalize_path).exists():
            # For VecNormalize, we need to seed the underlying environment
            # Use a different approach - just call reset and set numpy seed manually
            np.random.seed(episode * 42)
            obs, info = dynamic_reset(env)
        else:
            # For regular environment, seed directly
            obs, info = env.reset(seed=episode * 42)
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 1000:
            # Use deterministic=False for some randomness in evaluation
            action, _ = model.predict(obs, deterministic=(episode % 2 == 0))
            obs, reward, terminated, truncated, info = dynamic_step(env, action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        
        success = info.get('intercept_distance', float('inf')) < 20.0  # 20m kill distance
        
        results.append({
            'episode': episode + 1,
            'reward': total_reward,
            'steps': steps,
            'success': success,
            'final_distance': info.get('intercept_distance', 0),
            'fuel_remaining': info.get('fuel_remaining', 0)
        })
        
        status = "SUCCESS" if success else "FAILED"
        
        # Convert NumPy arrays/scalars to Python floats for formatting
        raw_reward = total_reward
        if hasattr(raw_reward, 'item'):
            reward = raw_reward.item()
        elif isinstance(raw_reward, np.ndarray):
            reward = float(raw_reward.flat[0])
        else:
            reward = float(raw_reward)
        
        raw_dist = info.get('intercept_distance', 0)
        if hasattr(raw_dist, 'item'):
            dist = raw_dist.item()
        elif isinstance(raw_dist, np.ndarray):
            dist = float(raw_dist.flat[0])
        else:
            dist = float(raw_dist)
        
        print(f"Episode {episode+1:2d}: {status} | Reward: {reward:6.2f} | Steps: {steps:3d} | Distance: {dist:6.1f}m")
    
    # Calculate statistics
    success_rate = sum(r['success'] for r in results) / len(results) * 100
    avg_reward = np.mean([r['reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    avg_distance = np.mean([r['final_distance'] for r in results])
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Episodes:     {num_episodes}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Avg Reward:   {avg_reward:.2f}")
    print(f"Avg Steps:    {avg_steps:.1f}")
    print(f"Avg Distance: {avg_distance:.1f}m")
    print(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test trained AegisIntercept model')
    parser.add_argument('--model', type=str, help='Path to trained model (.zip file)')
    parser.add_argument('--vec-normalize', type=str, help='Path to VecNormalize stats (.pkl file)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--checkpoint-dir', type=str, help='Load from checkpoint directory (auto-finds model and vec_normalize)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.checkpoint_dir and not args.model:
        parser.error("Either --checkpoint-dir or --model must be provided")
    
    if args.checkpoint_dir:
        # Auto-find files in checkpoint directory
        checkpoint_path = Path(args.checkpoint_dir)
        model_path = checkpoint_path / "model.zip"
        vec_normalize_path = checkpoint_path / "vec_normalize.pkl"
        
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            return 1
        
        test_model(str(model_path), str(vec_normalize_path) if vec_normalize_path.exists() else None, args.episodes)
    else:
        test_model(args.model, args.vec_normalize, args.episodes)
    
    return 0


if __name__ == "__main__":
    exit(main())